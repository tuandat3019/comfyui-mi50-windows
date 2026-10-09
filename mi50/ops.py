"""Inference-only math backends for the isolated MI50 TRELLIS profile."""
import math
import torch
REFERENCE_SDPA = torch.nn.functional.scaled_dot_product_attention
REFERENCE_BINCOUNT = torch.bincount
REFERENCE_UNIQUE = torch.unique


def metadata_bincount(input, weights=None, minlength=0):
    if input.device.type != 'cuda':
        return REFERENCE_BINCOUNT(input,weights,minlength)
    result = REFERENCE_BINCOUNT(input.cpu(),None if weights is None else weights.cpu(),minlength)
    return result.to(input.device)


def metadata_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if input.device.type != 'cuda' or dim is None:
        return REFERENCE_UNIQUE(input,sorted,return_inverse,return_counts,dim)
    result = REFERENCE_UNIQUE(input.cpu(),sorted,return_inverse,return_counts,dim)
    if isinstance(result,tuple):
        return tuple(value.to(input.device) for value in result)
    return result.to(input.device)


def bounded_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
    if dropout_p:
        raise ValueError('MI50 TRELLIS profile supports inference with dropout=0 only')
    if enable_gqa:
        if query.shape[-3] % key.shape[-3] or key.shape[-3] != value.shape[-3]:
            raise ValueError('Invalid GQA head counts')
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)
    leading = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2])
    length, source = query.shape[-2], key.shape[-2]
    result = torch.empty((*leading, length, value.shape[-1]), device=query.device, dtype=query.dtype)
    if length == 0:
        return result
    # Bound one FP32 attention buffer to 32 MiB, independent of token count.
    block = min(256, max(1, (32 * 1024**2) // (4 * max(1, math.prod(leading)) * max(1, source))))
    factor = (1 / math.sqrt(query.shape[-1])) if scale is None else scale
    kt = key.transpose(-2, -1)
    for start in range(0, length, block):
        stop = min(length, start + block)
        scores = ((query[..., start:stop, :] * factor) @ kt).float()
        if attn_mask is not None:
            mask = attn_mask if attn_mask.shape[-2] == 1 else attn_mask[..., start:stop, :]
            if mask.dtype == torch.bool:
                scores.masked_fill_(~mask, -float('inf'))
            else:
                scores.add_(mask)
        if is_causal:
            row = torch.arange(start, stop, device=query.device)[:, None]
            column = torch.arange(source, device=query.device)[None, :]
            scores.masked_fill_(column > row, -float('inf'))
        row_max = scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, 0.0)
        scores.sub_(row_max).exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True).clamp_min_(torch.finfo(torch.float32).tiny))
        result[..., start:stop, :] = scores.to(value.dtype) @ value
    return result


def verify_sdpa(device='cpu'):
    """Compare against PyTorch's CPU implementation, including empty masked rows."""
    reference = REFERENCE_SDPA
    generator = torch.Generator().manual_seed(621)
    dtype = torch.float16 if device != 'cpu' else torch.float32
    reports = []
    for length, source, causal, mask_kind, gqa in (
        (257, 1031, False, None, False),
        (37, 53, True, None, False),
        (11, 23, False, 'bool', False),
        (17, 31, False, 'float', True),
    ):
        q = torch.randn(1, 4, length, 32, generator=generator)
        k = torch.randn(1, 2 if gqa else 4, source, 32, generator=generator)
        v = torch.randn(1, 2 if gqa else 4, source, 16, generator=generator)
        mask = None
        if mask_kind == 'bool':
            mask = torch.ones(length, source, dtype=torch.bool)
            mask[0, :] = False
            mask[4:, ::3] = False
        elif mask_kind == 'float':
            mask = torch.randn(1, 1, length, source, generator=generator) / 4
        qd, kd, vd = [x.to(device=device, dtype=dtype) for x in (q,k,v)]
        md = None if mask is None else mask.to(device=device)
        actual = bounded_sdpa(qd,kd,vd,md,is_causal=causal,enable_gqa=gqa).float().cpu()
        expected = reference(qd.float().cpu(),kd.float().cpu(),vd.float().cpu(),
                             mask,is_causal=causal,enable_gqa=gqa)
        error = (actual-expected).abs().max().item()
        assert torch.isfinite(actual).all() and error < (0.003 if dtype == torch.float16 else 0.000002), error
        reports.append({'q':length,'kv':source,'causal':causal,'mask':mask_kind,'gqa':gqa,'max_error':error})
    return reports


def chunked_subm_forward(feats, neighbor_cache, weight, bias=None):
    """Explicit sparse convolution with a bounded im2col buffer."""
    count, channels = feats.shape
    outputs, *kernel, _ = weight.shape
    volume = math.prod(kernel)
    neighbors = neighbor_cache['neighbor_map']
    block = min(2048, max(1, (64 * 1024**2) // (volume * channels * feats.element_size())))
    flat_weight = weight.reshape(outputs, volume * channels).transpose(0,1)
    result = torch.empty((count, outputs), device=feats.device, dtype=feats.dtype)
    for start in range(0,count,block):
        stop = min(count,start+block)
        index = neighbors[start:stop].long()
        valid = (index >= 0) & (index != 0xffffffff)
        index = torch.where(valid,index,0).reshape(-1)
        column = feats.index_select(0,index).reshape(stop-start,volume,channels)
        column.mul_(valid[...,None])
        column = column.reshape(stop-start,volume*channels)
        product = column @ flat_weight
        if bias is not None:
            product.add_(bias)
        result[start:stop] = product
    return result


def weighted_gather(feats, indices, weight):
    """Triton's interpolation sum, using Torch GPU gathers and FP32 accumulation."""
    count, volume = indices.shape
    channels = feats.shape[-1]
    result = torch.zeros((count,channels), device=feats.device,dtype=feats.dtype)
    if feats.shape[0] == 0:
        return result
    block = min(32768,max(1,(32*1024**2)//(channels*4)))
    for start in range(0,count,block):
        stop = min(count,start+block)
        buffer = torch.zeros((stop-start,channels),device=feats.device,dtype=torch.float32)
        for corner in range(volume):
            index = indices[start:stop,corner].long()
            valid = (index >= 0) & (index != 0xffffffff)
            index = torch.where(valid,index,0)
            values = feats.index_select(0,index).float()
            factors = weight[start:stop,corner].float() * valid
            buffer.add_(values*factors[:,None])
        result[start:stop] = buffer.to(feats.dtype)
    return result


def verify_sparse_math(device='cpu'):
    import itertools
    generator = torch.Generator().manual_seed(455)
    shape = (2,3,4,5,6)
    dense = torch.randn(shape,generator=generator)
    coords = torch.tensor([p for p in itertools.product(range(2),range(4),range(5),range(6))
                           if sum(p)%4 != 0],dtype=torch.int32)
    lookup = {tuple(p):i for i,p in enumerate(coords.tolist())}
    feats = torch.stack([dense[b,:,x,y,z] for b,x,y,z in coords.tolist()])
    dense.zero_()
    for row,(b,x,y,z) in enumerate(coords.tolist()):
        dense[b,:,x,y,z] = feats[row]
    offsets = list(itertools.product(range(-1,2),repeat=3))
    neighbors = torch.tensor([[lookup.get((b,x+dx,y+dy,z+dz),0xffffffff) for dx,dy,dz in offsets]
                             for b,x,y,z in coords.tolist()],dtype=torch.int64)
    weight = torch.randn(7,3,3,3,3,generator=generator)/8
    bias = torch.randn(7,generator=generator)/8
    dtype = torch.float16 if device != 'cpu' else torch.float32
    ft,wt,bt = [x.to(device=device,dtype=dtype) for x in (feats,weight,bias)]
    actual = chunked_subm_forward(ft,{'neighbor_map':neighbors.to(device)},wt,bt).float().cpu()
    quantized_dense = dense.to(dtype).float()
    expected = torch.nn.functional.conv3d(quantized_dense,wt.cpu().float().permute(0,4,1,2,3),bt.cpu().float(),padding=1)
    expected = torch.stack([expected[b,:,x,y,z] for b,x,y,z in coords.tolist()])
    error = (actual-expected).abs().max().item()
    assert torch.isfinite(actual).all() and error < (0.005 if device != 'cpu' else 0.00001),error
    # A gather with missing neighbors must match an ordinary dense gather with a zero row.
    indexes = torch.randint(0,ft.shape[0],(19,8),generator=generator)
    indexes[:,3] = 0xffffffff
    factors = torch.rand(19,8,generator=generator)
    gathered = weighted_gather(ft,indexes.to(device),factors.to(device)).float().cpu()
    padded = torch.cat((ft.cpu().float(),torch.zeros(1,ft.shape[1])),dim=0)
    dense_indexes = indexes.clone()
    dense_indexes[indexes==0xffffffff] = ft.shape[0]
    expected_gather = (padded[dense_indexes]*factors[...,None]).sum(dim=1)
    gather_error = (gathered-expected_gather).abs().max().item()
    assert gather_error < (0.004 if device != 'cpu' else 0.00001),gather_error
    return {'sparse_conv_error':error,'weighted_gather_error':gather_error,'voxels':len(coords)}
