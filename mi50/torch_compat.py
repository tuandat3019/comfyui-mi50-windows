"""Inference-only Torch workarounds validated on MI50/HIP 5.7."""
import torch


def install():
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library('cublas')
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
    # Keep the API/device semantics while avoiding an expensive ZLUDA reduction
    # module for Comfy's scalar empty-latent check. Model forwards stay on MI50.
    native_count_nonzero = torch.count_nonzero
    def count_nonzero_compat(input, dim=None):
        if input.device.type == 'cuda' and dim is None and input.numel() <= 1024 * 1024:
            return native_count_nonzero(input.cpu()).to(input.device)
        return native_count_nonzero(input, dim=dim)
    torch.count_nonzero = count_nonzero_compat
    # Avoid the large-row CUDA softmax launch that faults in HIP5.7 on this card.
    # Stable max/exp/sum softmax remains GPU computation, with FP32 accumulation.
    native_softmax = torch.Tensor.softmax
    def softmax_compat(input, dim, dtype=None):
        if input.device.type != 'cuda' or input.shape[dim] <= 1024:
            return native_softmax(input, dim, dtype=dtype)
        out_dtype = input.dtype if dtype is None else dtype
        work_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
        # Own the buffer so FP32 inputs are never modified. Reuse it to avoid
        # multiple attention-sized temporaries in the legacy CUDA allocator.
        x = input.to(work_dtype, copy=True)
        x.sub_(x.amax(dim=dim, keepdim=True)).exp_()
        x.div_(x.sum(dim=dim, keepdim=True))
        return x.to(out_dtype)
    torch.Tensor.softmax = softmax_compat
    native_layer_norm = torch.nn.functional.layer_norm
    def layer_norm_compat(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if input.device.type != 'cuda':
            return native_layer_norm(input, normalized_shape, weight, bias, eps)
        work_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
        dims = tuple(range(input.ndim - len(normalized_shape), input.ndim))
        x = input.to(work_dtype)
        centered = x - x.mean(dim=dims, keepdim=True)
        out = centered * torch.rsqrt(centered.square().mean(dim=dims, keepdim=True) + eps)
        if weight is not None:
            out = out * weight.to(work_dtype)
        if bias is not None:
            out = out + bias.to(work_dtype)
        return out.to(input.dtype)
    torch.nn.functional.layer_norm = layer_norm_compat
    native_group_norm = torch.nn.functional.group_norm
    def group_norm_compat(input, num_groups, weight=None, bias=None, eps=1e-5):
        if input.device.type != 'cuda':
            return native_group_norm(input, num_groups, weight, bias, eps)
        work_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
        x = input.to(work_dtype).reshape(input.shape[0], num_groups, -1)
        centered = x - x.mean(dim=-1, keepdim=True)
        out = (centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)).reshape(input.shape)
        affine_shape = (1, input.shape[1]) + (1,) * (input.ndim - 2)
        if weight is not None:
            out = out * weight.to(work_dtype).reshape(affine_shape)
        if bias is not None:
            out = out + bias.to(work_dtype).reshape(affine_shape)
        return out.to(input.dtype)
    torch.nn.functional.group_norm = group_norm_compat
    # Match the ZLUDA Comfy fork: optional TorchScript helpers stay ordinary Python.
    def script_compat(function, *args, **kwargs):
        function.graph = torch._C.Graph()
        return function
    torch.jit.script = script_compat
