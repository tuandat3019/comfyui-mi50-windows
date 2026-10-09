"""Scoped MI50 inference overlay; upstream TRELLIS files remain unchanged on disk."""
from pathlib import Path
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ['MI50_COMFY']) / 'custom_nodes/ComfyUI-Trellis2-AMD'
PATCHES = {}


class SourceLoader(importlib.abc.Loader):
    def __init__(self, delegate, path):
        self.delegate, self.path = delegate, path
    def create_module(self, spec):
        return self.delegate.create_module(spec)
    def __getattr__(self, key):
        return getattr(self.delegate,key)
    def exec_module(self, module):
        original = self.path.read_text(encoding='utf-8-sig')
        source = original.replace('torch.bfloat16','torch.float16').replace('.bfloat16()', '.half()')
        if self.path.name == 'nodes.py':
            old = '["xformers","flash_attn","aule"]'
            assert old in source, 'Sparse backend selector changed upstream'
            source = source.replace(old,'["xformers","flash_attn","aule","sdpa"]')
        if source != original:
            PATCHES[self.path.relative_to(SOURCE).as_posix()] = {
                'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                'overlay_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
            diagnostic = Path(os.environ.get('MI50_DIAGNOSTIC_DIR',ROOT))
            diagnostic.mkdir(parents=True,exist_ok=True)
            (diagnostic/'mi50-source-patches.json').write_text(json.dumps(PATCHES,indent=2),encoding='utf-8')
        exec(compile(source,str(self.path),'exec',dont_inherit=True),module.__dict__)


class SourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.origin and isinstance(spec.loader,importlib.machinery.SourceFileLoader):
            origin = Path(spec.origin).resolve()
            if origin.is_relative_to(SOURCE):
                spec.loader = SourceLoader(spec.loader,origin)
                return spec
        return None


def install():
    from ops import bounded_sdpa, chunked_subm_forward, weighted_gather, metadata_bincount, metadata_unique
    torch.nn.functional.scaled_dot_product_attention = bounded_sdpa
    torch.bincount = metadata_bincount
    torch.unique = metadata_unique
    sys.meta_path.insert(0,SourceFinder())
    import flex_gemm
    import flex_gemm.ops.spconv as spconv
    setter = spconv.set_algorithm
    def force_explicit(_):
        setter(spconv.Algorithm.EXPLICIT_GEMM)
    spconv.set_algorithm = force_explicit
    force_explicit(None)
    conv = importlib.import_module('flex_gemm.ops.spconv.submanifold_conv3d')
    conv.SubMConv3dFunction._sparse_submanifold_conv_forward = staticmethod(chunked_subm_forward)
    flex_gemm.kernels.triton.indice_weighed_sum_fwd = weighted_gather
    print('TRELLIS_MI50_PROFILE fp16; bounded-sdpa; chunked-explicit-gemm; torch-interpolation; cpu-batch-counts-and-row-unique',flush=True)
