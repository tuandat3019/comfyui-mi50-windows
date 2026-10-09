"""ComfyUI import hook: release stage weights and audit CUDA allocator."""
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

class NodeLoader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate
    def create_module(self, spec):
        return self.delegate.create_module(spec)
    def __getattr__(self, name):
        return getattr(self.delegate, name)
    def exec_module(self, module):
        self.delegate.exec_module(module)
        import torch
        import comfy.model_management as mm
        import comfy.utils as utils
        def audit(label):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f'MI50_MEMORY {label}; allocated={torch.cuda.memory_allocated()/1024**3:.3f}GiB reserved={torch.cuda.memory_reserved()/1024**3:.3f}GiB free={free/1024**3:.3f}GiB total={total/1024**3:.3f}GiB', flush=True)
        original = module.CLIPTextEncode.encode
        def encode_and_release(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            import torch
            import comfy.model_management as mm
            torch.cuda.synchronize()
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
            free, total = torch.cuda.mem_get_info()
            print(f'MI50_STAGE_OFFLOAD encoder released; free={free / 1024**3:.3f}GiB total={total / 1024**3:.3f}GiB', flush=True)
            audit('encoder released')
            return output
        module.CLIPTextEncode.encode = encode_and_release
        original_load = mm.load_models_gpu
        def load_and_audit(models, *args, **kwargs):
            label = ','.join(type(m.model).__name__ for m in models)
            audit('before load ' + label)
            result = original_load(models, *args, **kwargs)
            audit('after load ' + label)
            return result
        mm.load_models_gpu = load_and_audit
        original_progress = utils.ProgressBar.update_absolute
        def progress_and_trim(self, value, total=None, preview=None):
            result = original_progress(self, value, total, preview)
            if self.total == 8 and value > 0:
                audit(f'step {value}/{self.total} before trim')
                mm.soft_empty_cache(force=True)
                audit(f'step {value}/{self.total} after trim')
            return result
        utils.ProgressBar.update_absolute = progress_and_trim
        original_decode = module.VAEDecodeTiled.decode
        def decode_and_release(self, *args, **kwargs):
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
            audit('diffusion released before VAE')
            return original_decode(self, *args, **kwargs)
        module.VAEDecodeTiled.decode = decode_and_release

class NodeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source):
        self.source = source.resolve()
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'nodes':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and Path(spec.origin).resolve() == self.source:
            spec.loader = NodeLoader(spec.loader)
            return spec
        return None

def install(source):
    if 'nodes' in sys.modules:
        raise RuntimeError('Offload hook must be installed before Comfy nodes import')
    sys.meta_path.insert(0, NodeFinder(Path(source)))
