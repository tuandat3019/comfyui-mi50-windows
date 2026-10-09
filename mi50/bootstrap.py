"""Configure this process, then run ComfyUI without modifying its core."""
from pathlib import Path
import faulthandler
import os
import runpy
import sys


def main():
    runtime = Path(os.environ['MI50_RUNTIME'])
    profile = os.environ['MI50_PROFILE']
    directories = [runtime / 'hip57/bin', runtime / 'zluda']
    if profile == 'trellis':
        directories.append(runtime / 'trellis/cuda11.8.0/bin')
    handles = [os.add_dll_directory(str(path)) for path in directories]
    import torch
    if torch.__version__ != '2.7.1+cu118':
        raise RuntimeError(f'Expected Torch 2.7.1+cu118, found {torch.__version__}')
    from torch_compat import install
    install()
    fraction = float(os.environ.get('MI50_ALLOCATOR_FRACTION', '.80'))
    if not .5 <= fraction <= .9:
        raise ValueError('MI50_ALLOCATOR_FRACTION must be between .5 and .9')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU; check --device (HIP ordinal).')
    properties = torch.cuda.get_device_properties(0)
    # Driver names vary. Selection is explicit; avoid silently selecting RX 6800.
    name = properties.name
    if not any(label in name.lower() for label in ('mi50', 'pro vii', 'radeon vii', 'mi60')):
        raise RuntimeError(f'The selected GPU is not a recognized gfx906 device: {name}')
    print(f'MI50_DEVICE {name}; VRAM={properties.total_memory / 1024**3:.3f} GiB', flush=True)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    if profile == 'trellis':
        sys.path.insert(0, str(runtime / 'trellis/built-extensions'))
        from trellis_compat import install as install_trellis
        install_trellis()
    target = Path(sys.argv[1]).resolve()
    if profile == 'image' and target.name == 'main.py':
        from stage_offload import install as install_offload
        install_offload(target.parent / 'nodes.py')
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    faulthandler.enable()
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
