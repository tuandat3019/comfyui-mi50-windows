"""Load test Torch libraries for compilation, with GPU discovery disabled."""
from pathlib import Path
import os
import runpy
import sys

root = Path(os.environ['MI50_RUNTIME'])
handles = [os.add_dll_directory(str(root / p)) for p in ('hip57/bin', 'zluda')]
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import torch.utils.cpp_extension as cpp_extension
# ZLUDA consumes PTX. NVIDIA SASS adds compile work and has no use on MI50.
cpp_extension._get_cuda_arch_flags = lambda cflags=None: ['-gencode=arch=compute_75,code=compute_75']
setup = Path(sys.argv[1]).resolve()
sys.argv = [str(setup), *sys.argv[2:]]
runpy.run_path(str(setup), run_name='__main__')
