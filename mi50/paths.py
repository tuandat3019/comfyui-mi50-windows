"""Runtime paths are relative to this checkout, with an optional SSD override."""
from pathlib import Path
import os
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def runtime_path(value=None):
    return Path(value or os.environ.get('MI50_RUNTIME', REPO / '.runtime')).resolve()


def find_seven_zip():
    value = os.environ.get('MI50_7ZIP') or shutil.which('7z')
    path = Path(value) if value else Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / '7-Zip/7z.exe'
    if not path.is_file():
        raise RuntimeError('Install 7-Zip, or set MI50_7ZIP to 7z.exe.')
    return path


def find_vcvars():
    value = os.environ.get('MI50_VCVARS')
    if value:
        path = Path(value).resolve()
    else:
        locator = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        if not locator.is_file():
            raise RuntimeError('Install Visual Studio 2022 Build Tools with Desktop development with C++.')
        result = subprocess.check_output([str(locator), '-latest', '-version', '[17.0,18.0)', '-products', '*',
                                          '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                          '-property', 'installationPath'], text=True).strip()
        path = Path(result) / 'VC/Auxiliary/Build/vcvars64.bat'
    if not path.is_file():
        raise RuntimeError('Cannot locate vcvars64.bat; set MI50_VCVARS explicitly.')
    return path


def launch_environment(runtime, profile, device):
    env = os.environ.copy()
    env.update(MI50_RUNTIME=str(runtime), MI50_COMFY=str(runtime / 'ComfyUI'), MI50_PROFILE=profile,
               HIP_PATH=str(runtime / 'hip57'), HIP_VISIBLE_DEVICES=str(device), CUDA_VISIBLE_DEVICES='0',
               ROCBLAS_TENSILE_LIBPATH=str(runtime / 'hip57/bin/rocblas/library'),
               ZLUDA_CACHE_DIR=str(runtime / 'zluda-cache'), PYTORCH_CUDA_ALLOC_CONF='backend:native',
               DISABLE_ADDMM_CUDA_LT='1', PYTHONUTF8='1',
               FLEX_GEMM_USE_AUTOTUNE_CACHE='0', FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE='0')
    directories = [runtime / 'hip57/bin', runtime / 'zluda']
    if profile == 'trellis':
        directories.insert(0, runtime / 'trellis/cuda11.8.0/bin')
    env['PATH'] = os.pathsep.join([*(str(p) for p in directories), env.get('PATH', '')])
    # Do not inherit a different project's Python overlays.
    env.pop('PYTHONPATH', None)
    return env
