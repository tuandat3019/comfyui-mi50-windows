"""Install isolated profiles without changing the display driver or starting GPU work."""
from pathlib import Path
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mi50.paths import REPO, runtime_path, find_seven_zip, find_vcvars
from mi50.downloads import fetch, verify, extract_zip


def run(command, **kwargs):
    print('[setup]', ' '.join(map(str, command[:5])), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def source_repo(path, specification):
    path = Path(path)
    if path.exists() and not (path / '.git').is_dir():
        raise RuntimeError(f'Refusing to overwrite a directory without a Git source stamp: {path}')
    if not path.exists():
        run(['git', 'init', path])
        run(['git', '-C', path, 'remote', 'add', 'origin', specification['url']])
    head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    if head.returncode == 0:
        if head.stdout.strip() == specification['commit']:
            return
        raise RuntimeError(f'Different source revision already exists: {path}. Choose a new --runtime directory.')
    run(['git', '-C', path, 'fetch', '--depth', '1', 'origin', specification['commit']])
    run(['git', '-C', path, 'checkout', '--detach', 'FETCH_HEAD'])


def patch_file(path, transform):
    path = Path(path)
    backup = path.with_suffix(path.suffix + '.original-mi50')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    text = transform(backup.read_text(encoding='utf-8-sig'))
    compile(text, str(path), 'exec')
    path.write_text(text, encoding='utf-8', newline='\n')


def patch_gguf(text):
    marker = '            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))'
    if text.count(marker) != 1:
        raise RuntimeError('Pinned GGUF padding-token patch no longer matches')
    return text.replace(marker, marker + '\n'
        '            if compat == "sd.cpp" and arch_str == "lumina2" and sd_key in '
        '{"x_pad_token", "cap_pad_token"} and len(shape) == 1:\n'
        '                shape = torch.Size((1, *shape))')


def patch_optional_profiler(text):
    marker = 'from .trellis2_profiler import NODE_CLASS_MAPPINGS as _P_MAP, NODE_DISPLAY_NAME_MAPPINGS as _P_DMAP\nNODE_CLASS_MAPPINGS.update(_P_MAP); NODE_DISPLAY_NAME_MAPPINGS.update(_P_DMAP)'
    if marker not in text:
        raise RuntimeError('Pinned missing-profiler patch no longer matches')
    return text.replace(marker, '')


def extract_7z(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    run([find_seven_zip(), 'x', archive, '-o' + str(destination), '-y'], stdout=subprocess.DEVNULL)


def extract_msi_dlls(package, destination, staging):
    """Read the File table and extract CABs; never execute an MSI installer."""
    extract_7z(package, staging)
    for cabinet in list(staging.glob('*.cab')):
        extract_7z(cabinet, staging)
    msi = ctypes.WinDLL('msi.dll')
    handle = ctypes.c_uint
    msi.MsiOpenDatabaseW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(handle)]
    msi.MsiDatabaseOpenViewW.argtypes = [handle, ctypes.c_wchar_p, ctypes.POINTER(handle)]
    msi.MsiViewExecute.argtypes = [handle, handle]
    msi.MsiViewFetch.argtypes = [handle, ctypes.POINTER(handle)]
    msi.MsiRecordGetStringW.argtypes = [handle, ctypes.c_uint, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint)]
    msi.MsiCloseHandle.argtypes = [handle]
    database, view = handle(), handle()

    def checked(code):
        if code != 0:
            raise RuntimeError(f'MSI extraction error {code}')

    checked(msi.MsiOpenDatabaseW(str(package), None, ctypes.byref(database)))
    try:
        checked(msi.MsiDatabaseOpenViewW(database, 'SELECT `File`, `FileName` FROM `File`', ctypes.byref(view)))
        try:
            checked(msi.MsiViewExecute(view, 0))
            while True:
                record = handle()
                code = msi.MsiViewFetch(view, ctypes.byref(record))
                if code == 259:
                    break
                checked(code)
                try:
                    values = []
                    for column in (1, 2):
                        size = ctypes.c_uint(4096)
                        buffer = ctypes.create_unicode_buffer(size.value)
                        checked(msi.MsiRecordGetStringW(record, column, buffer, ctypes.byref(size)))
                        values.append(buffer.value)
                    key, name = values[0], values[1].split('|')[-1]
                    if not name.lower().endswith('.dll'):
                        continue
                    if Path(key).name != key or Path(name).name != name:
                        raise RuntimeError('Unexpected MSI filename')
                    target = destination / name
                    if not target.exists():
                        shutil.copy2(staging / key, target)
                finally:
                    msi.MsiCloseHandle(record)
        finally:
            msi.MsiCloseHandle(view)
    finally:
        msi.MsiCloseHandle(database)


def prepare_hip(runtime, records, cache, args):
    bin_dir = runtime / 'hip57/bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    if args.hip_bin:
        source = Path(args.hip_bin).resolve()
        if source == bin_dir.resolve():
            raise ValueError('--hip-bin must point outside the new runtime')
        shutil.copytree(source, bin_dir, dirs_exist_ok=True)
    else:
        if not args.accept_runtime_licenses:
            raise RuntimeError('Read the AMD and ZLUDA terms linked in docs/setup.md, then pass --accept-runtime-licenses; or provide --hip-bin.')
        driver = Path(args.driver_archive) if args.driver_archive else fetch(records['amd-pro26q1.exe'], cache)
        verify(driver, records['amd-pro26q1.exe'])
        sdk = Path(args.sdk_archive) if args.sdk_archive else fetch(records['hip57-sdk.exe'], cache)
        verify(sdk, records['hip57-sdk.exe'])
        staging = runtime / 'staging'
        extract_7z(driver, staging / 'driver')
        extract_7z(sdk, staging / 'sdk')
        driver_bins = list((staging / 'driver').rglob('amdocl/amdhip64.dll'))
        if len(driver_bins) != 1:
            raise RuntimeError('Cannot uniquely locate the pinned driver HIP runtime')
        for name in ('amdhip64.dll', 'amd_comgr.dll'):
            shutil.copy2(driver_bins[0].parent / name, bin_dir / name)
        for name in ('rocblas-hip57.7z', 'rocblas-gfx906.7z'):
            extracted = staging / name.split('.')[0]
            extract_7z(fetch(records[name], cache), extracted)
            dlls = list(extracted.rglob('rocblas.dll'))
            if dlls:
                if len(dlls) != 1:
                    raise RuntimeError('Ambiguous rocBLAS DLL')
                shutil.copy2(dlls[0], bin_dir / 'rocblas.dll')
            libraries = list(extracted.rglob('library'))
            if libraries:
                for library in libraries:
                    shutil.copytree(library, bin_dir / 'rocblas/library', dirs_exist_ok=True)
            else:
                for path in extracted.rglob('*'):
                    if path.is_file() and path.suffix.lower() in ('.dat', '.co', '.hsaco', '.yaml'):
                        target = bin_dir / 'rocblas/library' / path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
        for name in ('ROCm_Libs_RT.msi', 'ROCm_RTC_RT.msi'):
            packages = list((staging / 'sdk').rglob(name))
            if len(packages) != 1:
                raise RuntimeError(f'Cannot uniquely locate {name}')
            extract_msi_dlls(packages[0], bin_dir, staging / name.replace('.msi', ''))
    expected = json.loads((REPO / 'config/hip-dlls.json').read_text(encoding='utf-8'))
    for item in expected:
        verify(bin_dir / item['name'], item)
    if not list((bin_dir / 'rocblas/library').glob('*gfx906*')):
        raise RuntimeError('Missing gfx906 rocBLAS tables')


def prepare_zluda(runtime, records, cache):
    staging = runtime / 'staging/zluda'
    extract_zip(fetch(records['zluda-rocm5.zip'], cache), staging)
    executables = list(staging.rglob('zluda.exe'))
    if len(executables) != 1:
        raise RuntimeError('Cannot uniquely locate zluda.exe')
    shutil.copytree(executables[0].parent, runtime / 'zluda', dirs_exist_ok=True)
    lib = runtime / 'venv/Lib/site-packages/torch/lib'
    for source, name in [('cublas.dll', 'cublas64_11.dll'), ('cusparse.dll', 'cusparse64_11.dll'), ('nvrtc.dll', 'nvrtc64_112_0.dll')]:
        target = lib / name
        backup = target.with_name(name + '.original-mi50')
        if not backup.exists():
            if source == 'nvrtc.dll':
                shutil.copy2(target, lib / 'nvrtc_cuda.dll')
            shutil.copy2(target, backup)
        shutil.copy2(runtime / 'zluda' / source, target)


def prepare_tiled_helper(runtime, records, cache):
    archive = fetch(records['o_voxel-amd-helper.whl'], cache)
    with zipfile.ZipFile(archive) as wheel:
        source = wheel.read('o_voxel/convert/flexible_dual_grid.py').decode('utf-8-sig').replace('\r\n', '\n')
    begin = source.index('@torch.no_grad()\ndef tiled_flexible_dual_grid_to_mesh(')
    end = source.index('\ndef _init_hashmap(', begin)
    helper = source[begin:end]
    target = runtime / 'trellis/built-extensions/o_voxel/convert/flexible_dual_grid.py'

    def transform(text):
        if 'tiled_flexible_dual_grid_to_mesh' in text:
            raise RuntimeError('Pinned o-voxel unexpectedly already contains the tiling helper')
        text = text.replace('from .. import _C', 'from .. import _C\nfrom tqdm import tqdm', 1)
        text = text.replace('    "flexible_dual_grid_to_mesh",', '    "flexible_dual_grid_to_mesh",\n    "tiled_flexible_dual_grid_to_mesh",', 1)
        return text.replace('\ndef _init_hashmap(', '\n' + helper + '\ndef _init_hashmap(', 1)

    patch_file(target, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['image', 'trellis', 'all'], default='image')
    parser.add_argument('--runtime')
    parser.add_argument('--download-cache')
    parser.add_argument('--hip-bin', help='A complete existing, compatible HIP5.7 bin directory including gfx906 tables')
    parser.add_argument('--driver-archive', help='Optional locally downloaded AMD PRO26Q1 Vega/Polaris archive')
    parser.add_argument('--sdk-archive', help='Optional locally downloaded HIP5.7 SDK archive')
    parser.add_argument('--accept-runtime-licenses', action='store_true')
    parser.add_argument('--skip-build', action='store_true', help='Prepare TRELLIS sources/dependencies; native extensions remain pending')
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Show paths and pinned revisions without downloading or writing')
    args = parser.parse_args()
    runtime = runtime_path(args.runtime)
    specifications = json.loads((REPO / 'config/sources.json').read_text(encoding='utf-8'))
    names = ['ComfyUI']
    if args.profile in ('image', 'all'):
        names.append('ComfyUI-GGUF')
    if args.profile in ('trellis', 'all'):
        names += ['ComfyUI-Trellis2-AMD', 'CuMesh', 'FlexGEMM', 'nvdiffrast', 'TRELLIS2-o-voxel', 'eigen-3.4.0']
    if args.plan:
        print(json.dumps({'profile': args.profile, 'runtime': str(runtime), 'sources': {name: specifications[name] for name in names},
                          'gpu_work': False, 'downloads_weights': False, 'build_pending': args.skip_build}, indent=2))
        return
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 12) or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError('This recipe requires 64-bit Windows and Python 3.12.')
    if not args.sources_only and not args.hip_bin and not args.accept_runtime_licenses:
        raise RuntimeError('Read docs/setup.md and pass --accept-runtime-licenses, or provide --hip-bin.')
    if args.profile in ('trellis', 'all') and not args.sources_only and not args.skip_build:
        find_vcvars()
    if runtime == REPO or runtime in REPO.parents:
        raise ValueError('--runtime must be a dedicated directory, not this checkout or its ancestor')
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(MI50_RUNTIME=str(runtime), PYTHONUTF8='1')
    for name in names:
        if name == 'ComfyUI':
            destination = runtime / name
        elif name.startswith('ComfyUI-'):
            destination = runtime / 'ComfyUI/custom_nodes' / name
        else:
            destination = runtime / 'trellis/sources' / name
        source_repo(destination, specifications[name])
    if 'ComfyUI-GGUF' in names:
        patch_file(runtime / 'ComfyUI/custom_nodes/ComfyUI-GGUF/loader.py', patch_gguf)
    if 'ComfyUI-Trellis2-AMD' in names:
        patch_file(runtime / 'ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/nodes.py', patch_optional_profiler)
    if args.sources_only:
        print('[setup] Source checkout complete. Python/native runtime has not been installed.')
        return
    python = runtime / 'venv/Scripts/python.exe'
    if not python.exists():
        run([sys.executable, '-m', 'venv', runtime / 'venv'])
    pip = [python, '-m', 'pip', 'install', '--disable-pip-version-check']
    run(pip + ['--upgrade', 'pip', 'wheel', 'setuptools<82'])
    run(pip + ['--index-url', 'https://download.pytorch.org/whl/cu118',
               'torch==2.7.1+cu118', 'torchvision==0.22.1+cu118', 'torchaudio==2.7.1+cu118'])
    requirements = []
    for line in (runtime / 'ComfyUI/requirements.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() not in ('torch', 'torchvision', 'torchaudio', 'comfy-angle'):
            requirements.append(line)
    requirement_file = runtime / 'comfy-requirements.txt'
    requirement_file.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    run(pip + ['-c', REPO / 'constraints.txt', '-r', requirement_file])
    if 'ComfyUI-GGUF' in names:
        run(pip + ['-c', REPO / 'constraints.txt', 'gguf==0.19.0', 'protobuf'])
    if 'ComfyUI-Trellis2-AMD' in names:
        run(pip + ['-c', REPO / 'constraints.txt', '-r', REPO / 'profiles/trellis2/requirements.txt'])
    records = {item['name']: item for item in json.loads((REPO / 'config/downloads.json').read_text(encoding='utf-8'))}
    cache = Path(args.download_cache).resolve() if args.download_cache else runtime / 'downloads'
    prepare_hip(runtime, records, cache, args)
    prepare_zluda(runtime, records, cache)
    if 'ComfyUI-Trellis2-AMD' in names:
        for item in records.values():
            if item.get('group') == 'cuda11':
                extract_zip(fetch(item, cache), runtime / 'trellis/cuda11.8.0', strip_root=True, headers_only=item.get('headers_only', False))
        if not args.skip_build:
            find_vcvars()
            run([sys.executable, REPO / 'scripts/patch_cub11.py'], env=env)
            run([sys.executable, REPO / 'scripts/patch_raster_video.py'], env=env)
            for name in ('CuMesh', 'nvdiffrast', 'FlexGEMM', 'o-voxel'):
                run([sys.executable, REPO / 'scripts/build_extensions.py', name], env=env)
            prepare_tiled_helper(runtime, records, cache)
    run([python, '-m', 'pip', 'check'])
    frozen = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (runtime / 'installed-requirements.txt').write_text(frozen, encoding='utf-8')
    state_path = runtime / 'install-state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    state.setdefault('profiles', [])
    state['profiles'] = sorted(set(state['profiles']) | ({'image', 'trellis'} if args.profile == 'all' else {args.profile}))
    state.update(sources=specifications, gpu_tested=False)
    if 'ComfyUI-Trellis2-AMD' in names:
        state['trellis_build_complete'] = not args.skip_build
    state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print('[setup] Complete. Weights are separate; no GPU inference has been run.')


if __name__ == '__main__':
    main()
