"""Build local CUDA11.8/PTX extension artifacts without installing packages."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import shutil
import subprocess
import sys

root = Path(os.environ['MI50_RUNTIME']) / 'trellis'
repo = Path(__file__).resolve().parents[1]
def write_changed(path,text):
    encoded = text.encode('utf-8')
    if not path.exists() or path.read_bytes() != encoded:
        path.write_bytes(encoded)

cuda = root / 'cuda11.8.0'
source = root / 'sources' / sys.argv[1]
if sys.argv[1] == 'o-voxel':
    source = root / 'sources/TRELLIS2-o-voxel/o-voxel'
if source.name in ('CuMesh','o-voxel'):
    # MSVC resolves the derived Ref's template argument to an inherited private
    # alias of the same name under nvcc 11.8. Rename only that private alias;
    # the type, visibility, arithmetic and class layout remain identical.
    ref = root / 'sources/eigen-3.4.0/Eigen/src/Core/Ref.h'
    baseline = ref.with_suffix('.h.original-mi50')
    if not baseline.exists():
        baseline.write_bytes(ref.read_bytes())
    original_ref = baseline.read_text(encoding='utf-8')
    begin = original_ref.index('template<typename Derived> class RefBase')
    end = original_ref.index('\n};',begin)+3
    block = original_ref[begin:end].replace('StrideType','RefBaseStrideType')
    block = block.replace('internal::traits<Derived>::RefBaseStrideType','internal::traits<Derived>::StrideType')
    write_changed(ref,original_ref[:begin]+block+original_ref[end:])
output = root / 'built-extensions'
output.mkdir(exist_ok=True)
from mi50.paths import find_vcvars
vcvars = find_vcvars()
env_script = root / 'load-build-env.cmd'
env_script.write_text('@echo off\ncall "' + str(vcvars) + '" >nul\nif errorlevel 1 exit /b 1\nset\n', encoding='ascii')
vc = subprocess.check_output(['cmd.exe', '/d', '/c', str(env_script)], text=True, errors='replace')
env = os.environ.copy()
for line in vc.splitlines():
    if '=' in line:
        key, value = line.split('=', 1)
        env[key] = value
env.update(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), TORCH_CUDA_ARCH_LIST='7.5+PTX',
           MAX_JOBS='1', PYTHONUTF8='1', DISTUTILS_USE_SDK='1', BUILD_TARGET='cuda',
           NVCC_PREPEND_FLAGS='-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1')
env['PATH'] = str(cuda / 'bin') + os.pathsep + env['PATH']
setup = source / 'setup.py'
original = setup.with_suffix('.py.original-mi50')
if not original.exists():
    original.write_bytes(setup.read_bytes())
text = original.read_text(encoding='utf-8')
text = text.replace('"cxx": ["-O3", "-std=c++20"]', '"cxx": ["/O2", "/std:c++17", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"]')
text = text.replace('"nvcc": ["-O3","-std=c++20"]', '"nvcc": ["-O3","-std=c++17", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"]')
text = text.replace('"cxx": ["-O3", "-std=c++17"]', '"cxx": ["/O2", "/std:c++17", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"]')
if source.name == 'CuMesh':
    eigen = root / 'sources/eigen-3.4.0'
    assert (eigen/'Eigen/Dense').is_file()
    text = text.replace('os.path.join(ROOT, "third_party/cubvh/third_party/eigen")',repr(eigen.as_posix()))
if source.name == 'o-voxel':
    eigen = root / 'sources/eigen-3.4.0'
    assert (eigen/'Eigen/Dense').is_file()
    text = text.replace('os.path.join(ROOT, "third_party/eigen")',repr(eigen.as_posix()))
    conversion = source/'src/convert/flexible_dual_grid.cpp'
    backup_conversion = conversion.with_suffix('.cpp.original-mi50')
    if not backup_conversion.exists():
        backup_conversion.write_bytes(conversion.read_bytes())
    conversion_text = backup_conversion.read_text(encoding='utf-8')
    # Unsuffixed floating literals already have type double; MSVC rejects d.
    assert '1e-6d' in conversion_text and '0.0d' in conversion_text
    write_changed(conversion,conversion_text.replace('1e-6d','1e-6').replace('0.0d','0.0'))
    neighbor = source/'src/io/filter_neighbor.cpp'
    backup_neighbor = neighbor.with_suffix('.cpp.original-mi50')
    if not backup_neighbor.exists():
        backup_neighbor.write_bytes(neighbor.read_bytes())
    neighbor_text = backup_neighbor.read_text(encoding='utf-8')
    assert neighbor_text.count('torch::zeros({N, C}') == 2
    write_changed(neighbor,neighbor_text.replace('torch::zeros({N, C}',
                        'torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)}'))
    parent_filter = source/'src/io/filter_parent.cpp'
    backup_parent = parent_filter.with_suffix('.cpp.original-mi50')
    if not backup_parent.exists():
        backup_parent.write_bytes(parent_filter.read_bytes())
    parent_text = backup_parent.read_text(encoding='utf-8')
    assert parent_text.count('torch::zeros({N_leaf, C}') == 2
    write_changed(parent_filter,parent_text.replace('torch::zeros({N_leaf, C}',
                  'torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}'))
    svo = source/'src/io/svo.cpp'
    backup_svo = svo.with_suffix('.cpp.original-mi50')
    if not backup_svo.exists():
        backup_svo.write_bytes(svo.read_bytes())
    svo_text = backup_svo.read_text(encoding='utf-8')
    for array in ('svo','codes'):
        old_shape = '{'+array+'.size()}'
        assert old_shape in svo_text
        svo_text = svo_text.replace(old_shape,'{static_cast<int64_t>('+array+'.size())}')
    write_changed(svo,svo_text)
if source.name == 'FlexGEMM':
    # Building an extension must not overwrite the user's global autotune cache.
    text = text.split('# Install autotune cache.')[0]
setup.write_text(text, encoding='utf-8')
if source.name == 'CuMesh':
    shutil.copytree(source / 'cumesh', output / 'cumesh', dirs_exist_ok=True)
elif source.name == 'nvdiffrast':
    shutil.copytree(source / 'nvdiffrast', output / 'nvdiffrast', dirs_exist_ok=True)
elif source.name == 'FlexGEMM':
    shutil.copytree(source / 'flex_gemm', output / 'flex_gemm', dirs_exist_ok=True)
elif source.name == 'o-voxel':
    shutil.copytree(source / 'o_voxel', output / 'o_voxel', dirs_exist_ok=True)
command = [str(root.parent / 'venv/Scripts/python.exe'), str(repo / 'mi50/build_entry.py'), str(setup), 'build_ext',
           '--build-lib', str(output), '--build-temp', str(root / 'build' / source.name)]
(root / ('build-' + source.name + '-command.json')).write_text(json.dumps({'command':command,
    'CUDA_HOME':str(cuda), 'PTX_ONLY':True, 'TORCH_CUDA_ARCH_LIST':env['TORCH_CUDA_ARCH_LIST'], 'MAX_JOBS':1}, indent=2), encoding='utf-8')
log_path = root / ('build-' + source.name + '.log')
print('BUILD_BEGIN', source.name, 'LOG', str(log_path), flush=True)
with log_path.open('w', encoding='utf-8') as log:
    code = subprocess.call(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
print('BUILD_DONE', source.name, 'EXIT', code, flush=True)
if code:
    print('\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-35:]), flush=True)
sys.exit(code)
