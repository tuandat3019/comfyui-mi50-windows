"""Launch one foreground ComfyUI profile on an explicitly selected HIP GPU."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mi50.paths import REPO, runtime_path, launch_environment


def command_for(runtime, profile, device, port):
    env = launch_environment(runtime, profile, device)
    custom_node = 'ComfyUI-GGUF' if profile == 'image' else 'ComfyUI-Trellis2-AMD'
    command = [runtime / 'zluda/zluda.exe', '--', runtime / 'venv/Scripts/python.exe', REPO / 'mi50/bootstrap.py',
               runtime / 'ComfyUI/main.py', '--listen', '127.0.0.1', '--port', str(port),
               '--disable-auto-launch', '--disable-all-custom-nodes', '--whitelist-custom-nodes', custom_node,
               '--disable-cuda-malloc', '--disable-triton-backend', '--disable-dynamic-vram',
               '--disable-pinned-memory', '--disable-async-offload', '--preview-method', 'none',
               '--reserve-vram', '1.5' if profile == 'image' else '2.5',
               '--user-directory', runtime / 'user' / profile, '--temp-directory', runtime / 'temp' / profile,
               '--output-directory', runtime / 'outputs' / profile]
    if profile == 'image':
        command += ['--fp16-unet', '--fp16-text-enc', '--fp32-vae', '--use-split-cross-attention']
    return list(map(str, command)), env


def missing_files(runtime, profile):
    required = ['venv/Scripts/python.exe', 'ComfyUI/main.py', 'zluda/zluda.exe', 'hip57/bin/amdhip64.dll',
                'hip57/bin/rocblas.dll', 'hip57/bin/rocsolver.dll', 'hip57/bin/rocsparse.dll']
    if profile == 'image':
        required += ['ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py']
    else:
        required += ['ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/nodes.py']
        required += ['trellis/built-extensions/' + name for name in
                     ('cumesh/_C.cp312-win_amd64.pyd', 'cumesh/_cubvh.cp312-win_amd64.pyd',
                      'cumesh/_xatlas.cp312-win_amd64.pyd', 'flex_gemm/kernels/cuda.cp312-win_amd64.pyd',
                      'o_voxel/_C.cp312-win_amd64.pyd', '_nvdiffrast_c.cp312-win_amd64.pyd')]
    return [name for name in required if not (runtime / name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['image', 'trellis'], default='image')
    parser.add_argument('--runtime')
    parser.add_argument('--device', type=int, default=0, help='HIP5.7 ordinal before visibility filtering; not Vulkan/DirectML ordinal')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true', help='Check installation files only; does not initialize a GPU')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.device < 0:
        parser.error('--device must be non-negative')
    port = args.port if args.port is not None else (8188 if args.profile == 'image' else 8189)
    if not 1 <= port <= 65535:
        parser.error('--port must be between 1 and 65535')
    runtime = runtime_path(args.runtime)
    command, env = command_for(runtime, args.profile, args.device, port)
    if args.dry_run:
        print(json.dumps({'command': command, 'environment': {name: env[name] for name in
                          ('MI50_RUNTIME', 'MI50_PROFILE', 'HIP_PATH', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                           'ZLUDA_CACHE_DIR', 'ROCBLAS_TENSILE_LIBPATH')}}, indent=2))
        return
    missing = missing_files(runtime, args.profile)
    if missing:
        raise RuntimeError('Incomplete runtime:\n' + '\n'.join(missing))
    if args.check:
        print('[check] Installation files present; GPU/weights have not been tested.')
        return
    if sys.platform != 'win32':
        raise RuntimeError('This launcher requires Windows.')
    import msvcrt
    from mi50.windows_job import ChildJob
    for name in ('user', 'temp', 'outputs', 'diagnostics'):
        (runtime / name / args.profile).mkdir(parents=True, exist_ok=True)
    env['MI50_DIAGNOSTIC_DIR'] = str(runtime / 'diagnostics' / args.profile)
    # The lock spans both profiles in this runtime. Do not unload other applications.
    with (runtime / 'mi50-session.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another profile from this runtime is still running.') from exc
        try:
            with ChildJob() as job:
                process = job.start(command, cwd=runtime / 'ComfyUI', env=env)
                print(f'[launch] {args.profile}: http://127.0.0.1:{port} — Ctrl+C to stop this worker', flush=True)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    code = 130
                finally:
                    job.close()
                    process.wait(timeout=30)
                if code:
                    raise SystemExit(code)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
