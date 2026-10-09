# Setup details

The default runtime is `.runtime/`, ignored by Git. It contains its own virtual environment, source checkouts, DLLs, native builds and cache. No production environment or display driver is modified.

## Dependency terms and downloads

Review the [AMD PRO26Q1 release/download page](https://www.amd.com/en/resources/support-articles/release-notes/RN-PRO-WIN-26-Q1.html), [AMD HIP SDK page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html), [ZLUDA project terms](https://github.com/lshqqytiger/ZLUDA), [community rocBLAS package](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.5.7) and [CUDA redistributable terms](https://docs.nvidia.com/cuda/eula/index.html). `--accept-runtime-licenses` is an acknowledgement for local download/extraction, not a driver-install command. Dependencies retain their own terms. See the [third-party notices](../THIRD_PARTY_NOTICES.md), particularly nvdiffrast.

The recipe uses two distinct AMD packages:

1. **PRO26Q1 Vega/Polaris** provides the tested `amdhip64.dll` and `amd_comgr.dll`. Its HIP runtime reports version `50731541` on the reference machine.
2. **HIP5.7 SDK / PRO23.Q4 For-HIP archive** supplies library/RTC DLL payloads. MSI File tables and CABs are read/extracted without executing installers. Community rocBLAS and gfx906 tables replace the generic rocBLAS payload.

Using a newer HIP runtime is not an interchangeable upgrade: the installed HIP7 production runtime did not enumerate MI50 in this machine. The repository records exact DLL/archive hashes, not a blanket assertion about every HIP version.

Downloads are cached and verified against [downloads.json](../config/downloads.json). All required runtime DLLs are checked against [hip-dlls.json](../config/hip-dlls.json). Git checkouts are pinned in [sources.json](../config/sources.json). Git transport verifies fetched objects; installers preserve modified local source at the pinned commit and refuse to reset a different checkout revision.

## Existing downloads or runtime

```powershell
py -3.12 scripts/setup.py --profile all --accept-runtime-licenses --driver-archive D:/Downloads/amd-software-pro-edition-26.q1-win11-vega-polaris.exe --sdk-archive D:/Downloads/AMD-Software-PRO-Edition-23.Q4-Win10-Win11-For-HIP.exe
```

Or provide a **complete** previously staged compatible HIP bin directory, including `rocblas/library/*gfx906*`:

```powershell
py -3.12 scripts/setup.py --profile image --hip-bin D:/AI/existing-hip57/bin
```

The source directory is copied and checked; it is not modified. Supplying only two HIP DLLs is insufficient. The ZLUDA archive is still fetched separately.

Set `MI50_7ZIP` to an explicit `7z.exe` path if 7-Zip is not installed in its usual location. `--download-cache` selects an existing archive cache; a mismatched checksum fails instead of using that file silently.

## TRELLIS native builds

Install VS2022 Build Tools with the C++ workload and Windows SDK separately. `vswhere` discovers its environment; `MI50_VCVARS` can specify `vcvars64.bat`. CUDA11.8 components are extracted locally from official redistributables. No NVIDIA GPU or system CUDA installation is required for the recipe.

The compiler is instructed to emit **compute_75 PTX only**. ZLUDA translates PTX for MI50; this is not an NVIDIA GPU selection. CuMesh's CUB calls, Eigen's MSVC alias, o-voxel literals/shapes and nvdiffrast scalar-video PTX helpers require the included compatibility patches. Build order is CuMesh, nvdiffrast, FlexGEMM, o-voxel, then the Python tiled-decoder helper.

`--skip-build` prepares dependencies and source but leaves TRELLIS unusable until a successful build. `--sources-only` checks out and patches the Comfy custom nodes without installing Python/native libraries. Neither option proves GPU compatibility. Re-running normal setup resumes through the build steps using retained intermediate objects.

Source compilation uses `MAX_JOBS=1` and below-normal priority. Allow substantial free RAM, disk space and time; this project has not measured minimal host-RAM requirements. Keep native sources in a short path and the ZLUDA cache on SSD.

## Verification levels

- `setup.py --plan`: read-only path/revision plan.
- `launch.py --dry-run`: read-only command/environment plan.
- `launch.py --check`: file presence, without GPU initialization or weight validation.
- Unit tests: CPU mathematical comparisons, archive/patch/runtime helpers, workflow structure and child-process cleanup.
- Published GPU examples: completed on the original isolated test runtime. Repeating the complete installation plus both GPU workflows on a clean machine remains a separate validation step.

Save the installer-generated `.runtime/installed-requirements.txt` when reporting a problem. It contains dependency versions; redact personal paths before sharing it. `setup.py` does not install arbitrary managers or the upstream author's full personal custom-node dependency collection.
