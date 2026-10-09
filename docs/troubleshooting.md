# Troubleshooting

**Wrong card or no MI50.** `--device` selects the HIP5.7 ordinal before filtering. The bootstrap expects one visible gfx906-style device name. Use the pinned runtime DLLs; Vulkan and DirectML ordinals differed on the reference machine. This installer cannot make an unsupported display-driver installation work and does not change the card's driver/BIOS.

**First run appears slow.** ZLUDA compiles kernels. A cold image attempt exceeded15minutes during encoder compilation, whereas the measured image run used cached kernels. Keep the cache on SSD and allow a longer request timeout. Read the foreground console for progress. Do not compare a cold run with a cached run as if they were repeated throughput samples.

**Missing CUDA/ROCm DLL.** Run `launch.py --check`, re-run setup and inspect the first error. `rocsolver.dll`, `rocsparse.dll` and HIP RTC DLLs are required; copying only `amdhip64.dll` is insufficient. Do not copy ROCm10/gfx1030 `.pyd` extensions into this Torchcu118 environment. TRELLIS extensions are built separately for this recipe.

**Native compilation fails.** Confirm VS2022 C++ tools, Windows SDK, Python3.12 and adequate free RAM. Check `.runtime/trellis/build-*.log`. CUDA11.8 and MSVC14.44 needed the compatibility changes included here; unsupported-compiler flags do not establish that another compiler has been tested. Choose a short runtime path. `--skip-build` intentionally leaves TRELLIS incomplete.

**Access violation or CUDA error500.** The original setup encountered HIP5.7 softmax/layer-norm crashes, unsupported PTX instructions and native sparse metadata crashes. Launch through this repository's bootstrap rather than plain `python ComfyUI/main.py`; retain the pinned source/runtime and selected backends. Upgrading a node or changing SDPA/conv backends needs new validation.

**Image memory pressure.** The tested image profile uses an allocator cap, encoder/diffusion offload, sampler cache trim and explicit tiled VAE. Disable other applications on the selected card before trying the measured profile. The reference snapshots are not a total peak-VRAM guarantee for arbitrary resolution/steps/custom nodes.

**Missing nodes or model filename errors.** The API workflows require the corresponding profile. Check `/object_info` and `history.json`; edit model selectors to match the exact installed names. These JSONs are API format, not editor graph format. A file check does not validate model loading.

**3D quality.** Review geometry and texture from several angles. The sample GLB is textured and structurally valid, but it is not watertight/print-prepared and small facial details are imperfect. The CPU preview does not fully reproduce PBR shading. These are model/export limitations, not proof of universally correct geometry.
