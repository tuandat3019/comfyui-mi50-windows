# Third-party notices

The root MIT license covers original code in this repository. It does not relicense dependency source, weights, binaries, or third-party-derived changes. Installers retrieve upstream sources/artifacts separately; those retain upstream notices and terms.

| Component | Provenance / terms |
| --- | --- |
| ComfyUI | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI), GPL-3.0; fetched as a separate pinned checkout |
| ComfyUI-GGUF padding compatibility | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF); original license preserved in [licenses/ComfyUI-GGUF.txt](licenses/ComfyUI-GGUF.txt) |
| TRELLIS.2 / AMD wrapper / o-voxel tiled helper | [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [dmonkman AMD wrapper](https://github.com/dmonkman/ComfyUI-Trellis2-AMD); [MIT notice](licenses/TRELLIS.txt). The helper is extracted from the pinned AMD wheel without importing its incompatible native binary. |
| CuMesh / CUB compatibility | [visualbruno/CuMesh](https://github.com/visualbruno/CuMesh); [MIT notice](licenses/CuMesh.txt). Bundled BVH/xatlas sources retain their upstream notices in the source checkout. |
| FlexGEMM | [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), upstream MIT terms |
| Eigen | [libeigen/eigen](https://gitlab.com/libeigen/eigen), upstream MPL-2.0 and source notices; fetched separately |
| nvdiffrast / `scripts/patch_raster_video.py` | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast), **Nvidia Source Code License (1-Way Commercial)**; full terms in [licenses/nvdiffrast.txt](licenses/nvdiffrast.txt). The derived PTX compatibility patch is supplied under the same license, including its research/evaluation/non-commercial use limitation. The root MIT license does not override those terms. |
| ZLUDA / community rocBLAS / AMD HIP DLLs | Downloaded from the projects/vendors linked in [setup.md](docs/setup.md); their respective licenses/EULAs apply |
| CUDA compiler/redistributables | [NVIDIA CUDA terms](https://docs.nvidia.com/cuda/eula/index.html), downloaded separately |
| Model weights | Respective model-repository terms linked in [models.md](docs/models.md), downloaded separately |

The code is a compatibility layer, not a claim of authorship of these upstream projects. No vendor endorsement or universal GPU/OS support is implied.
