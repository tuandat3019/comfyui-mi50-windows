# Architecture

```text
scripts/launch.py
  -> local ZLUDA executable
    -> isolated Torch2.7.1+cu118 Python
      -> mi50/bootstrap.py
        -> shared inference compatibility / allocator cap
        -> image: encoder/VAE stage offload + cache trim
        -> trellis: scoped FP16 source overlay + bounded math + rebuilt extensions
          -> pinned ComfyUI main.py
```

Source locations and runtime roots are computed from the checkout/configuration. No Kurai LLM, proxy, model router, media lease, private virtual-environment `.pth` or original absolute machine path is required.

The foreground launcher has a shared per-runtime lock and an unnamed Windows Job with kill-on-close. It owns only its launched process tree; it does not search for or terminate external model processes. Different runtime directories have independent locks, so users must still coordinate use of the same GPU across applications.

The TRELLIS import overlay is restricted to its pinned custom-node directory. It changes BF16 dtype references to FP16 in memory and adds `sdpa` to the sparse backend selector. It records source hashes by relative filename. It does not patch the Comfy core on disk. The node's missing optional profiler import is removed during installation because the pinned upstream commit references an absent file.

`bounded_sdpa` chunks queries while retaining all keys/values; one nominal FP32 score block is capped at32MiB, with a minimum single query. It supports the masks/GQA cases exercised by its tests and rejects nonzero dropout. Extremely large head/key dimensions can exceed the nominal budget for that minimum block; no arbitrary tensor-size memory guarantee is implied. Scaling before FP16 matmul differs numerically from some native attention implementations.

Sparse convolution uses explicit im2col/GEMM in chunks, nominal64MiB, minimum one voxel. Weighted interpolation uses Torch GPU gathers with FP32 accumulation. Native neighbor hashing/BVH/UV/rasterization/o-voxel remain native extensions translated by ZLUDA. Triton is installed for import compatibility; the selected model math routes avoid its convolution/interpolation kernels.

Batch-count metadata and row-wise uniqueness run on CPU because the legacy GPU kernels crashed in testing. Results return to the input device. Vector uniqueness/sort/scan remain on their original implementations. This is not an all-GPU claim.

The image hooks release encoder weights after conditioning, trim the allocator during the tested eight-step sampler's progress updates, and release diffusion weights before tiled VAE decode. They apply only to the image profile; the TRELLIS node retains its upstream low-VRAM stage behavior.

Native source patches retain backups and enforce expected upstream patterns. CuMesh uses stable scalar CUB sorts in z/y/x order for duplicate-face ordering; nvdiffrast scalar-video PTX helpers are lowered to ordinary CUDA arithmetic. CUDA11.8 compiler/runtime headers are staged alongside, separate from the AMD runtime that executes the translated kernels.
