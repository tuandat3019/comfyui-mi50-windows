# Reference measurements — 2026-10-09

One **MI5016GB on Windows**, reported as AMD Radeon Pro VII under the PRO26Q1 Vega/Polaris driver. ComfyUI commit `d91ed5f5b7fa60fa18464c2ad7c80254da2f0f29`, Torch2.7.1+cu118, Python3.12.10, HIP runtime50731541, ZLUDA rocm5 release854c58e. Source/runtime pins are in `config/`.

| Measurement | Z-Image NSFW v2 Q8 | TRELLIS.2-4B |
| --- | ---: | ---: |
| ComfyUI execution | 92.610 s | 515.38 s (console) |
| Sampling | 57.2 s, 8 steps | SS72.279 / shape66.832 / texture38.078 s, 12 steps each |
| Worker/controller generation timing | Worker111.141 s | Controller generation516.198 s |
| Original full harness wall time | 121.952 s | 569.375 s |

Z-Image: 1024×1024, seed42, Euler/simple, CFG1, Qwen3-4B encoder storedBF16/computeFP16, Flux VAE FP32 tiled512/64, split attention, allocator80%, stage offload and step cache trim. Existing ZLUDA cache, newly loaded worker/weights; not a warm persistent-server request.

TRELLIS: seed42, pipeline1024, texture1024, target100Kfaces, token cap24576, sparse structure32, low-VRAM/tiled decoder, Euler12/12/12, SDPA/FlexGEMM, FP16 overlay. The reference is the Z-Image robot after CPU background preparation/padding. This run performed all three sampling stages; no latent replay. Shape decode251.443s included cold decoder compilation. Texture decode8.114s. Model-run profile492.239s includes stage load/unload, and does not equal the whole export workflow.

The original full harness included reference preparation, device checks, server start and a private media controller. Its LLM was already manually stopped with an empty session; the measured wall time does **not** include saving/restoring a resident LLM. The standalone public launcher removes that private controller, so those original totals are historical evidence, not a predicted wall time for this package.

## Artifact validation

The actual `examples/robot.glb` has96,781faces and63,421 UV vertices, one material and two embedded textures; the base-color texture is1024². Structural inspection found finite geometry, valid indices/UVs, zero degenerate faces and consistent winding. After UV-seam merging there are48,378vertices; the mesh is not watertight and was not prepared for printing. Front/back/side views were physically reviewed: recognizable red robot, plausible limbs/back, weaker small mouth detail.

The preview uses per-triangle base-color sampling in a CPU inspection renderer. It is not a final PBR render. `examples/z-image-robot.png` retains the original image pixels with ancillary text metadata removed for publication; the prepared RGBA input is separate. Artifact hashes are in `results.json`.

## Limits and comparison discipline

- Single successful final example per profile. Two last image runs had similar sampling around57s, but used different VAE policies.
- No complete fully warm TRELLIS run, cold-install end-to-end rerun of this extracted repo, same-settings1024 RX6800 A/B, multi-GPU run or long-term stability study.
- Earlier image runs were424.891s/442.718s before the combined allocator/softmax/cache policy changes. The final92.610s is about4.59× faster than the first of those workflow measurements, but several changes were combined; this is not a per-patch speedup or comparison with another GPU.
- Model-run Torch peak allocated2905.653MiB/reserved3078MiB is not whole-export peak VRAM. Image WDDM snapshots are also not whole-workflow peaks; no minimum-VRAM promise is made from them.
- CPU work remains for metadata, input preparation and inspection. No claim that all operations run on GPU.

## Reproduce

Install the pinned profile(s), place/download the exact weights, launch one profile, and submit the included API workflow with seed42. Use the transparent `examples/robot-reference.png` for TRELLIS. Retain the launcher console, installed package list, `history.json`, `timing.json`, model revisions and cache condition. Record server startup separately from request execution. Measure a second run after complete compilation to distinguish cold cost from cached performance; this repository has not filled in that missing measurement.
