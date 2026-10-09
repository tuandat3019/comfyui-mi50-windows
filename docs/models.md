# Model files

Weights are not included. The separate downloader is opt-in:

```powershell
py -3.12 scripts/download_models.py --profile image --plan
.runtime/venv/Scripts/python.exe scripts/download_models.py --profile image
```

Use the runtime Python for actual downloads, which provides `huggingface_hub`. Choose `--profile trellis` for 3D weights. `--runtime` works with both. Review each model repository's license/access conditions first; authentication, if required, uses your local Hugging Face configuration. The script never stores tokens in this repository or starts GPU inference.

## Z-Image example

| File under `.runtime/ComfyUI/models/` | Source |
| --- | --- |
| `diffusion_models/z_image_nsfw_v2-Q8_0.gguf` | [lesliemore/z-image-turbo-nsfw-v2-GGUF](https://huggingface.co/lesliemore/z-image-turbo-nsfw-v2-GGUF), revision `7567e753c45bdd16ea9c1c052af97c27a8a4a071` |
| `text_encoders/qwen_3_4b.safetensors` | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo), revision `6fc90a3b1b653e935a0d175e260736de25b84df5` |
| `vae/ae.safetensors` | Same Comfy-Org repository/revision |

The published image measurement uses **NSFW v2 Q8**, with a normal toy-robot prompt. The original Z-Image Turbo checkpoint has not been validated with this extracted package. Other files may be selected by editing the API workflow; they do not inherit this benchmark. The Qwen encoder is stored in BF16 but computes in FP16 in this profile. VAE runs FP32 with tile512/overlap64.

## TRELLIS example

The wrapper expects these local directories:

- `models/microsoft/TRELLIS.2-4B/`: model JSON and `ckpts/*.json`, `ckpts/*.safetensors` from [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B), revision `af44b45f2e35a493886929c6d786e563ec68364d`.
- `models/facebook/dinov3-vitl16-pretrain-lvd1689m/`: `model.safetensors`, `config.json`, `preprocessor_config.json` from [visualbruno/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/visualbruno/dinov3-vitl16-pretrain-lvd1689m), revision `8463e34549282813c2cbf67241b27c6fe8fa6321`.
- `models/microsoft/TRELLIS-image-large/ckpts/`: `ss_dec_conv3d_16l8_fp16.json` and `.safetensors` from [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large), revision `25e0d31ffbebe4b5a97464dd851910efc3002d96`.

The example loads a prepared RGBA PNG and does not invoke an automatic background-removal model. Keep the whole object visible with padding; use `examples/robot-reference.png` to reproduce the reference. Image framing and the reference's unseen surfaces affect mesh quality.

TRELLIS's imported wrapper can download missing files during loading. Downloading in advance keeps network time out of inference measurements. Pinning repositories does not mean every file/resolution has been tested: the demonstrated workflow is pipeline1024/texture1024, low-VRAM, tiled decoder, SDPA and FlexGEMM, 12 steps per stage.
