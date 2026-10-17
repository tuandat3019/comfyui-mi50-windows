# RESTORE-IGNORED — ComfyUI MI50 Windows

Tài liệu khôi phục các file **không nằm trong Git**. Cập nhật 2026-10-17, trước khi chuyển
`C:\Projects\comfyui-mi50-windows` sang `F:\Projects\comfyui-mi50-windows`.

Repo hiện **không có file ignored nào trên đĩa** (chỉ source, docs, examples). `.gitignore`
đã chừa chỗ cho runtime/venv/model cho lần cài sau.

## 1. Các đường dẫn sẽ bị ignore khi cài

| Đường dẫn | Nguồn |
|---|---|
| `.runtime/` | runtime Python/Torch HIP/ZLUDA do `scripts/setup.py` tạo |
| `.venv/` | virtualenv |
| `models/`, `outputs/`, `logs/` | model và output |
| `*.safetensors`, `*.gguf`, `*.pt`, `*.onnx`, `*.dll`, `*.exe`, `*.zip`, `*.7z`, `*.whl` | binary/model |
| `local-config.json`, `.env` | cấu hình máy |

## 2. Khôi phục

```powershell
py -3.12 scripts/setup.py --profile image --plan
py -3.12 scripts/setup.py --profile image --accept-runtime-licenses
```

Profile 3D (TRELLIS.2):

```powershell
py -3.12 scripts/setup.py --profile trellis --accept-runtime-licenses
```

Tải model:

```powershell
py -3.12 scripts/download_models.py --profile image
```

Script dùng manifest/URL pin trong `scripts/` và `docs/`. Xem thêm:

- `docs/setup.md` — dependency, license, runtime.
- `scripts/build_extensions.py` — build native extension.
- `scripts/launch.py` — khởi chạy ComfyUI.
- `scripts/submit_workflow.py` — gửi workflow API.

Yêu cầu: Windows x64, Python 3.12, Git, 7-Zip, driver AMD; TRELLIS cần Visual Studio 2022
Build Tools + Windows SDK.

## 3. Không cần khôi phục

- `.runtime/`, `.venv/`, `outputs/`, `logs/`, cache kernel — tái tạo bằng setup.
- Model đã tải — tải lại theo `download_models.py`.

## 4. Kiểm tra

```powershell
py -3.12 scripts/setup.py --profile image --plan
py -3.12 scripts/setup.py --profile image --check
```
