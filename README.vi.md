# ComfyUI MI50 trên Windows

Repo gồm hai profile tùy chọn: **Z-Image GGUF tạo ảnh** và **TRELLIS.2 tạo mesh có texture**, dùng chung HIP5.7/ZLUDA/Torch. Đã chạy hoàn chỉnh trên một MI50 16GB, driver nhận tên `AMD Radeon Pro VII`. Chưa xác minh Linux, MI50 32GB hoặc các card gfx906 khác.

Kết quả mẫu robot seed42: Z-Image NSFW v2 Q8, 1024²/8 bước/Euler/simple/CFG1 chạy workflow **92,610 giây**; TRELLIS.2 pipeline1024/texture1024, 12/12/12 bước chạy **515,38 giây**, xuất **96.781 tam giác**. Lượt TRELLIS còn compile kernel decoder; chưa phải số đo toàn pipeline đã nóng cache. Xem [benchmark](benchmarks/README.md), [ảnh mẫu](examples/z-image-robot.png), [preview mesh](examples/robot-4views.png) và [GLB thật](examples/robot.glb).

## Cài và chạy

Cần Windows x64, Python3.12, Git, 7-Zip và driver AMD tương thích. TRELLIS cần thêm VS2022 Build Tools với C++/Windows SDK. Bộ cài chỉ giải nén runtime vào thư mục riêng, không cài driver hoặc sửa PATH hệ thống. Đọc [hướng dẫn cài và điều khoản runtime](docs/setup.md).

```powershell
git clone https://github.com/tuandat3019/comfyui-mi50-windows.git
cd comfyui-mi50-windows
py -3.12 scripts/setup.py --profile image --accept-runtime-licenses
```

Thêm 3D khi cần:

```powershell
py -3.12 scripts/setup.py --profile trellis --accept-runtime-licenses
```

Weights tải riêng theo [hướng dẫn model](docs/models.md). Sau đó bật profile ảnh:

```powershell
py -3.12 scripts/launch.py --profile image --device 0
```

ComfyUI ở `http://127.0.0.1:8188`. Để tạo 3D, Ctrl+C dừng profile ảnh rồi chạy `--profile trellis`; cổng mặc định8189. `--device` là ordinal HIP5.7, không suy từ Vulkan/DirectML. Chạy từng profile và nhường đúng GPU trước; launcher không tự dừng LLM hoặc chương trình khác.

Workflow mẫu là JSON API, gửi bằng `scripts/submit_workflow.py` như các lệnh trong [README chính](README.md). Có thể thay prompt/seed; ảnh TRELLIS cần đủ vật thể, nền trong suốt và có lề. Repo kèm ảnh robot đã chuẩn bị để thử lại kết quả mẫu.

Code đã tách khỏi Kurai; GPU evidence thuộc runtime thử gốc. Chưa chạy lại toàn bộ GPU workflow từ một lần cài sạch của package public. Bộ kiểm CPU kiểm math/patch/launcher/helpers; `launch.py --check` chỉ kiểm file, không chứng minh GPU chạy được. Cache ZLUDA giữ trên SSD để tránh compile lại.
