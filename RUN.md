# Hướng dẫn chạy Day 7

## Môi trường

Bài được chạy bằng Python 3.13.2 trên Windows.
Thực hiện các lệnh từ thư mục gốc repo.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dữ liệu và nguồn

- `data/day6/chunks.jsonl` chứa 71 chunk kế thừa từ cấu hình C Day 6.
- `data/day6/processed/` chứa 7 văn bản đã xử lý.
- `data/day6/source_manifest.json` chứa thông tin nguồn và snapshot.
- `data/day6/sources.md` chứa thông tin nguồn và giấy phép.
- `data/day6/ingestion_summary.json` ghi cấu hình embedding Day 6.

Các đường dẫn trong manifest và metadata được giữ theo repo Day 6.
Ví dụ `data/processed/giay_do.txt` tương ứng với bản sao
`data/day6/processed/giay_do.txt` trong repo này.

Khi phân phối lại dữ liệu cần giữ thông tin ghi công và giấy phép
được ghi trong hồ sơ nguồn.

## Tạo database Chroma

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.rebuild_store
```

Lần đầu cần mạng để tải snapshot embedding model.
Lệnh tạo collection `day6_c` trong `.runtime/chroma`,
sau đó kiểm tra ID, nội dung và metadata.

Model dùng revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`,
vector 384 chiều và cùng cách xử lý query/document của Day 6.

## Cấu hình Gemini API key

Chạy dòng đầu, nhập key ở lời nhắc rồi nhấn Enter.
PowerShell che nội dung nhập.

```powershell
$geminiSecret = Read-Host "Nhap Gemini API key" -AsSecureString
```

Sau đó chạy:

```powershell
$env:GEMINI_API_KEY = ([System.Net.NetworkCredential]::new("", $geminiSecret).Password).Trim()
Remove-Variable geminiSecret
```

Key chỉ được đặt trong phiên PowerShell hiện tại.
Các lần gọi Gemini phát sinh chi phí theo tài khoản sử dụng.

## Chạy thử retrieval

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.dense_search
.\.venv\Scripts\python.exe -X utf8 -m src.lexical_search
.\.venv\Scripts\python.exe -X utf8 -m src.hybrid_search
```

Mỗi chương trình chờ nhập một câu hỏi.

## Chạy agent

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.run_agent
```

Nhập câu hỏi khi xuất hiện `Nhap cau hoi:`.
Kết quả và trace được lưu trong `results/agent_smoke/`.

## Chạy benchmark

Retrieval trên dev:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.benchmark_dev
.\.venv\Scripts\python.exe -X utf8 -m src.benchmark_dev20
.\.venv\Scripts\python.exe -X utf8 -m src.benchmark_dev_filter
```

Agent trên dev:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_agent_dev
```

Chạy lại baseline dev bằng prompt và tham số baseline:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_agent_dev --configuration baseline
```

Có thể ghi rõ `--configuration selected` để chạy cấu hình chọn.
Mỗi lần chạy dev lưu prompt, model và tham số trong `protocol.json`.

Agent trên test với baseline và cấu hình đã chọn:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_agent_test
```

Retrieval trên test:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.benchmark_test
```

Chọn cấu hình bằng dev trước khi đánh giá test.
Các chương trình agent tạo thư mục riêng cho mỗi lần chạy.
Các chương trình benchmark retrieval ghi vào thư mục kết quả cố định,
nên chạy lại sẽ ghi đè các file cùng tên.

Mặc định `evaluate_agent_dev` chạy cấu hình đã chọn.
Kết quả baseline dev trước khi chỉnh được giữ trong artifact đã nộp.
Khi lỗi API hoặc retrieval, các lượt tìm đã hoàn thành cùng context
được giữ trong trace. Lượt gọi Gemini thất bại ghi trạng thái lỗi,
tham số đầu vào và thời gian; usage chỉ có khi API trả về.

## Kiểm tra code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bộ test có sẵn kiểm tra logic bằng mock.
Đánh giá chất lượng model thật nằm trong benchmark, trace và reviews.

## Kết quả đã báo cáo

- Dev baseline: `results/agent_dev/20261114T100229Z_992bf0eb/`
- Dev cấu hình chọn: `results/agent_dev/20261114T101320Z_78d7417a/`
- Test hai cấu hình: `results/agent_test/20261114T104435Z_03b82767/`
- Retrieval test: `results/retrieval_test/`
- Cấu hình chọn: `configs/selected_agent.json`
- Báo cáo: `report/individual/2A202603388_NguyenThuyDuong.md`

Nhãn trong `reviews.json` được chuẩn bị với AI hỗ trợ đối chiếu
câu trả lời và chunk trong trace. Việc xác nhận của học viên
cần được ghi nhận trong báo cáo.
