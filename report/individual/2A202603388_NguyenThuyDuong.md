# Báo cáo Day 7

## Dữ liệu và benchmark

### Dữ liệu kế thừa từ Day 6

Corpus gồm 7 nguồn Wikipedia tiếng Việt về giấy dó, hội họa dân gian,
khắc gỗ và các dòng tranh Đông Hồ, Hàng Trống, Kim Hoàng, làng Sình.

Day 7 sử dụng 71 chunk của cấu hình C đã được chọn trên dev ở Day 6.
File chunk được lưu tại `data/day6/chunks.jsonl`. Mỗi chunk giữ ID,
nội dung, nguồn, mã đơn vị văn bản và metadata.

Trường `page` là mã đơn vị văn bản, ví dụ `giay_do_u001`.
Nguồn gốc, snapshot và thông tin giấy phép được lưu trong
`data/manifest.json` và `data/sources.md` của repo Day 6.
Khi chia sẻ lại dữ liệu cần giữ thông tin ghi công và giấy phép tương ứng.

### Embedding và vector store

- Embedding model là
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Revision là `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
- Vector có 384 chiều, chạy trên CPU.
- Văn bản được chia thành cửa sổ tối đa 126 token nội dung.
  Vector cửa sổ được gộp theo trọng số số token rồi chuẩn hóa L2.
- Document và query sử dụng cùng cách xử lý embedding.
- Vector store là Chroma PersistentClient, collection `day6_c`,
  dùng cosine và HNSW với `ef_search=100`.
- Các thí nghiệm ban đầu dùng database Chroma trong repo Day 6.
  Sau đó database được dựng lại tại `.runtime/chroma` trong repo Day 7
  bằng `python -m src.rebuild_store`, với cùng 71 chunk và cấu hình
  embedding. Đã kiểm tra ID, nội dung, metadata và thử truy xuất thành công.
- Cấu hình ingestion được sao chép vào
  `data/day6/ingestion_summary.json`.

### Agent và môi trường

Agent sử dụng Gemini `gemini-3.8-flash` qua SDK `google-genai`.
Môi trường thực nghiệm dùng Python 3.13.2 trên Windows.

Model quyết định tìm kiếm hoặc trả lời bằng JSON.
Chương trình thực thi retrieval tool, chọn context và gửi lại cho model.
Agent trả lời bằng tiếng Việt, kèm citation theo chunk ID,
hoặc từ chối khi thiếu bằng chứng.

### Benchmark và cách chấm

| Split | Có đáp án | Thiếu bằng chứng | Tổng |
| ----- | --------: | ---------------: | ---: |
| Dev   |        15 |                2 |   17 |
| Test  |         5 |                2 |    7 |
| Tổng  |        20 |                4 |   24 |

Bộ dev nằm trong `data/questions.json`. Các câu dev có đáp án
được chuyển từ Day 6. Bộ test mới nằm trong `data/test_questions.json`,
được chuẩn bị sau khi ghi nhận cấu hình chọn trong
`configs/selected_agent.json` và trước khi chạy đánh giá test.

Mỗi câu có đáp án gồm câu hỏi, đáp án chuẩn, nguồn, mã đơn vị văn bản
và trích đoạn evidence. Câu thiếu bằng chứng có ghi cách kiểm tra
trong corpus. Gold chỉ được dùng để chấm sau khi agent chạy.

Retrieval được đánh giá bằng Hit@1 và Evidence Hit@3.
Context Evidence Rate đo tỷ lệ câu có đủ evidence trong context cuối.
Với câu yêu cầu nhiều bằng chứng, cần đủ các phần đã khai báo.

Câu trả lời và context được chấm theo thang 0–2.
Citation được kiểm tra cả sự hiện diện trong context và khả năng
hỗ trợ nội dung trả lời. Abstention được kiểm tra trên câu thiếu
bằng chứng và báo thêm tỷ lệ từ chối nhầm câu có đáp án.

AI hỗ trợ tạo benchmark và đối chiếu câu trả lời với các chunk trong trace.
Phương thức đánh giá được ghi trong từng nhãn review.
Việc cùng công cụ tham gia chuẩn bị gold và đánh giá có nguy cơ thiên lệch.

## Retrieval và thử nghiệm

### So sánh phương pháp retrieval

Các phương pháp chạy trên cùng 71 chunk, 15 câu dev có đáp án
và cùng bộ nhãn evidence. Câu thiếu bằng chứng được loại khỏi mẫu số
retrieval metrics.

Dense dùng Chroma Day 6. Lexical dùng TF-IDF của starter.
Hybrid hợp nhất thứ hạng dense và TF-IDF bằng RRF,
với điểm mỗi kết quả là 1 / (60 + rank).

| Cấu hình               | Candidate mỗi retriever | Filter       |         Hit@1 | Evidence Hit@3 | Trung bình ms/query |
| ---------------------- | ----------------------: | ------------ | ------------: | -------------: | ------------------: |
| Dense                  |                      10 | Không        |  9/15 (60,0%) |  10/15 (66,7%) |               27,00 |
| TF-IDF                 |                      10 | Không        | 10/15 (66,7%) |  12/15 (80,0%) |                0,41 |
| Hybrid                 |                      10 | Không        | 10/15 (66,7%) |  13/15 (86,7%) |               27,43 |
| Hybrid, tăng candidate |                      20 | Không        | 10/15 (66,7%) |  13/15 (86,7%) |               29,65 |
| Dense có filter        |                      10 | Theo câu hỏi | 10/15 (66,7%) |  11/15 (73,3%) |               23,98 |
| TF-IDF có filter       |                      10 | Theo câu hỏi | 11/15 (73,3%) |  14/15 (93,3%) |                0,92 |
| Hybrid có filter       |                      10 | Theo câu hỏi | 10/15 (66,7%) |  14/15 (93,3%) |               24,91 |

Thời gian được đo sau setup. Thời gian hybrid là tổng thời gian dense,
lexical và hợp nhất thứ hạng, tái sử dụng kết quả thành phần trong lần chạy.
Lexical có filter tính cả thời gian tạo index TF-IDF trên tập đã lọc.
Các phép đo là từng lần chạy, chưa có thống kê qua nhiều lần lặp.

Retrieval chạy local, chi phí API của bước này bằng 0.
Chi phí phần cứng và điện chưa được đo.

Kết quả và trace nằm trong:

- `results/retrieval_dev/`
- `results/retrieval_dev20/`
- `results/retrieval_dev_filter/`

### Retrieval trên test sau khi chốt cấu hình

Chạy hybrid có filter bằng câu hỏi gốc trên cùng 5 câu test có đáp án.
Baseline dùng 10 candidate mỗi retriever, cấu hình chọn dùng 12.
Cả hai được chấm tại cùng cutoff 1 và 3.

| Cấu hình      |       Hit@1 | Evidence Hit@3 |
| ------------- | ----------: | -------------: |
| Baseline      | 4/5 (80,0%) |     5/5 (100%) |
| Cấu hình chọn | 4/5 (80,0%) |     5/5 (100%) |

Mỗi câu chỉ được tính hit khi có đủ các phần evidence đã khai báo
trong kết quả tại cutoff tương ứng. Hai câu thiếu bằng chứng
được loại khỏi mẫu số retrieval metrics.

Kết quả và ranking từng câu được lưu tại
`results/retrieval_test/summary.json` và
`results/retrieval_test/traces.json`.

Trên bộ test này, tăng candidate từ 10 lên 12 giữ nguyên
Hit@1 và Evidence Hit@3. Cấu hình đã chọn được giữ theo quyết định
trên dev trước khi chạy test.

### Thí nghiệm cải thiện 1 — tăng candidate

Giả thuyết là tăng candidate từ 10 lên 20 cho mỗi retriever có thể
đưa thêm evidence vào danh sách hợp nhất và cải thiện top 3.

Kết quả hybrid giữ nguyên Hit@1 là 10/15 và Evidence Hit@3 là 13/15.
Thời gian trung bình tăng từ 27,43 lên 29,65 ms/query.
Thí nghiệm này chưa cải thiện hai metric xếp hạng đã đo.

### Thí nghiệm cải thiện 2 — metadata filter

Giả thuyết là câu hỏi nêu rõ một dòng tranh sẽ tìm chính xác hơn
khi giới hạn theo metadata `painting_tradition`.

Filter được suy ra từ tên Đông Hồ, Hàng Trống, Kim Hoàng hoặc làng Sình
trong câu hỏi. Câu nhắc nhiều dòng tranh hoặc không nêu dòng tranh
được tìm trên toàn corpus. Cách suy ra filter chỉ sử dụng câu hỏi.

Hybrid có filter tăng Evidence Hit@3 từ 13/15 lên 14/15.
Hit@1 giữ nguyên 10/15. Câu `dev_10` về gỗ làm ván khắc Hàng Trống
đã tìm được evidence trong top 3.

### Phân tích lỗi và cải thiện agent trên dev

Với `dev_06`, model viết lại câu hỏi về hai loại ván in Đông Hồ,
nhưng các kết quả đưa vào context thiếu evidence cần thiết.
Benchmark dùng câu hỏi gốc có evidence ở hạng 2 của hybrid có filter.
Sau khi yêu cầu lượt tìm đầu giữ nguyên câu hỏi, lần thử riêng
trả lời đúng ván in nét và ván in màu sau một lượt tìm.

Với `dev_15`, evidence về cây dướng và cây dó liệt nằm ở hạng 11
của hybrid trong benchmark câu hỏi gốc. Agent dùng top 3 thiếu
đoạn này trong context và dừng khi đạt giới hạn tìm kiếm.
Lần thử top 12, budget 16.000 ký tự và giữ câu hỏi gốc ở lượt đầu
đã trả lời đúng sau một lượt tìm.

Sau đó chạy lại toàn bộ dev. Context Evidence Rate tăng từ
13/15 (86,7%) lên 15/15 (100%). Cấu hình mới đồng thời thay đổi
prompt, top-k, số candidate và context budget, nên kết quả toàn bộ dev
phản ánh tác động kết hợp của các thay đổi.

### Cấu hình được chọn

Chọn hybrid có filter theo câu hỏi, 12 candidate mỗi retriever,
top 12, RRF constant 60, context budget 16.000 ký tự
và tối đa 3 lượt tìm. Lượt tìm đầu sử dụng nguyên câu hỏi.

Lý do chọn là chất lượng evidence và câu trả lời trên dev.
Cấu hình được ghi trong `configs/selected_agent.json` trước khi
chạy test. Đánh đổi thời gian và token được trình bày ở phần agent.

## Agent và đánh giá

### Thiết kế agent

Gemini `gemini-3.8-flash` quyết định hành động bằng JSON.
Hành động `search` được chương trình chuyển thành lời gọi
`HybridRetriever.search`. Hành động `answer` chứa câu trả lời
và danh sách citation ID.

Retrieval tool dùng dense và TF-IDF, hợp nhất bằng RRF.
Filter được suy ra từ câu hỏi gốc và giữ trong các lượt tìm tiếp theo.
Agent chỉ nhận câu hỏi cùng lịch sử truy xuất và context.

Context được tạo bằng `pack_context` của starter:
loại chunk trùng ID, ưu tiên thứ tự xuất hiện và giữ nguyên
chunk vừa với budget, gồm cả nguồn và vị trí.
Cách này có thể ưu tiên đoạn từ lượt tìm trước khi context gần đầy.

Vòng lặp giới hạn 3 lượt retrieval, tương ứng tối đa 4 quyết định
từ model. Khi model yêu cầu tìm tiếp sau giới hạn, hệ thống từ chối.
Khi trả lời, starter kiểm tra citation ID có trong context
và xuất hiện trong câu trả lời. Citation thiếu hoặc không hợp lệ
dẫn đến phản hồi thiếu bằng chứng.

Prompt yêu cầu coi nội dung tài liệu là dữ liệu và bỏ qua
chỉ dẫn nằm trong tài liệu.

### Cấu hình so sánh

| Thuộc tính              | Baseline             | Cấu hình chọn        |
| ----------------------- | -------------------- | -------------------- |
| Model                   | gemini-3.8-flash     | gemini-3.8-flash     |
| Retrieval               | Hybrid RRF có filter | Hybrid RRF có filter |
| Candidate mỗi retriever | 10                   | 12                   |
| Top-k mỗi lượt          | 3                    | 12                   |
| Context budget          | 6.000 ký tự          | 16.000 ký tự         |
| Số lượt tìm tối đa      | 3                    | 3                    |
| Query đầu tiên          | Model quyết định     | Giữ nguyên câu hỏi   |

### Nhãn đánh giá

Câu trả lời có đáp án được chấm 0–2 theo độ đúng và đầy đủ.
Context được chấm 0–2 theo evidence và khả năng xác định nguồn.
Citation được đối chiếu với nội dung chunk trong context.
Các nhãn có lý do và ghi phương thức AI hỗ trợ đánh giá.

Citation được đếm theo danh sách ID trong trace.
Một ID được dùng nhiều lần trong câu trả lời vẫn được tính
theo danh sách này. Khả năng hỗ trợ các ý được dẫn được đọc đối chiếu.

| Cấu hình / split | Tool use | Grounded answer 0/1/2 | Trung bình /2 | Context 0/1/2 | Citation hỗ trợ | Câu có citation / câu trả lời | Abstention đúng | Từ chối nhầm |
| ---------------- | -------- | --------------------- | ------------: | ------------- | --------------- | ----------------------------- | --------------- | ------------ |
| Chọn / dev       | 17/17    | 0/0/15                |          2,00 | 0/0/15        | 18/18           | 15/15                         | 2/2             | 0/15         |
| Baseline / test  | 7/7      | 0/0/5                 |          2,00 | 0/0/5         | 5/5             | 5/5                           | 2/2             | 0/5          |
| Chọn / test      | 7/7      | 0/0/5                 |          2,00 | 0/0/5         | 5/5             | 5/5                           | 2/2             | 0/5          |

Ở bảng trên, cột 0/1/2 lần lượt ghi số câu tại mỗi mức điểm.
Hai câu thiếu bằng chứng được đánh giá bằng abstention.

Trên test, cả hai cấu hình đều dùng một lượt retrieval cho
5 câu có đáp án và ba lượt cho 2 câu thiếu bằng chứng.
Tổng cộng mỗi cấu hình có 11 lượt retrieval và 18 lượt gọi model.

Baseline từ chối hai câu thiếu bằng chứng khi chạm giới hạn tìm.
Cấu hình chọn chủ động nêu thiếu thông tin sau ba lượt tìm,
sau đó starter chuyển thành trạng thái abstain do citation rỗng.

### Thời gian, token và chi phí

| Cấu hình / split | Tổng giây | Lượt gọi model | Tổng token API báo cáo |
| ---------------- | --------: | -------------: | ---------------------: |
| Baseline / dev   |    143,66 |             42 |                 45.850 |
| Chọn / dev       |    147,91 |             38 |                 89.179 |
| Baseline / test  |     64,42 |             18 |                 17.258 |
| Chọn / test      |     65,04 |             18 |                 26.261 |

Thời gian trên loại trừ setup model embedding và retriever.
Token được lấy từ usage metadata do API trả về.

Ảnh trang chi phí do học viên cung cấp hiển thị `Total cost = 4.59k`
cho ngày 14/11/2026, tương đương khoảng **4.590 VNĐ** theo
cách hiển thị rút gọn của trang. Đơn vị VNĐ do học viên xác nhận.

Số liệu này được ghi nhận ở mức tổng theo ngày trên trang chi phí.
Phạm vi có thể gồm các lần thử kết nối, smoke test, dev, test và những
lượt sử dụng khác của key trong ngày. Ảnh chưa cung cấp phân bổ chi phí
cho từng cấu hình hoặc từng lần chạy. Bảng token ở trên phục vụ so sánh
mức sử dụng của các lần chạy đã lưu trace.

Cấu hình chọn cải thiện Context Evidence Rate trên dev
từ 13/15 lên 15/15, đồng thời tăng lượng token.
Trên bộ test 7 câu, hai cấu hình đạt cùng chất lượng đã chấm,
trong khi cấu hình chọn dùng nhiều hơn khoảng 52% tổng token.

### Ví dụ và lỗi

| Query   | Kết quả và nhận xét                                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| dev_06  | Baseline thiếu evidence và từ chối. Sau thay đổi, trả lời đủ ván in nét và ván in màu, citation d97706bf1aa4024f76705bb5.       |
| dev_15  | Baseline thiếu đoạn về nguyên liệu thay thế. Cấu hình chọn trả lời cây dướng và cây dó liệt, citation eadf6a9943a9b4457c49e3bd. |
| test_01 | Cả hai cấu hình trả lời đúng in nhá và in đồ, citation 0bd27690a3583722fc7c9a06.                                                |
| test_06 | Cả hai cấu hình từ chối câu hỏi doanh thu Kim Hoàng năm 2025, không đưa số liệu suy đoán.                                       |

Một lần smoke test gặp lỗi API 503 UNAVAILABLE.
Lần chạy lại thành công. Trace lỗi được giữ trong
`results/agent_smoke/21f2bdaee5624845b0fd292481bf2598.json`.
Hai lần chạy toàn bộ dev và lần chạy test được báo cáo đều có 0 lỗi.

### Đường dẫn kết quả và cách chạy

- Dev baseline nằm trong `results/agent_dev/20261114T100229Z_992bf0eb/`.
- Dev cấu hình chọn nằm trong `results/agent_dev/20261114T101320Z_78d7417a/`.
- Test nằm trong `results/agent_test/20261114T104435Z_03b82767/`.
- Mỗi thư mục cấu hình test có `reviews.json`.
- Tổng hợp test nằm trong `evaluation_summary.json`.
- Prompt và cấu hình test được lưu trong `protocol.json`.

Lệnh chạy từ thư mục gốc repo, với môi trường Python đã cài thư viện
và biến môi trường `GEMINI_API_KEY`:

    python -X utf8 -m src.run_agent
    python -X utf8 -m src.evaluate_agent_dev
    python -X utf8 -m src.evaluate_agent_test

### Giới hạn

Benchmark nhỏ, bộ test chỉ có 5 câu có đáp án và 2 câu thiếu bằng chứng,
tập trung vào tranh Kim Hoàng và làng Sình.
Kết quả chưa phản ánh đầy đủ các chủ đề khác trong corpus.

AI tham gia chuẩn bị gold và đối chiếu câu trả lời với chunk trong trace,
nên có nguy cơ thiên lệch. Việc đối chiếu trực tiếp nguồn gốc
và xác nhận của học viên cần được ghi nhận khi hoàn thiện bài nộp.

Các phép đo thời gian là từng lần chạy, chịu ảnh hưởng của dịch vụ API.
Code truy xuất hiện dùng database tại `.runtime/chroma` trong repo Day 7.
Cần cài các thư viện trong `requirements.txt` và tải đúng revision
embedding model trước khi dựng lại database.
Các phép chạy đã báo cáo sử dụng môi trường Python của Day 6.

## Thử thách nâng cao

Bài hiện hoàn thành các thí nghiệm thuộc phần cơ bản.
Thử thách tài liệu từ 100 trang chưa được thực hiện.

## Kiểm tra code

Đã chạy lệnh sau từ thư mục gốc repo:

    python -m unittest discover -s tests -v

Kết quả kiểm tra cuối là 18/18 unit test đạt, gồm 15 test có sẵn
và 3 test hồi quy cho việc giữ trace khi model hoặc retrieval gặp lỗi.
Các test dùng mock kiểm tra logic code.
Kết quả agent thật được đánh giá riêng bằng benchmark và trace
đã trình bày ở trên.

Sau khi đánh giá test, code được bổ sung nhật ký lượt gọi Gemini thất bại,
giữ các retrieval event và context đã có khi xảy ra lỗi, cùng lựa chọn
`--configuration baseline` để chạy lại baseline trên dev.
Các thay đổi phục vụ ghi log và tái lập. Prompt và tham số retrieval
của hai cấu hình test được giữ theo `protocol.json` của lần chạy đã báo cáo.
Trace cũ ghi theo phiên bản code tại thời điểm chạy, nên trace lỗi 503 cũ
có thể thiếu các retrieval event đã diễn ra trước lỗi.
