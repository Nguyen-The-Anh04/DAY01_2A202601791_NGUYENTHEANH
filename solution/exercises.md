# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay phần placeholder bằng câu trả lời thật
(chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**
> Temperature 0.0: Hang Sơn Đòong ở tỉnh Quảng Bình là hang đỘng tự nhiên lớn nhất thế giới và thể tích. Trong hang có sông ngầm, rừng nhiệt đới và đám mây bên trong lòng hang.

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp (0.0), mô hình trả lời rất nhất quán, tập trung vào sự thật phổ biến và có cấu trúc rõ ràng. Ở temperature 0.5, câu trả lời vẫn chính xác nhưng có thêm chi tiết mô tả sinh động hơn. Khi temperature tăng lên 1.0 và 1.5, mô hình trở nên sáng tạo hơn, đôi khi đưa ra những sự thật ít phổ biến hoặc diễn đạt theo cách bất ngờ, nhưng cũng có nguy cơ lan man hoặc sai lệch khỏi chủ đề.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần trả lời chính xác, nhất quán và an toàn — không có chỗ cho sự sáng tạo hay ngẫu nhiên. Temperature thấp giúp mô hình bám sát thông tin đã được huấn luyện, giảm nguy cơ bịa đặt thông tin hoặc trả lời không nhất quán giữa các lượt hỏi.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với bảng giá hiện tại, GPT-4o có giá output khoảng $0.010/1K token, còn GPT-4o-mini khoảng $0.0006/1K token. Vậy GPT-4o đắt hơn khoảng **16.7 lần** cho output. Với 10.000 người dùng × 3 lần/ngày × 350 token = 10.5 triệu token output/ngày, chi phí hàng ngày với GPT-4o-mini chỉ khoảng $6.30, còn GPT-4o lên tới ~$105. Trường hợp nên dùng GPT-4o: khi cần suy luận phức tạp, phân tích dữ liệu nhạy cảm hoặc viết code với yêu cầu cao về độ chính xác. Trường hợp nên dùng mini: chatbot FAQ, phân loại email, hoặc bất kỳ tác vụ đơn giản, lặp lại nhiều lần mà chất lượng phản hồi không cần quá tinh vi.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi ngắn gọn, dùng từ đơn giản, ví dụ về "sổ cái chung" hoặc "trò chơi ghi chép", độ dài khoảng 2–3 câu. Ngược lại, với persona chuyên gia tài chính, phản hồi dài hơn nhiều, chứa thuật ngữ như "cơ sở dữ liệu phân tán", "hàm băm", "mining", "consensus mechanism". System prompt đóng vai trò "định hướng tư duy" — nó không chỉ thay đổi cách diễn đạt mà còn chọn lọc thông tin nào được nhấn mạnh, ví dụ nào phù hợp, và mức độ chi tiết phù hợp với đối tượng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn ~100 từ tiếng Việt, tiktoken thường đếm được khoảng 110–120 token, trong khi ước lượng từ/0.75 cho khoảng 133 token. Chênh nhau khoảng 10–20%. Tiếng Việt tốn nhiều token hơn tiếng Anh vì: (1) tiếng Việt có nhiều âm tiết/dấu thanh đi kèm, mỗi dấu thường được mã hóa thành token riêng; (2) từ đơn tiếng Việt thường ngắn hơn (ví dụ "xe", "ăn", "và") nên số từ nhiều hơn cho cùng ý; (3) tokenizer của OpenAI được huấn luyện chủ yếu trên tiếng Anh nên không tối ưu với cấu trúc tiếng Việt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng cần phản hồi tức thì để cảm nhận "độ trễ thấp" — ví dụ chatbot trò chuyện, hỗ trợ khách hàng trực tiếp, hoặc tạo văn bản dài. Người dùng thấy từng từ xuất hiện dần, tạo cảm giác mô hình đang "suy nghĩ" và phản hồi nhanh. Ngược lại, non-streaming phù hợp khi cần xử lý hàng loạt (batch), lưu toàn bộ phản hồi vào database trước khi hiển thị, hoặc khi phản hồi ngắn (< 50 token) mà việc stream không mang lại lợi ích đáng kể về trải nghiệm.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm áp lực lên server khi nó đang quá tải — các client retry ở những thời điểm khác nhau, tránh "đàn áp" đồng loạt. Delay tăng dần (0.1s → 0.2s → 0.4s) cũng phản ánh mức độ nghiêm trọng của lỗi: nếu server vẫn lỗi sau nhiều lần thử, có thể là lỗi kéo dài và cần chờ lâu hơn. Nếu hàng nghìn client cùng retry với delay cố định 1 giây, tất cả sẽ gửi request lại cùng một thời điểm, tạo ra "làn sóng thứ hai" có thể làm server sập hoàn toàn — hiện tượng này gọi là "thundering herd".

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **Persona:** "Bạn là trợ lý học tập AI của sinh viên VinUniversity. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt. Nếu không chắc chắn, hãy nói 'Tôi không chắc' thay vì bịa đặt. Luôn đưa ra ví dụ cụ thể khi giải thích khái niệm."

> Giải thích lựa chọn từ ngữ:
> 1. **"Trả lời ngắn gọn, rõ ràng"** — Sinh viên thường cần thông tin nhanh giữa các buổi học; câu trả lời dài dòng sẽ làm giảm hiệu quả sử dụng.
> 2. **"Nếu không chắc chắn, hãy nói 'Tôi không chắc'"** — Đây là kỹ thuật "admit uncertainty" giúp giảm hallucination, đặc biệt quan trọng trong bối cảnh học thuật.
> 3. **"Luôn đưa ra ví dụ cụ thể"** — Học tập hiệu quả nhất khi kết nối lý thuyết với thực tế; ví dụ giúp sinh viên ghi nhớ và hiểu sâu hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất:** History chỉ giữ 3 lượt hội thoại gần nhất (6 message), nghĩa là trợ lý "quên" hoàn toàn ngữ cảnh từ các lượt trước đó. Điều này gây khó chịu khi sinh viên hỏi các câu liên quan đến chủ đề đã thảo luận trước đó.

> **Cải thiện đề xuất:** Thêm "bộ nhớ ngữ cảnh dài hạn" (long-term memory) bằng cách lưu tóm tắt hội thoại vào file JSON hoặc SQLite. Cách triển khai: sau mỗi phiên chat, gọi một lần API với prompt "Tóm tắt nội dung hội thoại này thành 3–4 câu" và lưu kết quả. Khi bắt đầu phiên mới, nạp tóm tắt này vào system prompt để mô hình có "ký ức" về các chủ đề đã thảo luận, bổ sung vào 3 lượt gần nhất.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
