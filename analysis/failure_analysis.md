# Failure Analysis — Lab 18: Production RAG

**Học viên:** Phan Văn Hiếu (2A202600732) — cá nhân, tự làm M1→M5
**Ngày:** 2026-06-22
**LLM:** gpt-5.4-mini (ckey.vn, OpenAI-compatible) · **Embeddings:** BAAI/bge-m3 (local)
**Test set:** 20 câu hỏi · **RAGAS:** 4 metrics

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7317 | 0.6858 | **−0.0458** |
| Answer Relevancy | 0.8444 | 0.8290 | −0.0154 |
| Context Precision | 0.9083 | 0.8500 | −0.0583 |
| Context Recall | 0.8333 | 0.8250 | −0.0083 |

> ⚠️ **Phát hiện ngược kỳ vọng:** pipeline "production" (hierarchical chunking + enrichment +
> hybrid + rerank) **thấp hơn naive baseline (paragraph chunking + dense-only) ở cả 4 metric.**
> Đây là kết quả thật và là trọng tâm phân tích bên dưới — không phải lỗi pipeline (37/37 test pass,
> chạy end-to-end exit 0).

---

## Nguyên nhân gốc (vì sao Production < Baseline)

1. **Child chunk quá nhỏ cắt vụn thông tin.** `HIERARCHICAL_CHILD_SIZE = 256` ký tự. Nhiều câu trả lời
   nằm trong **bảng ngưỡng / bảng số liệu** (ngưỡng phê duyệt mua sắm, hạn mức bảo hiểm, bậc lương).
   Child 256 ký tự cắt rời dòng quyết định khỏi tiêu đề bảng → context retrieve được **thiếu đúng dòng
   chứa con số** → faithfulness/recall tụt. Pipeline hiện retrieve child nhưng **không return parent**
   (bước "child→parent" của hierarchical chưa được nối trong `run_query`), nên mất lợi thế cốt lõi.
2. **Enrichment thêm nhiễu vào context.** Combined enrichment prepend 1 câu "context" do LLM sinh +
   đổi metadata. Câu context này lọt vào đoạn đưa cho LLM trả lời → LLM bám vào diễn giải thay vì số liệu
   gốc → **faithfulness giảm mạnh nhất** (−0.046) và **precision giảm nhiều nhất** (−0.058).
3. **Corpus nhỏ (25 docs) → baseline vốn đã rất mạnh.** Dense-only trên paragraph chunk trả về nguyên
   đoạn đầy đủ ngữ cảnh → precision 0.91. Khi corpus nhỏ, "sophistication" của production chỉ thêm nhiễu.
   Đây là bài học **over-engineering trên corpus nhỏ**.

---

## Bottom-5 Failures (sắp theo avg 4 metric tăng dần)

### #1 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024 hiện hành: từ 3 năm trở lên cộng 1 ngày mỗi 3 năm (v2023 cũ yêu cầu 5 năm).
- **Got (suy từ metric):** faithfulness = 0.0 → câu trả lời chứa thông tin **không có trong context**
  (khả năng cao trộn nhầm số liệu v2023 "5 năm" do cả 2 phiên bản cùng được index).
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? *(có cả v2023 lẫn v2024 → nhiễu phiên bản)* → Query OK? *(có)*
- **Root cause:** version conflict — child chunk của `nghi_phep_nam_v2023` và `v2024` lẫn nhau, không có
  metadata "superseded" để lọc → LLM lấy nhầm bản cũ.
- **Suggested fix:** thêm metadata `version/effective_date` + lọc bản hiện hành; hoặc loại doc superseded khỏi index.

### #2 — Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** 200.000.000 VNĐ/năm (nội trú + ngoại trú + nha khoa).
- **Got (suy từ metric):** faithfulness = 0.0 → con số đưa ra không khớp/không có trong context retrieve.
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? *(child 256 ký tự cắt rời con số khỏi mục)* → Query OK? *(có)*
- **Root cause:** child chunk quá nhỏ → đoạn chứa "200 triệu" không lọt top-3 hoặc bị tách khỏi ngữ cảnh "PVI".
- **Suggested fix:** tăng `child_size`, hoặc structure-aware giữ nguyên mục; return parent khi trả lời.

### #3 — Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?  ⭐ (case study)
- **Expected:** Đơn > 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got (suy từ metric):** faithfulness = 0.0 → trả lời cấp phê duyệt không dựa trên context (bảng ngưỡng bị cắt).
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? *(bảng ngưỡng phê duyệt bị child-chunk cắt vụn)* → Query OK? *(có)*
- **Root cause:** `mua_sam.md` chứa bảng ngưỡng (5–50tr → Director; >50tr → CEO). Child 256 ký tự tách dòng
  ">50 triệu: CEO" khỏi tiêu đề bảng → context thiếu → LLM điền từ kiến thức nền.
- **Suggested fix:** structure-aware chunking giữ nguyên bảng; child→return parent; metadata filter `category=mua_sam`.

### #4 — Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG — phải báo trong 1 giờ qua helpdesk/hotline; tự xử lý là vi phạm nghiêm trọng.
- **Got (suy từ metric):** faithfulness = 0.5 → một phần câu trả lời không bám context (thiếu chi tiết "1 giờ"/"vi phạm").
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output một phần đúng → Context đúng? *(một phần)* → Query OK? *(có)* — đây là câu **negation**.
- **Root cause:** câu phủ định cần lấy đủ điều kiện; child chunk chỉ chứa "KHÔNG" mà thiếu quy trình báo cáo.
- **Suggested fix:** prompt nhấn "trả lời đầy đủ điều kiện/quy trình"; return parent để có đủ đoạn.

### #5 — Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần Giám đốc điều hành (CEO) phê duyệt (>14 ngày phải tự đóng BH phần mình).
- **Got (suy từ metric):** context_precision = 0.5 → context retrieve lẫn nhiều đoạn không liên quan.
- **Worst metric:** context_precision (0.5)
- **Error Tree:** Output ổn → Context lẫn nhiễu *(enrichment + hybrid kéo thêm đoạn thừa)* → Query OK? *(có)*
- **Root cause:** RRF + enriched chunk đưa lên đoạn về "nghỉ phép" chung chung trước đoạn về "ngưỡng ngày → cấp duyệt".
- **Suggested fix:** rerank đã có nhưng top-3 vẫn lẫn → tăng trọng số rerank / metadata filter theo loại nghỉ phép.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"* (#3)

**Error Tree walkthrough:**
1. **Output đúng?** → Không (faithfulness 0.0): cấp phê duyệt nêu ra không được context chống lưng.
2. **Context đúng?** → Không: bảng ngưỡng phê duyệt trong `mua_sam.md` bị child-chunk 256 ký tự cắt rời
   dòng ">50 triệu → CEO" khỏi tiêu đề bảng; top-3 không chứa dòng quyết định.
3. **Query rewrite OK?** → Có: câu hỏi rõ ràng, số tiền cụ thể.
4. **Fix ở bước:** **Chunking (M1) + ghép child→parent (pipeline).**

**Nếu có thêm 1 giờ, sẽ optimize:**
- Nối đúng cơ chế hierarchical: retrieve child (precision) → **return parent** (đủ ngữ cảnh bảng) trong `run_query`.
- Đổi corpus sang **structure-aware** cho các doc dạng bảng (mua sắm, lương, bảo hiểm).
- Thêm metadata `version/effective_date` + lọc bản hiện hành để diệt lỗi version conflict (#1).
- Tách câu "context" của enrichment ra khỏi đoạn đưa LLM trả lời (chỉ dùng để embed), tránh tụt faithfulness.
