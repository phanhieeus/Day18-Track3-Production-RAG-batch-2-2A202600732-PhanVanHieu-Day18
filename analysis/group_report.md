# Group Report — Lab 18: Production RAG

**Học viên:** Phan Văn Hiếu (2A202600732) — bài cá nhân, tự làm toàn bộ 5 module
**Ngày:** 2026-06-22

## Phân công Module (cá nhân làm tất cả)

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phan Văn Hiếu | M1: Chunking | ✅ | 13/13 |
| Phan Văn Hiếu | M2: Hybrid Search | ✅ | 5/5 |
| Phan Văn Hiếu | M3: Reranking | ✅ | 5/5 |
| Phan Văn Hiếu | M4: Evaluation | ✅ | 4/4 |
| Phan Văn Hiếu | M5: Enrichment | ✅ | 10/10 |

**Tổng: 37/37 test pass · 0 TODO còn lại · pipeline chạy end-to-end (exit 0).**

## Kết quả RAGAS (20 câu hỏi)

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7317 | 0.6858 | −0.0458 |
| Answer Relevancy | 0.8444 | 0.8290 | −0.0154 |
| Context Precision | 0.9083 | 0.8500 | −0.0583 |
| Context Recall | 0.8333 | 0.8250 | −0.0083 |

> Production đạt 3/4 metric ≥ 0.70 (answer_relevancy, context_precision, context_recall ≥ 0.82);
> chỉ faithfulness 0.686 dưới ngưỡng.

## Key Findings

1. **Biggest improvement:** Không có metric nào tăng — đây chính là **bài học giá trị nhất**: trên corpus
   nhỏ (25 docs), baseline dense-only trên paragraph chunk đã rất mạnh (precision 0.91). Thêm
   hierarchical child 256 ký tự + enrichment chỉ **thêm nhiễu** và **cắt vụn bảng số liệu**.
2. **Biggest challenge:** Cấu hình RAGAS chạy với LLM OpenAI-compatible (ckey.vn) **không có endpoint
   embeddings** → phải thay bằng embeddings local `bge-m3` qua `LangchainEmbeddingsWrapper`.
3. **Surprise finding:** "Production RAG" phức tạp hơn **không tự động tốt hơn**. faithfulness tụt nhiều
   nhất (−0.046) vì câu "context" do enrichment sinh ra lọt vào đoạn trả lời, khiến LLM bám diễn giải
   thay vì số liệu gốc; và pipeline retrieve child nhưng **chưa return parent** nên mất ngữ cảnh bảng.

## Presentation Notes (5 phút)

1. **RAGAS naive vs production:** bảng trên — production thấp hơn nhẹ ở cả 4 metric.
2. **Biggest win:** về kỹ thuật, M2 Hybrid (BM25 tiếng Việt + Dense + RRF) và M3 rerank chạy đúng; về
   *insight*, win lớn nhất là nhận ra cần đo trước khi tin "thêm tính năng = tốt hơn".
3. **Case study — 1 failure + Error Tree:** "Mua thiết bị 55 triệu cần ai duyệt?" → faithfulness 0.0 →
   bảng ngưỡng phê duyệt bị child-chunk 256 ký tự cắt rời → fix ở chunking + child→parent
   (chi tiết trong `failure_analysis.md`).
4. **Next optimization nếu có thêm 1 giờ:** nối child→return parent trong `run_query`; structure-aware cho
   doc dạng bảng; metadata version để diệt version-conflict; tách câu context enrichment khỏi đoạn trả lời.
