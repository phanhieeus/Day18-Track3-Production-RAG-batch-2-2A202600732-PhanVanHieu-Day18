# Individual Reflection — Lab 18: Production RAG

**Tên:** Phan Văn Hiếu — 2A202600732
**Module phụ trách:** Toàn bộ M1–M5 (bài cá nhân)
**Ngày:** 2026-06-22

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật từ run của tôi) |
|----------------|--------|-------------|--------------------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Nhóm câu theo cosine `all-MiniLM-L6-v2`, threshold 0.85. Tạo ít chunk hơn basic vì gộp câu cùng chủ đề; nhưng với corpus tiếng Việt model `MiniLM` (đa phần tiếng Anh) phân biệt chủ đề yếu hơn `bge-m3`. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` + `_add_parent()` | parent 2048 / child 256. Child có `parent_id` hợp lệ, nhỏ hơn parent (test pass). **Bài học đau:** child 256 ký tự cắt rời bảng số liệu → là nguyên nhân #1 khiến faithfulness tụt. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | `underthesea` nối từ ghép bằng `_` → bắt buộc `replace("_"," ")` nếu không BM25 không khớp query. RRF `1/(k+rank+1)` merge 2 danh sách → `method="hybrid"`. |
| Dense vector search | M2 | `DenseSearch.index/search` | Qdrant `query_points()` + `recreate_collection`, embed `bge-m3` (1024-dim). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder` (KHÔNG dùng FlagEmbedding). Rerank top-N → top-3, sort theo `rerank_score`; doc "nghỉ phép" lên trên "VPN" (test pass). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production: faithfulness **0.686** (thấp nhất), answer_relevancy 0.829, context_precision 0.850, context_recall 0.825. faithfulness thấp vì câu trả lời chứa thông tin ngoài context. |
| Diagnostic / Error Tree | M4 | `failure_analysis()` | Map worst-metric → (diagnosis, fix); sort theo avg tăng dần lấy bottom-N. 6/10 failure có worst_metric = faithfulness. |
| Contextual embeddings / enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()` | Combined mode 1 call/chunk (gpt-5.4-mini). Câu "context" sinh ra giúp embed nhưng khi lọt vào đoạn trả lời lại **kéo faithfulness xuống** — kết quả ngược với benchmark Anthropic vì corpus của tôi nhỏ. |

---

## Phần 2: Khó khăn & Cách giải quyết

1. **Provider LLM không phải OpenAI gốc (ckey.vn).**
   - *Lỗi:* các hàm hardcode `OpenAI()` + `model="gpt-4o-mini"` sẽ bắn sai endpoint/sai model.
   - *Giải quyết:* tập trung hoá trong `config.py`: `OPENAI_BASE_URL`, `LLM_MODEL`, helper
     `get_openai_client()`; mọi call dùng helper + `LLM_MODEL`.

2. **RAGAS không có endpoint embeddings của ckey.vn.**
   - *Lỗi tiềm tàng:* RAGAS mặc định gọi OpenAI embeddings → 404 trên ckey.vn.
   - *Giải quyết:* truyền tường minh `llm=ChatOpenAI(base_url=...)` + `embeddings=HuggingFaceEmbeddings("BAAI/bge-m3")`
     (local) vào `evaluate()`. Test mini RAGAS trên 1 mẫu trả về faithfulness 1.0 trước khi chạy full.

3. **Môi trường Python.**
   - *Lỗi:* `.venv` ban đầu là **Python 3.14 rỗng**; `ragas<0.2` (bản 2024) rủi ro trên 3.14. `pytest`
     cũng không có trong `requirements.txt`.
   - *Giải quyết:* tạo lại venv bằng `py -3.11 -m venv .venv`, cài requirements + `pytest`; thêm `pytest` vào requirements.

4. **Kết quả ngược kỳ vọng (production < baseline).**
   - *Khó khăn:* tưởng pipeline sai khi thấy cả 4 metric giảm.
   - *Debug:* kiểm tra 37/37 test pass + exit 0 → không phải bug; đọc `failures` thấy 6/10 là faithfulness=0
     trên câu hỏi numeric/version → suy ra child-chunk 256 ký tự cắt bảng + enrichment thêm nhiễu. Đây là
     finding chứ không phải lỗi.

**Thời gian debug chính:** ~setup môi trường + cấu hình RAGAS provider là phần tốn công nhất.

---

## Phần 3: Action Plan cho project cá nhân

### Project: RAG hỏi-đáp tài liệu nội bộ (áp dụng từ lab này)

**Hiện tại**
- Pipeline RAG: chunk + dense-only (giống naive baseline).
- Known issues: trả lời sai số liệu nằm trong bảng; lẫn lộn giữa các phiên bản tài liệu.

**Plan áp dụng (rút ra từ số liệu lab)**
1. [ ] **Chunking:** dùng **structure-aware** cho doc dạng bảng/điều khoản (giữ nguyên bảng); chỉ dùng
   hierarchical khi **thực sự nối child→return parent** — lab cho thấy child 256 ký tự rời rạc làm hại faithfulness.
2. [ ] **Search:** **Hybrid (BM25 tiếng Việt + Dense + RRF)** — giữ, vì BM25 bắt tốt từ khoá/số; nhớ
   `segment_vietnamese` + `replace("_"," ")`.
3. [ ] **Reranking:** **có**, `bge-reranker-v2-m3`; cân nhắc tăng ảnh hưởng rerank hoặc thêm metadata filter
   vì top-3 vẫn lẫn nhiễu (failure #5).
4. [ ] **Evaluation:** **RAGAS 4 metrics** làm thước đo bắt buộc *trước/sau* mỗi thay đổi — lab dạy rằng
   "thêm tính năng" có thể làm tệ đi; phải đo.
5. [ ] **Enrichment:** chỉ **contextual prepend / metadata** dùng để **embed**, KHÔNG đưa câu context sinh
   ra vào đoạn trả lời (tránh tụt faithfulness); thêm metadata `version/effective_date` để lọc bản hiện hành.

### Timeline
- **Tuần 1:** thêm RAGAS harness + đo baseline hiện tại; thêm metadata version, lọc doc superseded.
- **Tuần 2:** structure-aware chunking cho doc bảng + nối child→parent; đo lại, so Δ.
- **Tuần 3:** hybrid + rerank; tinh chỉnh prompt faithfulness; chốt cấu hình theo điểm RAGAS cao nhất.

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | N/A (bài cá nhân) |
| Problem solving | 5 |
