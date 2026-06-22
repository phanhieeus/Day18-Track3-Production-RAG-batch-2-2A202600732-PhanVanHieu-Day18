# Hướng dẫn hoàn thành Lab 18 — Production RAG Pipeline (từng bước)

> Tài liệu này gộp toàn bộ `README.md`, `ASSIGNMENT.md`, `RUBRIC.md` và phân tích code scaffold
> thành **một quy trình duy nhất**. Làm tuần tự từ trên xuống là xong bài.
>
> **Bài cá nhân.** Bạn cần điền code vào **9 hàm có `# TODO`** trong `src/`, chạy pipeline,
> rồi viết 3 file phân tích/reflection. Điểm: **100 + 10 bonus**.

---

## 0. Bức tranh tổng thể

Luồng pipeline production:

```
load_documents (M1)
   → chunk_hierarchical (M1)        # cắt nhỏ tài liệu, giữ parent-child
   → enrich_chunks (M5)             # làm giàu chunk bằng LLM trước khi embed
   → HybridSearch.index (M2)        # BM25 (tiếng Việt) + Dense (Qdrant)
   → search + reciprocal_rank_fusion (M2)
   → CrossEncoderReranker.rerank (M3)   # top-20 → top-3
   → LLM trả lời (đã viết sẵn trong pipeline.py)
   → evaluate_ragas + failure_analysis (M4)
```

So sánh với **naive baseline** (`naive_baseline.py`): chunk theo đoạn + chỉ dense, không hybrid/rerank/enrichment.

**Danh sách 9 TODO bạn phải làm** (đây là toàn bộ phần code chấm điểm):

| Module | File | Hàm cần điền |
|--------|------|--------------|
| M1 | `src/m1_chunking.py` | `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware` |
| M2 | `src/m2_search.py` | `segment_vietnamese`, `BM25Search.index/.search`, `DenseSearch.index/.search`, `reciprocal_rank_fusion` |
| M3 | `src/m3_rerank.py` | `CrossEncoderReranker._load_model`, `.rerank` |
| M4 | `src/m4_eval.py` | `evaluate_ragas`, `failure_analysis` |
| M5 | `src/m5_enrichment.py` | `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `_enrich_single_call` |

Các hàm khác (`load_documents`, `chunk_basic`, `compare_strategies`, `HybridSearch`, `enrich_chunks`, `save_report`, …) **đã viết sẵn** — đừng sửa.

---

## 1. Setup môi trường (≈10 phút)

> **Tin tốt:** trong máy bạn hiện tại Qdrant đã chạy (Docker container `...qdrant-1`), `.env` đã có
> `OPENAI_API_KEY`, Python là 3.11. Nên phần lớn bước này chỉ là kiểm tra lại.

### 1.1 Kích hoạt venv & cài dependencies

> ⚠️ **Cảnh báo môi trường:** `.venv` hiện tại là **Python 3.14 và đang rỗng** (chưa cài gì), dù
> `.python-version` ghi 3.11. RAGAS (`ragas<0.2`) + langchain `0.2.x` là bản cũ (2024), **rủi ro cao khi
> cài/chạy trên Python 3.14**. Khuyến nghị tạo lại venv bằng **Python 3.11**:
>
> ```powershell
> py -3.11 -m venv .venv        # cần đã cài Python 3.11; nếu không có thì tải về
> .\.venv\Scripts\Activate.ps1
> python --version              # phải in 3.11.x
> pip install -r requirements.txt
> ```
>
> Nếu buộc phải dùng 3.14 và RAGAS lỗi khi cài → gỡ pin cũ, thử `pip install -U "ragas>=0.2"` (API
> `evaluate(...)` ở §5.1 vẫn tương thích). Nhưng ưu tiên 3.11 để bám đúng đề.

```powershell
.\.venv\Scripts\Activate.ps1          # kích hoạt virtualenv
pip install -r requirements.txt
```

> `requirements.txt` có dòng `openai>=1.30` bị lặp 2 lần — vô hại, kệ nó. Nếu muốn gọn thì xoá 1 dòng.

### 1.1b Kiểm tra kết nối LLM (ckey.vn)

Sau khi cài xong, xác nhận provider hoạt động (đã test OK với key hiện tại → trả về "gpt-5.4-mini"):

```powershell
python -c "import config; c=config.get_openai_client(); print(c.chat.completions.create(model=config.LLM_MODEL, messages=[{'role':'user','content':'ping'}], max_tokens=5).model)"
```

### 1.2 Qdrant (vector DB cho M2 Dense)

```powershell
docker compose up -d        # nếu chưa chạy
docker ps                   # phải thấy container qdrant, port 6333
```

### 1.3 LLM provider (ckey.vn — OpenAI-compatible)

Lab này dùng provider tuỳ chỉnh **ckey.vn** qua endpoint OpenAI-compatible. File `.env` đã được cấu hình:

```
OPENAI_API_KEY=sk-...                       # key ckey.vn
OPENAI_BASE_URL=https://api.xah.io/v1       # endpoint OpenAI-compatible
LLM_MODEL=gpt-5.4-mini                       # model dùng cho M5 + pipeline + RAGAS
```

`config.py` đọc 3 biến này và cung cấp helper `get_openai_client()` (tự gắn `base_url`). **Mọi nơi gọi LLM
phải dùng helper này + `LLM_MODEL`, không dùng `OpenAI()` trần hay hardcode `"gpt-4o-mini"`** — nếu không
request sẽ bắn sai endpoint/sai model. `pipeline.py` và `naive_baseline.py` đã được sửa sẵn theo cách này.

> Thiếu key thì M4/M5 vẫn chạy ở chế độ fallback nhưng RAGAS sẽ ra 0.
> **Lưu ý embeddings:** endpoint ckey.vn có thể không hỗ trợ `/embeddings` của OpenAI. RAGAS mặc định cần
> OpenAI embeddings → ta sẽ thay bằng embeddings local `bge-m3` (xem [§5.1]). Dense search (M2) vốn đã dùng
> `bge-m3` local nên không phụ thuộc provider.

### 1.4 Pre-download model (tránh timeout giữa chừng)

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

> `all-MiniLM-L6-v2` dùng cho semantic chunking (M1), `bge-m3` cho dense embedding (M2),
> `bge-reranker-v2-m3` cho rerank (M3). Tải 1 lần, cache lại.

### 1.5 Chạy baseline TRƯỚC (bắt buộc)

> Làm bước này **sau khi đã code xong M4** (vì baseline gọi `evaluate_ragas`). Nếu chạy ngay bây giờ
> khi M4 còn rỗng thì RAGAS ra 0 — không sao, vẫn ghi nhận được số chunk. Khuyến nghị: cứ code M1→M5 trước,
> rồi mục [Phần 7] sẽ chạy `main.py` lo luôn cả baseline lẫn production.

---

## 2. Module 1 — Advanced Chunking (≈20 phút)

**File:** `src/m1_chunking.py` · **Test:** `pytest tests/test_m1.py`

Mỗi hàm trả về `list[Chunk]`. `Chunk` có 3 field: `text`, `metadata` (dict), `parent_id`.

### 2.1 `chunk_semantic` — nhóm câu theo độ tương đồng

Ý tưởng: tách câu → embed từng câu → nếu câu i giống câu i-1 (cosine ≥ threshold) thì gộp chung chunk, ngược lại mở chunk mới.

```python
def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    def cosine(a, b):
        return dot(a, b) / (norm(a) * norm(b) + 1e-9)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if cosine(embeddings[i - 1], embeddings[i]) < threshold:
            chunks.append(Chunk(text=" ".join(current),
                                metadata={**metadata, "strategy": "semantic"}))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(Chunk(text=" ".join(current),
                            metadata={**metadata, "strategy": "semantic"}))
    return chunks
```

### 2.2 `chunk_hierarchical` — parent (lớn) + child (nhỏ)

Trả về **tuple** `(parents, children)`. Mỗi child phải có `parent_id` trỏ tới parent của nó, và `parent.metadata["parent_id"]` phải khớp (test kiểm tra điều này).

```python
def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents, children = [], []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            _add_parent(parents, children, current, parent_size, child_size, metadata)
            current = ""
        current += para + "\n\n"
    if current.strip():
        _add_parent(parents, children, current, parent_size, child_size, metadata)
    return parents, children


def _add_parent(parents, children, text, parent_size, child_size, metadata):
    pid = f"parent_{len(parents)}"
    parents.append(Chunk(text=text.strip(),
                         metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                         parent_id=pid))
    # cắt parent thành child theo độ dài
    words = text.split()
    buf = ""
    for w in words:
        if len(buf) + len(w) + 1 > child_size and buf:
            children.append(Chunk(text=buf.strip(),
                                  metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
            buf = ""
        buf += w + " "
    if buf.strip():
        children.append(Chunk(text=buf.strip(),
                              metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
```

> Đặt `_add_parent` ở cùng file (ngay trên hoặc dưới `chunk_hierarchical`). Test dùng `parent_size=200, child_size=80` nên đảm bảo child luôn nhỏ hơn parent — cách cắt theo word ở trên thoả mãn.

### 2.3 `chunk_structure_aware` — cắt theo header markdown

```python
def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    for part in parts:
        if not part.strip():
            continue
        if re.match(r'^#{1,3}\s+', part):
            current_header = part.strip()
        else:
            chunks.append(Chunk(
                text=(current_header + "\n\n" + part.strip()).strip(),
                metadata={**metadata, "section": current_header, "strategy": "structure"}))
    return chunks
```

### 2.4 Test M1

```powershell
pytest tests/test_m1.py -v
python src/m1_chunking.py     # in bảng so sánh basic/semantic/hierarchical/structure
```

Pass khi: cả 3 strategy trả về chunk không rỗng, child có `parent_id` hợp lệ & nhỏ hơn parent, structure giữ header + có `section` trong metadata.

---

## 3. Module 2 — Hybrid Search (≈20 phút)

**File:** `src/m2_search.py` · **Test:** `pytest tests/test_m2.py`

### 3.1 `segment_vietnamese`

```python
def segment_vietnamese(text: str) -> str:
    from underthesea import word_tokenize
    segmented = word_tokenize(text, format="text")
    return segmented.replace("_", " ")
```

> **Quan trọng:** `underthesea` nối từ ghép bằng `_` (vd `nghỉ_phép`). Nếu không `replace("_", " ")`,
> BM25 coi `nghỉ_phép` là 1 token còn query `nghỉ phép` là 2 token → không khớp. Đây là lỗi bẫy phổ biến nhất của M2.

### 3.2 `BM25Search`

```python
def index(self, chunks: list[dict]) -> None:
    from rank_bm25 import BM25Okapi
    self.documents = chunks
    self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
    self.bm25 = BM25Okapi(self.corpus_tokens)

def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
    if self.bm25 is None:
        return []
    tokenized_query = segment_vietnamese(query).split()
    scores = self.bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_indices:
        if scores[i] > 0:                       # bỏ doc không liên quan
            doc = self.documents[i]
            results.append(SearchResult(text=doc["text"], score=float(scores[i]),
                                        metadata=doc.get("metadata", {}), method="bm25"))
    return results
```

### 3.3 `DenseSearch`

```python
def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
    from qdrant_client.models import Distance, VectorParams, PointStruct
    self.client.recreate_collection(
        collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    texts = [c["text"] for c in chunks]
    vectors = self._get_encoder().encode(texts, show_progress_bar=True)
    points = [PointStruct(id=i, vector=v.tolist(),
                          payload={**c.get("metadata", {}), "text": c["text"]})
              for i, (c, v) in enumerate(zip(chunks, vectors))]
    self.client.upsert(collection, points)

def search(self, query: str, top_k: int = DENSE_TOP_K,
           collection: str = COLLECTION_NAME) -> list[SearchResult]:
    query_vector = self._get_encoder().encode(query).tolist()
    response = self.client.query_points(collection, query=query_vector, limit=top_k)
    return [SearchResult(text=pt.payload["text"], score=pt.score,
                         metadata=pt.payload, method="dense")
            for pt in response.points]
```

> Dùng `query_points()` (qdrant-client mới), **không** dùng `search()` cũ.

### 3.4 `reciprocal_rank_fusion`

```python
def reciprocal_rank_fusion(results_list, k: int = 60, top_k: int = HYBRID_TOP_K):
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [SearchResult(text=e["result"].text, score=e["score"],
                         metadata=e["result"].metadata, method="hybrid")
            for e in ranked[:top_k]]
```

### 3.5 Test M2

```powershell
pytest tests/test_m2.py -v
```

> `test_bm25_*` không cần Qdrant. Các test trong file không gọi DenseSearch trực tiếp, nhưng pipeline thì có — nên giữ Qdrant chạy.

---

## 4. Module 3 — Reranking (≈15 phút)

**File:** `src/m3_rerank.py` · **Test:** `pytest tests/test_m3.py`

```python
def _load_model(self):
    if self._model is None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)
    return self._model

def rerank(self, query: str, documents: list[dict],
           top_k: int = RERANK_TOP_K) -> list[RerankResult]:
    if not documents:
        return []
    model = self._load_model()
    pairs = [(query, doc["text"]) for doc in documents]
    scores = model.predict(pairs)
    if isinstance(scores, (int, float)):
        scores = [scores]
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [RerankResult(text=doc["text"], original_score=doc.get("score", 0.0),
                         rerank_score=float(score), metadata=doc.get("metadata", {}), rank=i)
            for i, (score, doc) in enumerate(scored[:top_k])]
```

> Dùng `sentence_transformers.CrossEncoder`, **không** dùng `FlagEmbedding`/`FlagReranker`
> (crash với transformers ≥ 5.0). `FlashrankReranker` là optional — bỏ qua được.

```powershell
pytest tests/test_m3.py -v
```

Pass khi: trả về ≤ top_k kết quả, sort giảm dần theo `rerank_score`, doc "nghỉ phép" xếp trên "VPN"/"mật khẩu".

---

## 5. Module 4 — RAGAS Evaluation (≈15 phút)

**File:** `src/m4_eval.py` · **Test:** `pytest tests/test_m4.py`

### 5.1 `evaluate_ragas`

```python
def evaluate_ragas(questions, answers, contexts, ground_truths) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

        # LLM ckey.vn (OpenAI-compatible) + embeddings local bge-m3
        ragas_llm = LangchainLLMWrapper(ChatOpenAI(
            model=LLM_MODEL, api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None, temperature=0.0))
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=ragas_llm, embeddings=ragas_emb)
        df = result.to_pandas()
        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=row["contexts"], ground_truth=row["ground_truth"],
            faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
            answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
            context_precision=float(row.get("context_precision", 0.0) or 0.0),
            context_recall=float(row.get("context_recall", 0.0) or 0.0))
            for _, row in df.iterrows()]

        def avg(key):
            vals = [getattr(p, key) for p in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {"faithfulness": avg("faithfulness"),
                "answer_relevancy": avg("answer_relevancy"),
                "context_precision": avg("context_precision"),
                "context_recall": avg("context_recall"),
                "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
```

### 5.2 `failure_analysis`

```python
def failure_analysis(eval_results, bottom_n: int = 10) -> list[dict]:
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision, "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({"question": r.question, "worst_metric": worst_metric,
                       "score": round(metrics[worst_metric], 4), "avg": avg,
                       "diagnosis": diagnosis, "suggested_fix": fix})
    scored.sort(key=lambda x: x["avg"])
    for s in scored:
        s.pop("avg")
    return scored[:bottom_n]
```

```powershell
pytest tests/test_m4.py -v
```

> Test M4 chỉ kiểm tra cấu trúc dict trả về (không thật sự gọi OpenAI vì `evaluate_ragas(["q"],...)`
> sẽ rơi vào except và trả zeros — vẫn pass). RAGAS thật chỉ chạy khi bạn chạy pipeline ở [Phần 7].

---

## 6. Module 5 — Enrichment (≈20 phút)

**File:** `src/m5_enrichment.py` · **Test:** `pytest tests/test_m5.py`

Mỗi hàm có **nhánh OpenAI** (khi có API key) và **nhánh fallback** (không cần API). Phải viết cả hai —
test chạy được cả khi không có key.

> **Sửa dòng import đầu file** `m5_enrichment.py` thành:
> ```python
> import os, sys, re
> ...
> from config import OPENAI_API_KEY, LLM_MODEL, get_openai_client
> ```
> (các snippet bên dưới dùng `get_openai_client()` + `LLM_MODEL`; `re` cần cho fallback HyQA.)

### 6.1 Bốn hàm riêng lẻ

```python
def summarize_chunk(text: str) -> str:
    if OPENAI_API_KEY:
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text}],
                max_tokens=150)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if OPENAI_API_KEY:
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi 1 dòng."},
                    {"role": "user", "content": text}],
                max_tokens=200)
            qs = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in qs if q.strip()][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    if OPENAI_API_KEY:
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"}],
                max_tokens=80)
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    if OPENAI_API_KEY:
        try:
            import json as _json
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata. Trả JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text}],
                max_tokens=150)
            return _json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
```

> Cần `import re` ở đầu file `m5_enrichment.py` (hiện chưa có) vì `generate_hypothesis_questions` fallback dùng `re.split`. Thêm `import re` vào dòng import trên cùng.

### 6.2 `_enrich_single_call` — chế độ combined (1 call/chunk, ăn bonus +2)

```python
def _enrich_single_call(text: str, source: str) -> dict:
    if OPENAI_API_KEY:
        try:
            import json as _json
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"}],
                max_tokens=400)
            return _json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    return {}
```

```powershell
pytest tests/test_m5.py -v
python src/m5_enrichment.py     # demo 4 technique trên 1 câu mẫu
```

> `enrich_chunks` (đã viết sẵn) mặc định chạy combined mode → dùng `_enrich_single_call`. Test M5 gọi `methods=["contextual"]` nên `contextual_prepend` phải đúng.

---

## 7. Chạy pipeline end-to-end (≈20 phút)

Sau khi 5 module đã xong và `pytest tests/ -v` xanh:

```powershell
python main.py
```

`main.py` sẽ tự động:
1. Chạy `naive_baseline.py` → `naive_baseline_report.json`
2. Chạy production pipeline → `ragas_report.json`
3. Di chuyển cả 2 file vào `reports/`
4. In bảng so sánh Basic vs Production

> **Lưu ý:** chạy thẳng `python src/pipeline.py` cũng được, nhưng nó lưu `ragas_report.json` ở thư mục
> gốc, còn `check_lab.py` lại tìm trong `reports/`. Dùng `main.py` để khỏi phải tự di chuyển file.

Chạy mất vài phút (encode toàn bộ corpus + gọi LLM cho ~30 câu test + RAGAS). Theo dõi log `[1/4]…[4/4]` và `[Eval]`.

Sau khi xong, mở `reports/ragas_report.json`, điền bảng so sánh (để dùng cho reflection):

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|-----------|---|
| Faithfulness | ? | ? | ? |
| Answer Relevancy | ? | ? | ? |
| Context Precision | ? | ? | ? |
| Context Recall | ? | ? | ? |

**Mục tiêu điểm (RUBRIC):** ≥3 metric đạt 0.70 → 10đ. Bonus: Faithfulness ≥0.85 (+3), tất cả ≥0.75 (+3).

---

## 8. Viết phân tích (≈15 phút)

### 8.1 `analysis/failure_analysis.md`

Mở `reports/ragas_report.json` → mục `failures` (chính là output của `failure_analysis`, đã sort tệ nhất lên đầu).
Lấy **bottom-5 câu hỏi tệ nhất**, với mỗi câu ghi:

- Câu hỏi + score thấp nhất
- `worst_metric` (metric kém nhất)
- `diagnosis` (chẩn đoán) + `suggested_fix` (cách sửa) — đã có sẵn trong JSON
- **Error Tree**: vẽ cây quyết định "metric nào thấp → nguyên nhân → fix" (rubric cho 5đ khi có Error Tree, chỉ 3đ nếu thiếu).

Có thể dùng `templates/failure_analysis.md` làm khung.

### 8.2 `analysis/group_report.md`

Báo cáo tổng (dùng `templates/group_report.md`). Vì đây là bài cá nhân, điền thông tin của bạn + bảng so sánh ở [Phần 7].

---

## 9. Reflection cá nhân (≈30 phút) — 15 điểm

Tạo file **`analysis/reflections/reflection_PhanVanHieu.md`** (copy từ `reflection_TEMPLATE.md`), gồm 3 phần
mà ASSIGNMENT yêu cầu:

**Phần 1 — Mapping bài giảng → code** (5đ): bảng concept → hàm cụ thể bạn vừa viết → observation thực tế.

| Lecture Concept | Module | Hàm | Observation (số liệu thật từ run của bạn) |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | "threshold 0.85 tạo X chunks vs basic Y chunks" |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF giúp…" |
| Cross-encoder rerank | M3 | `CrossEncoderReranker.rerank()` | "latency ~X ms" |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "metric thấp nhất là … vì …" |
| Contextual embeddings | M5 | `contextual_prepend()` | "giảm retrieval failure bằng …" |

**Phần 2 — Khó khăn & cách giải quyết** (5đ): exact error message + cách debug + kiến thức đã bổ sung.
(Gợi ý lỗi hay gặp ở [Phụ lục].)

**Phần 3 — Action plan cho project cá nhân** (5đ): chunking/search/rerank/eval/enrichment nào sẽ áp dụng + timeline cụ thể.

---

## 10. Kiểm tra & nộp

```powershell
pytest tests/ -v                       # tất cả test xanh?
grep -r "# TODO:" src/m*.py            # phải = 0 dòng
python check_lab.py                    # checklist tổng
```

`check_lab.py` kiểm tra: 5 file source, `reports/ragas_report.json` (có key `aggregate`+`num_questions`),
file analysis, reflection cá nhân, số TODO còn lại, % test pass. Mục tiêu: in ra `🚀 Bài lab sẵn sàng để nộp!`.

### Checklist deliverable cuối

- [ ] `src/m1`…`m5` + `pipeline.py` — hết TODO, test pass
- [ ] `reports/ragas_report.json` + `reports/naive_baseline_report.json`
- [ ] `analysis/failure_analysis.md` — bottom-5 + Error Tree
- [ ] `analysis/group_report.md`
- [ ] `analysis/reflections/reflection_PhanVanHieu.md` — đủ 3 phần
- [ ] `python check_lab.py` ra "sẵn sàng nộp"
- [ ] Commit & push lên GitHub, nộp link repo

```powershell
git add -A
git commit -m "Complete Lab 18: implement M1-M5 + analysis + reflection"
git push
```

---

## Phụ lục — Lỗi thường gặp & cách xử lý

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| BM25 query "nghỉ phép" không ra kết quả | quên `replace("_", " ")` trong `segment_vietnamese` | sửa M2 §3.1 |
| `AttributeError: 'QdrantClient' object has no attribute 'search'` hoặc deprecated | dùng API cũ | dùng `query_points()` (M2 §3.3) |
| Reranker crash `XLMRobertaTokenizer` / transformers | dùng FlagEmbedding | dùng `sentence_transformers.CrossEncoder` (M3) |
| RAGAS trả toàn 0.0 | thiếu/ hết hạn key, Python < 3.11, hoặc base_url sai | kiểm tra `.env`, dùng venv 3.11 |
| `404 model not found` / `NotFoundError` | sai tên model hoặc còn hardcode `gpt-4o-mini` | dùng `LLM_MODEL` (=`gpt-5.4-mini`) ở mọi call |
| RAGAS lỗi `/embeddings 404` hoặc embeddings | ckey.vn không có endpoint embeddings của OpenAI | đã thay bằng `HuggingFaceEmbeddings("BAAI/bge-m3")` (§5.1) |
| `Connection error` tới api.xah.io | `OPENAI_BASE_URL` sai/thiếu | đảm bảo `.env` có `OPENAI_BASE_URL=https://api.xah.io/v1` |
| `Connection refused` cổng 6333 | Qdrant chưa chạy | `docker compose up -d` |
| `name 're' is not defined` trong M5 | thiếu `import re` | thêm `import re` đầu `m5_enrichment.py` |
| Encode/RAGAS rất chậm | tải model + gọi LLM 30 câu | bình thường vài phút; đã pre-download model ở §1.4 |
| `check_lab.py` báo thiếu `reports/ragas_report.json` | chạy `pipeline.py` trực tiếp | chạy `python main.py` (tự move vào `reports/`) |

---

### Thứ tự làm gọn nhất

1. Setup ([1]) → 2. Code M1→M5 ([2]–[6]), test từng module → 3. `python main.py` ([7]) →
4. Điền analysis + reflection ([8]–[9]) → 5. `python check_lab.py` → push ([10]).
