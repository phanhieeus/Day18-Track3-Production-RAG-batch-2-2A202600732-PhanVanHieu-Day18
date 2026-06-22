from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
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
        # (endpoint ckey.vn không chắc có /embeddings của OpenAI nên dùng HF local)
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

        def _val(row, key):
            v = row.get(key, 0.0)
            try:
                return float(v) if v == v else 0.0  # loại NaN
            except (TypeError, ValueError):
                return 0.0

        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
            faithfulness=_val(row, "faithfulness"),
            answer_relevancy=_val(row, "answer_relevancy"),
            context_precision=_val(row, "context_precision"),
            context_recall=_val(row, "context_recall"))
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


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({"question": r.question, "worst_metric": worst_metric,
                       "score": round(metrics[worst_metric], 4), "_avg": avg,
                       "diagnosis": diagnosis, "suggested_fix": fix})
    scored.sort(key=lambda x: x["_avg"])
    for s in scored:
        s.pop("_avg")
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
