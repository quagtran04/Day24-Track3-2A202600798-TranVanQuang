from __future__ import annotations

"""Module 4: RAGAS evaluation and failure analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAG_USE_RAGAS, TEST_SET_PATH


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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _token_f1(a: str, b: str) -> float:
    import re

    ta = re.findall(r"\w+", a.lower(), flags=re.UNICODE)
    tb = re.findall(r"\w+", b.lower(), flags=re.UNICODE)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    overlap = len(sa & sb)
    precision = overlap / len(sa)
    recall = overlap / len(sb)
    return 2 * precision * recall / (precision + recall + 1e-9)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS when available; otherwise return deterministic lexical scores."""
    try:
        if not RAG_USE_RAGAS:
            raise RuntimeError("set RAG_USE_RAGAS=1 to enable RAGAS LLM metrics")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
    except Exception as exc:
        print(f"  RAGAS evaluation fallback active: {exc}")
        per_question = []
        for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
            context_text = " ".join(ctxs)
            per_question.append(
                EvalResult(
                    question=question,
                    answer=answer,
                    contexts=ctxs,
                    ground_truth=ground_truth,
                    faithfulness=_token_f1(answer, context_text),
                    answer_relevancy=_token_f1(answer, question),
                    context_precision=_token_f1(context_text, question),
                    context_recall=_token_f1(context_text, ground_truth),
                )
            )

    def avg(name: str) -> float:
        if not per_question:
            return 0.0
        return sum(getattr(item, name) for item in per_question) / len(per_question)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or answer unsupported", "Tighten the prompt and cite context spans."),
        "context_recall": ("Relevant chunks are missing", "Improve chunking, add BM25 terms, or enrich chunks."),
        "context_precision": ("Retrieved context is noisy", "Add reranking and metadata filters."),
        "answer_relevancy": ("Answer does not directly address the question", "Improve prompt and answer format."),
    }

    rows = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        avg_score = sum(metrics.values()) / len(metrics)
        diagnosis, fix = diagnostic_tree[worst_metric]
        rows.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth,
                "worst_metric": worst_metric,
                "score": avg_score,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
            }
        )
    rows.sort(key=lambda row: row["score"])
    return rows[:bottom_n]


def _safe_float(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return 0.0 if math.isnan(value) or math.isinf(value) else value


def _json_safe(value):
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    serializable_per_question = [
        asdict(item) if isinstance(item, EvalResult) else item
        for item in results.get("per_question", [])
    ]
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(serializable_per_question),
        "per_question": serializable_per_question,
        "failures": failures,
    }
    report = _json_safe(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
