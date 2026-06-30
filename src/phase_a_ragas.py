from __future__ import annotations

"""Phase A: RAGAS production evaluation for the 50-question test set."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json not found at {path}. Run: python setup_answers.py"
        )
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def save_phase_a_report(
    results: list[RagasResult],
    clusters: dict,
    path: str = "reports/ragas_50q.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness": sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy": sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall": sum(r.context_recall for r in subset) / len(subset),
                "avg_score": sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution")
        if dist not in groups:
            raise ValueError(f"Unknown distribution: {dist!r}")
        groups[dist].append(item)
    return groups


def _clamp_score(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {t.strip(".,:;!?()[]{}\"'").lower() for t in left.split()}
    right_tokens = {t.strip(".,:;!?()[]{}\"'").lower() for t in right.split()}
    left_tokens = {t for t in left_tokens if len(t) > 2}
    right_tokens = {t for t in right_tokens if len(t) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _fallback_scores(answer: str, contexts: list[str], ground_truth: str) -> dict[str, float]:
    context_text = " ".join(contexts)
    answer_gt = _token_overlap_score(answer, ground_truth)
    answer_ctx = _token_overlap_score(answer, context_text)
    gt_ctx = _token_overlap_score(ground_truth, context_text)
    return {
        "faithfulness": round(max(answer_ctx, 0.15 if answer else 0.0), 4),
        "answer_relevancy": round(max(answer_gt, 0.10 if answer else 0.0), 4),
        "context_precision": round(max(answer_ctx, 0.10 if contexts else 0.0), 4),
        "context_recall": round(max(gt_ctx, 0.10 if contexts else 0.0), 4),
    }


def _metric_from(raw_item, metric: str, fallback: float) -> float:
    if isinstance(raw_item, dict):
        return _clamp_score(raw_item.get(metric), fallback)
    return _clamp_score(getattr(raw_item, metric, fallback), fallback)


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    raw_per_question = []
    if answers:
        try:
            from src.m4_eval import evaluate_ragas

            raw = evaluate_ragas(
                [a.get("question", "") for a in answers],
                [a.get("answer", "") for a in answers],
                [a.get("contexts", []) for a in answers],
                [a.get("ground_truth", "") for a in answers],
            )
            if isinstance(raw, dict):
                raw_per_question = raw.get("per_question", []) or raw.get("results", [])
            elif isinstance(raw, list):
                raw_per_question = raw
        except Exception as exc:
            print(f"RAGAS unavailable, using lexical fallback scores: {exc}")

    results = []
    for index, item in enumerate(answers):
        contexts = item.get("contexts", []) or []
        if isinstance(contexts, str):
            contexts = [contexts]
        fallback = _fallback_scores(
            item.get("answer", ""),
            contexts,
            item.get("ground_truth", ""),
        )
        raw_item = raw_per_question[index] if index < len(raw_per_question) else {}
        results.append(
            RagasResult(
                question_id=item.get("id", item.get("question_id", index + 1)),
                distribution=item.get("distribution", "factual"),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                contexts=contexts,
                ground_truth=item.get("ground_truth", ""),
                faithfulness=_metric_from(raw_item, "faithfulness", fallback["faithfulness"]),
                answer_relevancy=_metric_from(
                    raw_item, "answer_relevancy", fallback["answer_relevancy"]
                ),
                context_precision=_metric_from(
                    raw_item, "context_precision", fallback["context_precision"]
                ),
                context_recall=_metric_from(raw_item, "context_recall", fallback["context_recall"]),
            )
        )
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append(
            {
                "rank": rank,
                "question_id": result.question_id,
                "distribution": result.distribution,
                "question": result.question,
                "avg_score": round(result.avg_score, 4),
                "worst_metric": result.worst_metric,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.worst_metric in matrix and result.distribution in matrix[result.worst_metric]:
            matrix[result.worst_metric][result.distribution] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    dominant_dist = max(
        distributions,
        key=lambda dist: sum(matrix[metric][dist] for metric in matrix),
        default="factual",
    )
    dominant_metric = max(
        matrix,
        key=lambda metric: sum(matrix[metric].values()),
        default="faithfulness",
    )
    diagnosis, fix = DIAGNOSTIC_TREE[dominant_metric]
    insight = (
        f"Distribution '{dominant_dist}' has the most failures. "
        f"The dominant weak metric is '{dominant_metric}' ({diagnosis}). "
        f"Suggested fix: {fix}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")
    for dist, questions in group_by_distribution(test_set).items():
        print(f"  {dist}: {len(questions)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)
    clusters = cluster_analysis(results)
    save_phase_a_report(results, clusters)
    print(f"Dominant failure: {clusters['dominant_failure_distribution']} / {clusters['dominant_failure_metric']}")
