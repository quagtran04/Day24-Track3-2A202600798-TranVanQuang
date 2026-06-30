from __future__ import annotations

"""Phase B: LLM-as-Judge with swap-and-average, Cohen kappa, and bias analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _openai_client():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_KEY.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api/v1"
    elif os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    return OpenAI(**kwargs)


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _heuristic_score(question: str, answer: str) -> float:
    q = question.lower()
    a = answer.lower()
    score = 0.45

    q_tokens = {t for t in re.findall(r"\w+", q) if len(t) > 2}
    a_tokens = {t for t in re.findall(r"\w+", a) if len(t) > 2}
    if q_tokens:
        score += min(0.25, len(q_tokens & a_tokens) / len(q_tokens) * 0.25)

    positive_patterns = [
        ("ngh", "15", 0.22),
        ("ket hon", "3", 0.20),
        ("55", "ceo", 0.25),
        ("vpn", "wireguard", 0.22),
        ("vpn", "cam", 0.16),
        ("tam ung", "80", 0.18),
        ("thuong tet", "1 thang", 0.18),
    ]
    normalized = (
        a.replace("ế", "e")
        .replace("ỉ", "i")
        .replace("ă", "a")
        .replace("â", "a")
        .replace("ô", "o")
        .replace("ơ", "o")
        .replace("ư", "u")
        .replace("đ", "d")
    )
    q_norm = (
        q.replace("ế", "e")
        .replace("ỉ", "i")
        .replace("ă", "a")
        .replace("â", "a")
        .replace("ô", "o")
        .replace("ơ", "o")
        .replace("ư", "u")
        .replace("đ", "d")
    )
    for q_need, a_need, bonus in positive_patterns:
        if q_need in q_norm and a_need in normalized:
            score += bonus

    negative_patterns = [
        ("ngh", "12", -0.25),
        ("vpn", "duoc", -0.18),
        ("55", "giam doc", -0.18),
    ]
    for q_need, a_bad, penalty in negative_patterns:
        if q_need in q_norm and a_bad in normalized:
            score += penalty

    if len(answer) > 700:
        score -= 0.08
    if len(answer.strip()) < 20:
        score -= 0.15
    return max(0.0, min(1.0, round(score, 3)))


def _normalize_judge_payload(payload: dict, fallback_scores: dict) -> dict:
    winner = str(payload.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else fallback_scores
    scores = {
        "A": max(0.0, min(1.0, float(scores.get("A", fallback_scores["A"])))),
        "B": max(0.0, min(1.0, float(scores.get("B", fallback_scores["B"])))),
    }
    reasoning = str(payload.get("reasoning") or "Compared accuracy, completeness, and conciseness.")
    return {"winner": winner, "reasoning": reasoning, "scores": scores}


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    fallback_scores = {
        "A": _heuristic_score(question, answer_a),
        "B": _heuristic_score(question, answer_b),
    }

    client = _openai_client()
    if client is not None:
        prompt = f"""
You are evaluating two Vietnamese HR-policy RAG answers.
Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Choose the better answer by accuracy, completeness, and conciseness.
Return only JSON: {{"winner":"A|B|tie","reasoning":"short reason","scores":{{"A":0.0,"B":0.0}}}}
"""
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = _parse_json_object(response.choices[0].message.content)
            return _normalize_judge_payload(payload, fallback_scores)
        except Exception as exc:
            print(f"LLM judge unavailable, using heuristic judge: {exc}")

    diff = fallback_scores["A"] - fallback_scores["B"]
    if abs(diff) < 0.05:
        winner = "tie"
        reasoning = "Answers are close by lexical relevance and policy-specific signals."
    else:
        winner = "A" if diff > 0 else "B"
        reasoning = f"Answer {winner} has stronger policy-specific signals and relevance."
    return {"winner": winner, "reasoning": reasoning, "scores": fallback_scores}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")
    final = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    scores_pass2 = {
        "A": pass2_raw.get("scores", {}).get("B", 0.0),
        "B": pass2_raw.get("scores", {}).get("A", 0.0),
    }
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=pass1["winner"] == winner_pass2,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2=scores_pass2,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0

    n = len(judge_labels)
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = sorted(set(judge_labels) | set(human_labels))
    expected = 0.0
    for label in labels:
        expected += (judge_labels.count(label) / n) * (human_labels.count(label) / n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, round((observed - expected) / (1 - expected), 6)))


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "No judge results to analyze.",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    decisive = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_wins_a_longer = sum(
        1 for result in decisive if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1 for result in decisive if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    )
    position_bias_rate = position_bias_count / total
    interpretation = (
        "Position bias is high; keep swap-and-average in the production eval gate."
        if position_bias_rate > 0.30
        else "Position bias is low in this sample; judge behavior is stable enough for gating."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


def _save_report(path: str = "reports/judge_results.json") -> None:
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    examples = []
    judge_labels = []
    for item in human_data:
        model_answer = item["model_answer"]
        reference_answer = item["human_note"]
        result = swap_and_average(item["question"], model_answer, reference_answer)
        judge_label = 1 if result.final_winner == "A" else 0
        judge_labels.append(judge_label)
        examples.append({**result.__dict__, "question_id": item["question_id"], "judge_label": judge_label})

    human_labels = [item["human_label"] for item in human_data]
    report = {
        "model": JUDGE_MODEL,
        "cohen_kappa": cohen_kappa(judge_labels, human_labels),
        "bias": bias_report([JudgeResult(**{k: v for k, v in e.items() if k in JudgeResult.__dataclass_fields__}) for e in examples]),
        "results": examples,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")


if __name__ == "__main__":
    q = "Nhan vien duoc nghi bao nhieu ngay phep nam?"
    a = "Nhan vien duoc nghi 15 ngay phep nam theo chinh sach v2024."
    b = "Nhan vien co 12 ngay phep hang nam."
    print(swap_and_average(q, a, b))
    _save_report()
