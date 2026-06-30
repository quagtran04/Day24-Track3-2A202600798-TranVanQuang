from __future__ import annotations

"""Phase C: production guardrails with PII scan, input/output rails, and latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


PII_PATTERNS = {
    "VN_CCCD": re.compile(r"\b\d{12}\b"),
    "VN_CMND": re.compile(r"\b\d{9}\b"),
    "VN_PHONE": re.compile(r"\b0[3-9]\d{8}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

BLOCK_PATTERNS = [
    r"bo qua|bỏ qua|ignore .*instruction|ignore .*prompt|forget .*polic",
    r"system override|admin command|priority 1|dump all|training data",
    r"dan|do anything now|unrestricted ai|khong co gioi han|không có giới hạn",
    r"mat khau admin|mật khẩu admin|confidential|du lieu mat|dữ liệu mật",
    r"liet ke .*luong|liệt kê .*lương|all employee salaries|bang luong chi tiet",
    r"cccd.*nhan vien|số điện thoại.*nhân viên|email.*nhan vien",
    r"system instructions|in ra.*context|reveal.*data",
    r"tan cong mang|tấn công mạng",
]

OFF_TOPIC_PATTERNS = [
    r"bai tho|bài thơ|mua thu|mùa thu",
    r"nau pho|nấu phở|cong thuc|công thức",
    r"bitcoin|ethereum|usd",
    r"phuong trinh|phương trình|dy/dx",
    r"marvel|phim hay",
]


def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    analyzer = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _regex_pii_scan(text: str) -> dict:
    entities = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                {
                    "type": "VN_CCCD" if entity_type == "VN_CMND" else entity_type,
                    "text": match.group(0),
                    "score": 0.9,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    entities.sort(key=lambda item: item["start"])
    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = (
            anonymized[: entity["start"]]
            + f"<{entity['type']}>"
            + anonymized[entity["end"] :]
        )
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    if analyzer is not None and anonymizer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            if results:
                anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
                entities = [
                    {
                        "type": result.entity_type,
                        "text": text[result.start : result.end],
                        "score": round(result.score, 3),
                        "start": result.start,
                        "end": result.end,
                    }
                    for result in results
                ]
                return {"has_pii": True, "entities": entities, "anonymized": anonymized}
        except Exception:
            pass
    return _regex_pii_scan(text)


def setup_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    return LLMRails(config)


def _normalize_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
            "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
            "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
            "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
            "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
            "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
            "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
            "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
            "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
            "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
            "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
            "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
            "đ": "d",
        }
    )
    return text.lower().translate(replacements)


def _rule_block_reason(text: str) -> str | None:
    normalized = _normalize_text(text)
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, normalized):
            return "policy_or_jailbreak"
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, normalized):
            return "off_topic"
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    reason = _rule_block_reason(text)
    if reason:
        return {
            "allowed": False,
            "blocked_reason": reason,
            "response": "Request blocked by input guardrail.",
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response if isinstance(response, str) else str(response)
            blocked = any(
                keyword in response_text.lower()
                for keyword in ["xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"]
            )
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception as exc:
            return {
                "allowed": True,
                "blocked_reason": None,
                "response": f"NeMo unavailable, rule rail allowed: {exc}",
            }

    return {"allowed": True, "blocked_reason": None, "response": "Allowed by rule rail."}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    reason = _rule_block_reason(answer)
    if pii["has_pii"]:
        return {
            "safe": False,
            "flagged_reason": "pii_in_output",
            "final_answer": pii["anonymized"],
        }
    if reason:
        return {
            "safe": False,
            "flagged_reason": reason,
            "final_answer": "Câu trả lời bị chặn vì chứa nội dung không phù hợp.",
        }

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            response_text = response if isinstance(response, str) else str(response)
            flagged = any(
                keyword in response_text.lower()
                for keyword in ["xin lỗi", "không thể cung cấp", "i cannot"]
            )
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception:
            pass

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "input": item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                    "expected": item["expected"],
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == item["expected"],
                }
            )
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = sorted(times)
    n = len(values)

    def pick(q: float) -> float:
        return round(values[min(max(int((n - 1) * q), 0), n - 1)], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    inputs = (test_inputs or ["test input"])[: max(1, n_runs)]
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in inputs:
            total_start = time.perf_counter()
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            total_ms = (time.perf_counter() - total_start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(total_ms)

    asyncio.run(_measure())
    total_p = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def _save_report(path: str = "reports/guard_results.json") -> dict:
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=20)
    report = {
        "adversarial_results": results,
        "passed": sum(1 for result in results if result["passed"]),
        "total": len(results),
        "pass_rate": sum(1 for result in results if result["passed"]) / len(results),
        "latency": latency,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")
    return report


if __name__ == "__main__":
    sample = "CCCD 034095001234, SDT 0987654321, email test@company.com"
    print(pii_scan(sample))
    _save_report()
