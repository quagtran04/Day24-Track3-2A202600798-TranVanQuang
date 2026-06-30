from __future__ import annotations

"""Module 5: chunk enrichment before embedding."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, RAG_USE_LLM


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _client():
    if not RAG_USE_LLM or not OPENAI_API_KEY:
        return None
    import httpx
    from openai import OpenAI

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    kwargs["http_client"] = httpx.Client(headers={"User-Agent": "python-httpx/0.27.0"})
    return OpenAI(**kwargs)


def _chat(system: str, user: str, max_tokens: int = 200) -> str:
    client = _client()
    if client is None:
        return ""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max(max_tokens, 512),
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def summarize_chunk(text: str) -> str:
    try:
        result = _chat(
            "Tom tat doan van sau trong 1-2 cau ngan gon bang tieng Viet.",
            text,
            max_tokens=150,
        )
        if result:
            return result
    except Exception as exc:
        print(f"  Summarize fallback active: {exc}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return ". ".join(sentences[:2]) + ("." if sentences and not sentences[:2][-1].endswith((".", "!", "?")) else "")


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    try:
        result = _chat(
            f"Tao {n_questions} cau hoi ma doan van co the tra loi. Moi cau tren mot dong.",
            text,
            max_tokens=200,
        )
        if result:
            questions = [q.strip().lstrip("0123456789.-) ") for q in result.splitlines() if q.strip()]
            return questions[:n_questions]
    except Exception as exc:
        print(f"  HyQA fallback active: {exc}")

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    questions = []
    for sentence in sentences[:n_questions]:
        lower = sentence.lower()
        if any(word in lower for word in ["ngay", "ngày", "bao", "muc", "mức"]):
            questions.append(f"{sentence.rstrip('.')} la bao nhieu?")
        else:
            questions.append(f"{sentence.rstrip('.')}?")
    return questions


def contextual_prepend(text: str, document_title: str = "") -> str:
    try:
        result = _chat(
            "Viet 1 cau ngan mo ta doan van nay thuoc tai lieu nao va noi ve chu de gi.",
            f"Tai lieu: {document_title}\n\nDoan van:\n{text}",
            max_tokens=80,
        )
        if result:
            return f"{result}\n\n{text}"
    except Exception as exc:
        print(f"  Contextual fallback active: {exc}")

    prefix = f"Trich tu {document_title}. " if document_title else "Ngu canh tai lieu noi bo. "
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    try:
        result = _chat(
            'Trich xuat metadata va chi tra ve JSON: {"topic": "...", "entities": [], "category": "policy|hr|it|finance", "language": "vi|en"}',
            text,
            max_tokens=150,
        )
        if result:
            match = re.search(r"\{.*\}", result, flags=re.DOTALL)
            return json.loads(match.group(0) if match else result)
    except Exception as exc:
        print(f"  Metadata fallback active: {exc}")

    lower = text.lower()
    category = "policy"
    if any(term in lower for term in ["mat khau", "mật khẩu", "vpn", "du lieu", "dữ liệu"]):
        category = "it"
    elif any(term in lower for term in ["luong", "lương", "thuong", "thưởng", "chi phi"]):
        category = "finance"
    elif any(term in lower for term in ["nghi", "nghỉ", "thu viec", "thử việc", "dao tao"]):
        category = "hr"
    return {"topic": "internal_policy", "entities": [], "category": category, "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    try:
        result = _chat(
            """Phan tich doan van va chi tra ve JSON:
{
  "summary": "tom tat 1-2 cau",
  "questions": ["cau hoi 1", "cau hoi 2", "cau hoi 3"],
  "context": "1 cau mo ta ngu canh",
  "metadata": {"topic": "...", "entities": [], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
            f"Tai lieu: {source}\n\nDoan van:\n{text}",
            max_tokens=400,
        )
        if result:
            match = re.search(r"\{.*\}", result, flags=re.DOTALL)
            return json.loads(match.group(0) if match else result)
    except Exception as exc:
        print(f"  Combined enrichment fallback active: {exc}")

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trich tu {source}, noi dung lien quan den quy dinh noi bo." if source else "Ngu canh quy dinh noi bo.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(_enrich_single_call(sample, "sample.md"))
