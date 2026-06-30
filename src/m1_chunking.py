from __future__ import annotations

"""Module 1: advanced chunking strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Skip {os.path.basename(fp)}: scanned PDF has no text layer.")
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group neighboring sentences when their semantic similarity is high."""
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    def token_vector(sentence: str) -> dict[str, float]:
        vec: dict[str, float] = {}
        for token in re.findall(r"\w+", sentence.lower(), flags=re.UNICODE):
            vec[token] = vec.get(token, 0.0) + 1.0
        return vec

    def cosine(a, b) -> float:
        try:
            from numpy import dot
            from numpy.linalg import norm

            return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
        except Exception:
            keys = set(a) | set(b)
            numerator = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
            norm_a = sqrt(sum(v * v for v in a.values()))
            norm_b = sqrt(sum(v * v for v in b.values()))
            return numerator / (norm_a * norm_b + 1e-9)

    try:
        if os.getenv("RAG_USE_HF_MODELS", "0") != "1":
            raise RuntimeError("set RAG_USE_HF_MODELS=1 to enable sentence-transformer chunking")
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(sentences)
    except Exception:
        embeddings = [token_vector(sentence) for sentence in sentences]

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(" ".join(group).strip(), {**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
        if group
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Build parent chunks for context and child chunks for retrieval."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            pid = f"parent_{len(parents)}"
            parents.append(
                Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid})
            )
            current = ""
        current = f"{current}\n\n{para}".strip() if current else para
    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid}))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        pieces = [p.strip() for p in parent.text.split("\n\n") if p.strip()]
        buffer = ""
        for piece in pieces:
            words = piece.split()
            parts = [piece] if len(piece) <= child_size else []
            if not parts:
                tmp = ""
                for word in words:
                    if tmp and len(tmp) + len(word) + 1 > child_size:
                        parts.append(tmp)
                        tmp = ""
                    tmp = f"{tmp} {word}".strip()
                if tmp:
                    parts.append(tmp)

            for part in parts:
                if buffer and len(buffer) + len(part) + 2 > child_size:
                    children.append(
                        Chunk(
                            buffer.strip(),
                            {**metadata, "chunk_type": "child", "child_index": len(children)},
                            parent_id=pid,
                        )
                    )
                    buffer = ""
                buffer = f"{buffer}\n\n{part}".strip() if buffer else part
        if buffer.strip():
            children.append(
                Chunk(
                    buffer.strip(),
                    {**metadata, "chunk_type": "child", "child_index": len(children)},
                    parent_id=pid,
                )
            )
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split markdown by headers while preserving header text in each chunk."""
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def flush() -> None:
        nonlocal current_content
        body = "\n".join(current_content).strip()
        if not current_header and not body:
            return
        section = re.sub(r"^#{1,6}\s*", "", current_header).strip() or "root"
        chunk_text = f"{current_header}\n\n{body}".strip() if current_header else body
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_text,
                    {**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
                )
            )
        current_content = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+.+", line):
            flush()
            current_header = line.strip()
        else:
            current_content.append(line)
    flush()

    return chunks or chunk_basic(text, metadata={**metadata, "strategy": "structure"})


def compare_strategies(documents: list[dict]) -> dict:
    def stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": stats(basic),
        "semantic": stats(semantic),
        "hierarchical": {**stats(children), "parents": len(parents)},
        "structure": stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, values in results.items():
        print(f"{name:<15} {values['count']:>7} {values['avg_len']:>5} {values['min_len']:>5} {values['max_len']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    compare_strategies(docs)
