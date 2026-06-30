from __future__ import annotations

"""Module 3: cross-encoder reranking with a lexical fallback."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class _LexicalReranker:
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q = self._tokens(query)
            d = self._tokens(doc)
            overlap = len(q & d)
            scores.append(overlap / (len(q) ** 0.5 + 1e-9))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if os.getenv("RAG_USE_HF_MODELS", "0") != "1":
                    raise RuntimeError("set RAG_USE_HF_MODELS=1 to enable CrossEncoder reranking")
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  CrossEncoder fallback active: {exc}")
                self._model = _LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
            return [
                RerankResult(
                    r["text"],
                    float(documents[r["id"]].get("score", 0.0)),
                    float(r.get("score", 0.0)),
                    r.get("meta", {}),
                    i,
                )
                for i, r in enumerate(ranked)
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [{"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}}]
    print(CrossEncoderReranker().rerank(query, docs))
