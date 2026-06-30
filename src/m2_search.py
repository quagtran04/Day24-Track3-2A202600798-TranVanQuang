from __future__ import annotations

"""Module 2: hybrid search with BM25, dense retrieval, and RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM, EMBEDDING_MODEL, HYBRID_TOP_K
from config import QDRANT_HOST, QDRANT_PORT


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into BM25-friendly tokens."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if not self.documents:
            return []
        query_tokens = segment_vietnamese(query).split()
        if not query_tokens:
            return []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            query_set = set(query_tokens)
            scores = [len(query_set & set(tokens)) for tokens in self.corpus_tokens]

        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        results = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue
            doc = self.documents[i]
            results.append(SearchResult(doc["text"], score, doc.get("metadata", {}), "bm25"))
        return results


class DenseSearch:
    def __init__(self):
        self._encoder = None
        self._fallback_docs: list[dict] = []
        self._fallback_vectors: list[dict[str, float]] = []
        if os.getenv("RAG_USE_HF_MODELS", "0") != "1":
            self.client = None
            return
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if os.getenv("RAG_USE_HF_MODELS", "0") != "1":
            raise RuntimeError("set RAG_USE_HF_MODELS=1 to enable dense embeddings")
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    @staticmethod
    def _token_vector(text: str) -> dict[str, float]:
        vec: dict[str, float] = {}
        for token in segment_vietnamese(text).split():
            vec[token] = vec.get(token, 0.0) + 1.0
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        numerator = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        return numerator / (norm_a * norm_b + 1e-9)

    def _fallback_index(self, chunks: list[dict]) -> None:
        self._fallback_docs = chunks
        self._fallback_vectors = [self._token_vector(chunk["text"]) for chunk in chunks]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        self._fallback_index(chunks)
        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [c["text"] for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                )
                for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection, points)
        except Exception as exc:
            print(f"  Dense indexing fallback active: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        if self.client is not None:
            try:
                query_vector = self._get_encoder().encode(query).tolist()
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        pt.payload.get("text", ""),
                        float(pt.score),
                        {k: v for k, v in pt.payload.items() if k != "text"},
                        "dense",
                    )
                    for pt in response.points
                ]
            except Exception as exc:
                print(f"  Dense search fallback active: {exc}")

        query_vec = self._token_vector(query)
        scored = [
            (self._cosine(query_vec, vector), doc)
            for vector, doc in zip(self._fallback_vectors, self._fallback_docs)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(doc["text"], float(score), doc.get("metadata", {}), "dense")
            for score, doc in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            item["result"].text,
            float(item["score"]),
            item["result"].metadata,
            "hybrid",
        )
        for item in merged
    ]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(segment_vietnamese("Nhan vien duoc nghi phep nam"))
