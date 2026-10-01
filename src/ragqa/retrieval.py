"""Retrievers over chunks: BM25 (keywords), a FAISS vector store (embeddings) and a hybrid of the two."""
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ragqa.chunking import Chunk
from ragqa.corpus import tokenize

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60


class BM25:
    """Okapi BM25 over an inverted index, written out so every term of the score is visible."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        docs = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self.n = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        self.k1 = k1
        index: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                index.setdefault(term, []).append((i, tf))
        self.postings = {t: (np.array([i for i, _ in e]), np.array([tf for _, tf in e], dtype=np.float32))
                         for t, e in index.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term in self.postings:
                ids, tf = self.postings[term]
                idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
                out[ids] += idf * tf * (self.k1 + 1) / (tf + self.norm[ids])
        return out

    def search(self, query: str, k: int) -> np.ndarray:
        scores = self.scores(query)
        top = np.argpartition(-scores, min(k, self.n - 1))[:k]
        return top[np.argsort(-scores[top], kind="stable")]


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype("float32")


class VectorStore:
    """Chunk embeddings in a FAISS index. `flat` is exact search; `hnsw` is an approximate graph index."""

    def __init__(self, vectors: np.ndarray, kind: str = "flat"):
        import faiss

        dim = vectors.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction, self.index.hnsw.efSearch = 200, 64
        else:
            raise ValueError(f"unknown index kind {kind!r}")
        self.index.add(vectors)
        self.kind = kind

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        return self.index.search(np.atleast_2d(queries), k)[1]

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        import faiss

        store = cls.__new__(cls)
        store.index, store.kind = faiss.read_index(str(path)), "loaded"
        return store


def reciprocal_rank_fusion(rankings: list[np.ndarray], k: int = RRF_K, top: int | None = None) -> np.ndarray:
    """Merge rankings by summing 1 / (k + rank): a chunk both retrievers like beats one only one loves."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[int(item)] = score.get(int(item), 0.0) + 1.0 / (k + rank)
    merged = sorted(score, key=lambda i: (-score[i], i))
    return np.array(merged[:top] if top else merged)


def first_hit(ranking: np.ndarray, gold: set[int]) -> int:
    """0-based rank of the first correct chunk, or a large number when none was retrieved."""
    for rank, item in enumerate(ranking):
        if int(item) in gold:
            return rank
    return 10**6


def recall_and_mrr(ranks: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    out = {f"recall_at_{k}": round(float((ranks < k).mean()), 6) for k in ks}
    out["mrr"] = round(float((1.0 / (ranks + 1)).mean()), 6)
    return out
