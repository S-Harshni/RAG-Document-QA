"""Three retrievers behind one interface: BM25 (keywords), dense (embeddings) and a hybrid of the two."""
import math
from collections import Counter

import numpy as np

from ragqa.corpus import Passage, tokenize

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60


class BM25:
    """Okapi BM25 over an inverted index, written out so every term of the score is visible."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        docs = [tokenize(f"{p.title} {p.text}") for p in passages]
        self.n = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        self.k1 = k1
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        index: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                index.setdefault(term, []).append((i, tf))
        for term, entries in index.items():
            ids, tfs = zip(*entries, strict=True)
            self.postings[term] = (np.array(ids), np.array(tfs, dtype=np.float32))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            out[ids] += idf * tf * (self.k1 + 1) / (tf + self.norm[ids])
        return out

    def rank(self, query: str) -> np.ndarray:
        return np.argsort(-self.scores(query), kind="stable")


class Dense:
    """Cosine similarity between sentence embeddings of the question and of each passage."""

    def __init__(self, passages: list[Passage], model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.vectors = self.encode([f"{p.title}. {p.text}" for p in passages])

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def rank_many(self, queries: list[str]) -> np.ndarray:
        return np.argsort(-(self.encode(queries) @ self.vectors.T), axis=1, kind="stable")

    def rank(self, query: str) -> np.ndarray:
        return self.rank_many([query])[0]


def reciprocal_rank_fusion(rankings: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Merge rankings by summing 1 / (k + rank): a passage both retrievers like beats one only one loves."""
    score = np.zeros(len(rankings[0]))
    for ranking in rankings:
        score[ranking] += 1.0 / (k + np.arange(1, len(ranking) + 1))
    return np.argsort(-score, kind="stable")


def recall_and_mrr(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """`ranks` holds the 0-based position of the correct passage for each question."""
    out = {f"recall_at_{k}": round(float((ranks < k).mean()), 6) for k in ks}
    out["mrr"] = round(float((1.0 / (ranks + 1)).mean()), 6)
    return out
