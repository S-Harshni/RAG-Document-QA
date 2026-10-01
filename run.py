"""Retrieval experiments: chunk size, retriever, and exact against approximate vector search.

    python run.py        # writes out/ (index, chunks) and results/retrieval.json; a few minutes on a laptop CPU
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ragqa import chunking, corpus, retrieval  # noqa: E402

SIZES = [60, 120, 240, 480]     # words per chunk
OVERLAP = 0.2
DEFAULT_SIZE = 120
CANDIDATES = 50                 # each retriever hands this many to the fusion step
OUT, RESULTS = ROOT / "out", ROOT / "results"


def evaluate(chunks, questions, q_vectors, embed) -> tuple[dict, dict]:
    by_doc = chunking.index_by_document(chunks)
    gold = [chunking.gold_chunks(q, chunks, by_doc) for q in questions]
    bm25 = retrieval.BM25(chunks)
    vectors = embed([f"{c.title}. {c.text}" for c in chunks])
    store = retrieval.VectorStore(vectors, "flat")
    dense = store.search(q_vectors, CANDIDATES)
    ranks = {"BM25 (keywords)": [], "Dense (vector store)": [], "Hybrid (rank fusion)": []}
    context_words = []
    for i, q in enumerate(questions):
        keyword = bm25.search(q.text, CANDIDATES)
        hybrid = retrieval.reciprocal_rank_fusion([keyword, dense[i]])
        ranks["BM25 (keywords)"].append(retrieval.first_hit(keyword, gold[i]))
        ranks["Dense (vector store)"].append(retrieval.first_hit(dense[i], gold[i]))
        ranks["Hybrid (rank fusion)"].append(retrieval.first_hit(hybrid, gold[i]))
        context_words.append(sum(chunks[int(c)].words for c in hybrid[:5]))
    table = {name: retrieval.recall_and_mrr(np.array(r)) for name, r in ranks.items()}
    info = {
        "chunks": len(chunks), "mean_words": round(float(np.mean([c.words for c in chunks])), 1),
        "answer_inside_one_chunk": round(float(np.mean([bool(g) for g in gold])), 6),
        "top5_context_words": round(float(np.mean(context_words))),
    }
    return {"retrievers": table, **info}, {"bm25": bm25, "vectors": vectors, "gold": gold}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    documents, questions = corpus.load()
    embed = retrieval.Embedder()
    q_vectors = embed([q.text for q in questions])
    by_size, keep = {}, None
    for size in SIZES:
        chunks = chunking.chunk_corpus(documents, size, int(size * OVERLAP))
        by_size[size], extra = evaluate(chunks, questions, q_vectors, embed)
        print(size, json.dumps(by_size[size]["retrievers"]["Hybrid (rank fusion)"]), by_size[size]["chunks"], flush=True)
        if size == DEFAULT_SIZE:
            keep = (chunks, extra)

    chunks, extra = keep
    flat, hnsw = retrieval.VectorStore(extra["vectors"], "flat"), retrieval.VectorStore(extra["vectors"], "hnsw")
    timing = {}
    for name, store in (("flat", flat), ("hnsw", hnsw)):
        t0 = time.perf_counter()
        found = np.vstack([store.search(v, 10) for v in q_vectors[:2000]])
        timing[name] = {"ms_per_query": round(1000 * (time.perf_counter() - t0) / 2000, 4), "found": found}
    agreement = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(timing["flat"]["found"], timing["hnsw"]["found"], strict=True)])
    stores = {
        "vectors": len(chunks), "dimensions": int(extra["vectors"].shape[1]),
        "flat_ms_per_query": timing["flat"]["ms_per_query"], "hnsw_ms_per_query": timing["hnsw"]["ms_per_query"],
        "hnsw_recall_of_exact_top10": round(float(agreement), 6),
    }
    flat.save(OUT / "chunks.faiss")
    (OUT / "chunks.pkl").write_bytes(pickle.dumps(chunks))
    reloaded = retrieval.VectorStore.load(OUT / "chunks.faiss")
    assert (reloaded.search(q_vectors[:5], 5) == flat.search(q_vectors[:5], 5)).all()

    result = {
        "dataset": {"name": "SQuAD v1.1 (dev)", "documents": len(documents), "questions": len(questions),
                    "words": sum(len(d.text.split()) for d in documents), "licence": "CC BY-SA 4.0"},
        "embedding_model": retrieval.EMBEDDING_MODEL, "overlap": OVERLAP, "default_size": DEFAULT_SIZE,
        "chunk_sizes": [{"size": s, **v} for s, v in by_size.items()], "vector_store": stores,
    }
    (RESULTS / "retrieval.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"vector_store": stores, "sizes": {s: (v["chunks"], v["answer_inside_one_chunk"], v["top5_context_words"],
          v["retrievers"]["Hybrid (rank fusion)"]["recall_at_5"]) for s, v in by_size.items()}}, indent=1))


if __name__ == "__main__":
    main()
