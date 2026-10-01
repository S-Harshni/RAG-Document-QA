"""Evaluate the retrievers on every SQuAD dev question and export what the demo page needs.

    python run.py            # writes docs/data.json (about two minutes on a laptop CPU)
    python run.py --ask "Who led the Normans at Hastings?"
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ragqa import corpus, generation, retrieval  # noqa: E402

TOP_K = 5


def positions(rankings: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.argmax(rankings == gold[:, None], axis=1)


def main() -> None:
    passages, questions = corpus.load()
    bm25, dense = retrieval.BM25(passages), retrieval.Dense(passages)
    if "--ask" in sys.argv:
        q = sys.argv[sys.argv.index("--ask") + 1]
        top = [passages[i] for i in retrieval.reciprocal_rank_fusion([bm25.rank(q), dense.rank(q)])[:TOP_K]]
        print(json.dumps(generation.answer(q, top), indent=1))
        return
    gold = np.array([q.passage_id for q in questions])
    texts = [q.text for q in questions]
    r_bm25 = np.stack([bm25.rank(t) for t in texts])
    r_dense = dense.rank_many(texts)
    r_hybrid = np.stack([retrieval.reciprocal_rank_fusion([a, b]) for a, b in zip(r_bm25, r_dense, strict=True)])
    rankings = {"BM25 (keywords)": r_bm25, "Dense (MiniLM embeddings)": r_dense, "Hybrid (rank fusion)": r_hybrid}
    table = [{"retriever": name, **retrieval.recall_and_mrr(positions(r, gold))} for name, r in rankings.items()]

    # Extractive fallback: is a reference answer inside the sentence it quotes from the top passages?
    hits = 0
    for q, ranking in zip(questions, r_hybrid, strict=True):
        sentence, _ = generation.best_sentence(q.text, [passages[i] for i in ranking[:TOP_K]])
        hits += any(a.lower() in sentence.lower() for a in q.answers)
    pos = positions(r_hybrid, gold)
    examples = []
    for i in np.random.default_rng(42).choice(len(questions), 12, replace=False):
        q = questions[int(i)]
        examples.append({"question": q.text, "answer": q.answers[0], "rank": int(pos[i]) + 1})
    out = {
        "dataset": {"name": "SQuAD v1.1 (dev)", "articles": len({p.title for p in passages}), "passages": len(passages),
                    "questions": len(questions), "licence": "CC BY-SA 4.0"},
        "embedding_model": retrieval.EMBEDDING_MODEL, "top_k": TOP_K, "retrievers": table,
        "extractive": {"questions": len(questions), "answer_in_quoted_sentence": round(hits / len(questions), 6)},
        "examples": examples,
        "passages": [{"t": p.title, "x": p.text} for p in passages],
    }
    (ROOT / "docs" / "data.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("dataset", "retrievers", "extractive")}, indent=1))


if __name__ == "__main__":
    main()
