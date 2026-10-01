import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ragqa import corpus, generation, retrieval  # noqa: E402
from ragqa.corpus import Passage  # noqa: E402

DOCS = [
    Passage(0, "Normans", "The Normans were descended from Norse raiders. William the Conqueror led them at Hastings in 1066."),
    Passage(1, "Photosynthesis", "Plants convert light into chemical energy. Chlorophyll absorbs mostly blue and red light."),
    Passage(2, "Rivers", "The Nile flows north through Egypt. The Amazon carries more water than any other river."),
]


def test_corpus_loads_squad():
    passages, questions = corpus.load()
    assert len(passages) == 2067 and len(questions) == 10570
    assert all(0 <= q.passage_id < len(passages) and q.answers for q in questions)
    q = questions[0]
    assert any(a in passages[q.passage_id].text for a in q.answers)


def test_tokenize_drops_stopwords_and_punctuation():
    assert corpus.tokenize("Who led the Normans at Hastings?") == ["led", "normans", "hastings"]


def test_bm25_ranks_the_matching_passage_first():
    bm25 = retrieval.BM25(DOCS)
    assert bm25.rank("Who led the Normans at Hastings?")[0] == 0
    assert bm25.rank("Which river carries the most water?")[0] == 2
    assert bm25.scores("zebra xylophone").sum() == 0
    # A rare word counts for more than a common one.
    assert bm25.scores("chlorophyll")[1] > bm25.scores("light")[1] / 2


def test_reciprocal_rank_fusion():
    a, b = np.array([0, 1, 2]), np.array([1, 0, 2])
    fused = retrieval.reciprocal_rank_fusion([a, b])
    assert fused[2] == 2 and set(fused[:2]) == {0, 1}
    assert list(retrieval.reciprocal_rank_fusion([a, a])) == [0, 1, 2]


def test_recall_and_mrr():
    m = retrieval.recall_and_mrr(np.array([0, 4, 20]))
    assert m["recall_at_1"] == pytest.approx(1 / 3, abs=1e-6) and m["recall_at_5"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["mrr"] == pytest.approx((1 + 1 / 5 + 1 / 21) / 3, abs=1e-6)


def test_extractive_answer_quotes_the_right_sentence(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    out = generation.answer("Who led the Normans at Hastings?", DOCS)
    assert "William the Conqueror" in out["answer"] and out["sources"] == [1] and out["mode"] == "extractive"


def test_llm_answer_sends_numbered_passages_and_parses_citations(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "William the Conqueror [1][7]."}}]})

    out = generation.answer("Who led the Normans?", DOCS, httpx.Client(transport=httpx.MockTransport(handler)))
    assert seen["auth"] == "Bearer test-key" and "[1] (Normans)" in seen["body"] and "[3] (Rivers)" in seen["body"]
    assert out == {"answer": "William the Conqueror [1][7].", "sources": [1], "mode": "llm"}  # [7] is not a passage


def test_prompt_tells_the_model_to_decline():
    assert "I cannot answer that from the documents" in generation.SYSTEM_PROMPT
    assert generation.build_prompt("q?", DOCS[:1]).endswith("Question: q?")
