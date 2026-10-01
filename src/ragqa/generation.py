"""Answering from retrieved passages: an LLM over any OpenAI-compatible API, or an extractive fallback.

Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to use a hosted model. Without a key the
system still answers, by quoting the sentence of the retrieved passages that best matches the question.
"""
import os
import re

import httpx

from ragqa.corpus import Passage, tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
SYSTEM_PROMPT = (
    "You answer questions using only the numbered passages provided. "
    "Give a short, direct answer and cite the passages you used in square brackets, like [2]. "
    "If the passages do not contain the answer, reply exactly: I cannot answer that from the documents."
)


def build_prompt(question: str, passages: list[Passage]) -> str:
    context = "\n\n".join(f"[{i}] ({p.title}) {p.text}" for i, p in enumerate(passages, start=1))
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


def best_sentence(question: str, passages: list[Passage]) -> tuple[str, int]:
    """The sentence sharing the most distinct question terms; ties go to the higher-ranked passage."""
    terms = set(tokenize(question))
    best, where, top = "", 0, -1.0
    for i, passage in enumerate(passages, start=1):
        for sentence in SENTENCE.split(passage.text):
            words = set(tokenize(sentence))
            score = len(terms & words) / (1 + 0.1 * len(words) ** 0.5)
            if score > top:
                best, where, top = sentence.strip(), i, score
    return best, where


def answer(question: str, passages: list[Passage], client: httpx.Client | None = None) -> dict:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        sentence, source = best_sentence(question, passages)
        return {"answer": sentence, "sources": [source], "mode": "extractive"}
    client = client or httpx.Client(timeout=60)
    response = client.post(
        os.environ.get("LLM_BASE_URL", "https://api.studio.nebius.com/v1").rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
            "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_prompt(question, passages)}],
        },
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(passages)})
    return {"answer": text, "sources": cited, "mode": "llm"}
