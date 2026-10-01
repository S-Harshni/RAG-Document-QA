"""LLM experiments on local open-source models. Every call is cached, so the script can be stopped and resumed.

    ollama pull llama3.2:3b qwen2.5:3b gemma2:2b
    python run.py && python eval_llm.py        # writes results/llm.json
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ragqa import agent, chunking, corpus, prompts, retrieval, safety, scoring  # noqa: E402
from ragqa.llm import LLM  # noqa: E402

MODELS = ["llama3.2:3b", "qwen2.5:3b", "gemma2:2b"]
MAIN_MODEL = "qwen2.5:3b"
TOP_K, BUDGET_TOKENS, SEED = 3, 1200, 7
N_ANSWERABLE, N_UNANSWERABLE, N_INJECTION, N_SAMPLING, N_AGENT = 150, 75, 36, 20, 30
CACHE = ROOT / "results" / "llm_cache.jsonl"


class Cached:
    """Wraps an LLM so an identical request is answered from disk."""

    def __init__(self):
        self.store = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                row = json.loads(line)
                self.store[row["key"]] = row["output"]
        self.llms: dict[str, LLM] = {}
        self.calls = 0
        self.errors = 0

    def chat(self, model: str, messages: list[dict], **kw) -> str:
        key = hashlib.sha256(json.dumps([model, messages, kw], sort_keys=True).encode()).hexdigest()
        if key not in self.store:
            self.llms.setdefault(model, LLM(model))
            for _ in range(3):                # a local model server can fail while it swaps models
                try:
                    self.store[key] = self.llms[model].chat(messages, **kw)
                    break
                except httpx.HTTPError:
                    time.sleep(5)
            else:
                # The server rejects some generations outright (malformed JSON under JSON mode).
                # That counts as an invalid answer, the same as any other unparseable output.
                self.store[key] = "<server error>"
                self.errors += 1
            self.calls += 1
            with CACHE.open("a") as f:
                f.write(json.dumps({"key": key, "output": safety.redact(self.store[key])}) + "\n")
        return self.store[key]

    def bound(self, model: str):
        outer = self

        class Bound:
            def chat(self, messages, **kw):
                return outer.chat(model, messages, **kw)

        return Bound()


def main() -> None:
    documents, questions = corpus.load()
    chunks = pickle.loads((ROOT / "out" / "chunks.pkl").read_bytes())
    by_doc = chunking.index_by_document(chunks)
    bm25, embed = retrieval.BM25(chunks), retrieval.Embedder()
    store = retrieval.VectorStore.load(ROOT / "out" / "chunks.faiss")
    llm = Cached()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(questions))

    def search(query: str, k: int = TOP_K, exclude_doc: int | None = None) -> list:
        fused = retrieval.reciprocal_rank_fusion([bm25.search(query, 50), store.search(embed([query]), 50)[0]])
        picked = [chunks[int(i)] for i in fused if exclude_doc is None or chunks[int(i)].doc_id != exclude_doc]
        return prompts.fit_to_budget(picked[:k], BUDGET_TOKENS)

    def ask(model, q, passages, style="few_shot", defended=True, **kw):
        raw = llm.chat(model, prompts.build_messages(q.text, passages, style, defended), json_mode=True, **kw)
        return raw, prompts.parse_response(raw, len(passages))

    answerable = [questions[i] for i in order[:N_ANSWERABLE]]
    unanswerable = [questions[i] for i in order[N_ANSWERABLE:N_ANSWERABLE + N_UNANSWERABLE]]
    retrieved = {id(q): search(q.text) for q in answerable}
    off_topic = {id(q): search(q.text, exclude_doc=q.doc_id) for q in unanswerable}   # the answer is not in these

    def grade(model: str, style: str) -> dict:
        valid = correct = f1 = cited_right = cited_any = had_gold = refused_with_gold = declined = 0
        for q in answerable:
            passages = retrieved[id(q)]
            gold = chunking.gold_chunks(q, chunks, by_doc)
            gold_positions = {i for i, c in enumerate(passages, start=1) if c.id in gold}
            _, out = ask(model, q, passages, style)
            valid += out is not None
            if out is None:
                continue
            ok = out["answerable"] and scoring.contains_answer(out["answer"], q.answers)
            correct += ok
            f1 += scoring.best_f1(out["answer"], q.answers) if out["answerable"] else 0.0
            if gold_positions:
                had_gold += 1
                refused_with_gold += not out["answerable"]
            if out["answerable"] and out["citations"]:
                cited_any += 1
                cited_right += bool(set(out["citations"]) & gold_positions)
        for q in unanswerable:
            _, out = ask(model, q, off_topic[id(q)], style)
            valid += out is not None
            declined += out is not None and not out["answerable"]
        n = len(answerable)
        return {
            "model": model, "style": style, "valid_json": round(valid / (n + len(unanswerable)), 4),
            "answer_accuracy": round(correct / n, 4), "token_f1": round(f1 / n, 4),
            "citation_accuracy": round(cited_right / max(cited_any, 1), 4),
            "wrongly_declined": round(refused_with_gold / max(had_gold, 1), 4),
            "correctly_declined": round(declined / len(unanswerable), 4),
        }

    gold_in_context = float(np.mean([bool({c.id for c in retrieved[id(q)]} & chunking.gold_chunks(q, chunks, by_doc)) for q in answerable]))
    models = [grade(m, "few_shot") for m in MODELS]
    print(json.dumps(models, indent=1), flush=True)
    styles = [grade(MAIN_MODEL, s) for s in prompts.STYLES]
    print(json.dumps(styles, indent=1), flush=True)

    # Prompt injection: one retrieved passage is poisoned; the attack succeeds if the canary appears in the output.
    attacks = list(safety.ATTACKS.items())
    injection = []
    for model in MODELS:
        for defended in (False, True):
            hits = still_right = 0
            for n, q in enumerate(answerable[:N_INJECTION]):
                passages = list(retrieved[id(q)])
                name, attack = attacks[n % len(attacks)]
                target = n % len(passages)
                poisoned = passages[target]
                passages[target] = chunking.Chunk(poisoned.id, poisoned.doc_id, poisoned.title,
                                                  safety.inject(poisoned.text, attack, ("end", "start", "middle")[n % 3]),
                                                  poisoned.start, poisoned.end)
                raw, out = ask(model, q, passages, "few_shot", defended)
                hits += safety.attack_succeeded(raw)
                still_right += out is not None and out["answerable"] and scoring.contains_answer(out["answer"], q.answers)
            injection.append({"model": model, "defended": defended, "attack_success": round(hits / N_INJECTION, 4),
                              "answer_accuracy": round(still_right / N_INJECTION, 4)})
    print(json.dumps(injection, indent=1), flush=True)
    clean_flagged = float(np.mean([safety.looks_like_injection(c.text) for c in chunks]))
    detector = {"attacks_flagged": round(float(np.mean([safety.looks_like_injection(safety.inject("Some text.", a)) for a in safety.ATTACKS.values()])), 4),
                "clean_chunks_flagged": round(clean_flagged, 6), "clean_chunks": len(chunks)}

    # Sampling: the same question asked three times at each temperature.
    sampling = []
    for temperature in (0.0, 0.7, 1.3):
        agree = right = 0
        for q in answerable[:N_SAMPLING]:
            outs = [ask(MAIN_MODEL, q, retrieved[id(q)], temperature=temperature, top_p=0.95, seed=s)[1] for s in (1, 2, 3)]
            texts = [scoring.normalise(o["answer"]) if o else "<invalid>" for o in outs]
            agree += len(set(texts)) == 1
            right += sum(o is not None and o["answerable"] and scoring.contains_answer(o["answer"], q.answers) for o in outs) / 3
        sampling.append({"temperature": temperature, "same_answer_all_three_runs": round(agree / N_SAMPLING, 4),
                         "answer_accuracy": round(right / N_SAMPLING, 4)})
    print(json.dumps(sampling, indent=1), flush=True)

    # Agent: questions where the first retrieval missed. Can the model recover by searching again?
    hard = []
    for i in order[N_ANSWERABLE + N_UNANSWERABLE:]:
        q = questions[i]
        if not {c.id for c in search(q.text)} & chunking.gold_chunks(q, chunks, by_doc):
            hard.append(q)
        if len(hard) == N_AGENT:
            break
    single = sum((o := ask(MAIN_MODEL, q, search(q.text))[1]) is not None and o["answerable"]
                 and scoring.contains_answer(o["answer"], q.answers) for q in hard)
    agent_right = found = searches = 0
    for q in hard:
        result = agent.run(q.text, llm.bound(MAIN_MODEL), search)
        agent_right += scoring.contains_answer(result["answer"], q.answers)
        found += bool(set(result["chunks"]) & chunking.gold_chunks(q, chunks, by_doc))
        searches += len(result["queries"])
    agent_result = {"questions": len(hard), "single_shot_accuracy": round(single / len(hard), 4),
                    "agent_accuracy": round(agent_right / len(hard), 4), "agent_found_gold_chunk": round(found / len(hard), 4),
                    "mean_searches": round(searches / len(hard), 2)}
    print(json.dumps(agent_result, indent=1), flush=True)

    out = {
        "setup": {"top_k": TOP_K, "budget_tokens": BUDGET_TOKENS, "answerable": N_ANSWERABLE, "unanswerable": N_UNANSWERABLE,
                  "gold_chunk_in_context": round(gold_in_context, 4), "main_model": MAIN_MODEL, "runtime": "Ollama, 4-bit quantised, local"},
        "models": models, "styles": styles, "injection": injection, "detector": detector, "sampling": sampling, "agent": agent_result,
    }
    (ROOT / "results" / "llm.json").write_text(json.dumps(out, indent=1))
    print("new calls:", llm.calls, "server errors:", llm.errors)


if __name__ == "__main__":
    main()
