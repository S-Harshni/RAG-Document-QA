# RAG Document QA

![tests](https://github.com/S-Harshni/RAG-Document-QA/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.12-blue)
![faiss](https://img.shields.io/badge/vector_store-FAISS-0467df)
![llm](https://img.shields.io/badge/LLM-Llama_·_Qwen_·_Gemma-6b46c1)
![license](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented question-answering pipeline in which every stage is measured: chunking, embeddings in a FAISS vector store, hybrid retrieval, three open-source language models, prompt-injection tests and a tool-using agent.

**Live demo:** https://s-harshni.github.io/RAG-Document-QA/

![Demo](docs/img/demo.png)

## Pipeline

```
documents ──► sentence-aware chunks (overlap) ──► embeddings ──► FAISS vector store ─┐
                                              └─► BM25 inverted index ───────────────┤
question ────────────────────────────────────────────────────────────────────────────┴─► rank fusion ──► top chunks
        ──► token budget ──► prompt (rules, examples, tagged chunks) ──► LLM ──► validated JSON: answer + citations, or a refusal
```

| Stage | What it does | Code |
| --- | --- | --- |
| Chunking | Packs whole sentences into chunks of a target size with 20% overlap | [`chunking.py`](src/ragqa/chunking.py) |
| Retrieval | BM25 written from scratch, MiniLM embeddings in a FAISS index (flat or HNSW), reciprocal rank fusion | [`retrieval.py`](src/ragqa/retrieval.py) |
| Prompting | Zero-shot, few-shot and chain-of-thought prompts; a context-window token budget; JSON output that must cite a chunk or decline | [`prompts.py`](src/ragqa/prompts.py) |
| Models | Any OpenAI-compatible API; by default Llama 3.2, Qwen 2.5 and Gemma 2 run locally through Ollama | [`llm.py`](src/ragqa/llm.py) |
| Safety | Six prompt-injection attacks, an injection detector, redaction of keys and personal data before logging | [`safety.py`](src/ragqa/safety.py) |
| Agent | The model may search again with its own query before answering (tool use with a step limit) | [`agent.py`](src/ragqa/agent.py) |

## Results

The corpus is the SQuAD v1.1 dev set, re-chunked: 48 Wikipedia articles (253,780 words) and 10,570 questions with known answer spans. A retriever is right when a chunk containing the answer span is in its top results.

### Retrieval (120-word chunks, 2,691 chunks, all 10,570 questions)

| Retriever | Right chunk ranked 1st | In top 5 | In top 10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25 (keywords) | 73.0% | 90.1% | 93.4% | 0.807 |
| Dense (MiniLM embeddings, FAISS) | 55.0% | 81.5% | 88.4% | 0.668 |
| **Hybrid (rank fusion)** | 68.6% | 91.0% | 95.3% | 0.785 |

Hybrid retrieval is the best at the depth a language model actually reads (top 5 and top 10). Keywords alone are strong at rank 1 here because SQuAD questions were written by people looking at the text and reuse its words; real users paraphrase more, which is why both are kept.

### Chunk size

| Chunk size (words) | Chunks | BM25, top 5 | Dense, top 5 | Hybrid, top 5 | Words sent to the model (top 5) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 5,520 | 86.0% | 80.3% | 88.3% | 236 |
| 120 | 2,691 | 90.1% | 81.5% | 91.0% | 524 |
| 240 | 1,330 | 92.8% | 80.0% | 91.6% | 1,117 |
| 480 | 671 | 95.1% | 68.0% | 85.5% | 2,285 |

Larger chunks help keyword search and hurt embeddings, which blur when one chunk covers several topics; they also multiply the prompt cost. 120 words is the default: close to the best hybrid recall at a quarter of the context of the largest setting.

![Retrieval evaluation](docs/img/evaluation.png)

### Vector store

FAISS index of 2,691 vectors (384 dimensions), saved and reloaded from disk. The approximate HNSW index returns 99.9% of the exact top-10 results at 0.20 ms per query (exact search: 0.22 ms). At this size exact search is already fast; the test shows the approximate index gives up almost nothing.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                       # retrieval, chunk sizes, vector store -> results/retrieval.json (a few minutes on CPU)

ollama pull llama3.2:3b qwen2.5:3b gemma2:2b
python eval_llm.py                  # language models, injection, sampling, agent -> results/llm.json (cached; resumable)
python build_demo.py                # docs/data.json for the demo page
pytest -q                           # 18 tests, no model or network needed
python -m http.server -d docs 8000  # demo at http://localhost:8000
```

Set `LLM_BASE_URL` and `LLM_API_KEY` to use a hosted OpenAI-compatible provider instead of Ollama.

## Tests

18 tests, run in CI: chunk boundaries and overlap, BM25 against hand-computed scores, rank fusion, the vector store round trip, token budgeting, parsing and validation of model output, every injection attack against the detector, redaction, and the agent loop with a scripted model.

## Limitations

- SQuAD questions reuse the wording of the text, which flatters keyword search; results on paraphrased questions would differ.
- One small embedding model and no re-ranker.
- The language models are 2 to 3 billion parameters, quantised, on a laptop; larger models would answer more accurately. The language-model tests use samples of a few dozen to 150 questions, so differences of a few points are not meaningful.
- Answer correctness is a string match against reference answers; it does not judge fluency.
- The live demo runs keyword retrieval in the browser; the vector store and the language models run in the Python pipeline.

## Data

Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* CC BY-SA 4.0.

## Author

[S Harshni](https://github.com/S-Harshni)
