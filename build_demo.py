"""Collect the measured results and the chunks into docs/data.json for the demo page.

    python run.py && python eval_llm.py && python build_demo.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ragqa import corpus  # noqa: E402

N_EXAMPLES = 8


def main() -> None:
    chunks = pickle.loads((ROOT / "out" / "chunks.pkl").read_bytes())
    _, questions = corpus.load()
    picked = np.random.default_rng(3).choice(len(questions), N_EXAMPLES, replace=False)
    llm = ROOT / "results" / "llm.json"
    data = {
        "retrieval": json.loads((ROOT / "results" / "retrieval.json").read_text()),
        "llm": json.loads(llm.read_text()) if llm.exists() else None,
        "examples": [questions[int(i)].text for i in picked],
        "chunks": [{"t": c.title.replace("_", " "), "x": c.text} for c in chunks],
    }
    (ROOT / "docs" / "data.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print("chunks", len(chunks), "llm results", data["llm"] is not None)


if __name__ == "__main__":
    main()
