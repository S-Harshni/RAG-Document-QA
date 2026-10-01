"""Documents and questions from SQuAD v1.1 (dev).

Each of the 48 Wikipedia articles is rebuilt as one long document, so chunking is a real decision
here: the published paragraph boundaries are thrown away and `chunking.py` chooses its own.
Every question keeps the character span of its answer inside the document, which is what lets
retrieval be scored for any chunking.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQUAD = ROOT / "data" / "squad-dev-v1.1.json"
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were what when where which who "
    "whom whose why how with did do does this these those there their than then into".split())


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    text: str
    doc_id: int
    start: int                 # character offset of the first reference answer in the document
    end: int
    answers: tuple[str, ...]


def load(path: Path = SQUAD) -> tuple[list[Document], list[Question]]:
    documents, questions = [], []
    for article in json.loads(path.read_text())["data"]:
        doc_id, parts, offset = len(documents), [], 0
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                first = qa["answers"][0]
                start = offset + first["answer_start"]
                answers = tuple(dict.fromkeys(a["text"] for a in qa["answers"]))
                questions.append(Question(qa["question"].strip(), doc_id, start, start + len(first["text"]), answers))
            parts.append(context)
            offset += len(context) + 1      # paragraphs are joined with one newline
        documents.append(Document(doc_id, article["title"].replace("_", " "), "\n".join(parts)))
    return documents, questions


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]
