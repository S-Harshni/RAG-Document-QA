"""Passages and questions from SQuAD v1.1 (dev): 48 Wikipedia articles, one passage per paragraph."""
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
class Passage:
    id: int
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    text: str
    passage_id: int
    answers: tuple[str, ...]


def load(path: Path = SQUAD) -> tuple[list[Passage], list[Question]]:
    passages, questions = [], []
    for article in json.loads(path.read_text())["data"]:
        title = article["title"].replace("_", " ")
        for paragraph in article["paragraphs"]:
            pid = len(passages)
            passages.append(Passage(pid, title, paragraph["context"]))
            for qa in paragraph["qas"]:
                answers = tuple(dict.fromkeys(a["text"] for a in qa["answers"]))
                questions.append(Question(qa["question"].strip(), pid, answers))
    return passages, questions


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]
