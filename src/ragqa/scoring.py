"""SQuAD-style answer scoring."""
import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalise(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(ARTICLES.sub(" ", text).split())


def f1(prediction: str, reference: str) -> float:
    p, r = normalise(prediction).split(), normalise(reference).split()
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or not common:
        return float(p == r)
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, references: tuple[str, ...]) -> float:
    return max(f1(prediction, ref) for ref in references)


def contains_answer(prediction: str, references: tuple[str, ...]) -> bool:
    """True when a reference answer appears in the prediction, or the prediction is a reference's core."""
    p = normalise(prediction)
    return bool(p) and any(normalise(ref) in p or (len(p) > 2 and p in normalise(ref)) for ref in references if normalise(ref))
