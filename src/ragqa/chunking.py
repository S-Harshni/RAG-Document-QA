"""Split documents into overlapping chunks along sentence boundaries.

Small chunks are precise but can cut an answer off from its context; large chunks keep context but
dilute the match and cost more tokens in the prompt. `run.py` measures that trade-off.
"""
import re
from dataclasses import dataclass

from ragqa.corpus import Document, Question

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    id: int
    doc_id: int
    title: str
    text: str
    start: int      # character span inside the document
    end: int

    @property
    def words(self) -> int:
        return len(self.text.split())


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in `text` (whitespace between them is left out)."""
    spans, at = [], 0
    for match in SENTENCE_END.finditer(text):
        if match.start() > at:
            spans.append((at, match.start()))
        at = match.end()
    if at < len(text):
        spans.append((at, len(text)))
    return spans


def chunk_document(doc: Document, size: int, overlap: int, first_id: int = 0) -> list[Chunk]:
    """Pack whole sentences into chunks of about `size` words; consecutive chunks share about `overlap` words."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than the chunk size")
    spans = sentences(doc.text)
    lengths = [len(doc.text[a:b].split()) for a, b in spans]
    chunks, i = [], 0
    while i < len(spans):
        j, words = i, 0
        while j < len(spans) and (words == 0 or words + lengths[j] <= size):
            words += lengths[j]
            j += 1
        start, end = spans[i][0], spans[j - 1][1]
        chunks.append(Chunk(first_id + len(chunks), doc.id, doc.title, doc.text[start:end], start, end))
        if j >= len(spans):
            break
        back, shared = j, 0
        while back - 1 > i and shared + lengths[back - 1] <= overlap:   # step back to create the overlap
            back -= 1
            shared += lengths[back]
        i = back
    return chunks


def chunk_corpus(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, size, overlap, first_id=len(chunks)))
    return chunks


def gold_chunks(question: Question, chunks: list[Chunk], by_doc: dict[int, list[int]]) -> set[int]:
    """Chunks that contain the whole answer span. Overlap means there can be more than one."""
    return {i for i in by_doc[question.doc_id] if chunks[i].start <= question.start and question.end <= chunks[i].end}


def index_by_document(chunks: list[Chunk]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for c in chunks:
        out.setdefault(c.doc_id, []).append(c.id)
    return out
