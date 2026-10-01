"""Retrieval-augmented question answering over a document collection."""
import os

# PyTorch and FAISS each ship their own OpenMP runtime; on macOS the two deadlock unless OpenMP is
# limited to one thread. This must be set before either library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
