import os
from typing import List
# pyrefly: ignore [missing-import]
import torch
from FlagEmbedding import BGEM3FlagModel

VECTOR_SIZE = 1024

_model = None

def get_model() -> BGEM3FlagModel:
    """Loads and caches the BAAI/bge-m3 embedding model."""
    global _model
    if _model is None:
        has_cuda = torch.cuda.is_available()
        # Use fp16 only if CUDA is available
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=has_cuda)
    return _model

def embed_text(text: str) -> List[float]:
    """
    Embeds a single string into a 1024-dimensional dense vector.
    """
    model = get_model()
    out = model.encode(
        [text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return out["dense_vecs"][0].tolist()

def embed_batch(texts: List[str], batch_size: int = 12) -> List[List[float]]:
    """
    Embeds a list of strings into 1024-dimensional dense vectors in batches.
    """
    if not texts:
        return []
    model = get_model()
    out = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return out["dense_vecs"].tolist()
