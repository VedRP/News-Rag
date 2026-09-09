"""Embeddings package."""
from .embedder import embed_text, embed_batch, get_model, VECTOR_SIZE

__all__ = ["embed_text", "embed_batch", "get_model", "VECTOR_SIZE"]
