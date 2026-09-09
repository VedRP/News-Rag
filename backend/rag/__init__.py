"""RAG (Retrieval-Augmented Generation) package."""
from .prompts import (
    GROUNDED_RAG_SYSTEM_PROMPT,
    format_retrieved_context,
    build_rag_user_prompt,
)

__all__ = [
    "GROUNDED_RAG_SYSTEM_PROMPT",
    "format_retrieved_context",
    "build_rag_user_prompt",
]
