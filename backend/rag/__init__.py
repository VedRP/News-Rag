"""RAG (Retrieval-Augmented Generation) package."""
from .prompts import (
    GROUNDED_RAG_SYSTEM_PROMPT,
    format_retrieved_context,
    build_rag_user_prompt,
)
from .answer import answer_question, retrieve_chunks, RAGResult

__all__ = [
    "GROUNDED_RAG_SYSTEM_PROMPT",
    "format_retrieved_context",
    "build_rag_user_prompt",
    "answer_question",
    "retrieve_chunks",
    "RAGResult",
]
