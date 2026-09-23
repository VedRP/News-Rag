"""RAG (Retrieval-Augmented Generation) package."""
from .prompts import (
    GROUNDED_RAG_SYSTEM_PROMPT,
    build_grounded_system_prompt,
    format_retrieved_context,
    build_rag_user_prompt,
)
from .answer import answer_question, answer_structured_question, generate_grounded_answer, retrieve_chunks, RAGResult

__all__ = [
    "GROUNDED_RAG_SYSTEM_PROMPT",
    "build_grounded_system_prompt",
    "format_retrieved_context",
    "build_rag_user_prompt",
    "answer_question",
    "answer_structured_question",
    "generate_grounded_answer",
    "retrieve_chunks",
    "RAGResult",
]
