from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from qdrant_client.models import Filter
from backend.embeddings.embedder import embed_text
from backend.retrieval.qdrant_client import get_client, ensure_collection, COLLECTION_NAME
from backend.llm.groq_client import chat, GEN_MODEL
from .prompts import (
    GROUNDED_RAG_SYSTEM_PROMPT,
    format_retrieved_context,
    build_rag_user_prompt,
)

SIMILARITY_THRESHOLD = 0.32

@dataclass
class RAGResult:
    """Structured result from the grounded RAG answering pipeline."""
    question: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    chunks_used: int = 0
    top_similarity: float = 0.0

def retrieve_chunks(
    question: str,
    top_k: int = 4,
    qdrant_filter: Optional[Filter] = None,
) -> List[Dict[str, Any]]:
    """
    Embeds the question and retrieves the top-k most similar chunks from Qdrant,
    optionally narrowed by a metadata filter (see backend.retrieval.metadata_filter).
    """
    client = get_client()
    ensure_collection(client)

    query_vector = embed_text(question)
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    chunks: List[Dict[str, Any]] = []
    for point in search_response.points:
        payload = point.payload or {}
        payload["_similarity_score"] = point.score
        chunks.append(payload)

    return chunks


def _generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    min_similarity: float,
    model: str,
) -> "RAGResult":
    """Shared generation step used by both plain-question and structured-intent answering."""
    top_score = retrieved_chunks[0].get("_similarity_score", 0.0) if retrieved_chunks else 0.0

    if not retrieved_chunks or top_score < min_similarity:
        return RAGResult(
            question=question,
            answer="The provided newspaper sources do not contain sufficient information to answer this question.",
            citations=[],
            chunks_used=0,
            top_similarity=top_score,
        )

    citations: List[Dict[str, Any]] = []
    for c in retrieved_chunks:
        citations.append({
            "source": c.get("source"),
            "page": c.get("page"),
            "date": c.get("date"),
            "title": c.get("title"),
            "score": round(float(c.get("_similarity_score", 0.0)), 4),
        })

    context_str = format_retrieved_context(retrieved_chunks)
    user_prompt = build_rag_user_prompt(question, context_str)

    answer_text = chat(
        system_prompt=GROUNDED_RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=0.2,
    )

    return RAGResult(
        question=question,
        answer=answer_text,
        citations=citations,
        chunks_used=len(retrieved_chunks),
        top_similarity=top_score,
    )

def answer_question(
    question: str,
    top_k: int = 4,
    min_similarity: float = SIMILARITY_THRESHOLD,
    model: str = GEN_MODEL,
) -> RAGResult:
    """
    End-to-end grounded RAG answering:
    Question -> Qdrant vector retrieval -> Grounding verification -> LLM response with citations.
    """
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    return _generate_grounded_answer(question, retrieved_chunks, min_similarity, model)


def answer_structured_question(
    utterance: str,
    top_k: int = 4,
    min_similarity: float = SIMILARITY_THRESHOLD,
    model: str = GEN_MODEL,
) -> RAGResult:
    """
    Phase 4 entry point: raw utterance -> intent parse -> validation -> metadata-filtered +
    vector retrieval -> grounded generation (Phases 1-3 unchanged underneath).

    If the parsed intent isn't actionable (unclear/low-confidence/no search query), this
    returns a clarification RAGResult with zero chunks rather than guessing at retrieval.

    If a metadata filter is built but returns zero matches, retrieval falls back to an
    unfiltered vector search rather than reporting "insufficient information" -- the
    intent parser's topic taxonomy (context.md Section 9) is narrower than the topic tags
    ingestion actually produces (backend/ingestion/metadata.py), so an over-strict filter
    is a taxonomy mismatch, not genuine absence of evidence.
    """
    from backend.llm.prompts.intent_parser import parse_intent
    from backend.backend_validation import validate_intent
    from backend.retrieval.metadata_filter import build_filter

    parsed = parse_intent(utterance)
    validated = validate_intent(parsed)

    if not validated.is_actionable:
        return RAGResult(
            question=utterance,
            answer=(
                "I couldn't confidently understand that request. Could you rephrase it, "
                "e.g. with a topic and/or location? "
                f"(parsed intent: {validated.intent}, confidence: {validated.confidence:.2f})"
            ),
            citations=[],
            chunks_used=0,
            top_similarity=0.0,
        )

    qdrant_filter = build_filter(validated)
    search_query = validated.raw_query_for_search or utterance

    retrieved_chunks = retrieve_chunks(search_query, top_k=top_k, qdrant_filter=qdrant_filter)
    if qdrant_filter is not None and not retrieved_chunks:
        retrieved_chunks = retrieve_chunks(search_query, top_k=top_k, qdrant_filter=None)

    return _generate_grounded_answer(search_query, retrieved_chunks, min_similarity, model)
