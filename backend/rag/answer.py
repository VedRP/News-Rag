from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
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

def retrieve_chunks(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Embeds the question and retrieves the top-k most similar chunks from Qdrant.
    """
    client = get_client()
    ensure_collection(client)
    
    query_vector = embed_text(question)
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    
    chunks: List[Dict[str, Any]] = []
    for point in search_response.points:
        payload = point.payload or {}
        payload["_similarity_score"] = point.score
        chunks.append(payload)
        
    return chunks

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
    
    # Check if retrieval returned any plausible matches
    top_score = retrieved_chunks[0].get("_similarity_score", 0.0) if retrieved_chunks else 0.0
    
    # If no results or top similarity is too low, reject immediately to avoid hallucination
    if not retrieved_chunks or top_score < min_similarity:
        return RAGResult(
            question=question,
            answer="The provided newspaper sources do not contain sufficient information to answer this question.",
            citations=[],
            chunks_used=0,
            top_similarity=top_score,
        )
        
    # Build citations list
    citations: List[Dict[str, Any]] = []
    for c in retrieved_chunks:
        citations.append({
            "source": c.get("source"),
            "page": c.get("page"),
            "date": c.get("date"),
            "title": c.get("title"),
            "score": round(float(c.get("_similarity_score", 0.0)), 4),
        })
        
    # Format evidence context and build user prompt
    context_str = format_retrieved_context(retrieved_chunks)
    user_prompt = build_rag_user_prompt(question, context_str)
    
    # Generate grounded response
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
