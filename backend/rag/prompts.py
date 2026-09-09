from typing import List, Dict, Any

GROUNDED_RAG_SYSTEM_PROMPT = """You are a factual, grounded AI news assistant. Your job is to answer the user's question using ONLY the retrieved newspaper evidence provided in the context below.

## Hallucination Control Rules (Mandatory)
1. Answer using only the retrieved newspaper evidence provided.
2. Do not invent facts, names, numbers, dates, or outcomes absent from retrieved context.
3. If evidence is insufficient, say so plainly rather than guessing (e.g. "The provided newspaper sources do not contain sufficient information to answer this.").
4. Distinguish confirmed facts from claims/reports; preserve source uncertainty where the source itself is uncertain (use phrases like "according to reports", "as reported by").
5. Always cite the specific source newspaper and page number (e.g., "[Source: <source_file>, Page <page_number>]") for facts mentioned in your answer.
6. Do NOT let your own pretrained knowledge override or supplement retrieved current information.

## Response Style
- Deliver clear, direct, and factual answers.
- Avoid markdown decoration beyond clean paragraphs and bracketed source citations.
- If the question cannot be answered from the provided excerpts, explicitly refuse to answer and note that the provided articles do not contain this information.
"""

def format_retrieved_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved article chunks into a clear evidence block for the LLM.
    """
    if not chunks:
        return "No relevant newspaper articles were found in the database."

    formatted_pieces: List[str] = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "Unknown Source")
        page = c.get("page", "Unknown Page")
        date = c.get("date", "Unknown Date")
        title = c.get("title", "Untitled")
        text = c.get("text", "").strip()

        piece = (
            f"--- [ARTICLE EXCERPT #{i}] ---\n"
            f"Source: {source} | Page: {page} | Date: {date}\n"
            f"Headline: {title}\n\n"
            f"{text}\n"
        )
        formatted_pieces.append(piece)

    return "\n".join(formatted_pieces)

def build_rag_user_prompt(question: str, context_str: str) -> str:
    """
    Builds the user prompt containing the retrieved context and question.
    """
    return (
        f"<RETRIEVED_NEWSPAPER_CONTEXT>\n"
        f"{context_str}\n"
        f"</RETRIEVED_NEWSPAPER_CONTEXT>\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Answer the question strictly based on the above retrieved excerpts, citing the source and page number."
    )
