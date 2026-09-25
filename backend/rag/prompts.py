from typing import List, Dict, Any

# Human-readable names the LLM is told to write in. Phase 6 officially supports these
# three (per task.md's "start with 2-3 languages" guidance) -- see
# backend.backend_validation.SUPPORTED_LANGUAGES for the enforced allowlist.
DEFAULT_TARGET_LANGUAGE = "english"


def build_grounded_system_prompt(target_language: str = DEFAULT_TARGET_LANGUAGE) -> str:
    """
    Builds the grounded-RAG system prompt for a specific response language.

    The hallucination-control rules are language-independent and stay verbatim
    regardless of target_language (context.md Section 12 -- these are load-bearing,
    not stylistic, so they aren't translated, only the instruction of which language
    to answer IN changes).
    """
    language_instruction = (
        "Respond ONLY in English."
        if target_language.lower() == "english"
        else (
            f"Respond ONLY in {target_language.title()}, using its native script "
            f"(e.g. Devanagari for Hindi/Marathi), not a transliteration. Proper nouns "
            f"that have no natural translation (names, places, organizations) may stay "
            f"as-is, but do not mix languages otherwise. Citations in square brackets "
            f"(e.g. \"[Source: ..., Page ...]\") stay in English regardless."
        )
    )

    return f"""You are a factual, grounded AI news assistant. Your job is to answer the user's question using ONLY the retrieved newspaper evidence provided in the context below.

## Hallucination Control Rules (Mandatory)
1. Answer using only the retrieved newspaper evidence provided.
2. Do not invent facts, names, numbers, dates, or outcomes absent from retrieved context.
3. If evidence is insufficient, say so plainly rather than guessing (e.g. "The provided newspaper sources do not contain sufficient information to answer this.") -- in the target response language below, not necessarily English.
4. Distinguish confirmed facts from claims/reports; preserve source uncertainty where the source itself is uncertain (use phrases like "according to reports", "as reported by").
5. Always cite the source for facts mentioned in your answer, using whatever locator the excerpt actually provides -- "[Source: <name>, Page <n>]" when a page number is given, "[Source: <name>, Link: <url>]" when a link is given instead, or just "[Source: <name>]" if neither is given. Never invent a page number or link that wasn't in the excerpt.
6. Do NOT let your own pretrained knowledge override or supplement retrieved current information.

## Response Language
{language_instruction}

## Response Style
- Deliver clear, direct, and factual answers.
- Avoid markdown decoration beyond clean paragraphs and bracketed source citations.
- If the question cannot be answered from the provided excerpts, explicitly refuse to answer and note that the provided articles do not contain this information.
"""


# Backward-compatible default (English) prompt, for any caller that hasn't been
# updated to pass a target_language yet.
GROUNDED_RAG_SYSTEM_PROMPT = build_grounded_system_prompt(DEFAULT_TARGET_LANGUAGE)

def format_retrieved_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved article chunks into a clear evidence block for the LLM.
    """
    if not chunks:
        return "No relevant newspaper articles were found in the database."

    formatted_pieces: List[str] = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "Unknown Source")
        date = c.get("date", "Unknown Date")
        title = c.get("title", "Untitled")
        text = c.get("text", "").strip()

        # PDF-sourced chunks have a page number; API-sourced chunks (newsdata.io) don't
        # but may have a link instead. Only show what's actually meaningful, so the LLM
        # never parrots back a literal "Page: None" in its citation.
        locator = f"Page: {c['page']}" if c.get("page") is not None else None
        if not locator and c.get("link"):
            locator = f"Link: {c['link']}"
        locator_str = f" | {locator}" if locator else ""

        piece = (
            f"--- [ARTICLE EXCERPT #{i}] ---\n"
            f"Source: {source}{locator_str} | Date: {date}\n"
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
