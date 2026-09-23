"""
Phase 5 conversational turn handler: ties intent parsing + validation + session
state + retrieval + grounded generation together so multi-turn follow-ups work
without the user repeating context (context.md Section 10).

One call per user utterance: handle_turn(state, utterance) -> TurnResult.
The caller (a CLI loop, later a voice loop) owns the SessionState and passes
it back in on every turn; this module only ever mutates it via
backend.conversation.state's helpers.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.backend_validation import SUPPORTED_LANGUAGES, normalize_language, validate_intent
from backend.conversation.phrases import phrase
from backend.conversation.state import (
    SessionState,
    find_story_by_number,
    record_turn,
    session_context_for_intent_parser,
    set_story_listing,
)
from backend.llm.prompts.intent_parser import parse_intent
from backend.rag.answer import SIMILARITY_THRESHOLD, generate_grounded_answer, retrieve_chunks
from backend.retrieval.metadata_filter import build_filter

NOT_YET_SUPPORTED_INTENTS = {"weather", "customize"}


@dataclass
class TurnResult:
    """What a single conversational turn produced, for the CLI/voice layer to render."""
    utterance: str
    kind: str  # "news_listing" | "story_detail" | "clarify" | "repeat" | "language_ack" | "end" | "unsupported"
    spoken_answer: str
    stories: List[Dict[str, Any]] = field(default_factory=list)
    citation: Optional[Dict[str, Any]] = None


def _format_story_listing(numbered_stories: List[Dict[str, Any]], language: str) -> str:
    """
    Formats the numbered listing header/footer in the session's language, but leaves
    each story's title/snippet as the raw source excerpt (source text is whatever
    language it was published in -- translating quoted evidence risks introducing
    inaccuracies outside grounded generation's hallucination-control rules). The
    follow-up-driven detail answers (story_detail) are the ones that are actually
    LLM-translated into the target language.
    """
    if not numbered_stories:
        return phrase(language, "no_stories")

    lines = [phrase(language, "listing_header", n=len(numbered_stories))]
    for story in numbered_stories:
        snippet = (story.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160].rsplit(" ", 1)[0] + "..."
        source = story.get("source", "Unknown Source")
        page = story.get("page", "?")
        lines.append(
            f"{story['number']}. {story.get('title', 'Untitled')} -- {snippet} "
            f"[Source: {source}, Page {page}]"
        )
    lines.append(phrase(language, "listing_footer"))
    return "\n".join(lines)


def _resolve_reference(state: SessionState, reference: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ref_type = reference.get("type")
    if ref_type == "story_number":
        return find_story_by_number(state, reference["value"])
    if ref_type == "current_story":
        return state.current_story
    return None


def _clarify(utterance: str, message: str, state: SessionState) -> TurnResult:
    result = TurnResult(utterance=utterance, kind="clarify", spoken_answer=message)
    record_turn(state, utterance, result.spoken_answer)
    return result


def handle_turn(state: SessionState, utterance: str, top_k: int = 5) -> TurnResult:
    session_context = session_context_for_intent_parser(state)
    parsed = parse_intent(utterance, session_state=session_context)
    validated = validate_intent(parsed)

    if validated.intent == "unclear" or (
        not validated.is_actionable and validated.reference.get("type") == "none"
    ):
        return _clarify(utterance, phrase(state.language, "clarify"), state)

    requested_language = normalize_language(validated.language) if validated.language else None

    if validated.intent == "change_language":
        if requested_language:
            state.language = requested_language
            message = phrase(state.language, "language_ack", language=state.language.title())
        else:
            message = phrase(
                state.language,
                "language_unsupported",
                language=validated.language or "that",
                supported=", ".join(sorted(SUPPORTED_LANGUAGES)),
                current=state.language,
            )
        result = TurnResult(utterance=utterance, kind="language_ack", spoken_answer=message)
        record_turn(state, utterance, result.spoken_answer)
        return result

    # A non-language-switch turn can still name a language (e.g. "Marathi mein cricket
    # news batao" -- context.md Section 9's own example): honor it for this turn's
    # generation and carry it forward, without requiring a separate "change to X" turn.
    if requested_language:
        state.language = requested_language

    if validated.intent == "end":
        result = TurnResult(utterance=utterance, kind="end", spoken_answer=phrase(state.language, "end"))
        record_turn(state, utterance, result.spoken_answer)
        return result

    if validated.intent == "repeat":
        if state.conversation_history:
            message = state.conversation_history[-1]["answer"]
        else:
            message = phrase(state.language, "nothing_to_repeat")
        result = TurnResult(utterance=utterance, kind="repeat", spoken_answer=message)
        record_turn(state, utterance, result.spoken_answer)
        return result

    if validated.intent in ("follow_up", "more_details") or validated.reference.get("type") != "none":
        target = _resolve_reference(state, validated.reference)
        if target is None:
            return _clarify(utterance, phrase(state.language, "clarify_reference"), state)
        state.current_story = target
        gen = generate_grounded_answer(
            validated.raw_query_for_search or utterance,
            [target],
            SIMILARITY_THRESHOLD,
            target_language=state.language,
        )
        citation = gen.citations[0] if gen.citations else None
        result = TurnResult(
            utterance=utterance, kind="story_detail", spoken_answer=gen.answer, citation=citation
        )
        record_turn(state, utterance, result.spoken_answer)
        return result

    if validated.intent in NOT_YET_SUPPORTED_INTENTS:
        message = phrase(state.language, "unsupported_intent", intent=validated.intent)
        result = TurnResult(utterance=utterance, kind="unsupported", spoken_answer=message)
        record_turn(state, utterance, result.spoken_answer)
        return result

    # Fresh "news" request: filtered retrieval -> numbered listing.
    qdrant_filter = build_filter(validated)
    search_query = validated.raw_query_for_search or utterance
    chunks = retrieve_chunks(search_query, top_k=top_k, qdrant_filter=qdrant_filter)
    if qdrant_filter is not None and not chunks:
        chunks = retrieve_chunks(search_query, top_k=top_k, qdrant_filter=None)

    numbered = set_story_listing(state, chunks)
    if validated.topic:
        state.current_topic = validated.topic

    spoken = _format_story_listing(numbered, state.language)
    result = TurnResult(utterance=utterance, kind="news_listing", spoken_answer=spoken, stories=numbered)
    record_turn(state, utterance, result.spoken_answer)
    return result
