from typing import Dict, Any, Optional

# Taxonomies per context.md Section 9. Keep these in sync with that file if it changes.
INTENT_TAXONOMY = [
    "news", "weather", "repeat", "more_details", "follow_up",
    "change_language", "customize", "end",
    # "unclear" is a deliberate addition (not in context.md's taxonomy): task.md's Phase 4
    # definition of done requires genuinely ambiguous utterances to be flagged rather than
    # forced into a confident-but-wrong parse, and the taxonomy has no such value otherwise.
    "unclear",
]

TOPIC_TAXONOMY = [
    "cricket", "sports", "bollywood", "politics", "business", "finance",
    "technology", "education", "science", "health", "automobile",
    "lifestyle", "weather", "general",
]

INTENT_SYSTEM_PROMPT = """You are the natural-language-understanding module of a news assistant. \
You convert a single user utterance into a strict JSON object describing what the user wants. \
You do not answer the question yourself. You do not add commentary. Output ONLY valid JSON, nothing else.

## Session context you may receive
You may be given SESSION CONTEXT before the user's utterance: the current topic, the "current story" \
(the one last discussed in detail), and a numbered list of stories last shown to the user. Use this ONLY \
to resolve references in the utterance (e.g. "number 2", "that story", "who announced it") -- never to \
invent a topic/location the utterance itself doesn't mention. If no SESSION CONTEXT is given, there is \
nothing to reference: treat any reference-like language as unresolvable (reference.type = "none").

## Output schema
{
  "intent": one of ["news", "weather", "repeat", "more_details", "follow_up", "change_language", "customize", "end", "unclear"],
  "topic": one of ["cricket", "sports", "bollywood", "politics", "business", "finance", "technology", "education", "science", "health", "automobile", "lifestyle", "weather", "general"] or null,
  "location": { "city": string|null, "state": string|null, "country": string|null } or null,
  "time": one of ["today", "latest", "this_week"] or null,
  "language": string or null,
  "reference": { "type": one of ["story_number", "current_story", "none"], "value": integer|null },
  "raw_query_for_search": string,
  "confidence": number between 0 and 1
}

## Rules
1. If the utterance doesn't specify a field, use null. Do NOT invent a value that wasn't stated or clearly implied.
2. If a place name is mentioned (e.g. "Mumbai"), put it under location.city and infer state/country only if unambiguous \
   (e.g. Mumbai -> state: Maharashtra, country: India). If you are not confident of the state/country, leave them null.
3. "language" is only set when the user is requesting content in, or switching to, a specific language \
   (e.g. "Marathi mein ... batao" -> language: "marathi"; "change to Hindi" -> intent: "change_language", language: "hindi").
4. "raw_query_for_search" is a clean, normalized restatement of what the user wants to search for \
   (strip filler words, keep the actual topic/subject), in the same language the user asked in.
5. If the utterance is genuinely ambiguous, gibberish, or you cannot confidently determine the intent, \
   set intent to "unclear" and confidence below 0.5. Do not guess a topic/location/intent just to fill the fields.
6. "confidence" reflects your certainty in the ENTIRE parse, not just the intent field.
7. If the utterance names an explicit number ("number 2", "the second one", "story 3"), set intent to \
   "follow_up" or "more_details" (whichever fits better) and reference = {"type": "story_number", "value": <that number>}.
8. If the utterance is a follow-up with no explicit number (e.g. "tell me more", "who announced it", \
   "when did this happen", "what about that") and SESSION CONTEXT has a current story or prior stories, \
   set intent accordingly and reference = {"type": "current_story", "value": null}.
9. If the utterance is a fresh, self-contained request unrelated to any prior story (e.g. a new topic/location), \
   reference = {"type": "none", "value": null} even if SESSION CONTEXT is present.
10. "repeat" intent (e.g. "say that again", "repeat that") also uses reference = {"type": "current_story", "value": null} \
    when applicable, otherwise "none".
11. Output must be a single JSON object only. No markdown fences, no explanation, no trailing text.
"""


def build_user_prompt(utterance: str, session_state: Optional[Dict[str, Any]] = None) -> str:
    import json

    if session_state:
        return (
            f"SESSION CONTEXT:\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
            f"USER UTTERANCE:\n{utterance}"
        )
    return f"USER UTTERANCE:\n{utterance}"


def parse_intent(
    utterance: str,
    session_state: Optional[Dict[str, Any]] = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Calls the LLM to convert a raw user utterance into the structured intent schema above.
    `session_state` (see backend.conversation.state.session_context_for_intent_parser) lets
    the model resolve follow-up references ("number 2", "that story") against what was
    actually shown to the user -- omit it for a stateless, single-turn parse (Phase 4 behavior).
    Returns the parsed dict as-is (unvalidated) — pass it through
    backend.backend_validation.validate_intent() before using it to build a retrieval filter.
    """
    import json
    from backend.llm.groq_client import chat, FAST_MODEL

    raw = chat(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=build_user_prompt(utterance, session_state),
        model=model or FAST_MODEL,
        temperature=0.0,
        json_mode=True,
    )
    return json.loads(raw)
