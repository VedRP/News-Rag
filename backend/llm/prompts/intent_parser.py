from typing import Dict, Any

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

## Output schema
{
  "intent": one of ["news", "weather", "repeat", "more_details", "follow_up", "change_language", "customize", "end", "unclear"],
  "topic": one of ["cricket", "sports", "bollywood", "politics", "business", "finance", "technology", "education", "science", "health", "automobile", "lifestyle", "weather", "general"] or null,
  "location": { "city": string|null, "state": string|null, "country": string|null } or null,
  "time": one of ["today", "latest", "this_week"] or null,
  "language": string or null,
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
7. Output must be a single JSON object only. No markdown fences, no explanation, no trailing text.
"""


def build_user_prompt(utterance: str) -> str:
    return f"USER UTTERANCE:\n{utterance}"


def parse_intent(utterance: str, model: str | None = None) -> Dict[str, Any]:
    """
    Calls the LLM to convert a raw user utterance into the structured intent schema above.
    Returns the parsed dict as-is (unvalidated) — pass it through
    backend.backend_validation.validate_intent() before using it to build a retrieval filter.
    """
    import json
    from backend.llm.groq_client import chat, FAST_MODEL

    raw = chat(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=build_user_prompt(utterance),
        model=model or FAST_MODEL,
        temperature=0.0,
        json_mode=True,
    )
    return json.loads(raw)
