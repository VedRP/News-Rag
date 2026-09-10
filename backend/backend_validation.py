from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.llm.prompts.intent_parser import INTENT_TAXONOMY, TOPIC_TAXONOMY

VALID_TIME_VALUES = {"today", "latest", "this_week"}
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class ValidatedIntent:
    """Result of validating a raw intent-parser JSON object before it drives retrieval."""
    is_actionable: bool
    intent: str
    topic: Optional[str]
    location: Optional[Dict[str, Optional[str]]]
    time: Optional[str]
    language: Optional[str]
    raw_query_for_search: str
    confidence: float
    errors: List[str] = field(default_factory=list)


def _clean_location(raw_location: Any, errors: List[str]) -> Optional[Dict[str, Optional[str]]]:
    if raw_location is None:
        return None
    if not isinstance(raw_location, dict):
        errors.append(f"location was not an object: {raw_location!r}")
        return None

    cleaned: Dict[str, Optional[str]] = {}
    for key in ("city", "state", "country"):
        value = raw_location.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"location.{key} was not a string: {value!r}")
            value = None
        cleaned[key] = value

    if not any(cleaned.values()):
        return None
    return cleaned


def validate_intent(parsed: Dict[str, Any]) -> ValidatedIntent:
    """
    Validates a raw JSON object from backend.llm.prompts.intent_parser.parse_intent()
    against the known taxonomies before it's allowed to drive a Qdrant filter + retrieval.

    An intent is only "actionable" (safe to use for retrieval/generation) if:
    - intent is a recognized, non-"unclear" value
    - confidence is at least LOW_CONFIDENCE_THRESHOLD
    Anything else should be routed to a clarifying response instead of a guessed answer.
    """
    errors: List[str] = []

    intent = parsed.get("intent")
    if intent not in INTENT_TAXONOMY:
        errors.append(f"unknown intent: {intent!r}")
        intent = "unclear"

    topic = parsed.get("topic")
    if topic is not None and topic not in TOPIC_TAXONOMY:
        # Not fatal: ingestion's own keyword-based topic tagger (backend/ingestion/metadata.py)
        # produces some tags (e.g. "judiciary", "infrastructure") that aren't in context.md's
        # canonical taxonomy. Drop the topic rather than filtering on a value that will never
        # match stored data, but don't downgrade the whole parse to "unclear" over it.
        errors.append(f"topic not in taxonomy, dropped: {topic!r}")
        topic = None

    location = _clean_location(parsed.get("location"), errors)

    time_value = parsed.get("time")
    if time_value is not None and time_value not in VALID_TIME_VALUES:
        errors.append(f"time not in {sorted(VALID_TIME_VALUES)}, dropped: {time_value!r}")
        time_value = None

    language = parsed.get("language")
    if language is not None and not isinstance(language, str):
        errors.append(f"language was not a string, dropped: {language!r}")
        language = None

    raw_query = parsed.get("raw_query_for_search") or ""
    if not isinstance(raw_query, str):
        errors.append(f"raw_query_for_search was not a string: {raw_query!r}")
        raw_query = ""

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)):
        errors.append(f"confidence missing or non-numeric: {confidence!r}")
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    is_actionable = (
        intent not in ("unclear",)
        and confidence >= LOW_CONFIDENCE_THRESHOLD
        and bool(raw_query.strip())
    )

    return ValidatedIntent(
        is_actionable=is_actionable,
        intent=intent,
        topic=topic,
        location=location,
        time=time_value,
        language=language,
        raw_query_for_search=raw_query,
        confidence=confidence,
        errors=errors,
    )
