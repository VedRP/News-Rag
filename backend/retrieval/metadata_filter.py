from typing import List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchAny

from backend.backend_validation import ValidatedIntent


def build_filter(intent: ValidatedIntent) -> Optional[Filter]:
    """
    Builds a Qdrant metadata filter from a validated structured intent.

    Chunk payloads (backend/ingestion/metadata.py) store a single flat "location" list
    containing both city and state names, and a "topics" list -- there are no separate
    city/state/country payload fields. So any location term the user gave (city and/or
    state) is matched against that one "location" field with MatchAny.

    "time" and "language" are intentionally NOT filtered on here:
    - Indexed chunks carry the newspaper's actual publication date, not a relative
      "today"/"this_week" label, so a hard filter on those would just return nothing.
    - Cross-lingual retrieval is BGE-M3's job (context.md Section 11) -- filtering out
      chunks by language before Phase 6 exists would break retrieval for any non-English
      query against the current (English-only) indexed content.
    """
    must: List[FieldCondition] = []

    if intent.topic:
        must.append(FieldCondition(key="topics", match=MatchAny(any=[intent.topic])))

    if intent.location:
        location_terms = [v for v in intent.location.values() if v]
        if location_terms:
            must.append(FieldCondition(key="location", match=MatchAny(any=location_terms)))

    return Filter(must=must) if must else None
