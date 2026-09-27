from typing import List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchAny

from backend.backend_validation import ValidatedIntent
from backend.retrieval.topic_mapping import to_newsdata_categories


def build_filter(intent: ValidatedIntent) -> Optional[Filter]:
    """
    Builds a Qdrant metadata filter from a validated structured intent.

    Chunk payloads store a single flat "location" list containing both city and state
    names, and a "topics" list -- there are no separate city/state/country payload
    fields. So any location term the user gave (city and/or state) is matched against
    that one "location" field with MatchAny.

    "topics" on newsdata.io-sourced chunks (backend/ingestion/newsdata_pipeline.py)
    holds the API's own real category values verbatim, not our intent taxonomy's
    richer topic names -- confirmed live against indexed data (e.g. "business",
    "politics", "top", never "cricket" or "real_estate"). to_newsdata_categories()
    maps the intent's topic to the matching real category value(s) before filtering;
    a topic with no real equivalent (e.g. "agriculture") maps to an empty list, which
    means no topic filter at all -- safe, since retrieval still runs via semantic
    search on raw_query_for_search.

    "time" and "language" are intentionally NOT filtered on here:
    - Indexed articles carry their actual publication date, not a relative
      "today"/"this_week" label, so a hard filter on those would just return nothing.
    - Cross-lingual retrieval is BGE-M3's job (context.md Section 11) -- filtering out
      chunks by language before Phase 6 exists would break retrieval for any non-English
      query against the current (English-only) indexed content.
    """
    must: List[FieldCondition] = []

    if intent.topic:
        categories = to_newsdata_categories(intent.topic)
        if categories:
            must.append(FieldCondition(key="topics", match=MatchAny(any=categories)))

    if intent.location:
        location_terms = [v for v in intent.location.values() if v]
        if location_terms:
            must.append(FieldCondition(key="location", match=MatchAny(any=location_terms)))

    return Filter(must=must) if must else None
