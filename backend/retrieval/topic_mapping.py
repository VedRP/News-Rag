"""
Maps our intent taxonomy's topics (backend.llm.prompts.intent_parser.TOPIC_TAXONOMY --
richer, meant for natural-language understanding) to the real, fixed category
vocabularies of our two news sources:

- newsdata.io (confirmed from their docs): business, crime, domestic, education,
  entertainment, environment, food, health, lifestyle, politics, science, sports,
  technology, top, tourism, world, other -- 17 values.
- gnews.io (confirmed from their docs): general, world, nation, business, technology,
  entertainment, sports, science, health -- 9 values.

Why this exists: chunks indexed from either source (backend/ingestion/
newsdata_pipeline.py, backend/ingestion/gnews_pipeline.py) carry that source's OWN
category values verbatim in their "topics" field -- confirmed live against actually-
indexed data. Neither source uses our richer taxonomy's exact words (e.g. "cricket",
"automobile", "real_estate" don't exist in either API's vocabulary). A Qdrant topic
filter built directly from an unmapped intent topic would silently match nothing.

Two mapping directions are needed:
- FETCH: when pre-filtering a request TO one specific API by category, each API
  needs its OWN category name (to_newsdata_category / to_gnews_category).
- FILTER: when narrowing a Qdrant search over ALREADY-INDEXED chunks (which may be
  from either source), match against the union of both sources' equivalent values
  (to_filter_categories).

A topic with no reasonable equivalent in a given taxonomy maps to None/empty, meaning
"don't filter/pre-filter on this" -- safe, since backend.rag.answer already falls back
to unfiltered semantic search whenever a filter yields zero hits.
"""
from typing import Dict, List, Optional

NEWSDATA_CATEGORIES = {
    "business", "crime", "domestic", "education", "entertainment", "environment",
    "food", "health", "lifestyle", "politics", "science", "sports", "technology",
    "top", "tourism", "world", "other",
}

GNEWS_CATEGORIES = {
    "general", "world", "nation", "business", "technology", "entertainment",
    "sports", "science", "health",
}

# topic -> (newsdata.io category or None, gnews.io category or None)
_TOPIC_TO_SOURCE_CATEGORIES: Dict[str, tuple[Optional[str], Optional[str]]] = {
    "politics": ("politics", "nation"),
    "national": ("domestic", "nation"),
    "international": ("world", "world"),
    "sports": ("sports", "sports"),
    "business": ("business", "business"),
    "finance": ("business", "business"),  # neither has a dedicated finance/markets category
    "technology": ("technology", "technology"),
    "science": ("science", "science"),
    "health": ("health", "health"),
    "education": ("education", None),  # gnews has no education category
    "entertainment": ("entertainment", "entertainment"),
    "lifestyle": ("lifestyle", None),  # gnews has no lifestyle category
    "automobile": ("technology", "technology"),  # closest real category; weak match
    "environment": ("environment", None),  # gnews has no environment category
    "weather": ("environment", None),  # weak match; neither has a weather category
    "crime": ("crime", None),  # gnews has no crime category
    "real_estate": (None, None),  # no real equivalent in either -- semantic search only
    "agriculture": (None, None),
    "travel": ("tourism", None),  # gnews has no tourism category
    "culture": ("lifestyle", None),  # weak match
    "local": ("domestic", "nation"),
    "defence": (None, None),  # too broad/ambiguous to map safely
    "energy": (None, None),
    "jobs": ("business", "business"),  # weak match
    "general": (None, "general"),
}


def to_newsdata_category(topic: str) -> Optional[str]:
    """The single newsdata.io category to request when fetching by this topic, if any."""
    return _TOPIC_TO_SOURCE_CATEGORIES.get(topic, (None, None))[0]


def to_gnews_category(topic: str) -> Optional[str]:
    """The single gnews.io category to request when fetching by this topic, if any."""
    return _TOPIC_TO_SOURCE_CATEGORIES.get(topic, (None, None))[1]


def to_filter_categories(topic: str) -> List[str]:
    """
    The set of real category values (from either source) to match against the "topics"
    field when filtering already-indexed chunks -- used by backend.retrieval.metadata_filter.
    """
    newsdata_cat, gnews_cat = _TOPIC_TO_SOURCE_CATEGORIES.get(topic, (None, None))
    return [c for c in (newsdata_cat, gnews_cat) if c]
