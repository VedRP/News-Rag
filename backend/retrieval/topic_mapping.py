"""
Maps our intent taxonomy's topics (backend.llm.prompts.intent_parser.TOPIC_TAXONOMY --
richer, meant for natural-language understanding) to newsdata.io's actual, fixed
category vocabulary (confirmed from their docs: business, crime, domestic, education,
entertainment, environment, food, health, lifestyle, politics, science, sports,
technology, top, tourism, world, other -- 17 total).

Why this exists: chunks indexed from newsdata.io (backend/ingestion/newsdata_pipeline.py)
carry the API's own category values verbatim in their "topics" field. Confirmed live
against actually-indexed data: only newsdata.io's real category strings ever appear
there (e.g. "business", "politics", "top") -- never anything from our richer taxonomy
(e.g. "cricket", "automobile", "real_estate" don't exist in the data at all). A Qdrant
topic filter built directly from an unmapped intent topic would silently match nothing
for most of the taxonomy.

Topics with no reasonable newsdata.io equivalent map to an empty list, meaning "don't
filter on this" -- retrieval still runs (semantic search via raw_query_for_search), it
just isn't narrowed by category. That's a safe default: backend.rag.answer already
falls back to unfiltered search whenever a filter yields zero hits.
"""
from typing import Dict, List

TOPIC_TO_NEWSDATA_CATEGORY: Dict[str, List[str]] = {
    "politics": ["politics"],
    "national": ["domestic", "top"],
    "international": ["world"],
    "sports": ["sports"],
    "business": ["business"],
    "finance": ["business"],  # newsdata.io has no dedicated finance/markets category
    "technology": ["technology"],
    "science": ["science"],
    "health": ["health"],
    "education": ["education"],
    "entertainment": ["entertainment"],
    "lifestyle": ["lifestyle"],
    "automobile": ["technology"],  # closest real category; weak match
    "environment": ["environment"],
    "weather": ["environment"],  # weak match; newsdata.io has no weather category
    "crime": ["crime"],
    "real_estate": [],  # no real equivalent -- semantic search only
    "agriculture": [],
    "travel": ["tourism"],
    "culture": ["lifestyle"],  # weak match
    "local": ["domestic"],
    "defence": [],  # too broad/ambiguous to map safely (world/politics/top all overlap)
    "energy": [],
    "jobs": ["business"],  # weak match
    "general": [],
}


def to_newsdata_categories(topic: str) -> List[str]:
    """Returns the newsdata.io category values to filter/fetch by for a given intent topic."""
    return TOPIC_TO_NEWSDATA_CATEGORY.get(topic, [])
