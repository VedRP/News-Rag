"""
Converts gnews.io articles (backend/ingestion/gnews_client.py) into the same
chunk-dict schema newsdata_pipeline.py produces, so both sources' chunks can be
indexed into the same Qdrant collection and retrieval/generation stay unchanged.

GNews's category taxonomy (general, world, nation, business, technology,
entertainment, sports, science, health -- 9 fixed values, confirmed from their
docs) differs from newsdata.io's (17 values). backend/retrieval/topic_mapping.py
maps our intent taxonomy to the union of both, so filtering works regardless of
which source a chunk came from.
"""
from typing import Any, Dict, List, Optional

from .gnews_client import GNewsArticle, fetch_search_articles, fetch_top_headlines_articles


def article_to_chunk(article: GNewsArticle, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Maps one GNewsArticle to our standard chunk dict. `text` prefers the (already
    truncation-marker-stripped) `content` over `description`, since GNews's free-tier
    content is substantially fuller than newsdata.io's.

    GNews's API doesn't return a category per-article (unlike newsdata.io) -- the
    category was the *request* filter, so it's passed in explicitly by the caller
    (fetch_and_convert below) rather than read off the article itself.
    """
    text = article.content or article.description or article.title or ""
    date = (article.published_at or "")[:10] or None  # "YYYY-MM-DDTHH:MM:SSZ" -> "YYYY-MM-DD"

    return {
        "text": text,
        "title": article.title or "Untitled",
        "source": article.source_name or article.source_url or "gnews.io",
        "page": None,
        "date": date,
        "language": (article.language or "english").lower(),
        "location": [],  # gnews.io doesn't return per-article country/location tags
        "topics": [category.lower()] if category else ["general"],
        "article_id": article.article_id,
        "link": article.url,
    }


def fetch_and_convert(
    category: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    query: Optional[str] = None,
    max_articles: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetches from gnews.io and returns chunk dicts. Uses /search when `query` is given
    (gnews.io requires q there), otherwise /top-headlines (category-based, no query
    needed -- returns trending articles).
    """
    if query:
        articles = fetch_search_articles(query=query, country=country, language=language, max_articles=max_articles)
    else:
        articles = fetch_top_headlines_articles(
            category=category, country=country, language=language, max_articles=max_articles
        )
    return [article_to_chunk(a, category=category) for a in articles]
