"""
Converts newsdata.io articles (backend/ingestion/newsdata_client.py) into the same
chunk-dict schema the PDF pipeline produces (context.md Section 7: text, title,
source, date, language, location, topics), so everything downstream -- embedding,
Qdrant indexing, retrieval, grounded generation -- works unchanged regardless of
which ingestion source built the chunk.

Unlike the PDF pipeline, most metadata here comes directly from the API (language,
country, category) rather than being inferred via keyword heuristics -- newsdata.io
already classifies these per article.
"""
from typing import Any, Dict, List, Optional

from .newsdata_client import NewsArticle, fetch_latest_articles

# newsdata.io returns plan-gated fields as a literal placeholder STRING (e.g.
# "ONLY AVAILABLE IN PAID PLANS") rather than null/omitting them -- confirmed via a
# live raw response. A naive `article.content or article.description or ...` fallback
# picks this placeholder (it's a non-empty, truthy string) instead of ever falling
# through to real data. Filter it out explicitly wherever we use a plan-gated field.
_GATED_PLACEHOLDER_MARKER = "ONLY AVAILABLE IN"


def _real_value(value: Optional[str]) -> Optional[str]:
    if not value or _GATED_PLACEHOLDER_MARKER in value:
        return None
    return value


def article_to_chunk(article: NewsArticle) -> Dict[str, Any]:
    """
    Maps one NewsArticle to our standard chunk dict. `text` prefers full `content`
    (paid plans) and falls back to `description` (available on the free plan) --
    see newsdata_client.py's module docstring, and _real_value() above for why a
    naive truthiness check on `content` isn't enough to detect "not available".
    """
    text = _real_value(article.content) or _real_value(article.description) or article.title or ""
    date = (article.pub_date or "")[:10] or None  # "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DD"

    return {
        "text": text,
        "title": article.title or "Untitled",
        "source": article.source_name or article.source_url or "newsdata.io",
        "page": None,  # not applicable to API-sourced articles
        "date": date,
        "language": (article.language or "english").lower(),
        "location": [c.title() for c in article.country],
        "topics": [c.lower() for c in article.category] or ["general"],
        "article_id": article.article_id,
        "link": article.link,
    }


def fetch_and_convert(
    query: Optional[str] = None,
    country: Optional[List[str]] = None,
    category: Optional[List[str]] = None,
    language: Optional[List[str]] = None,
    size: int = 10,
) -> List[Dict[str, Any]]:
    """Fetches latest articles from newsdata.io and returns them as chunk dicts."""
    articles = fetch_latest_articles(
        query=query, country=country, category=category, language=language, size=size
    )
    return [article_to_chunk(a) for a in articles]
