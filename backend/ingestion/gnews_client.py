"""
Thin client for gnews.io (github/docs: https://docs.gnews.io/), a second news
source alongside newsdata.io (backend/ingestion/newsdata_client.py). Chunks from
both sources share the same schema and can coexist in the same Qdrant collection.

Free-plan constraints (confirmed live against the real API, not assumed):
- 100 requests/day, max 10 articles per request.
- `content` is present and substantially longer than newsdata.io's free-tier
  `description`, but is truncated with a literal "... [N chars]" marker appended
  -- e.g. "Russia and Ukraine have traded attacks... [4112 chars]". Must be
  stripped before use (see gnews_pipeline.py), or it reads/speaks as garbage.
- 12-hour publication delay ("real-time" news needs a paid plan).
- Free plan is non-commercial (fine for this personal project).
"""
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://gnews.io/api/v4"

# GNews appends "... [N chars]" (or similar) to indicate the true, untruncated
# length of `content` on the free plan. It's not part of the article text.
TRUNCATION_MARKER_RE = re.compile(r"\s*\.\.\.\s*\[\d+\s*chars?\]\s*$", re.IGNORECASE)


class GNewsAPIError(Exception):
    """Raised for a non-2xx response from gnews.io, with the API's own error message."""


@dataclass
class GNewsArticle:
    """One article from gnews.io, keeping only the fields we use."""
    article_id: str
    title: Optional[str]
    description: Optional[str]
    content: Optional[str]
    url: Optional[str]
    published_at: Optional[str]
    language: Optional[str]
    source_name: Optional[str] = None
    source_url: Optional[str] = None


def strip_truncation_marker(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return TRUNCATION_MARKER_RE.sub("", text).strip()


def _get_api_key() -> str:
    api_key = os.environ.get("GNEWS_API_KEY")
    if not api_key:
        raise ValueError(
            "GNEWS_API_KEY environment variable is missing. "
            "Set it in .env -- get a key at https://gnews.io/"
        )
    return api_key


def fetch_top_headlines(
    category: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    query: Optional[str] = None,
    max_articles: int = 10,
) -> Dict[str, Any]:
    """Calls GET /top-headlines. Raises GNewsAPIError on a non-2xx response."""
    params: Dict[str, Any] = {"apikey": _get_api_key(), "max": min(max_articles, 10)}
    if category:
        params["category"] = category
    if country:
        params["country"] = country
    if language:
        params["lang"] = language
    if query:
        params["q"] = query

    resp = requests.get(f"{BASE_URL}/top-headlines", params=params, timeout=15)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise GNewsAPIError(f"gnews.io returned {resp.status_code}: {detail}")
    return resp.json()


def fetch_search(
    query: str,
    country: Optional[str] = None,
    language: Optional[str] = None,
    max_articles: int = 10,
) -> Dict[str, Any]:
    """Calls GET /search (q is mandatory here). Raises GNewsAPIError on a non-2xx response."""
    params: Dict[str, Any] = {"apikey": _get_api_key(), "q": query, "max": min(max_articles, 10)}
    if country:
        params["country"] = country
    if language:
        params["lang"] = language

    resp = requests.get(f"{BASE_URL}/search", params=params, timeout=15)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise GNewsAPIError(f"gnews.io returned {resp.status_code}: {detail}")
    return resp.json()


def _parse_articles(data: Dict[str, Any]) -> List[GNewsArticle]:
    articles: List[GNewsArticle] = []
    for item in data.get("articles", []):
        source = item.get("source") or {}
        articles.append(
            GNewsArticle(
                article_id=item.get("id") or item.get("url", ""),
                title=item.get("title"),
                description=item.get("description"),
                content=strip_truncation_marker(item.get("content")),
                url=item.get("url"),
                published_at=item.get("publishedAt"),
                language=item.get("lang"),
                source_name=source.get("name"),
                source_url=source.get("url"),
            )
        )
    return articles


def fetch_top_headlines_articles(
    category: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    query: Optional[str] = None,
    max_articles: int = 10,
) -> List[GNewsArticle]:
    data = fetch_top_headlines(
        category=category, country=country, language=language, query=query, max_articles=max_articles
    )
    return _parse_articles(data)


def fetch_search_articles(
    query: str,
    country: Optional[str] = None,
    language: Optional[str] = None,
    max_articles: int = 10,
) -> List[GNewsArticle]:
    data = fetch_search(query=query, country=country, language=language, max_articles=max_articles)
    return _parse_articles(data)
