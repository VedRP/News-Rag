"""
Thin client for newsdata.io's /latest endpoint (real-time news), replacing the
PDF pipeline (backend/ingestion/extract.py etc.) as the news source per user
direction. See https://newsdata.io/documentation (OpenAPI spec at
https://newsdata.io/openapi.json) for the full parameter/response reference.

Free-plan constraints that shape this module:
- `content` (full article text) is null on the free plan -- only `description`
  (a short snippet) is available. Chunks built from this source will have much
  less material than the PDF pipeline's full article text.
- Max `size` per request is 10 articles (50 on paid plans).
- 200 credits/day; a `size=10` request costs 1 credit (so up to ~2000 articles/day).
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://newsdata.io/api/1/latest"


class NewsDataAPIError(Exception):
    """Raised for a non-2xx response from newsdata.io, with the API's own error message."""


@dataclass
class NewsArticle:
    """One article from newsdata.io's /latest response, keeping only the fields we use."""
    article_id: str
    title: Optional[str]
    description: Optional[str]
    content: Optional[str]
    link: Optional[str]
    pub_date: Optional[str]
    language: Optional[str]
    country: List[str] = field(default_factory=list)
    category: List[str] = field(default_factory=list)
    source_name: Optional[str] = None
    source_url: Optional[str] = None


def _get_api_key() -> str:
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        raise ValueError(
            "NEWSDATA_API_KEY environment variable is missing. "
            "Set it in .env -- get a key at https://newsdata.io/"
        )
    return api_key


def fetch_latest(
    query: Optional[str] = None,
    country: Optional[List[str]] = None,
    category: Optional[List[str]] = None,
    language: Optional[List[str]] = None,
    size: int = 10,
    page: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls GET /1/latest and returns the raw parsed JSON response (status, totalResults,
    results, nextPage). Raises NewsDataAPIError on a non-2xx response.
    """
    params: Dict[str, Any] = {"apikey": _get_api_key(), "size": min(size, 10)}
    if query:
        params["q"] = query
    if country:
        params["country"] = ",".join(country)
    if category:
        params["category"] = ",".join(category)
    if language:
        params["language"] = ",".join(language)
    if page:
        params["page"] = page

    resp = requests.get(BASE_URL, params=params, timeout=15)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise NewsDataAPIError(f"newsdata.io returned {resp.status_code}: {detail}")

    return resp.json()


def fetch_latest_articles(
    query: Optional[str] = None,
    country: Optional[List[str]] = None,
    category: Optional[List[str]] = None,
    language: Optional[List[str]] = None,
    size: int = 10,
) -> List[NewsArticle]:
    """Convenience wrapper: fetch_latest() -> a list of NewsArticle (single page, no pagination)."""
    data = fetch_latest(query=query, country=country, category=category, language=language, size=size)
    articles: List[NewsArticle] = []
    for item in data.get("results", []):
        articles.append(
            NewsArticle(
                article_id=item["article_id"],
                title=item.get("title"),
                description=item.get("description"),
                content=item.get("content"),
                link=item.get("link"),
                pub_date=item.get("pubDate"),
                language=item.get("language"),
                country=item.get("country") or [],
                category=item.get("category") or [],
                source_name=item.get("source_name"),
                source_url=item.get("source_url"),
            )
        )
    return articles
