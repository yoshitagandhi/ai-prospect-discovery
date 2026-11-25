"""Tavily web search client: the general web search used for discovery.

Google's own Custom Search JSON API is closed to new customers, so an authorised search
API is used instead. Search results are used within one request only and never cached,
to respect the search provider's terms. Search-result URLs are never returned as
prospects; only profile pages recognised by a platform module are.
"""

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_QUERY_CHARS = 400
MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 1_000


class SearchError(Exception):
    """The search API could not be used (network error, bad status, bad response)."""


class SearchResult(BaseModel):
    """One search hit: page title, URL and the text snippet the search engine showed."""

    title: str
    url: str
    snippet: str


class _TavilyItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    title: str = ""
    content: str = ""


class _TavilyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[_TavilyItem] = []


async def search_web(
    client: httpx.AsyncClient,
    settings: Settings,
    query: str,
    include_domains: tuple[str, ...] = (),
    exclude_domains: tuple[str, ...] = (),
) -> list[SearchResult]:
    """Run one basic Tavily search (1 credit) and return validated results."""
    body: dict = {
        "query": query[:MAX_QUERY_CHARS],
        "search_depth": "basic",
        "max_results": settings.max_results_per_platform,
        "include_answer": False,
        "include_raw_content": False,
    }
    if include_domains:
        body["include_domains"] = list(include_domains)
    if exclude_domains:
        body["exclude_domains"] = list(exclude_domains)

    try:
        response = await client.post(
            TAVILY_SEARCH_URL,
            json=body,
            headers={
                "Authorization": "Bearer " + settings.tavily_api_key.get_secret_value()
            },
            timeout=settings.http_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise SearchError("search timed out") from None
    except httpx.HTTPError:
        raise SearchError("search request failed") from None

    if response.status_code != 200:
        raise SearchError(f"search API returned HTTP {response.status_code}")
    try:
        parsed = _TavilyResponse.model_validate_json(response.content)
    except ValidationError:
        raise SearchError("search API returned an unexpected response") from None

    return [
        SearchResult(
            title=item.title[:MAX_TITLE_CHARS],
            url=item.url,
            snippet=item.content[:MAX_SNIPPET_CHARS],
        )
        for item in parsed.results
    ]
