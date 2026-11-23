"""Tests for app/platforms/web_search.py. respx fakes the Tavily API: no network."""

import asyncio
import json

import httpx
import pytest
import respx

from app.config import Settings
from app.platforms.web_search import (
    MAX_QUERY_CHARS,
    MAX_SNIPPET_CHARS,
    TAVILY_SEARCH_URL,
    SearchError,
    search_web,
)

SETTINGS = Settings(
    tavily_api_key="tvly-test-not-a-real-key",
    gemini_api_key="gemini-test-not-a-real-key",
)


def run_search(query="chess players Springfield", **kwargs):
    async def run():
        async with httpx.AsyncClient() as client:
            return await search_web(client, SETTINGS, query, **kwargs)

    return asyncio.run(run())


def test_results_returned_and_request_built_correctly():
    payload = {
        "results": [
            {
                "title": "Jane Doe - Chess Coach | LinkedIn",
                "url": "https://www.linkedin.com/in/jane-doe",
                "content": "x" * (MAX_SNIPPET_CHARS + 50),
                "score": 0.9,
            }
        ]
    }
    with respx.mock() as router:
        route = router.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )
        results = run_search(
            "q" * (MAX_QUERY_CHARS + 10), include_domains=("linkedin.com",)
        )
    request = route.calls.last.request
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer tvly-test-not-a-real-key"
    assert body["include_domains"] == ["linkedin.com"]
    assert body["search_depth"] == "basic"
    assert len(body["query"]) == MAX_QUERY_CHARS
    assert "exclude_domains" not in body
    assert results[0].url == "https://www.linkedin.com/in/jane-doe"
    assert len(results[0].snippet) == MAX_SNIPPET_CHARS


def test_empty_results_are_fine():
    with respx.mock() as router:
        router.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        assert run_search() == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"detail": "Unauthorized"}),
        httpx.Response(429, json={"detail": "Too many requests"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"results": [{"title": "no url"}]}),
    ],
)
def test_bad_responses_raise_search_error(response):
    with respx.mock() as router:
        router.post(TAVILY_SEARCH_URL).mock(return_value=response)
        with pytest.raises(SearchError):
            run_search()


def test_timeout_raises_search_error():
    with respx.mock() as router:
        router.post(TAVILY_SEARCH_URL).mock(side_effect=httpx.ReadTimeout)
        with pytest.raises(SearchError, match="timed out"):
            run_search()
