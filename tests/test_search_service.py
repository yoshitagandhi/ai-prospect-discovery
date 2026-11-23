"""Tests for app/services/search_service.py. Search and YouTube calls are replaced."""

import asyncio

import pytest

from app.config import Settings
from app.platforms import web_search, youtube
from app.platforms.web_search import SearchError, SearchResult
from app.platforms.youtube import YouTubeChannel, YouTubeError
from app.schemas import ParsedQuery
from app.services import search_service
from app.services.search_service import build_search_text, discover

SETTINGS = Settings(
    tavily_api_key="tvly-test-not-a-real-key",
    gemini_api_key="gemini-test-not-a-real-key",
)

PLATFORM_RESULTS = {
    "linkedin.com": [
        SearchResult(
            title="Jane Doe - Chess Coach | LinkedIn",
            url="https://www.linkedin.com/in/jane-doe",
            snippet="Chess coach in Springfield",
        ),
        SearchResult(
            title="Jane Doe - Chess Coach | LinkedIn",
            url="https://uk.linkedin.com/in/Jane-Doe/",
            snippet="duplicate of the first profile",
        ),
        SearchResult(
            title="Chess tips | LinkedIn",
            url="https://www.linkedin.com/pulse/chess-tips",
            snippet="an article, not a profile",
        ),
    ],
    "instagram.com": [
        SearchResult(
            title="Instagram",
            url="https://www.instagram.com/someone/",
            snippet="generic title, no usable name",
        )
    ],
    "x.com": [
        SearchResult(
            title='Sam Lee on X: "Chess club tonight in Springfield" / X',
            url="https://x.com/samlee/status/123456",
            snippet="Chess club tonight in Springfield",
        )
    ],
}


def parsed(target_type="individual", **location):
    return ParsedQuery.model_validate(
        {
            "is_prospect_query": True,
            "interest": "Chess",
            "audience": "Chess players",
            "location": location or {"city": "Springfield", "country": "USA"},
            "target_type": target_type,
        }
    )


@pytest.fixture
def fake_sources(monkeypatch):
    calls = []

    async def fake_search_web(
        client, settings, query, include_domains=(), exclude_domains=()
    ):
        calls.append(
            {
                "query": query,
                "include": include_domains,
                "exclude": exclude_domains,
            }
        )
        if not include_domains:
            return [
                SearchResult(
                    title="Bright Chess Shop",
                    url="https://brightchess.example/contact",
                    snippet="Contact us",
                ),
                SearchResult(
                    title="Chess shop | Facebook",
                    url="https://www.facebook.com/brightchess",
                    snippet="should be filtered",
                ),
            ]
        return PLATFORM_RESULTS.get(include_domains[0], [])

    async def fake_find_channels(client, settings, query, country_code):
        calls.append({"youtube": query, "country_code": country_code})
        return [
            YouTubeChannel(
                channel_id="UC" + "a" * 22,
                title="Chess With Jane",
                description="Weekly chess lessons",
                country="US",
                link="https://www.youtube.com/@chesswithjane",
            )
        ]

    monkeypatch.setattr(web_search, "search_web", fake_search_web)
    monkeypatch.setattr(youtube, "find_channels", fake_find_channels)
    return calls


def run_discover(query):
    return asyncio.run(discover(None, SETTINGS, query))


def test_search_text_removes_repeated_words_and_prefers_city():
    query = parsed(city="Springfield", state="Illinois", country="USA")
    assert build_search_text(query) == "Chess players Springfield USA"


def test_search_text_uses_state_when_no_city():
    query = parsed(state="Illinois", country="USA")
    assert build_search_text(query) == "Chess players Illinois USA"


def test_candidates_collected_from_every_platform(fake_sources):
    outcome = run_discover(parsed())
    found = {(c.platform, c.name, c.profile_url) for c in outcome.candidates}
    assert found == {
        ("LinkedIn", "Jane Doe", "https://www.linkedin.com/in/jane-doe"),
        ("X", "Sam Lee", "https://x.com/samlee"),
        ("YouTube", "Chess With Jane", "https://www.youtube.com/@chesswithjane"),
    }
    youtube_candidate = next(c for c in outcome.candidates if c.platform == "YouTube")
    assert youtube_candidate.location_hint == "US"
    assert youtube_candidate.found_via == "youtube_api"
    assert outcome.warnings == []


def test_each_platform_searched_on_its_own_domains(fake_sources):
    run_discover(parsed())
    searched = {call["include"] for call in fake_sources if "include" in call}
    assert searched == {
        ("linkedin.com",),
        ("instagram.com",),
        ("facebook.com",),
        ("x.com", "twitter.com"),
        ("threads.com", "threads.net"),
    }
    assert {"youtube": "Chess players Springfield USA", "country_code": None} in (
        fake_sources
    )


def test_websites_only_searched_for_business_queries(fake_sources):
    individual = run_discover(parsed(target_type="individual"))
    assert individual.websites == []
    assert all(call.get("include") != () for call in fake_sources)

    business = run_discover(parsed(target_type="business"))
    assert [site.url for site in business.websites] == [
        "https://brightchess.example/contact"
    ]
    website_call = next(c for c in fake_sources if c.get("include") == ())
    assert "facebook.com" in website_call["exclude"]
    assert website_call["query"].endswith(" contact")


def test_failing_platforms_become_warnings(fake_sources, monkeypatch):
    original_search = web_search.search_web

    async def failing_for_instagram(client, settings, query, **kwargs):
        if kwargs.get("include_domains") == ("instagram.com",):
            raise SearchError("search API returned HTTP 429")
        return await original_search(client, settings, query, **kwargs)

    async def no_youtube(client, settings, query, country_code):
        raise YouTubeError("no API key configured")

    monkeypatch.setattr(web_search, "search_web", failing_for_instagram)
    monkeypatch.setattr(youtube, "find_channels", no_youtube)
    outcome = run_discover(parsed())
    assert outcome.warnings == [
        "Instagram unavailable: search API returned HTTP 429",
        "YouTube unavailable: no API key configured",
    ]
    assert {c.platform for c in outcome.candidates} == {"LinkedIn", "X"}


def test_unexpected_errors_are_not_hidden(fake_sources, monkeypatch):
    async def buggy(client, settings, query, country_code):
        raise AttributeError("our bug")

    monkeypatch.setattr(youtube, "find_channels", buggy)
    with pytest.raises(AttributeError):
        run_discover(parsed())


def test_search_service_uses_the_shared_platform_registry():
    assert search_service.SEARCH_PLATFORMS[0].NAME == "LinkedIn"
