"""Tests for app/services/prospect_service.py with every pipeline step replaced."""

import asyncio

import pytest

from app import config
from app.platforms.web_search import SearchResult
from app.schemas import Location, ParsedQuery, SocialLink
from app.services import (
    profile_extractor,
    prospect_service,
    query_service,
    relevance_service,
    search_service,
)
from app.services.profile_extractor import SiteContacts
from app.services.relevance_service import RelevanceResult, ScoredCandidate
from app.services.search_service import Candidate, SearchOutcome

PARSED = ParsedQuery.model_validate(
    {
        "is_prospect_query": True,
        "interest": "Chess",
        "audience": "Chess coaches",
        "location": {"city": "Springfield", "country": "USA", "country_code": "US"},
        "target_type": "individual",
    }
)
JANE = Candidate(
    platform="LinkedIn",
    name="Jane Doe",
    profile_url="https://www.linkedin.com/in/jane-doe",
    evidence="Chess coach in Springfield",
    source_url="https://www.linkedin.com/in/jane-doe",
    found_via="web_search",
)
SAM = Candidate(
    platform="X",
    name="Sam Lee",
    profile_url="https://x.com/samlee",
    evidence="Chess every weekend",
    source_url="https://x.com/samlee/status/1",
    found_via="web_search",
)


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-not-a-real-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-not-a-real-key")
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)
    config.get_settings.cache_clear()

    async def parse_query(query):
        return PARSED, None

    async def discover(client, settings, parsed):
        return SearchOutcome(candidates=[JANE, SAM])

    async def score_candidates(settings, parsed, candidates):
        return RelevanceResult(
            kept=[
                ScoredCandidate(
                    candidate=JANE,
                    score=1.0,
                    interest_verified=True,
                    location_verified=True,
                    audience_match=True,
                    specific=True,
                    quotes=["Chess coach", "Springfield"],
                ),
                ScoredCandidate(
                    candidate=SAM,
                    score=0.4,
                    interest_verified=True,
                    location_verified=False,
                    audience_match=False,
                    specific=False,
                    quotes=["Chess"],
                ),
            ]
        )

    monkeypatch.setattr(query_service, "parse_query", parse_query)
    monkeypatch.setattr(search_service, "discover", discover)
    monkeypatch.setattr(relevance_service, "score_candidates", score_candidates)
    yield
    config.get_settings.cache_clear()


def test_prospects_built_with_location_only_when_verified():
    result = asyncio.run(prospect_service.find_prospects("chess coaches"))
    jane, sam = result.prospects
    assert jane.name == "Jane Doe"
    assert jane.location == "Springfield, USA"
    assert sam.location is None
    assert jane.social_link[0].link == "https://www.linkedin.com/in/jane-doe"
    assert jane.evidence is None and jane.relevance_score is None
    assert "Relevance: 2 of 2 candidates kept" in result.warnings


def test_evidence_mode_adds_quotes_sources_and_score():
    result = asyncio.run(
        prospect_service.find_prospects("chess coaches", include_evidence=True)
    )
    jane = result.prospects[0]
    assert jane.evidence == ["Chess coach", "Springfield"]
    assert jane.source_urls == ["https://www.linkedin.com/in/jane-doe"]
    assert jane.relevance_score == 1.0


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (Location(city="Washington", state="DC", country="USA"), "Washington, DC, USA"),
        (Location(city="Dubai", country="UAE"), "Dubai, UAE"),
        (Location(state="Gujarat", country="India"), "Gujarat, India"),
        (Location(city="Singapore", country="Singapore"), "Singapore"),
        (Location(), None),
    ],
)
def test_location_text(location, expected):
    assert prospect_service.location_text(location) == expected


def test_kept_website_becomes_prospect_with_its_contact_links(monkeypatch):
    hit = SearchResult(
        title="Bright Bistro | Contact",
        url="https://brightbistro.example/contact",
        snippet="Chess cafe in Springfield",
    )
    site_candidate = profile_extractor.website_candidate(hit)
    contact_link = SocialLink(platform="Email", link="mailto:hi@brightbistro.example")
    requested = []

    async def discover(client, settings, parsed):
        return SearchOutcome(candidates=[], websites=[hit])

    async def score_candidates(settings, parsed, candidates):
        return RelevanceResult(
            kept=[
                ScoredCandidate(
                    candidate=candidates[0],
                    score=0.9,
                    interest_verified=True,
                    location_verified=True,
                    audience_match=True,
                    specific=False,
                    quotes=["Chess cafe", "Springfield"],
                )
            ]
        )

    async def extract_contacts(client, settings, candidates, country_code):
        requested.extend(candidates)
        return [
            SiteContacts(
                candidate=site_candidate,
                page_url="https://brightbistro.example/contact",
                links=[contact_link],
            )
        ], []

    monkeypatch.setattr(search_service, "discover", discover)
    monkeypatch.setattr(relevance_service, "score_candidates", score_candidates)
    monkeypatch.setattr(profile_extractor, "extract_contacts", extract_contacts)
    result = asyncio.run(
        prospect_service.find_prospects("chess cafes", include_evidence=True)
    )
    assert [c.name for c in requested] == ["Bright Bistro"]
    prospect = result.prospects[0]
    assert prospect.name == "Bright Bistro"
    assert prospect.social_link == [contact_link]
    assert prospect.source_urls == ["https://brightbistro.example/contact"]
    assert "Contacts: 1 websites with public contacts" in result.warnings
