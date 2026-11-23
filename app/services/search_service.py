"""Discovery orchestration: one search per platform, all run at the same time.

For each search-based platform (LinkedIn, Instagram, Facebook, X, Threads) one Tavily
search restricted to that platform's domains is made, and only results that the
platform module recognises as profile pages become candidates. YouTube uses its official
API. For business-type queries one extra search finds business websites, which are later
read for public emails, phones and WhatsApp links.

A failing platform becomes a warning; the request still succeeds with the others.
Only expected failures (SearchError, YouTubeError) are turned into warnings: anything
else is a bug and is raised.
"""

import asyncio
from collections.abc import Awaitable
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from app.config import Settings
from app.platforms import SEARCH_PLATFORMS, web_search, youtube
from app.platforms.web_search import SearchError, SearchResult
from app.platforms.youtube import YouTubeError
from app.schemas import ParsedQuery
from app.utils.normalizers import clean_display_name
from app.utils.safe_http import NEVER_FETCH_DOMAINS

EXPECTED_FAILURES = (SearchError, YouTubeError)
WEBSITE_TARGET_TYPES = {"business", "both"}
DIRECTORY_DOMAINS = (
    "clutch.co",
    "glassdoor.com",
    "indeed.com",
    "justdial.com",
    "tripadvisor.com",
    "wikipedia.org",
    "yellowpages.com",
    "yelp.com",
    "zomato.com",
)
WEBSITE_EXCLUDED_DOMAINS = NEVER_FETCH_DOMAINS + DIRECTORY_DOMAINS


class Candidate(BaseModel):
    """A possible prospect, with the evidence text and where it came from."""

    platform: str
    name: str
    profile_url: str
    evidence: str
    source_url: str
    location_hint: str | None = None
    found_via: Literal["web_search", "youtube_api"]


class SearchOutcome(BaseModel):
    """Everything discovery found for one query."""

    candidates: list[Candidate] = []
    websites: list[SearchResult] = []
    warnings: list[str] = []


def build_search_text(parsed: ParsedQuery) -> str:
    """Audience, interest, city (or state) and country, with repeated words removed.

    'Football lovers' + 'Football' + 'Washington DC' + 'USA'
    -> 'Football lovers Washington DC USA'
    """
    location = parsed.location
    place = location.city or location.state
    parts = [parsed.audience, parsed.interest, place, location.country]
    words: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for word in (part or "").split():
            if word.lower() not in seen:
                seen.add(word.lower())
                words.append(word)
    return " ".join(words)


def _host_matches(url: str, domains: tuple[str, ...]) -> bool:
    """True when the URL's host is one of the domains or a subdomain of one."""
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


async def _search_platform(
    client: httpx.AsyncClient, settings: Settings, module, text: str
) -> list[Candidate]:
    """One domain-restricted search; keep only recognised, named profile pages."""
    results = await web_search.search_web(
        client, settings, text, include_domains=module.SEARCH_DOMAINS
    )
    candidates: list[Candidate] = []
    seen_links: set[str] = set()
    for result in results:
        link = module.profile_url(result.url)
        name = clean_display_name(result.title)
        if link is None or name is None or link in seen_links:
            continue
        seen_links.add(link)
        candidates.append(
            Candidate(
                platform=module.NAME,
                name=name,
                profile_url=link,
                evidence=result.snippet,
                source_url=result.url,
                found_via="web_search",
            )
        )
    return candidates


async def _search_youtube(
    client: httpx.AsyncClient, settings: Settings, parsed: ParsedQuery, text: str
) -> list[Candidate]:
    """Channels from the official YouTube API, with their description as evidence."""
    channels = await youtube.find_channels(
        client, settings, text, parsed.location.country_code
    )
    candidates: list[Candidate] = []
    for channel in channels:
        name = clean_display_name(channel.title)
        if name is None:
            continue
        candidates.append(
            Candidate(
                platform=youtube.NAME,
                name=name,
                profile_url=channel.link,
                evidence=channel.description,
                source_url=channel.link,
                location_hint=channel.country,
                found_via="youtube_api",
            )
        )
    return candidates


async def _search_websites(
    client: httpx.AsyncClient, settings: Settings, text: str
) -> list[SearchResult]:
    """Business websites (not social platforms or directories) for contact extraction."""
    results = await web_search.search_web(
        client, settings, text + " contact", exclude_domains=WEBSITE_EXCLUDED_DOMAINS
    )
    return [
        result
        for result in results
        if not _host_matches(result.url, WEBSITE_EXCLUDED_DOMAINS)
    ]


async def discover(
    client: httpx.AsyncClient, settings: Settings, parsed: ParsedQuery
) -> SearchOutcome:
    """Run every platform search at the same time and collect what succeeded."""
    text = build_search_text(parsed)
    jobs: dict[str, Awaitable] = {
        module.NAME: _search_platform(client, settings, module, text)
        for module in SEARCH_PLATFORMS
    }
    jobs[youtube.NAME] = _search_youtube(client, settings, parsed, text)
    if parsed.target_type in WEBSITE_TARGET_TYPES:
        jobs["Websites"] = _search_websites(client, settings, text)

    results = await asyncio.gather(*jobs.values(), return_exceptions=True)

    outcome = SearchOutcome()
    for name, result in zip(jobs, results, strict=True):
        if isinstance(result, EXPECTED_FAILURES):
            outcome.warnings.append(f"{name} unavailable: {result}")
        elif isinstance(result, BaseException):
            raise result
        elif name == "Websites":
            outcome.websites = result
        else:
            outcome.candidates.extend(result)
    return outcome
