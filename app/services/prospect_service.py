"""Prospect discovery pipeline.

Steps:
1. Understand the query (LLM + deterministic checks).
2. Discover candidate profiles on every platform, and business websites for business
   queries, all at the same time.
3. Score profiles and websites together; keep only those whose relevance is verified.
4. Read the kept business websites for public emails, phones, WhatsApp links and the
   social profiles they link to (each website becomes one prospect).
5. Build prospects, best first, and merge the ones that share a link
   (deduplication_service.py). The prospect cap is applied after merging.

In evidence mode each prospect also carries its verified quotes, source URLs and
relevance score, and the warnings list reports what each step found.
"""

import asyncio
from collections import Counter

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.platforms.web_search import SearchResult
from app.schemas import Location, ParsedQuery, Prospect, SocialLink
from app.services import (
    deduplication_service,
    profile_extractor,
    query_service,
    relevance_service,
    search_service,
)
from app.services.llm_service import LlmUsage
from app.services.profile_extractor import WEBSITE_PLATFORM, website_candidate
from app.services.relevance_service import ScoredCandidate
from app.services.search_service import Candidate


class DiscoveryResult(BaseModel):
    """Everything the API needs to build its response."""

    prospects: list[Prospect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parsed_query: ParsedQuery | None = None
    llm_usage: list[LlmUsage] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    websites: list[SearchResult] = Field(default_factory=list)


def location_text(location: Location) -> str | None:
    """'Washington, DC, USA' from the parsed city, state and country (missing parts skipped)."""
    parts: list[str] = []
    for part in (location.city, location.state, location.country):
        if part and part not in parts:
            parts.append(part)
    return ", ".join(parts) or None


def build_prospect(
    item: ScoredCandidate,
    parsed: ParsedQuery,
    include_evidence: bool,
    links: list[SocialLink] | None = None,
    source_url: str | None = None,
) -> Prospect:
    """Turn a kept candidate into a Prospect. The location is only set when verified.

    Profiles get their own profile link; websites pass in the links found on the site.
    """
    candidate = item.candidate
    if links is None:
        links = [SocialLink(platform=candidate.platform, link=candidate.profile_url)]
    return Prospect(
        name=candidate.name,
        location=location_text(parsed.location) if item.location_verified else None,
        social_link=links,
        evidence=item.quotes if include_evidence else None,
        source_urls=[source_url or candidate.source_url] if include_evidence else None,
        relevance_score=item.score if include_evidence else None,
    )


class DeadlineExceededError(Exception):
    """The whole pipeline took longer than request_deadline_seconds."""


async def find_prospects(query: str, include_evidence: bool = False) -> DiscoveryResult:
    """Run the pipeline for an already validated query, within the request deadline.

    Raises DeadlineExceededError when the deadline is reached, and lets the pipeline's
    own errors through (QueryNotProspectError, QueryUnderstandingError,
    SearchUnavailableError, RelevanceUnavailableError); the API turns them into
    422 and 503 responses.
    """
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.request_deadline_seconds):
            return await _run_pipeline(query, include_evidence)
    except TimeoutError as error:
        raise DeadlineExceededError("request deadline reached") from error


async def _run_pipeline(query: str, include_evidence: bool) -> DiscoveryResult:
    """Parse, discover, check relevance, read websites, merge and cap."""
    settings = get_settings()
    parsed_query, usage = await query_service.parse_query(query)
    result = DiscoveryResult(parsed_query=parsed_query)
    if usage is not None:
        result.llm_usage.append(usage)

    async with httpx.AsyncClient() as client:
        outcome = await search_service.discover(client, settings, parsed_query)
        result.candidates = outcome.candidates
        result.websites = outcome.websites
        result.warnings.extend(outcome.warnings)

        site_candidates = [
            candidate
            for candidate in map(website_candidate, outcome.websites)
            if candidate is not None
        ]
        relevance = await relevance_service.score_candidates(
            settings, parsed_query, outcome.candidates + site_candidates
        )
        result.warnings.extend(relevance.warnings)
        if relevance.usage is not None:
            result.llm_usage.append(relevance.usage)

        kept_sites = [
            item
            for item in relevance.kept
            if item.candidate.platform == WEBSITE_PLATFORM
        ]
        sites, site_warnings = await profile_extractor.extract_contacts(
            client,
            settings,
            [item.candidate for item in kept_sites],
            parsed_query.location.country_code,
        )
        result.warnings.extend(site_warnings)

    contacts_by_url = {site.candidate.profile_url: site for site in sites}
    built: list[Prospect] = []
    for item in relevance.kept:
        try:
            if item.candidate.platform == WEBSITE_PLATFORM:
                site = contacts_by_url.get(item.candidate.profile_url)
                if site is None:
                    continue
                prospect = build_prospect(
                    item, parsed_query, include_evidence, site.links, site.page_url
                )
            else:
                prospect = build_prospect(item, parsed_query, include_evidence)
        except ValidationError:
            result.warnings.append(
                f"{item.candidate.platform}: skipped a profile with an invalid link"
            )
            continue
        built.append(prospect)
    merged, merged_count = deduplication_service.merge_duplicates(built)
    result.prospects = merged[: settings.max_prospects]

    counts = Counter(candidate.platform for candidate in outcome.candidates)
    for platform, count in sorted(counts.items()):
        result.warnings.append(f"{platform}: {count} candidate profiles found")
    result.warnings.append(
        f"Websites: {len(outcome.websites)} business sites found, "
        f"{len(kept_sites)} kept by relevance"
    )
    result.warnings.append(
        f"Relevance: {len(relevance.kept)} of "
        f"{len(outcome.candidates) + len(site_candidates)} candidates kept"
    )
    result.warnings.append(f"Contacts: {len(sites)} websites with public contacts")
    result.warnings.append(f"Duplicates: {merged_count} merged")
    return result
