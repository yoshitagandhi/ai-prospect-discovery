"""Prospect discovery pipeline.

Steps: understand the query, discover candidate profiles on every platform, then keep
only candidates whose relevance is verified, best first. Contact extraction from
business websites and cross-platform deduplication come next.

In evidence mode each prospect also carries its verified quotes, source URLs and
relevance score, and the warnings list reports what each platform found.
"""

from collections import Counter

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.platforms.web_search import SearchResult
from app.schemas import Location, ParsedQuery, Prospect, SocialLink
from app.services import query_service, relevance_service, search_service
from app.services.llm_service import LlmUsage
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
    item: ScoredCandidate, parsed: ParsedQuery, include_evidence: bool
) -> Prospect:
    """Turn a kept candidate into a Prospect. The location is only set when verified."""
    candidate = item.candidate
    return Prospect(
        name=candidate.name,
        location=location_text(parsed.location) if item.location_verified else None,
        social_link=[
            SocialLink(platform=candidate.platform, link=candidate.profile_url)
        ],
        evidence=item.quotes if include_evidence else None,
        source_urls=[candidate.source_url] if include_evidence else None,
        relevance_score=item.score if include_evidence else None,
    )


async def find_prospects(query: str, include_evidence: bool = False) -> DiscoveryResult:
    """Run the pipeline for an already validated query.

    Raises query_service.QueryNotProspectError or QueryUnderstandingError, which the
    API turns into 422 and 503 responses.
    """
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

    relevance = await relevance_service.score_candidates(
        settings, parsed_query, outcome.candidates
    )
    result.warnings.extend(relevance.warnings)
    if relevance.usage is not None:
        result.llm_usage.append(relevance.usage)

    for item in relevance.kept[: settings.max_prospects]:
        try:
            result.prospects.append(
                build_prospect(item, parsed_query, include_evidence)
            )
        except ValidationError:
            result.warnings.append(
                f"{item.candidate.platform}: skipped a profile with an invalid link"
            )

    counts = Counter(candidate.platform for candidate in outcome.candidates)
    for platform, count in sorted(counts.items()):
        result.warnings.append(f"{platform}: {count} candidate profiles found")
    result.warnings.append(f"Websites: {len(outcome.websites)} business sites found")
    result.warnings.append(
        f"Relevance: {len(relevance.kept)} of {len(outcome.candidates)} candidates kept"
    )
    return result
