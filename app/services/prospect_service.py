"""Prospect discovery pipeline.

Current steps: understand the query, then discover candidate profiles on every platform.
Relevance scoring, contact extraction and deduplication come next; until they exist no
prospects are returned (results are never padded). In evidence mode the warnings list
reports how many candidates each platform found.
"""

from collections import Counter

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.platforms.web_search import SearchResult
from app.schemas import ParsedQuery, Prospect
from app.services import query_service, search_service
from app.services.llm_service import LlmUsage
from app.services.search_service import Candidate


class DiscoveryResult(BaseModel):
    """Everything the API needs to build its response."""

    prospects: list[Prospect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parsed_query: ParsedQuery | None = None
    llm_usage: list[LlmUsage] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    websites: list[SearchResult] = Field(default_factory=list)


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

    counts = Counter(candidate.platform for candidate in outcome.candidates)
    for platform, count in sorted(counts.items()):
        result.warnings.append(f"{platform}: {count} candidate profiles found")
    result.warnings.append(f"Websites: {len(outcome.websites)} business sites found")
    return result
