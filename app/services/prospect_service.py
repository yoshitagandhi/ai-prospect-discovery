"""Prospect discovery pipeline.

Current steps: understand the query. Discovery, extraction, relevance scoring and
deduplication are added next; until then no prospects are returned (never padded).
"""

from pydantic import BaseModel, Field

from app.schemas import ParsedQuery, Prospect
from app.services import query_service
from app.services.llm_service import LlmUsage


class DiscoveryResult(BaseModel):
    """Everything the API needs to build its response."""

    prospects: list[Prospect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parsed_query: ParsedQuery | None = None
    llm_usage: list[LlmUsage] = Field(default_factory=list)


async def find_prospects(query: str, include_evidence: bool = False) -> DiscoveryResult:
    """Run the pipeline for an already validated query.

    Raises query_service.QueryNotProspectError or QueryUnderstandingError, which the
    API turns into 422 and 503 responses.
    """
    parsed_query, usage = await query_service.parse_query(query)
    result = DiscoveryResult(parsed_query=parsed_query)
    if usage is not None:
        result.llm_usage.append(usage)
    return result
