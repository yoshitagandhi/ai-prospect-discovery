"""Query understanding: turn a free-text query into a validated ParsedQuery.

Steps: check the cache, ask the LLM for structured JSON, then run deterministic checks
that do not trust the model:
- a query the model marks as not being a prospect search is rejected (422 in the API);
- the extracted interest or related terms must share at least one word with the query,
  so an injected or invented interest is rejected.

Results are cached in memory (ParsedQuery only, never search results) for
cache_ttl_seconds, keyed on the lower-cased query with spaces collapsed.
"""

from cachetools import TTLCache

from app.config import get_settings
from app.schemas import ParsedQuery
from app.services import llm_service
from app.services.llm_service import LlmError, LlmUsage
from app.utils.normalizers import content_words

CACHE_SIZE = 256
SYSTEM_PROMPT = """You extract the search intent from a prospect discovery query.
Return JSON with these fields:
- is_prospect_query: true only if the query asks to find people or businesses (an
  audience, interest or role, optionally with a place). Greetings, general questions
  and unrelated requests are false.
- interest: the main topic, activity or industry, using the query's own words where
  possible (for example "Football" for "football lovers").
- audience: the kind of people or businesses wanted, in a few words.
- location: city, state and country only when the query states or clearly implies them.
  country_code is the ISO 3166-1 alpha-2 code of that country (for example "US").
- target_type: "individual" for people, "business" for companies or venues, "both"
  when either fits (for example owners of businesses).
- related_terms: up to 5 short synonyms or closely related search terms.
Never invent a place, name or detail the query does not contain. Use null for unknown
fields."""


class QueryNotProspectError(Exception):
    """The query does not describe a search for people or businesses."""


class QueryUnderstandingError(Exception):
    """The query could not be understood (LLM unavailable or output failed checks)."""


_cache: TTLCache | None = None


def _get_cache(ttl_seconds: int) -> TTLCache:
    """Create the cache on first use, so its lifetime follows the settings."""
    global _cache
    if _cache is None:
        _cache = TTLCache(maxsize=CACHE_SIZE, ttl=max(ttl_seconds, 1))
    return _cache


def reset_query_cache() -> None:
    """Forget every cached result. Used by tests."""
    global _cache
    _cache = None


def _cache_key(query: str) -> str:
    """'  Chess  PLAYERS ' and 'chess players' share one cache entry."""
    return " ".join(query.lower().split())


def interest_matches_query(parsed: ParsedQuery, query: str) -> bool:
    """True when the interest or a related term shares a word with the query."""
    extracted = parsed.interest or ""
    for term in parsed.related_terms:
        extracted += " " + term
    return bool(content_words(extracted) & content_words(query))


async def parse_query(query: str) -> tuple[ParsedQuery, LlmUsage | None]:
    """Return the parsed query and the LLM usage (None when served from the cache)."""
    settings = get_settings()
    cache = _get_cache(settings.cache_ttl_seconds)
    key = _cache_key(query)
    if settings.cache_ttl_seconds > 0 and key in cache:
        parsed = cache[key]
        usage = None
    else:
        try:
            parsed, usage = await llm_service.generate_json(
                settings, SYSTEM_PROMPT, query, ParsedQuery
            )
        except LlmError:
            raise QueryUnderstandingError("LLM unavailable") from None
        if settings.cache_ttl_seconds > 0:
            cache[key] = parsed

    if not parsed.is_prospect_query:
        raise QueryNotProspectError("not a prospect search")
    if not interest_matches_query(parsed, query):
        raise QueryUnderstandingError("interest does not match the query")
    return parsed, usage
