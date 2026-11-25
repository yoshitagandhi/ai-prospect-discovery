"""Relevance scoring: which candidates really match the query, with evidence.

1. One batched LLM call judges every candidate (at most MAX_CANDIDATES, each evidence
   text cut to MAX_EVIDENCE_CHARS) and returns booleans plus exact quotes.
2. Deterministic checks never trust the model: a quote counts only if it really appears
   in that candidate's name or evidence AND shares a word with what it should prove
   (the interest and related terms, or the most specific place the query names: the
   city if there is one, else the state, else the country). For a country-level query
   only, a YouTube channel's own country equal to the query's country code also
   verifies the location (a country is too coarse to prove a city or state).
   The interest quote must come from the evidence text, not only from the name:
   a name match alone is never enough.
3. Keep rule: the interest must be verified, the candidate must be the kind of
   prospect wanted (audience match), it must be a specific person or business (not a
   list, directory, media outlet or large brand) and, when the query names a place,
   the location must be verified too. A name match alone is never enough.
4. Score: 0.4 interest + 0.3 location + 0.2 audience + 0.1 specific person or
   business. Because the keep rule already requires interest, audience and specific,
   kept candidates score 1.0 with a verified place and 0.7 when the query names none.

If no LLM provider can judge the candidates, RelevanceUnavailableError is raised and
the API answers 503. There is no keyword fallback: keyword matching let vendors, news
articles and directories through, and weak results are worse than no results.

Diagnostics: at DEBUG log level (LOG_LEVEL=DEBUG in .env, for local checks only) one
line per candidate shows what the LLM said, what our code verified and whether the
candidate was kept. Only the platform and public name are logged, never evidence text;
the name is logged with repr() so a crafted name cannot fake extra log lines.
"""

import json
import logging

from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import ParsedQuery
from app.services import llm_service
from app.services.llm_service import LlmError, LlmUsage
from app.services.query_service import BUSINESS_ROLE_WORDS
from app.services.search_service import Candidate
from app.utils.normalizers import content_words

logger = logging.getLogger("prospect_discovery.relevance")

MAX_CANDIDATES = 40
MAX_EVIDENCE_CHARS = 300
MAX_QUOTE_CHARS = 300
WEIGHTS = {"interest": 0.4, "location": 0.3, "audience": 0.2, "specific": 0.1}
NON_INTEREST_WORDS = BUSINESS_ROLE_WORDS | content_words(
    "fan fans lover lovers enthusiast enthusiasts supporter supporters community "
    "communities people person group groups service services management "
    "professional professionals member members team teams club clubs"
)

SYSTEM_PROMPT = """You judge whether search results describe a prospect that matches a query.
The data contains the query intent and a list of candidates, each with an id, platform,
name and evidence text. For every candidate return:
- candidate_id: the id given.
- interest_match: true only if the name or evidence shows the interest or a close synonym.
- interest_quote: the exact words from the name or evidence that show it, copied
  character for character; null if there are none.
- location_match and location_quote: true only if the text shows the candidate is
  based in or active in the requested place (city, state or country). A place that is
  only mentioned as news, an event or a topic does not count. If the query has no
  place, use false and null.
- audience_match: true only if the candidate is the kind of prospect wanted. When the
  target type is "individual", organisations, media outlets and brands are false.
  For fans, lovers or enthusiasts of something, a person who shows the interest
  personally counts: they post about it, play it, support a team or belong to a fan
  club. Teams, leagues, clubs, venues and media are not fans.
  For founders, owners and other roles, the candidate must hold that role.
  Communities, events, accelerators, investors and media for founders are not
  founders.
  When the target type is "business" or "both", a business of the requested kind
  counts (a restaurant counts for "restaurant owners", because its owner is reached
  through it). A business that sells to, advises, serves or markets to the audience
  does not count: a consultancy, supplier or app for restaurant owners is not a
  restaurant owner.
- is_specific_person_or_business: true for a specific person, local business or
  organisation; false for news agencies, media outlets, governing bodies, large
  brands, generic topic pages, directories or lists.
Judge only from the given text. Never guess or invent quotes."""


class CandidateJudgement(BaseModel):
    """The LLM's view of one candidate, before our own checks."""

    candidate_id: int = Field(ge=0)
    interest_match: bool
    interest_quote: str | None = Field(default=None, max_length=MAX_QUOTE_CHARS)
    location_match: bool
    location_quote: str | None = Field(default=None, max_length=MAX_QUOTE_CHARS)
    audience_match: bool
    is_specific_person_or_business: bool


class RelevanceJudgements(BaseModel):
    """The LLM reply: one judgement per candidate."""

    judgements: list[CandidateJudgement] = Field(max_length=MAX_CANDIDATES)


class ScoredCandidate(BaseModel):
    """A candidate with its verified evidence and score."""

    candidate: Candidate
    score: float
    interest_verified: bool
    location_verified: bool
    audience_match: bool
    specific: bool
    quotes: list[str] = []


class RelevanceUnavailableError(Exception):
    """No LLM provider could judge the candidates."""


class RelevanceResult(BaseModel):
    """Candidates that passed the keep rule (best first), warnings and the LLM usage."""

    kept: list[ScoredCandidate] = []
    warnings: list[str] = []
    usage: LlmUsage | None = None


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace, so quotes match regardless of spacing."""
    return " ".join(text.lower().split())


def _candidate_text(candidate: Candidate) -> str:
    """The exact text the LLM was shown for this candidate."""
    return f"{candidate.name} {candidate.evidence[:MAX_EVIDENCE_CHARS]}"


def interest_words(parsed: ParsedQuery) -> set[str]:
    """Topic words that can prove the interest: from the interest and related terms.

    Role and generic words (owner, fan, service, management...) are removed, because a
    quote such as "cafe owners" or "sports fans" does not prove the topic.
    """
    words = content_words(" ".join([parsed.interest or "", *parsed.related_terms]))
    return words - NON_INTEREST_WORDS


def location_words(parsed: ParsedQuery) -> set[str]:
    """Words that can prove the place: only the most specific place the query names.

    A city query needs the city, a state query the state, and only a country query
    accepts the country or its code. So "India" alone never proves "Ahmedabad".
    """
    location = parsed.location
    if location.city:
        parts = [location.city]
    elif location.state:
        parts = [location.state]
    else:
        parts = [location.country, location.country_code]
    return content_words(" ".join(part for part in parts if part))


def query_has_location(parsed: ParsedQuery) -> bool:
    """True when the query names a place."""
    location = parsed.location
    return any([location.city, location.state, location.country])


def quote_is_verified(
    quote: str | None,
    candidate: Candidate,
    target_words: set[str],
    evidence_only: bool = False,
) -> bool:
    """A quote counts only if it appears in the candidate's text and proves the target.

    With evidence_only=True the quote must appear in the evidence, not just the name.
    """
    if not quote or not quote.strip():
        return False
    if evidence_only:
        text = candidate.evidence[:MAX_EVIDENCE_CHARS]
    else:
        text = _candidate_text(candidate)
    if _normalise(quote) not in _normalise(text):
        return False
    return bool(content_words(quote) & target_words)


def _youtube_country_matches(candidate: Candidate, parsed: ParsedQuery) -> bool:
    """For country-level queries only: the channel's own country equals the query's."""
    if parsed.location.city or parsed.location.state:
        return False
    code = parsed.location.country_code
    return bool(code and candidate.location_hint and candidate.location_hint == code)


def _score(scored: ScoredCandidate) -> float:
    """Weighted sum of the four signals, between 0 and 1."""
    return round(
        WEIGHTS["interest"] * scored.interest_verified
        + WEIGHTS["location"] * scored.location_verified
        + WEIGHTS["audience"] * scored.audience_match
        + WEIGHTS["specific"] * scored.specific,
        4,
    )


def _keep(scored: ScoredCandidate, parsed: ParsedQuery) -> bool:
    """Interest verified, audience matched, specific, and location verified if needed."""
    if not (scored.interest_verified and scored.audience_match and scored.specific):
        return False
    return scored.location_verified or not query_has_location(parsed)


def _from_judgement(
    candidate: Candidate, judgement: CandidateJudgement, parsed: ParsedQuery
) -> ScoredCandidate:
    """Apply our own checks to the LLM's judgement of one candidate."""
    interest_ok = judgement.interest_match and quote_is_verified(
        judgement.interest_quote, candidate, interest_words(parsed), evidence_only=True
    )
    location_ok = (
        judgement.location_match
        and quote_is_verified(
            judgement.location_quote, candidate, location_words(parsed)
        )
    ) or _youtube_country_matches(candidate, parsed)
    quotes = [
        quote
        for quote, verified in (
            (judgement.interest_quote, interest_ok),
            (judgement.location_quote, location_ok),
        )
        if verified and quote
    ]
    scored = ScoredCandidate(
        candidate=candidate,
        score=0.0,
        interest_verified=interest_ok,
        location_verified=location_ok,
        audience_match=judgement.audience_match,
        specific=judgement.is_specific_person_or_business,
        quotes=quotes,
    )
    scored.score = _score(scored)
    logger.debug(
        "Relevance %s %r kept=%s | LLM interest=%s location=%s audience=%s"
        " | verified interest=%s location=%s",
        candidate.platform,
        candidate.name,
        _keep(scored, parsed),
        judgement.interest_match,
        judgement.location_match,
        judgement.audience_match,
        interest_ok,
        location_ok,
    )
    return scored


def _llm_payload(parsed: ParsedQuery, candidates: list[Candidate]) -> str:
    """The query intent and candidates as JSON data for the LLM."""
    location = parsed.location
    return json.dumps(
        {
            "query_intent": {
                "interest": parsed.interest,
                "audience": parsed.audience,
                "related_terms": parsed.related_terms,
                "target_type": parsed.target_type,
                "place": {
                    "city": location.city,
                    "state": location.state,
                    "country": location.country,
                },
            },
            "candidates": [
                {
                    "id": index,
                    "platform": candidate.platform,
                    "name": candidate.name,
                    "evidence": candidate.evidence[:MAX_EVIDENCE_CHARS],
                }
                for index, candidate in enumerate(candidates)
            ],
        },
        ensure_ascii=False,
    )


async def score_candidates(
    settings: Settings, parsed: ParsedQuery, candidates: list[Candidate]
) -> RelevanceResult:
    """Judge, verify, filter and rank candidates. No LLM call when there are none.

    Raises RelevanceUnavailableError when no LLM provider returns valid judgements.
    """
    candidates = candidates[:MAX_CANDIDATES]
    if not candidates:
        return RelevanceResult()

    try:
        reply, usage = await llm_service.generate_json(
            settings,
            SYSTEM_PROMPT,
            _llm_payload(parsed, candidates),
            RelevanceJudgements,
        )
    except LlmError as error:
        raise RelevanceUnavailableError("no LLM provider could judge") from error

    result = RelevanceResult(usage=usage)
    judgements: dict[int, CandidateJudgement] = {}
    for judgement in reply.judgements:
        judgements.setdefault(judgement.candidate_id, judgement)
    scored = []
    for index, candidate in enumerate(candidates):
        if index in judgements:
            scored.append(_from_judgement(candidate, judgements[index], parsed))
        else:
            logger.debug(
                "Relevance %s %r kept=False | no LLM judgement",
                candidate.platform,
                candidate.name,
            )

    if len(scored) < len(candidates):
        result.warnings.append(
            f"Relevance: LLM judged only {len(scored)} of {len(candidates)} candidates"
        )
    kept = [item for item in scored if _keep(item, parsed)]
    kept.sort(key=lambda item: item.score, reverse=True)
    result.kept = kept
    return result
