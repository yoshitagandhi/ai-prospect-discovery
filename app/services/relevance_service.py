"""Relevance scoring: which candidates really match the query, with evidence.

1. One batched LLM call judges every candidate (at most MAX_CANDIDATES, each evidence
   text cut to MAX_EVIDENCE_CHARS) and returns booleans plus exact quotes.
2. Deterministic checks never trust the model: a quote counts only if it really appears
   in that candidate's name or evidence AND shares a word with what it should prove
   (the interest and related terms, or the requested place). For YouTube, a channel
   country equal to the query's country code also verifies the location.
3. Keep rule: the interest must be verified and, when the query names a place, the
   location must be verified too. A name match alone is never enough.
4. Score (for ranking): 0.4 interest + 0.3 location + 0.2 audience + 0.1 specific
   person or business.

If the LLM is unavailable, plain keyword matching is used instead and a warning says so.
"""

import json

from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import ParsedQuery
from app.services import llm_service
from app.services.llm_service import LlmError, LlmUsage
from app.services.search_service import Candidate
from app.utils.normalizers import content_words

MAX_CANDIDATES = 40
MAX_EVIDENCE_CHARS = 400
MAX_QUOTE_CHARS = 300
WEIGHTS = {"interest": 0.4, "location": 0.3, "audience": 0.2, "specific": 0.1}

SYSTEM_PROMPT = """You judge whether search results describe a prospect that matches a query.
The data contains the query intent and a list of candidates, each with an id, platform,
name and evidence text. For every candidate return:
- candidate_id: the id given.
- interest_match: true only if the name or evidence shows the interest or a close synonym.
- interest_quote: the exact words from the name or evidence that show it, copied
  character for character; null if there are none.
- location_match and location_quote: the same for the requested place (city, state or
  country). If the query has no place, use false and null.
- audience_match: true if the candidate is the kind of person or business wanted.
- is_specific_person_or_business: true for a specific person, business or organisation;
  false for news articles, generic topic pages, directories or lists.
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


class RelevanceResult(BaseModel):
    """Candidates that passed the keep rule (best first), plus warnings and LLM usage."""

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
    """Words that can prove the interest: the interest itself and related terms."""
    return content_words(" ".join([parsed.interest or "", *parsed.related_terms]))


def location_words(parsed: ParsedQuery) -> set[str]:
    """Words that can prove the place: city, state, country and country code."""
    location = parsed.location
    parts = [location.city, location.state, location.country, location.country_code]
    return content_words(" ".join(part for part in parts if part))


def query_has_location(parsed: ParsedQuery) -> bool:
    """True when the query names a place."""
    location = parsed.location
    return any([location.city, location.state, location.country])


def quote_is_verified(
    quote: str | None, candidate: Candidate, target_words: set[str]
) -> bool:
    """A quote counts only if it appears in the candidate's text and proves the target."""
    if not quote or not quote.strip():
        return False
    if _normalise(quote) not in _normalise(_candidate_text(candidate)):
        return False
    return bool(content_words(quote) & target_words)


def _youtube_country_matches(candidate: Candidate, parsed: ParsedQuery) -> bool:
    """A YouTube channel's own country equal to the query's country code."""
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
    """Interest must be verified; location too when the query names a place."""
    if not scored.interest_verified:
        return False
    return scored.location_verified or not query_has_location(parsed)


def _from_judgement(
    candidate: Candidate, judgement: CandidateJudgement, parsed: ParsedQuery
) -> ScoredCandidate:
    """Apply our own checks to the LLM's judgement of one candidate."""
    interest_ok = judgement.interest_match and quote_is_verified(
        judgement.interest_quote, candidate, interest_words(parsed)
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
    return scored


def _from_keywords(candidate: Candidate, parsed: ParsedQuery) -> ScoredCandidate:
    """Fallback without an LLM: shared words between the candidate and the query."""
    words = content_words(_candidate_text(candidate))
    scored = ScoredCandidate(
        candidate=candidate,
        score=0.0,
        interest_verified=bool(words & interest_words(parsed)),
        location_verified=bool(words & location_words(parsed))
        or _youtube_country_matches(candidate, parsed),
        audience_match=bool(words & content_words(parsed.audience or "")),
        specific=False,
    )
    scored.score = _score(scored)
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
    """Judge, verify, filter and rank candidates. No LLM call when there are none."""
    candidates = candidates[:MAX_CANDIDATES]
    if not candidates:
        return RelevanceResult()

    result = RelevanceResult()
    try:
        reply, usage = await llm_service.generate_json(
            settings,
            SYSTEM_PROMPT,
            _llm_payload(parsed, candidates),
            RelevanceJudgements,
        )
        result.usage = usage
        judgements: dict[int, CandidateJudgement] = {}
        for judgement in reply.judgements:
            judgements.setdefault(judgement.candidate_id, judgement)
        scored = [
            _from_judgement(candidate, judgements[index], parsed)
            for index, candidate in enumerate(candidates)
            if index in judgements
        ]
    except LlmError:
        result.warnings.append("Relevance: LLM unavailable, keyword matching used")
        scored = [_from_keywords(candidate, parsed) for candidate in candidates]

    kept = [item for item in scored if _keep(item, parsed)]
    kept.sort(key=lambda item: item.score, reverse=True)
    result.kept = kept
    return result
