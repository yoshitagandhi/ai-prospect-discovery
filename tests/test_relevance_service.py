"""Tests for app/services/relevance_service.py. The LLM call is replaced: no network."""

import asyncio
import json

import pytest

from app.config import Settings
from app.schemas import ParsedQuery
from app.services import llm_service
from app.services.llm_service import LlmError, LlmUsage
from app.services.relevance_service import (
    MAX_CANDIDATES,
    RelevanceJudgements,
    quote_is_verified,
    score_candidates,
)
from app.services.search_service import Candidate

SETTINGS = Settings(
    tavily_api_key="tvly-test-not-a-real-key",
    gemini_api_key="gemini-test-not-a-real-key",
)
USAGE = LlmUsage(
    provider="gemini", model="test", input_tokens=900, output_tokens=200, latency_ms=9
)


def parsed(**location):
    return ParsedQuery.model_validate(
        {
            "is_prospect_query": True,
            "interest": "Chess",
            "audience": "Chess coaches",
            "location": location,
            "target_type": "individual",
            "related_terms": ["chess club"],
        }
    )


SPRINGFIELD = {"city": "Springfield", "country": "USA", "country_code": "US"}


def candidate(name, evidence, platform="LinkedIn", location_hint=None):
    return Candidate(
        platform=platform,
        name=name,
        profile_url="https://www.linkedin.com/in/" + name.lower().replace(" ", "-"),
        evidence=evidence,
        source_url="https://www.linkedin.com/in/" + name.lower().replace(" ", "-"),
        location_hint=location_hint,
        found_via="web_search",
    )


def judgement(candidate_id, **changes):
    data = {
        "candidate_id": candidate_id,
        "interest_match": True,
        "interest_quote": "chess coach",
        "location_match": True,
        "location_quote": "Springfield",
        "audience_match": True,
        "is_specific_person_or_business": True,
    }
    data.update(changes)
    return data


def fake_llm(monkeypatch, judgements=None, error=None):
    payloads = []

    async def generate_json(settings, system_prompt, untrusted_text, schema):
        payloads.append(json.loads(untrusted_text))
        if error is not None:
            raise error
        return RelevanceJudgements.model_validate({"judgements": judgements}), USAGE

    monkeypatch.setattr(llm_service, "generate_json", generate_json)
    return payloads


def run(query, candidates):
    return asyncio.run(score_candidates(SETTINGS, query, candidates))


JANE = candidate("Jane Doe", "Jane Doe is a chess coach in Springfield, Illinois.")


def test_verified_candidate_kept_with_quotes(monkeypatch):
    fake_llm(monkeypatch, [judgement(0)])
    result = run(parsed(**SPRINGFIELD), [JANE])
    assert len(result.kept) == 1
    kept = result.kept[0]
    assert kept.interest_verified and kept.location_verified
    assert kept.quotes == ["chess coach", "Springfield"]
    assert kept.score == 1.0
    assert result.usage == USAGE


def test_invented_quote_is_rejected(monkeypatch):
    fake_llm(monkeypatch, [judgement(0, interest_quote="grandmaster chess champion")])
    assert run(parsed(**SPRINGFIELD), [JANE]).kept == []


def test_quote_that_proves_nothing_is_rejected(monkeypatch):
    fake_llm(monkeypatch, [judgement(0, interest_quote="Jane Doe is a")])
    assert run(parsed(**SPRINGFIELD), [JANE]).kept == []


def test_location_required_when_query_names_a_place(monkeypatch):
    no_place = candidate("Sam Lee", "Sam Lee, chess coach and chess club organiser.")
    fake_llm(monkeypatch, [judgement(0, location_match=False, location_quote=None)])
    assert run(parsed(**SPRINGFIELD), [no_place]).kept == []


def test_location_not_needed_when_query_has_no_place(monkeypatch):
    no_place = candidate("Sam Lee", "Sam Lee, chess coach and chess club organiser.")
    fake_llm(monkeypatch, [judgement(0, location_match=False, location_quote=None)])
    result = run(parsed(), [no_place])
    assert [item.candidate.name for item in result.kept] == ["Sam Lee"]


def test_youtube_country_verifies_location_for_country_queries(monkeypatch):
    channel = candidate(
        "Chess With Jane",
        "Weekly chess lessons",
        platform="YouTube",
        location_hint="US",
    )
    fake_llm(
        monkeypatch,
        [judgement(0, interest_quote="chess lessons", location_quote=None)],
    )
    result = run(parsed(country="USA", country_code="US"), [channel])
    assert result.kept[0].location_verified


def test_youtube_country_does_not_verify_a_city(monkeypatch):
    channel = candidate(
        "Chess With Jane",
        "Weekly chess lessons",
        platform="YouTube",
        location_hint="US",
    )
    fake_llm(
        monkeypatch,
        [judgement(0, interest_quote="chess lessons", location_quote=None)],
    )
    assert run(parsed(**SPRINGFIELD), [channel]).kept == []


def test_audience_mismatch_is_dropped(monkeypatch):
    fake_llm(monkeypatch, [judgement(0, audience_match=False)])
    assert run(parsed(**SPRINGFIELD), [JANE]).kept == []


def test_kept_candidates_ranked_by_score(monkeypatch):
    club = candidate("Chess Club News", "Chess club news from Springfield.")
    fake_llm(
        monkeypatch,
        [
            judgement(
                0, interest_quote="Chess club", is_specific_person_or_business=False
            ),
            judgement(1),
        ],
    )
    result = run(parsed(**SPRINGFIELD), [club, JANE])
    assert [item.candidate.name for item in result.kept] == [
        "Jane Doe",
        "Chess Club News",
    ]
    assert result.kept[1].score == 0.9


def test_candidate_without_judgement_is_dropped(monkeypatch):
    fake_llm(monkeypatch, [judgement(5)])
    assert run(parsed(**SPRINGFIELD), [JANE]).kept == []


def test_only_max_candidates_sent_to_llm(monkeypatch):
    payloads = fake_llm(monkeypatch, [])
    run(parsed(**SPRINGFIELD), [JANE] * (MAX_CANDIDATES + 5))
    assert len(payloads[0]["candidates"]) == MAX_CANDIDATES


def test_no_candidates_means_no_llm_call(monkeypatch):
    payloads = fake_llm(monkeypatch, [])
    assert run(parsed(**SPRINGFIELD), []).kept == []
    assert payloads == []


def test_keyword_fallback_when_llm_unavailable(monkeypatch):
    unrelated = candidate("Bob Ray", "Bob Ray bakes bread in Springfield.")
    fake_llm(monkeypatch, error=LlmError("down"))
    result = run(parsed(**SPRINGFIELD), [JANE, unrelated])
    assert [item.candidate.name for item in result.kept] == ["Jane Doe"]
    assert result.warnings == ["Relevance: LLM unavailable, keyword matching used"]
    assert result.usage is None


@pytest.mark.parametrize(
    ("quote", "expected"),
    [("CHESS   coach", True), ("chess player", False), ("", False), (None, False)],
)
def test_quote_check_ignores_case_and_spacing(quote, expected):
    assert quote_is_verified(quote, JANE, {"chess"}) is expected
