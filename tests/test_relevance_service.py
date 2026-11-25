"""Tests for app/services/relevance_service.py. The LLM call is replaced: no network."""

import asyncio
import json
import logging

import pytest

from app.config import Settings
from app.schemas import ParsedQuery
from app.services import llm_service
from app.services.llm_service import LlmError, LlmUsage
from app.services.relevance_service import (
    MAX_CANDIDATES,
    RelevanceJudgements,
    RelevanceUnavailableError,
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


def test_prompt_says_the_candidate_must_be_the_audience(monkeypatch):
    prompts = []

    async def generate_json(settings, system_prompt, untrusted_text, schema):
        prompts.append(system_prompt)
        judgements = RelevanceJudgements.model_validate({"judgements": [judgement(0)]})
        return judgements, USAGE

    monkeypatch.setattr(llm_service, "generate_json", generate_json)
    run(parsed(**SPRINGFIELD), [JANE])
    assert "sells to, advises, serves or markets to the audience" in prompts[0]


def test_prompt_counts_a_business_of_the_kind_for_business_queries(monkeypatch):
    prompts = []

    async def generate_json(settings, system_prompt, untrusted_text, schema):
        prompts.append(system_prompt)
        judgements = RelevanceJudgements.model_validate({"judgements": []})
        return judgements, USAGE

    monkeypatch.setattr(llm_service, "generate_json", generate_json)
    run(parsed(**SPRINGFIELD), [JANE])
    assert "a business of the requested kind" in prompts[0]
    assert "counts" in prompts[0]


def test_debug_log_explains_each_decision(monkeypatch, caplog):
    fake_llm(monkeypatch, [judgement(0, audience_match=False)])
    sam = candidate("Sam Lee", "Sam Lee is a chess coach in Springfield.")
    with caplog.at_level(logging.DEBUG, logger="prospect_discovery.relevance"):
        run(parsed(**SPRINGFIELD), [JANE, sam])
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2
    assert lines[0].startswith("Relevance LinkedIn 'Jane Doe' kept=False")
    assert "audience=False" in lines[0]
    assert "verified interest=True location=True" in lines[0]
    assert lines[1] == "Relevance LinkedIn 'Sam Lee' kept=False | no LLM judgement"
    assert all("Springfield, Illinois" not in line for line in lines)


def test_debug_log_off_at_default_level(monkeypatch, caplog):
    fake_llm(monkeypatch, [judgement(0)])
    with caplog.at_level(logging.INFO, logger="prospect_discovery.relevance"):
        run(parsed(**SPRINGFIELD), [JANE])
    assert caplog.records == []


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


def test_lists_media_and_brands_are_dropped(monkeypatch):
    listing = candidate("Top 25 Chess Coaches", "Chess coaches in Springfield.")
    fake_llm(
        monkeypatch,
        [
            judgement(
                0, interest_quote="Chess coaches", is_specific_person_or_business=False
            ),
            judgement(1),
        ],
    )
    result = run(parsed(**SPRINGFIELD), [listing, JANE])
    assert [item.candidate.name for item in result.kept] == ["Jane Doe"]
    assert result.kept[0].score == 1.0


def test_score_is_lower_when_the_query_names_no_place(monkeypatch):
    fake_llm(monkeypatch, [judgement(0, location_match=False, location_quote=None)])
    assert run(parsed(), [JANE]).kept[0].score == 0.7


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


@pytest.mark.parametrize(
    ("quote", "expected"),
    [("CHESS   coach", True), ("chess player", False), ("", False), (None, False)],
)
def test_quote_check_ignores_case_and_spacing(quote, expected):
    assert quote_is_verified(quote, JANE, {"chess"}) is expected


def test_role_and_generic_words_do_not_prove_the_interest(monkeypatch):
    owner = candidate("Sam Lee", "Sam Lee, business owner in Springfield.")
    fake_llm(monkeypatch, [judgement(0, interest_quote="business owner")])
    query = parsed(**SPRINGFIELD).model_copy(
        update={"related_terms": ["cafe owners", "food service"]}
    )
    assert run(query, [owner]).kept == []


def test_name_alone_does_not_prove_the_interest(monkeypatch):
    named = candidate("Chess Secrets Inc", "Consulting company in Springfield.")
    fake_llm(monkeypatch, [judgement(0, interest_quote="Chess Secrets")])
    assert run(parsed(**SPRINGFIELD), [named]).kept == []


def test_llm_unavailable_raises_instead_of_guessing(monkeypatch):
    fake_llm(monkeypatch, error=LlmError("down"))
    with pytest.raises(RelevanceUnavailableError):
        run(parsed(**SPRINGFIELD), [JANE])
