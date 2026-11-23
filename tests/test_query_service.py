"""Tests for app/services/query_service.py. The LLM call is replaced: no network."""

import asyncio

import pytest

from app import config
from app.schemas import ParsedQuery
from app.services import llm_service, query_service
from app.services.llm_service import LlmError, LlmUsage
from app.services.query_service import (
    QueryNotProspectError,
    QueryUnderstandingError,
    parse_query,
)

USAGE = LlmUsage(
    provider="gemini", model="test", input_tokens=100, output_tokens=40, latency_ms=5
)


def parsed(**changes):
    data = {
        "is_prospect_query": True,
        "interest": "Chess",
        "audience": "Chess players",
        "location": {"city": "Springfield", "country": "USA", "country_code": "US"},
        "target_type": "individual",
        "related_terms": [],
    }
    data.update(changes)
    return ParsedQuery.model_validate(data)


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-not-a-real-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-not-a-real-key")
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)
    config.get_settings.cache_clear()
    query_service.reset_query_cache()
    yield
    query_service.reset_query_cache()
    config.get_settings.cache_clear()


def fake_llm(monkeypatch, result=None, error=None):
    calls = []

    async def generate_json(settings, system_prompt, untrusted_text, schema):
        calls.append(untrusted_text)
        if error is not None:
            raise error
        return result, USAGE

    monkeypatch.setattr(llm_service, "generate_json", generate_json)
    return calls


def test_query_parsed_and_usage_returned(monkeypatch):
    fake_llm(monkeypatch, result=parsed())
    result, usage = asyncio.run(parse_query("chess players in Springfield USA"))
    assert result.interest == "Chess"
    assert usage == USAGE


def test_repeated_query_served_from_cache(monkeypatch):
    calls = fake_llm(monkeypatch, result=parsed())
    asyncio.run(parse_query("chess players in Springfield USA"))
    result, usage = asyncio.run(parse_query("  CHESS players in   springfield usa "))
    assert len(calls) == 1
    assert usage is None
    assert result.interest == "Chess"


def test_non_prospect_query_rejected(monkeypatch):
    fake_llm(
        monkeypatch,
        result=parsed(is_prospect_query=False, interest=None, audience=None),
    )
    with pytest.raises(QueryNotProspectError):
        asyncio.run(parse_query("hello how are you"))


def test_llm_failure_becomes_understanding_error(monkeypatch):
    fake_llm(monkeypatch, error=LlmError("down"))
    with pytest.raises(QueryUnderstandingError):
        asyncio.run(parse_query("chess players in Springfield USA"))


def test_interest_not_in_query_rejected(monkeypatch):
    fake_llm(monkeypatch, result=parsed(interest="Crypto", audience="Investors"))
    with pytest.raises(QueryUnderstandingError):
        asyncio.run(parse_query("chess players in Springfield USA"))


def test_plural_and_related_terms_count_as_matches(monkeypatch):
    fake_llm(monkeypatch, result=parsed(interest="Restaurants", audience="Owners"))
    asyncio.run(parse_query("restaurant owners in Springfield"))
    query_service.reset_query_cache()
    fake_llm(
        monkeypatch,
        result=parsed(interest="Entrepreneurship", related_terms=["startup founders"]),
    )
    asyncio.run(parse_query("startup founders in Springfield"))


def test_connecting_words_alone_do_not_count_as_a_match(monkeypatch):
    fake_llm(monkeypatch, result=parsed(interest="Crypto in the", audience="Investors"))
    with pytest.raises(QueryUnderstandingError):
        asyncio.run(parse_query("chess players in the park"))
