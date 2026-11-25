"""Tests for app/main.py using FastAPI's TestClient. Fake keys only, no network."""

import pytest
from fastapi.testclient import TestClient

from app import config
from app.config import ConfigError
from app.main import app
from app.schemas import Location, ParsedQuery
from app.services import prospect_service, query_service, search_service
from app.services.prospect_service import DeadlineExceededError, DiscoveryResult
from app.services.query_service import QueryNotProspectError, QueryUnderstandingError
from app.services.relevance_service import RelevanceUnavailableError
from app.services.search_service import SearchUnavailableError
from app.utils.rate_limit import reset_rate_limits

ENDPOINT = "/prospect-discovery"

FAKE_PARSED = ParsedQuery(
    is_prospect_query=True,
    interest="Chess",
    audience="Chess players",
    location=Location(city="Springfield", country="USA", country_code="US"),
    target_type="individual",
)


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    async def fake_parse_query(query):
        return FAKE_PARSED, None

    async def fake_discover(client, settings, parsed):
        return search_service.SearchOutcome()

    monkeypatch.setattr(query_service, "parse_query", fake_parse_query)
    monkeypatch.setattr(search_service, "discover", fake_discover)


@pytest.fixture
def fake_keys(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-not-a-real-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-not-a-real-key")
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def client(fake_keys):
    reset_rate_limits()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    reset_rate_limits()


def test_valid_query_returns_exact_spec_shape(client):
    response = client.post(ENDPOINT, json={"query": "  chess players in a small town "})
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "query": "chess players in a small town",
        "total_prospects": 0,
        "prospects": [],
    }


def test_invalid_query_returns_422_without_echoing_input(client):
    secret_looking = "zzsecretzz" * 40
    response = client.post(ENDPOINT, json={"query": secret_looking})
    assert response.status_code == 422
    assert response.json()["success"] is False
    assert "query" in response.json()["error"]
    assert "zzsecretzz" not in response.text


def test_missing_query_returns_422(client):
    response = client.post(ENDPOINT, json={})
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_extra_field_returns_422(client):
    response = client.post(ENDPOINT, json={"query": "chess players", "limit": 5})
    assert response.status_code == 422


def test_malformed_json_returns_422(client):
    response = client.post(
        ENDPOINT, content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_oversized_body_returns_413(client):
    response = client.post(ENDPOINT, json={"query": "a" * 20_000})
    assert response.status_code == 413
    assert response.json() == {"success": False, "error": "Request body too large"}


def test_body_without_content_length_returns_411(client):
    response = client.post(
        ENDPOINT,
        content=iter([b'{"query": "chess players"}']),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 411


def test_get_on_endpoint_returns_405(client):
    response = client.get(ENDPOINT)
    assert response.status_code == 405
    assert response.json() == {"success": False, "error": "Method not allowed"}


def test_unknown_path_returns_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["success"] is False


def test_docs_and_health_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}


def test_unexpected_error_returns_generic_500(client, monkeypatch):
    async def broken_find_prospects(query, include_evidence=False):
        raise RuntimeError("internal detail at C:/hidden/path")

    monkeypatch.setattr(prospect_service, "find_prospects", broken_find_prospects)
    response = client.post(ENDPOINT, json={"query": "chess players"})
    assert response.status_code == 500
    assert response.json() == {"success": False, "error": "Internal server error"}
    assert "hidden" not in response.text


def test_warnings_only_shown_with_include_evidence(client, monkeypatch):
    async def find_with_warning(query, include_evidence=False):
        return DiscoveryResult(warnings=["YouTube skipped: no API key"])

    monkeypatch.setattr(prospect_service, "find_prospects", find_with_warning)
    default = client.post(ENDPOINT, json={"query": "chess players"}).json()
    with_evidence = client.post(
        ENDPOINT + "?include_evidence=true", json={"query": "chess players"}
    ).json()
    assert "warnings" not in default
    assert with_evidence["warnings"] == ["YouTube skipped: no API key"]


def test_missing_required_key_stops_startup(fake_keys, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    config.get_settings.cache_clear()
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"), TestClient(app):
        pass


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_sixth_request_in_a_minute_is_rate_limited(client):
    for _ in range(5):
        assert client.post(ENDPOINT, json={"query": "chess players"}).status_code == 200
    response = client.post(ENDPOINT, json={"query": "chess players"})
    assert response.status_code == 429
    assert response.json() == {
        "success": False,
        "error": "Too many requests, please try again later",
    }
    assert int(response.headers["retry-after"]) > 0


def test_health_is_never_rate_limited(client):
    for _ in range(20):
        assert client.get("/health").status_code == 200


def test_parsed_query_only_shown_with_include_evidence(client):
    default = client.post(ENDPOINT, json={"query": "chess players"}).json()
    with_evidence = client.post(
        ENDPOINT + "?include_evidence=true", json={"query": "chess players"}
    ).json()
    assert "parsed_query" not in default
    assert with_evidence["parsed_query"]["interest"] == "Chess"
    assert with_evidence["parsed_query"]["location"] == {
        "city": "Springfield",
        "country": "USA",
        "country_code": "US",
    }


def test_non_prospect_query_returns_422(client, monkeypatch):
    async def not_prospect(query):
        raise QueryNotProspectError("not a prospect search")

    monkeypatch.setattr(query_service, "parse_query", not_prospect)
    response = client.post(ENDPOINT, json={"query": "hello how are you"})
    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "Query does not describe a prospect search",
    }


def test_llm_unavailable_returns_503(client, monkeypatch):
    async def unavailable(query):
        raise QueryUnderstandingError("LLM unavailable")

    monkeypatch.setattr(query_service, "parse_query", unavailable)
    response = client.post(ENDPOINT, json={"query": "chess players"})
    assert response.status_code == 503
    assert response.json()["success"] is False


def test_json_responses_declare_utf8(client):
    ok = client.post(ENDPOINT, json={"query": "chess players"})
    error = client.post(ENDPOINT, json={})
    assert ok.headers["content-type"] == "application/json; charset=utf-8"
    assert error.headers["content-type"] == "application/json; charset=utf-8"


def test_relevance_unavailable_returns_503_without_results(client, monkeypatch):
    async def unavailable(query, include_evidence=False):
        raise RelevanceUnavailableError("LLM unavailable")

    monkeypatch.setattr(prospect_service, "find_prospects", unavailable)
    response = client.post(ENDPOINT, json={"query": "chess players"})
    assert response.status_code == 503
    assert response.json() == {
        "success": False,
        "error": "Prospect checking is temporarily unavailable, please try again later",
    }


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            SearchUnavailableError(
                "LinkedIn unavailable: search API returned HTTP 432"
            ),
            "Search is temporarily unavailable, please try again later",
        ),
        (
            DeadlineExceededError("request deadline reached"),
            "The search took too long, please try again later",
        ),
    ],
)
def test_outages_return_503_not_an_empty_success(client, monkeypatch, error, message):
    async def failing(query, include_evidence=False):
        raise error

    monkeypatch.setattr(prospect_service, "find_prospects", failing)
    response = client.post(ENDPOINT, json={"query": "chess players"})
    assert response.status_code == 503
    assert response.json() == {"success": False, "error": message}
    assert "HTTP 432" not in response.text


def test_daily_limit_protects_the_free_search_credits(fake_keys, monkeypatch):
    monkeypatch.setenv("DAILY_REQUEST_LIMIT", "2")
    config.get_settings.cache_clear()
    reset_rate_limits()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        responses = [
            test_client.post(ENDPOINT, json={"query": "chess players"})
            for _ in range(3)
        ]
    reset_rate_limits()
    assert [r.status_code for r in responses] == [200, 200, 429]
    assert int(responses[2].headers["retry-after"]) > 3600
