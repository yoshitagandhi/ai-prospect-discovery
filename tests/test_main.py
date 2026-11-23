"""Tests for app/main.py using FastAPI's TestClient. Fake keys only, no network."""

import pytest
from fastapi.testclient import TestClient

from app import config
from app.config import ConfigError
from app.main import app
from app.services import prospect_service

ENDPOINT = "/prospect-discovery"


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
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


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
        return [], ["YouTube skipped: no API key"]

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
