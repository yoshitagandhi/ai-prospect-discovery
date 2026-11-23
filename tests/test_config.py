"""Tests for app/config.py. They use fake keys only: no real .env file and no network."""

import pytest
from pydantic import ValidationError

from app import config
from app.config import ConfigError, load_settings

FAKE_TAVILY = "tvly-test-not-a-real-key"
FAKE_GEMINI = "gemini-test-not-a-real-key"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove every setting from the environment so tests never see real keys."""
    for field_name in config.Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def set_required_keys(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", FAKE_TAVILY)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI)


def test_defaults_applied_when_only_required_keys_set(monkeypatch):
    set_required_keys(monkeypatch)
    settings = load_settings(env_file=None)
    assert settings.port == 4000
    assert settings.http_timeout_seconds == 10.0
    assert settings.global_rate_limit_per_minute == 30
    assert settings.llm_timeout_seconds == 20.0
    assert settings.groq_api_key is None
    assert settings.youtube_api_key is None


def test_missing_required_key_names_the_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI)
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        load_settings(env_file=None)


def test_blank_value_counts_as_missing(monkeypatch):
    set_required_keys(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "   ")
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        load_settings(env_file=None)


def test_secrets_hidden_in_repr_and_str(monkeypatch):
    set_required_keys(monkeypatch)
    settings = load_settings(env_file=None)
    assert FAKE_TAVILY not in repr(settings)
    assert FAKE_TAVILY not in str(settings)
    assert settings.tavily_api_key.get_secret_value() == FAKE_TAVILY


def test_invalid_number_rejected_without_echoing_value(monkeypatch):
    set_required_keys(monkeypatch)
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "-999")
    with pytest.raises(ConfigError) as error_info:
        load_settings(env_file=None)
    message = str(error_info.value)
    assert "HTTP_TIMEOUT_SECONDS" in message
    assert "-999" not in message


def test_environment_variable_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TAVILY_API_KEY=from-file\nGEMINI_API_KEY=from-file\n", encoding="utf-8"
    )
    monkeypatch.setenv("TAVILY_API_KEY", "from-environment")
    settings = load_settings(env_file=env_file)
    assert settings.tavily_api_key.get_secret_value() == "from-environment"
    assert settings.gemini_api_key.get_secret_value() == "from-file"


def test_lowercase_log_level_accepted(monkeypatch):
    set_required_keys(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert load_settings(env_file=None).log_level == "DEBUG"


def test_settings_are_frozen(monkeypatch):
    set_required_keys(monkeypatch)
    settings = load_settings(env_file=None)
    with pytest.raises(ValidationError):
        settings.port = 1


def test_get_settings_returns_same_object(monkeypatch):
    set_required_keys(monkeypatch)
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)
    assert config.get_settings() is config.get_settings()
