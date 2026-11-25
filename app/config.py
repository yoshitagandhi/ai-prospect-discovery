"""Application settings, loaded once from environment variables.

Where values come from (highest priority first):
1. Real environment variables (for example, set in Render's dashboard).
2. The local .env file in the project root (development only, never committed).
3. The defaults defined in Settings below.

API keys are stored as SecretStr, so they never appear in logs, reprs or error messages.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE: Path | None = PROJECT_ROOT / ".env"

REQUIRED_VARIABLES = ("TAVILY_API_KEY", "GEMINI_API_KEY")


class ConfigError(Exception):
    """Raised when a required setting is missing or a setting has an invalid value."""


class Settings(BaseModel):
    """All runtime settings. Frozen, so nothing can change them after start-up."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tavily_api_key: SecretStr
    gemini_api_key: SecretStr
    groq_api_key: SecretStr | None = None
    youtube_api_key: SecretStr | None = None

    llm_primary_model: str = Field(
        default="gemini-3.5-flash-lite", min_length=1, max_length=100
    )
    llm_fallback_model: str = Field(
        default="openai/gpt-oss-20b", min_length=1, max_length=100
    )

    http_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    llm_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    request_deadline_seconds: float = Field(default=45.0, gt=0, le=120)
    max_response_bytes: int = Field(default=1_000_000, ge=10_000, le=5_000_000)
    max_redirects: int = Field(default=3, ge=0, le=5)

    max_results_per_platform: int = Field(default=10, ge=1, le=20)
    max_pages_to_fetch: int = Field(default=10, ge=0, le=30)
    max_prospects: int = Field(default=25, ge=1, le=100)

    rate_limit_per_minute: int = Field(default=5, ge=1, le=60)
    global_rate_limit_per_minute: int = Field(default=30, ge=1, le=600)
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86_400)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    port: int = Field(default=4000, ge=1, le=65_535)


def _env_name(field_name: str) -> str:
    """Map a Settings field name to its environment variable name (tavily_api_key -> TAVILY_API_KEY)."""
    return field_name.upper()


def _collect_raw_values(env_file: Path | None) -> dict[str, str]:
    """Gather the non-blank value of every setting. Real environment variables win over the .env file.

    dotenv_values() only reads the file; it does not copy secrets into os.environ.
    """
    file_values: dict[str, str | None] = {}
    if env_file is not None and env_file.is_file():
        file_values = dotenv_values(env_file)

    raw_values: dict[str, str] = {}
    for field_name in Settings.model_fields:
        env_name = _env_name(field_name)
        value = os.environ.get(env_name)
        if value is None or not value.strip():
            value = file_values.get(env_name)
        if value is not None and value.strip():
            raw_values[field_name] = value.strip()
    return raw_values


def load_settings(env_file: Path | None = None) -> Settings:
    """Build and validate Settings. Raises ConfigError with a message that never contains values."""
    raw_values = _collect_raw_values(env_file)

    missing = [name for name in REQUIRED_VARIABLES if name.lower() not in raw_values]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )

    if "log_level" in raw_values:
        raw_values["log_level"] = raw_values["log_level"].upper()

    try:
        return Settings(**raw_values)
    except ValidationError as error:
        bad_names = sorted({_env_name(str(item["loc"][0])) for item in error.errors()})
        raise ConfigError(
            "Invalid value for setting(s): " + ", ".join(bad_names)
        ) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared Settings object, loading it on first use."""
    return load_settings(DEFAULT_ENV_FILE)
