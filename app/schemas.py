"""Request, response and query-understanding models for the prospect discovery API.

Every link that reaches the response passes a per-platform check, so the API can never
return a malformed link, an unexpected URL scheme or a look-alike domain.

Google is deliberately not an output platform: it is used for discovery only, and the
spec asks for profiles, not generic search-result URLs.
"""

import re
import unicodedata
from typing import Annotated, Literal
from urllib.parse import urlsplit

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

PlatformName = Literal[
    "LinkedIn",
    "X",
    "YouTube",
    "Instagram",
    "WhatsApp",
    "Facebook",
    "Email",
    "Threads",
    "Phone",
]

TargetType = Literal["individual", "business", "both"]

PLATFORM_DOMAINS: dict[str, tuple[str, ...]] = {
    "LinkedIn": ("linkedin.com",),
    "X": ("x.com", "twitter.com"),
    "YouTube": ("youtube.com",),
    "Instagram": ("instagram.com",),
    "Facebook": ("facebook.com",),
    "Threads": ("threads.net", "threads.com"),
    "WhatsApp": ("wa.me", "api.whatsapp.com"),
}

PHONE_LINK_PATTERN = re.compile(r"tel:\+[1-9][0-9]{6,14}")


def _reject_control_characters(value: str) -> str:
    """Reject invisible control characters such as NUL or newlines inside text fields."""
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("text must not contain control characters")
    return value


def _blank_to_none(value: object) -> object:
    """Treat empty or whitespace-only strings as 'not provided'."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=300),
    AfterValidator(_reject_control_characters),
]
NameText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=150),
    AfterValidator(_reject_control_characters),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    AfterValidator(_reject_control_characters),
]
TermText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=40),
    AfterValidator(_reject_control_characters),
]
LinkText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    AfterValidator(_reject_control_characters),
]


def _check_email_link(link: str) -> None:
    """Email links must be mailto: followed by a syntactically valid address (no network check)."""
    if not link.startswith("mailto:"):
        raise ValueError("Email link must start with mailto:")
    address = link.removeprefix("mailto:")
    try:
        validate_email(address, check_deliverability=False)
    except EmailNotValidError:
        raise ValueError("Email link contains an invalid address") from None


def _check_phone_link(link: str) -> None:
    """Phone links must be tel:+ followed by 7 to 15 digits (E.164), first digit not zero."""
    if PHONE_LINK_PATTERN.fullmatch(link) is None:
        raise ValueError("Phone link must look like tel:+ followed by 7 to 15 digits")


def _check_web_link(platform: str, link: str) -> None:
    """Web links must be https, on the platform's own domain, with no credentials or custom port."""
    parts = urlsplit(link)
    if parts.scheme != "https":
        raise ValueError(f"{platform} link must use https")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{platform} link must not contain credentials")
    if parts.port not in (None, 443):
        raise ValueError(f"{platform} link must not use a custom port")
    host = (parts.hostname or "").lower()
    allowed_domains = PLATFORM_DOMAINS[platform]
    if not any(
        host == domain or host.endswith("." + domain) for domain in allowed_domains
    ):
        raise ValueError(f"{platform} link is not on an allowed {platform} domain")


class ProspectRequest(BaseModel):
    """Body of POST /prospect-discovery."""

    model_config = ConfigDict(extra="forbid")

    query: QueryText

    @field_validator("query")
    @classmethod
    def query_must_contain_letters(cls, value: str) -> str:
        if sum(char.isalpha() for char in value) < 2:
            raise ValueError("query must contain at least 2 letters")
        return value


class Location(BaseModel):
    """City, state, country and ISO country code from the query. Any part may be unknown."""

    model_config = ConfigDict(extra="ignore")

    city: ShortText | None = None
    state: ShortText | None = None
    country: ShortText | None = None
    country_code: str | None = None

    @field_validator("city", "state", "country", mode="before")
    @classmethod
    def blank_means_unknown(cls, value: object) -> object:
        return _blank_to_none(value)

    @field_validator("country_code", mode="before")
    @classmethod
    def unknown_country_code_means_none(cls, value: object) -> str | None:
        """Keep only 2-letter region codes that the phonenumbers library knows.

        "UK" is accepted as "GB" because models often write it. Anything unknown
        becomes None instead of failing the whole parsed query.
        """
        if not isinstance(value, str):
            return None
        code = value.strip().upper()
        if code == "UK":
            code = "GB"
        if code in phonenumbers.SUPPORTED_REGIONS:
            return code
        return None


class ParsedQuery(BaseModel):
    """Structured understanding of the query, as returned by the LLM and then validated here."""

    model_config = ConfigDict(extra="ignore")

    is_prospect_query: bool
    interest: ShortText | None = None
    audience: ShortText | None = None
    location: Location = Field(default_factory=Location)
    target_type: TargetType
    related_terms: list[TermText] = Field(default_factory=list, max_length=5)

    @field_validator("interest", "audience", mode="before")
    @classmethod
    def blank_means_missing(cls, value: object) -> object:
        return _blank_to_none(value)

    @model_validator(mode="after")
    def prospect_query_needs_interest_and_audience(self) -> "ParsedQuery":
        if self.is_prospect_query and (not self.interest or not self.audience):
            raise ValueError("a prospect query needs an interest and an audience")
        return self


class SocialLink(BaseModel):
    """One public profile or contact link, using the spec's platform names."""

    model_config = ConfigDict(extra="forbid")

    platform: PlatformName
    link: LinkText

    @model_validator(mode="after")
    def link_must_match_platform(self) -> "SocialLink":
        if self.platform == "Email":
            _check_email_link(self.link)
        elif self.platform == "Phone":
            _check_phone_link(self.link)
        else:
            _check_web_link(self.platform, self.link)
        return self


class Prospect(BaseModel):
    """One discovered person or business.

    evidence, source_urls and relevance_score are only filled in evidence mode.
    """

    model_config = ConfigDict(extra="forbid")

    name: NameText
    location: NameText | None = None
    social_link: list[SocialLink] = Field(min_length=1, max_length=20)
    evidence: list[str] | None = None
    source_urls: list[str] | None = None
    relevance_score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("location", mode="before")
    @classmethod
    def blank_location_means_unknown(cls, value: object) -> object:
        return _blank_to_none(value)

    @field_validator("social_link")
    @classmethod
    def remove_duplicate_links(cls, links: list[SocialLink]) -> list[SocialLink]:
        unique_links = []
        seen = set()
        for item in links:
            key = (item.platform, item.link)
            if key not in seen:
                seen.add(key)
                unique_links.append(item)
        return unique_links


class ProspectResponse(BaseModel):
    """Successful response. warnings and parsed_query only appear in evidence mode."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    query: str
    total_prospects: int = Field(ge=0)
    prospects: list[Prospect]
    warnings: list[str] | None = None
    parsed_query: ParsedQuery | None = None

    @model_validator(mode="after")
    def total_must_match_prospects(self) -> "ProspectResponse":
        if self.total_prospects != len(self.prospects):
            raise ValueError("total_prospects must equal the number of prospects")
        return self


class ErrorResponse(BaseModel):
    """Error body. Never contains stack traces, internal paths or the rejected input."""

    model_config = ConfigDict(extra="forbid")

    success: Literal[False] = False
    error: str = Field(min_length=1, max_length=300)
