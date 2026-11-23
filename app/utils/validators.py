"""Find and validate public contact details in page text. Nothing is ever guessed.

- Emails: the text is first split into words, and only words containing "@" are checked
  against a pattern with fixed maximum lengths and no nested repeats, so a long or
  hostile page cannot make it hang. Each match is then checked with email-validator.
  Image file names, placeholder domains and no-reply addresses are dropped.
- Phones: the phonenumbers library's own matcher in strict "valid number" mode, so no
  hand-written phone pattern is used. Without a country code only numbers written
  with a leading "+" can be read.
- WhatsApp: only wa.me/<digits> or api.whatsapp.com/send?phone=<digits> links,
  rewritten to the single form https://wa.me/<digits>.
"""

import re
from urllib.parse import parse_qs, urlsplit

import phonenumbers
from email_validator import EmailNotValidError, validate_email

WORD_SEPARATORS = re.compile(r"[\s<>()\[\]{}\"',;:|/?&=!*]+")
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}"
)
FILE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
)
PLACEHOLDER_DOMAINS = (
    "example.com",
    "example.org",
    "example.net",
    "domain.com",
    "email.com",
    "yourdomain.com",
    "yoursite.com",
    "sentry.io",
    "wixpress.com",
)
NO_REPLY_NAMES = {"noreply", "no-reply", "donotreply", "do-not-reply"}


def _is_unwanted_email(email: str) -> bool:
    """True for image names (logo@2x.png), placeholders and no-reply addresses."""
    local_part, _, domain = email.partition("@")
    if domain.endswith(FILE_EXTENSIONS):
        return True
    if any(
        domain == item or domain.endswith("." + item) for item in PLACEHOLDER_DOMAINS
    ):
        return True
    return local_part in NO_REPLY_NAMES


def extract_emails(text: str) -> list[str]:
    """Return unique, lower-cased, valid public emails in order of appearance."""
    emails: list[str] = []
    for word in WORD_SEPARATORS.split(text):
        if "@" not in word:
            continue
        candidate = word.strip(".")
        if EMAIL_PATTERN.fullmatch(candidate) is None:
            continue
        try:
            validate_email(candidate, check_deliverability=False)
        except EmailNotValidError:
            continue
        email = candidate.lower()
        if not _is_unwanted_email(email) and email not in emails:
            emails.append(email)
    return emails


def _region_or_none(country_code: str | None) -> str | None:
    """Use the country code only if the phonenumbers library knows it."""
    if country_code in phonenumbers.SUPPORTED_REGIONS:
        return country_code
    return None


def parse_phone(value: str, country_code: str | None) -> str | None:
    """Parse one phone number (for example from a tel: link) into +E.164, or None."""
    text = value.strip().removeprefix("tel:")
    try:
        number = phonenumbers.parse(text, _region_or_none(country_code))
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def extract_phones(text: str, country_code: str | None) -> list[str]:
    """Return unique valid phone numbers found in text, in +E.164 format."""
    phones: list[str] = []
    matcher = phonenumbers.PhoneNumberMatcher(
        text, _region_or_none(country_code), leniency=phonenumbers.Leniency.VALID
    )
    for match in matcher:
        phone = phonenumbers.format_number(
            match.number, phonenumbers.PhoneNumberFormat.E164
        )
        if phone not in phones:
            phones.append(phone)
    return phones


def parse_whatsapp_link(url: str) -> str | None:
    """Return https://wa.me/<digits> for a valid WhatsApp click-to-chat link, else None."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()
    if host == "wa.me":
        digits = parts.path.strip("/")
    elif host == "api.whatsapp.com" and parts.path.rstrip("/") == "/send":
        digits = parse_qs(parts.query).get("phone", [""])[0]
    else:
        return None

    digits = digits.strip().lstrip("+")
    if not (digits.isascii() and digits.isdigit()):
        return None
    if parse_phone("+" + digits, None) is None:
        return None
    return "https://wa.me/" + digits


def email_to_link(email: str) -> str:
    """Build the Email link used in social_link."""
    return "mailto:" + email


def phone_to_link(phone: str) -> str:
    """Build the Phone link used in social_link from a +E.164 number."""
    return "tel:" + phone
