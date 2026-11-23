"""Tests for app/utils/validators.py. Pure text processing: no network, no keys."""

import time

import phonenumbers
import pytest

from app.schemas import SocialLink
from app.utils.validators import (
    email_to_link,
    extract_emails,
    extract_phones,
    parse_phone,
    parse_whatsapp_link,
    phone_to_link,
)


def example_phone(region):
    """A number the phonenumbers library itself guarantees is valid for the region."""
    number = phonenumbers.example_number(region)
    national = phonenumbers.format_number(
        number, phonenumbers.PhoneNumberFormat.NATIONAL
    )
    international = phonenumbers.format_number(
        number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
    )
    e164 = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    return national, international, e164


def test_emails_found_lowercased_and_unique():
    text = (
        "Contact: Info@Shop-Example.co.uk or sales@shop-example.co.uk. "
        "Again: info@shop-example.co.uk"
    )
    assert extract_emails(text) == [
        "info@shop-example.co.uk",
        "sales@shop-example.co.uk",
    ]


def test_email_inside_mailto_link_with_subject():
    assert extract_emails('<a href="mailto:hello@shop-example.com?subject=Hi">') == [
        "hello@shop-example.com"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "logo@2x.png",
        "you@example.com",
        "name@domain.com",
        "noreply@shop-example.com",
        "abc123@o12345.ingest.sentry.io",
        "name [at] shop-example [dot] com",
        "not.an.email@",
        "a..b@shop-example.com",
    ],
)
def test_unwanted_or_invalid_emails_ignored(text):
    assert extract_emails(text) == []


def test_email_scan_is_fast_on_hostile_input():
    started = time.perf_counter()
    extract_emails("a" * 500_000)
    extract_emails("a@" * 250_000)
    extract_emails(("x" * 64 + "@") * 5_000)
    assert time.perf_counter() - started < 1.0


def test_local_phone_found_with_country_code():
    national, _, e164 = example_phone("IN")
    assert extract_phones(f"Call us on {national} today", "IN") == [e164]


def test_local_phone_not_found_without_country_code():
    national, _, _ = example_phone("IN")
    assert extract_phones(f"Call us on {national} today", None) == []


def test_international_phone_found_without_country_code():
    _, international, e164 = example_phone("GB")
    assert extract_phones(f"Phone: {international}", None) == [e164]


def test_invalid_phone_ignored():
    assert extract_phones("Order number 12345 and 0000000", "US") == []


def test_parse_phone_from_tel_link():
    _, _, e164 = example_phone("US")
    assert parse_phone("tel:" + e164, None) == e164
    assert parse_phone("tel:12", "US") is None


def test_whatsapp_link_forms_normalised():
    _, _, e164 = example_phone("AE")
    digits = e164.removeprefix("+")
    expected = "https://wa.me/" + digits
    assert parse_whatsapp_link("https://wa.me/" + digits) == expected
    assert parse_whatsapp_link("https://wa.me/+" + digits + "/") == expected
    assert (
        parse_whatsapp_link(f"https://api.whatsapp.com/send?phone={digits}&text=Hi")
        == expected
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://wa.me/message/ABC123",
        "https://wa.me/12",
        "https://evil.example/wa.me/971501234567",
        "javascript:alert(1)",
        "https://api.whatsapp.com/other?phone=971501234567",
    ],
)
def test_bad_whatsapp_links_rejected(url):
    assert parse_whatsapp_link(url) is None


def test_built_links_pass_the_output_schema():
    _, _, e164 = example_phone("US")
    email_link = SocialLink(
        platform="Email", link=email_to_link("info@shop-example.com")
    )
    phone_link = SocialLink(platform="Phone", link=phone_to_link(e164))
    assert email_link.link == "mailto:info@shop-example.com"
    assert phone_link.link == "tel:" + e164
