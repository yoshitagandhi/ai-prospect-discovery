"""Tests for app/services/profile_extractor.py. Pages are fake HTML: no network."""

import asyncio

import phonenumbers
import pytest

from app.config import Settings
from app.platforms.web_search import SearchResult
from app.services import profile_extractor
from app.services.profile_extractor import (
    extract_contacts,
    parse_contacts,
    website_candidate,
)
from app.utils import safe_http
from app.utils.safe_http import FetchedPage, FetchError, UnsafeUrlError

SETTINGS = Settings(
    tavily_api_key="tvly-test-not-a-real-key",
    gemini_api_key="gemini-test-not-a-real-key",
    max_pages_to_fetch=3,
)
PAGE_URL = "https://brightbistro.example/contact"


def example_phone(region):
    number = phonenumbers.example_number(region)
    national = phonenumbers.format_number(
        number, phonenumbers.PhoneNumberFormat.NATIONAL
    )
    e164 = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    return national, e164


def links_of(html, country_code="AE"):
    return [
        (link.platform, link.link)
        for link in parse_contacts(html, PAGE_URL, country_code)
    ]


def test_links_and_text_contacts_extracted_in_page_order():
    national, e164 = example_phone("AE")
    digits = e164.removeprefix("+")
    html = f"""
    <html><body>
      <a href="mailto:Hello@BrightBistro.example?subject=Booking">Email us</a>
      <a href="tel:{e164}">Call</a>
      <a href="https://wa.me/{digits}">WhatsApp</a>
      <a href="https://www.instagram.com/BrightBistro/">Instagram</a>
      <a href="https://www.youtube.com/@BrightBistro">YouTube</a>
      <a href="/menu">Menu</a>
      <p>Reservations: events@brightbistro.example or {national}</p>
    </body></html>
    """
    assert links_of(html) == [
        ("Email", "mailto:hello@brightbistro.example"),
        ("Phone", "tel:" + e164),
        ("WhatsApp", "https://wa.me/" + digits),
        ("Instagram", "https://www.instagram.com/brightbistro/"),
        ("YouTube", "https://www.youtube.com/@brightbistro"),
        ("Email", "mailto:events@brightbistro.example"),
    ]


def test_hidden_and_unsafe_content_ignored():
    html = """
    <html><head><script>var contact = "secret@brightbistro.example";</script>
    <style>.x { content: "style@brightbistro.example"; }</style></head>
    <body>
      <a href="javascript:alert('x@brightbistro.example')">bad</a>
      <a href="https://www.instagram.com/p/AbC123/">a post, not a profile</a>
      <a href="https://linkedin.com.evil.example/in/someone">look-alike</a>
    </body></html>
    """
    assert links_of(html) == []


def test_duplicates_appear_once():
    html = """
    <a href="mailto:info@brightbistro.example">a</a>
    <p>info@brightbistro.example</p>
    <a href="https://x.com/BrightBistro">X</a>
    <a href="https://twitter.com/brightbistro">X again</a>
    """
    assert links_of(html) == [
        ("Email", "mailto:info@brightbistro.example"),
        ("X", "https://x.com/brightbistro"),
    ]


def test_website_candidate_uses_cleaned_title():
    hit = SearchResult(
        title="Bright Bistro | Contact Us",
        url=PAGE_URL,
        snippet="Restaurant in Dubai",
    )
    candidate = website_candidate(hit)
    assert candidate.name == "Bright Bistro"
    assert candidate.platform == "Website"
    assert website_candidate(hit.model_copy(update={"title": "Home"})) is None


@pytest.fixture
def fake_web(monkeypatch):
    pages = {
        "https://good.example/contact": '<a href="mailto:hi@good.example">mail</a>',
        "https://empty.example/": "<p>No contact details here</p>",
    }

    async def robots(client, url, settings):
        return "blocked.example" not in url

    async def fetch(client, url, settings):
        if "broken.example" in url:
            raise FetchError("HTTP status 500", 500)
        if "unsafe.example" in url:
            raise UnsafeUrlError("host resolves to a non-public address")
        if "buggy.example" in url:
            raise AttributeError("our bug")
        return FetchedPage(
            requested_url=url, final_url=url, status_code=200, text=pages[url]
        )

    monkeypatch.setattr(safe_http, "is_allowed_by_robots", robots)
    monkeypatch.setattr(safe_http, "fetch_page", fetch)


def site(url):
    return website_candidate(SearchResult(title="Some Bistro", url=url, snippet="x"))


def run_extract(urls):
    return asyncio.run(
        extract_contacts(None, SETTINGS, [site(url) for url in urls], "AE")
    )


def test_extract_contacts_keeps_sites_with_links_and_warns_about_others(fake_web):
    sites, warnings = run_extract(
        [
            "https://good.example/contact",
            "https://empty.example/",
            "https://blocked.example/",
        ]
    )
    assert [s.page_url for s in sites] == ["https://good.example/contact"]
    assert sites[0].links[0].link == "mailto:hi@good.example"
    assert warnings == [
        "Website empty.example: no public contact details found",
        "Website blocked.example skipped: blocked by robots.txt",
    ]


def test_fetch_failures_become_warnings(fake_web):
    sites, warnings = run_extract(
        ["https://broken.example/", "https://unsafe.example/"]
    )
    assert sites == []
    assert warnings == [
        "Website broken.example skipped: HTTP status 500",
        "Website unsafe.example skipped: host resolves to a non-public address",
    ]


def test_only_max_pages_are_fetched(fake_web, monkeypatch):
    fetched = []
    original = safe_http.fetch_page

    async def counting_fetch(client, url, settings):
        fetched.append(url)
        return await original(client, url, settings)

    monkeypatch.setattr(safe_http, "fetch_page", counting_fetch)
    run_extract(["https://good.example/contact"] * 5)
    assert len(fetched) == SETTINGS.max_pages_to_fetch


def test_unexpected_errors_are_not_hidden(fake_web):
    with pytest.raises(AttributeError):
        run_extract(["https://buggy.example/"])


def test_semaphore_limit_is_small():
    assert profile_extractor.MAX_CONCURRENT_FETCHES <= 5
