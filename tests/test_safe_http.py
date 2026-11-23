"""Tests for app/utils/safe_http.py.

No real network: respx fakes every HTTP response and the DNS lookup is replaced by
a fixed table. Async functions are run with asyncio.run(), so no pytest plugin is needed.
"""

import asyncio

import httpx
import pytest
import respx

from app.config import Settings
from app.utils import safe_http
from app.utils.safe_http import (
    FetchError,
    UnsafeUrlError,
    check_url_is_safe,
    fetch_page,
    is_allowed_by_robots,
    is_public_ip,
)

PUBLIC_IP = "93.184.216.34"
FAKE_DNS = {
    "shop.example": [PUBLIC_IP],
    "other.example": [PUBLIC_IP],
    "loop.example": [PUBLIC_IP],
    "private.example": ["10.0.0.5"],
    "mixed.example": [PUBLIC_IP, "127.0.0.1"],
    "metadata.example": ["169.254.169.254"],
    "2130706433": ["127.0.0.1"],
}
HTML = {"content-type": "text/html; charset=utf-8"}
SETTINGS = Settings(
    tavily_api_key="tvly-test-not-a-real-key",
    gemini_api_key="gemini-test-not-a-real-key",
    max_response_bytes=10_000,
    max_redirects=3,
)


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    async def resolve(host):
        if host not in FAKE_DNS:
            raise FetchError("host name could not be resolved")
        return FAKE_DNS[host]

    monkeypatch.setattr(safe_http, "_resolve_host", resolve)


def fetch(url):
    async def run():
        async with httpx.AsyncClient() as client:
            return await fetch_page(client, url, SETTINGS)

    return asyncio.run(run())


def robots_allows(url):
    async def run():
        async with httpx.AsyncClient() as client:
            return await is_allowed_by_robots(client, url, SETTINGS)

    return asyncio.run(run())


@pytest.mark.parametrize("address", ["8.8.8.8", PUBLIC_IP, "2606:4700:4700::1111"])
def test_public_addresses_accepted(address):
    assert is_public_ip(address)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:10.0.0.1",
        "not-an-ip",
    ],
)
def test_non_public_addresses_rejected(address):
    assert not is_public_ip(address)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://shop.example/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://user:pass@shop.example/",
        "http://shop.example:8080/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://localhost/",
        "http://printer.local/",
        "https://www.linkedin.com/in/someone",
        "https://www.google.com/search?q=x",
        "http://private.example/",
        "http://mixed.example/",
        "http://metadata.example/",
        "http://2130706433/",
        "http:///no-host",
    ],
)
def test_unsafe_urls_rejected(url):
    with pytest.raises(UnsafeUrlError):
        asyncio.run(check_url_is_safe(url))


def test_safe_url_accepted():
    asyncio.run(check_url_is_safe("https://shop.example/contact"))


def test_unresolvable_host_is_fetch_error():
    with pytest.raises(FetchError):
        asyncio.run(check_url_is_safe("https://unknown.example/"))


def test_html_page_fetched():
    with respx.mock() as router:
        router.get("https://shop.example/").mock(
            return_value=httpx.Response(200, headers=HTML, text="<p>Hello</p>")
        )
        page = fetch("https://shop.example/")
    assert page.status_code == 200
    assert page.final_url == "https://shop.example/"
    assert "Hello" in page.text


def test_safe_redirect_followed():
    with respx.mock() as router:
        router.get("https://shop.example/old").mock(
            return_value=httpx.Response(301, headers={"location": "/new"})
        )
        router.get("https://shop.example/new").mock(
            return_value=httpx.Response(200, headers=HTML, text="moved")
        )
        page = fetch("https://shop.example/old")
    assert page.final_url == "https://shop.example/new"
    assert page.requested_url == "https://shop.example/old"


def test_redirect_to_private_address_blocked():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://shop.example/").mock(
            return_value=httpx.Response(
                302, headers={"location": "http://private.example/admin"}
            )
        )
        private_route = router.get("http://private.example/admin")
        with pytest.raises(UnsafeUrlError):
            fetch("https://shop.example/")
    assert not private_route.called


def test_too_many_redirects():
    with respx.mock() as router:
        router.get(url__regex=r"https://loop\.example/.*").mock(
            return_value=httpx.Response(302, headers={"location": "/again"})
        )
        with pytest.raises(FetchError, match="too many redirects"):
            fetch("https://loop.example/start")


def test_declared_large_page_rejected():
    with respx.mock() as router:
        router.get("https://shop.example/").mock(
            return_value=httpx.Response(200, headers=HTML, content=b"a" * 20_000)
        )
        with pytest.raises(FetchError, match="too large"):
            fetch("https://shop.example/")


def test_streamed_large_page_without_length_rejected():
    async def endless_chunks():
        for _ in range(100):
            yield b"a" * 1_000

    with respx.mock() as router:
        router.get("https://shop.example/").mock(
            return_value=httpx.Response(200, headers=HTML, content=endless_chunks())
        )
        with pytest.raises(FetchError, match="too large"):
            fetch("https://shop.example/")


def test_non_html_rejected():
    with respx.mock() as router:
        router.get("https://shop.example/file.pdf").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"
            )
        )
        with pytest.raises(FetchError, match="content type"):
            fetch("https://shop.example/file.pdf")


def test_http_error_status_rejected():
    with respx.mock() as router:
        router.get("https://shop.example/missing").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(FetchError) as error_info:
            fetch("https://shop.example/missing")
    assert error_info.value.status_code == 404


def test_timeout_becomes_fetch_error():
    with respx.mock() as router:
        router.get("https://shop.example/").mock(side_effect=httpx.ConnectTimeout)
        with pytest.raises(FetchError, match="timed out"):
            fetch("https://shop.example/")


def test_robots_disallow_respected():
    robots = "User-agent: *\nDisallow: /private\n"
    with respx.mock() as router:
        router.get("https://shop.example/robots.txt").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/plain"}, text=robots
            )
        )
        assert robots_allows("https://shop.example/about") is True
        assert robots_allows("https://shop.example/private/page") is False


def test_missing_robots_allows_everything():
    with respx.mock() as router:
        router.get("https://other.example/robots.txt").mock(
            return_value=httpx.Response(404)
        )
        assert robots_allows("https://other.example/contact") is True


def test_robots_server_error_skips_site():
    with respx.mock() as router:
        router.get("https://other.example/robots.txt").mock(
            return_value=httpx.Response(503)
        )
        assert robots_allows("https://other.example/contact") is False
