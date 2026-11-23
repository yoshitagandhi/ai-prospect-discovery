"""Safe fetching of public web pages (SSRF protection).

This is the only module allowed to download arbitrary web pages. Every URL, including
every redirect target, must pass check_url_is_safe() before any connection is made:

- http or https only, no user:pass@ part, ports 80 or 443 only, a host name (not a raw IP);
- not localhost or a local network name, and not a platform whose pages we never fetch;
- every IP address the host resolves to must be a public internet address.

Downloads are limited by a timeout, a maximum number of redirects and a maximum size,
and only the expected content types are accepted.

Known limit (documented in the README): httpx looks the host up again when it connects,
so a hostile DNS server could answer differently the second time (DNS rebinding).
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel

from app.config import Settings

USER_AGENT = "AIProspectDiscovery/0.1"
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
ROBOTS_CONTENT_TYPES = ("text/plain",)
LOCAL_HOST_NAMES = {"localhost"}
LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home")
NEVER_FETCH_DOMAINS = (
    "linkedin.com",
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "threads.net",
    "threads.com",
    "youtube.com",
    "whatsapp.com",
    "wa.me",
    "google.com",
)


class UnsafeUrlError(ValueError):
    """The URL must never be fetched (bad scheme, private address, blocked domain...)."""


class FetchError(Exception):
    """The page could not be fetched (timeout, HTTP error, too large, wrong type...)."""

    def __init__(self, reason: str, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.status_code = status_code


class FetchedPage(BaseModel):
    """A successfully downloaded page. final_url differs from requested_url after redirects."""

    requested_url: str
    final_url: str
    status_code: int
    text: str


def is_public_ip(address: str) -> bool:
    """True only for public internet addresses (IPv4 or IPv6).

    Private, loopback, link-local (including 169.254.169.254, the cloud metadata
    address), carrier-grade NAT, reserved, documentation and multicast ranges are
    all rejected. IPv4 addresses hidden inside IPv6 (::ffff:10.0.0.1) are unwrapped first.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _is_ip_literal(host: str) -> bool:
    """True when the host part of a URL is a raw IP address rather than a name."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _matches_domain(host: str, domains: tuple[str, ...]) -> bool:
    """True when host is one of the domains or a subdomain of one of them."""
    return any(host == domain or host.endswith("." + domain) for domain in domains)


async def _resolve_host(host: str) -> list[str]:
    """Return every IP address the host name resolves to."""
    loop = asyncio.get_running_loop()
    try:
        results = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise FetchError("host name could not be resolved") from None
    return [result[4][0] for result in results]


async def check_url_is_safe(url: str) -> None:
    """Raise UnsafeUrlError unless the URL is safe to fetch. Makes no HTTP request.

    Raises FetchError instead when the host name cannot be resolved at all.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise UnsafeUrlError("URL is malformed") from None

    if parts.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError("only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeUrlError("URL must not contain credentials")
    if port not in ALLOWED_PORTS:
        raise UnsafeUrlError("only ports 80 and 443 are allowed")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeUrlError("URL has no host")
    if _is_ip_literal(host):
        raise UnsafeUrlError("raw IP addresses are not allowed")
    if host in LOCAL_HOST_NAMES or host.endswith(LOCAL_HOST_SUFFIXES):
        raise UnsafeUrlError("local network names are not allowed")
    if _matches_domain(host, NEVER_FETCH_DOMAINS):
        raise UnsafeUrlError("pages on this platform are never fetched")

    addresses = await _resolve_host(host)
    if not addresses or not all(is_public_ip(address) for address in addresses):
        raise UnsafeUrlError("host resolves to a non-public address")


def _media_type(content_type_header: str) -> str:
    """'text/html; charset=utf-8' -> 'text/html'."""
    return content_type_header.split(";")[0].strip().lower()


def _decode(body: bytes, charset: str | None) -> str:
    """Decode bytes using the declared charset, falling back to UTF-8."""
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


async def _download_once(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
    allowed_types: tuple[str, ...],
) -> tuple[int, str | None, str]:
    """GET one URL without following redirects.

    Returns (status_code, redirect_location, text). The body is read in chunks and
    the download stops as soon as it exceeds the size limit.
    """
    try:
        async with client.stream(
            "GET",
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.http_timeout_seconds,
            follow_redirects=False,
        ) as response:
            if response.status_code in REDIRECT_STATUS_CODES:
                return response.status_code, response.headers.get("location"), ""
            if response.status_code >= 400:
                raise FetchError(
                    f"HTTP status {response.status_code}", response.status_code
                )
            media_type = _media_type(response.headers.get("content-type", ""))
            if media_type not in allowed_types:
                raise FetchError("unexpected content type")

            declared_size = response.headers.get("content-length", "")
            if (
                declared_size.isdigit()
                and int(declared_size) > settings.max_response_bytes
            ):
                raise FetchError("page is too large")

            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > settings.max_response_bytes:
                    raise FetchError("page is too large")
            text = _decode(bytes(body), response.charset_encoding)
            return response.status_code, None, text
    except httpx.TimeoutException:
        raise FetchError("request timed out") from None
    except httpx.HTTPError:
        raise FetchError("request failed") from None


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
    allowed_types: tuple[str, ...] = HTML_CONTENT_TYPES,
) -> FetchedPage:
    """Fetch a public page safely, re-checking every redirect target.

    Raises UnsafeUrlError for URLs that must never be fetched and FetchError for
    everything else that goes wrong. Callers skip the site and record a warning.
    """
    current_url = url
    for _ in range(settings.max_redirects + 1):
        await check_url_is_safe(current_url)
        status_code, location, text = await _download_once(
            client, current_url, settings, allowed_types
        )
        if status_code not in REDIRECT_STATUS_CODES:
            return FetchedPage(
                requested_url=url,
                final_url=current_url,
                status_code=status_code,
                text=text,
            )
        if not location:
            raise FetchError("redirect without a location")
        current_url = urljoin(current_url, location)
    raise FetchError("too many redirects")


async def is_allowed_by_robots(
    client: httpx.AsyncClient, url: str, settings: Settings
) -> bool:
    """Check the site's robots.txt before fetching a page from it.

    Rules (the same as major crawlers): a missing robots.txt (any 4xx) allows
    everything; a server error, timeout or unreadable robots.txt means we skip the site.
    """
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        page = await fetch_page(
            client, robots_url, settings, allowed_types=ROBOTS_CONTENT_TYPES
        )
    except FetchError as error:
        return error.status_code is not None and 400 <= error.status_code < 500
    parser = RobotFileParser()
    parser.parse(page.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)
