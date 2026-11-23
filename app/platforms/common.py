"""Shared URL parsing for the platform modules."""

from urllib.parse import parse_qs, urlsplit

ALLOWED_PORTS = {None, 80, 443}


def split_platform_url(
    url: str, domains: tuple[str, ...]
) -> tuple[list[str], dict[str, list[str]]] | None:
    """Return (path segments, query parameters) if url is on one of the domains.

    Returns None for other domains (including look-alikes such as linkedin.com.evil.io),
    non-web schemes, credentials in the URL or unusual ports.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    if parts.username is not None or parts.password is not None:
        return None
    if port not in ALLOWED_PORTS:
        return None
    host = (parts.hostname or "").lower()
    if not any(host == domain or host.endswith("." + domain) for domain in domains):
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    return segments, parse_qs(parts.query)
