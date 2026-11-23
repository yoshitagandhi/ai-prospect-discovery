"""Instagram: public profiles found through web search.

Instagram pages are never fetched; only URLs and search snippets are used. Post and
reel URLs (/p/..., /reel/...) do not name their author, so they are dropped.
"""

import re

from app.platforms.common import split_platform_url

NAME = "Instagram"
SEARCH_DOMAINS = ("instagram.com",)
RESERVED_PATHS = {
    "about",
    "accounts",
    "developer",
    "direct",
    "explore",
    "legal",
    "p",
    "popular",
    "reel",
    "reels",
    "stories",
    "tv",
    "web",
}
HANDLE_PATTERN = re.compile(r"[A-Za-z0-9._]{1,30}")


def profile_url(url: str) -> str | None:
    """'https://instagram.com/Jane.Doe/' -> 'https://www.instagram.com/jane.doe/'."""
    parsed = split_platform_url(url, SEARCH_DOMAINS)
    if parsed is None:
        return None
    segments, _ = parsed
    if len(segments) != 1:
        return None
    handle = segments[0].lower()
    if handle in RESERVED_PATHS or HANDLE_PATTERN.fullmatch(handle) is None:
        return None
    return f"https://www.instagram.com/{handle}/"
