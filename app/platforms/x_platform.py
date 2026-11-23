"""X (Twitter): public profiles found through web search.

X pages are never fetched; only URLs and search snippets are used. A post URL
(x.com/<handle>/status/<id>) names its author, so it is turned into that author's
profile and the post snippet is kept as evidence.
"""

import re

from app.platforms.common import split_platform_url

NAME = "X"
SEARCH_DOMAINS = ("x.com", "twitter.com")
RESERVED_PATHS = {
    "compose",
    "explore",
    "hashtag",
    "home",
    "i",
    "intent",
    "login",
    "messages",
    "notifications",
    "privacy",
    "search",
    "settings",
    "share",
    "signup",
    "tos",
}
HANDLE_PATTERN = re.compile(r"[A-Za-z0-9_]{1,15}")


def profile_url(url: str) -> str | None:
    """'https://twitter.com/JaneDoe/status/123' -> 'https://x.com/janedoe'."""
    parsed = split_platform_url(url, SEARCH_DOMAINS)
    if parsed is None:
        return None
    segments, _ = parsed
    is_profile = len(segments) == 1
    is_post = len(segments) == 3 and segments[1] == "status"
    if not (is_profile or is_post):
        return None
    handle = segments[0].lower()
    if handle in RESERVED_PATHS or HANDLE_PATTERN.fullmatch(handle) is None:
        return None
    return f"https://x.com/{handle}"
