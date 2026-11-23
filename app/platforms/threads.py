"""Threads: public profiles found through web search.

Threads pages are never fetched; only URLs and search snippets are used. A post URL
(threads.com/@<handle>/post/<id>) names its author, so it is turned into that
author's profile and the post snippet is kept as evidence.
"""

import re

from app.platforms.common import split_platform_url

NAME = "Threads"
SEARCH_DOMAINS = ("threads.com", "threads.net")
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9._]{1,30}")


def profile_url(url: str) -> str | None:
    """'https://www.threads.net/@Jane.Doe/post/abc' -> 'https://www.threads.com/@jane.doe'."""
    parsed = split_platform_url(url, SEARCH_DOMAINS)
    if parsed is None:
        return None
    segments, _ = parsed
    is_profile = len(segments) == 1
    is_post = len(segments) == 3 and segments[1] == "post"
    if not (is_profile or is_post):
        return None
    handle = segments[0].lower()
    if HANDLE_PATTERN.fullmatch(handle) is None:
        return None
    return f"https://www.threads.com/{handle}"
