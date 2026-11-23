"""LinkedIn: public profile and company pages found through web search.

LinkedIn pages are never fetched (no permitted access path); only their URLs and the
search snippets are used. Posts, articles and job pages are not profiles and are dropped.
"""

import re

from app.platforms.common import split_platform_url

NAME = "LinkedIn"
SEARCH_DOMAINS = ("linkedin.com",)
PROFILE_SECTIONS = {"in", "company"}
SLUG_PATTERN = re.compile(r"[A-Za-z0-9_%-]{2,100}")


def profile_url(url: str) -> str | None:
    """'https://uk.linkedin.com/in/Jane-Doe/details/' -> 'https://www.linkedin.com/in/jane-doe'."""
    parsed = split_platform_url(url, SEARCH_DOMAINS)
    if parsed is None:
        return None
    segments, _ = parsed
    if len(segments) < 2 or segments[0].lower() not in PROFILE_SECTIONS:
        return None
    if SLUG_PATTERN.fullmatch(segments[1]) is None:
        return None
    return f"https://www.linkedin.com/{segments[0].lower()}/{segments[1].lower()}"
