"""Facebook: public pages and profiles found through web search.

Facebook pages are never fetched; only URLs and search snippets are used. Accepted
forms: facebook.com/<username>, facebook.com/profile.php?id=<digits> and
facebook.com/people/<name>/<digits>. Groups, events, posts and tools are dropped.
"""

import re

from app.platforms.common import split_platform_url

NAME = "Facebook"
SEARCH_DOMAINS = ("facebook.com",)
RESERVED_PATHS = {
    "ads",
    "business",
    "dialog",
    "events",
    "gaming",
    "groups",
    "hashtag",
    "help",
    "login",
    "login.php",
    "marketplace",
    "pages",
    "permalink.php",
    "photo",
    "photo.php",
    "plugins",
    "policies",
    "privacy",
    "reel",
    "reels",
    "search",
    "share",
    "sharer",
    "sharer.php",
    "story.php",
    "watch",
}
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9.]{5,50}")
NUMERIC_ID_PATTERN = re.compile(r"[0-9]{5,25}")
PEOPLE_NAME_PATTERN = re.compile(r"[^/?#\s]{1,100}")


def profile_url(url: str) -> str | None:
    """Return a clean https://www.facebook.com/... profile link, or None."""
    parsed = split_platform_url(url, SEARCH_DOMAINS)
    if parsed is None:
        return None
    segments, query = parsed

    if segments == ["profile.php"]:
        profile_id = query.get("id", [""])[0]
        if NUMERIC_ID_PATTERN.fullmatch(profile_id):
            return f"https://www.facebook.com/profile.php?id={profile_id}"
        return None

    if len(segments) == 3 and segments[0] == "people":
        name, profile_id = segments[1], segments[2]
        if PEOPLE_NAME_PATTERN.fullmatch(name) and NUMERIC_ID_PATTERN.fullmatch(
            profile_id
        ):
            return f"https://www.facebook.com/people/{name}/{profile_id}"
        return None

    if len(segments) != 1:
        return None
    username = segments[0].lower()
    if username in RESERVED_PATHS or USERNAME_PATTERN.fullmatch(username) is None:
        return None
    return f"https://www.facebook.com/{username}"
