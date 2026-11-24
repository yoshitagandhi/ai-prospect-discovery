"""Clean-up helpers shared by discovery and deduplication.

Dash separators are written as escapes (\\u2013 en dash, \\u2014 em dash) so they are
not mistaken for an ordinary hyphen when reading the code.
"""

import re
import unicodedata

TITLE_SEPARATORS = (
    " | ",
    " - ",
    " \u2013 ",
    " \u2014 ",
    " • ",
    " · ",
    " (@",
    " / ",
    " on X:",
    " on Threads",
    " on Instagram",
)
GENERIC_NAMES = {
    "facebook",
    "home",
    "instagram",
    "linkedin",
    "log in",
    "login",
    "page not found",
    "profile",
    "sign up",
    "threads",
    "twitter",
    "x",
    "youtube",
}
MAX_NAME_LENGTH = 100
EDGE_CHARACTERS = " -|\u2022\u00b7\"'\u201c\u201d\u2018\u2019\u00ab\u00bb"


def clean_display_name(title: str) -> str | None:
    """Turn a page title into a display name, or None if it is not a usable name.

    'Jane Doe - Owner - Agency | LinkedIn' -> 'Jane Doe'
    'Jane Doe (@jane) • Instagram photos and videos' -> 'Jane Doe'
    Generic titles such as 'Login' or 'Instagram' give None.
    Invisible formatting characters (such as direction marks) are removed and quotes
    around the name are stripped.
    """
    name = "".join(char for char in title if unicodedata.category(char) != "Cf")
    name = name.strip()
    for separator in TITLE_SEPARATORS:
        name = name.split(separator)[0]
    name = " ".join(name.split()).strip(EDGE_CHARACTERS)
    if not name or len(name) > MAX_NAME_LENGTH:
        return None
    if name.lower() in GENERIC_NAMES:
        return None
    if not any(char.isalpha() for char in name):
        return None
    return name


WORD_PATTERN = re.compile(r"[^\W_]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "near",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def content_words(text: str) -> set[str]:
    """Lower-case words with a trailing plural 's' removed ('restaurants' -> 'restaurant').

    Words ending in 'ss' keep their ending ('chess', 'business' are not plurals).

    Small connecting words such as "in" or "the" are ignored, so they never count as a
    match when two texts are compared.
    """
    words = set()
    for word in WORD_PATTERN.findall(text.lower()):
        if word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return words
