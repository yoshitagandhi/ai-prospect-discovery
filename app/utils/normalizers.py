"""Clean-up helpers shared by discovery and deduplication.

Dash separators are written as escapes (\\u2013 en dash, \\u2014 em dash) so they are
not mistaken for an ordinary hyphen when reading the code.
"""

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


def clean_display_name(title: str) -> str | None:
    """Turn a page title into a display name, or None if it is not a usable name.

    'Jane Doe - Owner - Agency | LinkedIn' -> 'Jane Doe'
    'Jane Doe (@jane) • Instagram photos and videos' -> 'Jane Doe'
    Generic titles such as 'Login' or 'Instagram' give None.
    """
    name = title.strip()
    for separator in TITLE_SEPARATORS:
        name = name.split(separator)[0]
    name = " ".join(name.split()).strip(" -|•·")
    if not name or len(name) > MAX_NAME_LENGTH:
        return None
    if name.lower() in GENERIC_NAMES:
        return None
    if not any(char.isalpha() for char in name):
        return None
    return name
