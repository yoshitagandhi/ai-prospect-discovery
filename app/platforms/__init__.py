"""Registry of platforms discovered through web search.

Each module provides NAME (the spec's platform name), SEARCH_DOMAINS and
profile_url(url). To add a platform, write such a module and add it to
SEARCH_PLATFORMS. YouTube is separate because it uses its own official API.
"""

from app.platforms import facebook, instagram, linkedin, threads, x_platform

SEARCH_PLATFORMS = (linkedin, instagram, facebook, x_platform, threads)
