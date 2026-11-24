"""Tests for the platform modules' profile_url(): which URLs count as profiles."""

import pytest

from app.platforms import (
    SEARCH_PLATFORMS,
    facebook,
    instagram,
    linkedin,
    threads,
    x_platform,
    youtube,
)
from app.schemas import SocialLink

GOOD_URLS = [
    (
        linkedin,
        "https://www.linkedin.com/in/Jane-Doe",
        "https://www.linkedin.com/in/jane-doe",
    ),
    (
        linkedin,
        "https://uk.linkedin.com/in/jane-doe/details/experience/",
        "https://www.linkedin.com/in/jane-doe",
    ),
    (
        linkedin,
        "https://www.linkedin.com/company/bright-agency/",
        "https://www.linkedin.com/company/bright-agency",
    ),
    (
        instagram,
        "https://instagram.com/Jane.Doe/",
        "https://www.instagram.com/jane.doe/",
    ),
    (
        facebook,
        "https://www.facebook.com/BrightBakery",
        "https://www.facebook.com/brightbakery",
    ),
    (
        facebook,
        "https://m.facebook.com/profile.php?id=1000123456",
        "https://www.facebook.com/profile.php?id=1000123456",
    ),
    (
        facebook,
        "https://www.facebook.com/people/Jane-Doe/1000123456/",
        "https://www.facebook.com/people/Jane-Doe/1000123456",
    ),
    (x_platform, "https://twitter.com/JaneDoe", "https://x.com/janedoe"),
    (x_platform, "https://x.com/janedoe/status/1234567890", "https://x.com/janedoe"),
    (threads, "https://www.threads.net/@Jane.Doe", "https://www.threads.com/@jane.doe"),
    (
        threads,
        "https://www.threads.com/@jane.doe/post/AbC123",
        "https://www.threads.com/@jane.doe",
    ),
]

BAD_URLS = [
    (linkedin, "https://www.linkedin.com/posts/jane-doe_activity-123"),
    (linkedin, "https://www.linkedin.com/pulse/some-article"),
    (linkedin, "https://www.linkedin.com/jobs/view/123"),
    (linkedin, "https://linkedin.com.evil.io/in/jane-doe"),
    (linkedin, "https://www.google.com/search?q=linkedin.com/in/jane"),
    (instagram, "https://www.instagram.com/p/AbC123/"),
    (instagram, "https://www.instagram.com/reel/AbC123/"),
    (instagram, "https://www.instagram.com/explore/tags/chess/"),
    (instagram, "https://www.instagram.com/janedoe/tagged/"),
    (facebook, "https://www.facebook.com/groups/chessclub"),
    (facebook, "https://www.facebook.com/events/123456"),
    (facebook, "https://www.facebook.com/brightbakery/posts/123"),
    (facebook, "https://www.facebook.com/profile.php?id=abc"),
    (facebook, "https://www.facebook.com/abc"),
    (x_platform, "https://x.com/search?q=chess"),
    (x_platform, "https://x.com/i/lists/123"),
    (x_platform, "https://x.com/hashtag/chess"),
    (x_platform, "https://x.com/name_that_is_far_too_long"),
    (threads, "https://www.threads.com/search?q=chess"),
    (threads, "https://www.threads.com/janedoe"),
    (threads, "javascript:alert(1)"),
    (x_platform, "https://user:pass@x.com/janedoe"),
    (x_platform, "https://x.com:8443/janedoe"),
]


@pytest.mark.parametrize(("module", "url", "expected"), GOOD_URLS)
def test_profile_urls_recognised_and_cleaned(module, url, expected):
    assert module.profile_url(url) == expected


@pytest.mark.parametrize(("module", "url"), BAD_URLS)
def test_non_profile_urls_rejected(module, url):
    assert module.profile_url(url) is None


@pytest.mark.parametrize(("module", "url", "expected"), GOOD_URLS)
def test_clean_links_pass_the_output_schema(module, url, expected):
    assert (
        SocialLink(platform=module.NAME, link=module.profile_url(url)).link == expected
    )


def test_registry_uses_spec_platform_names():
    names = [module.NAME for module in SEARCH_PLATFORMS]
    assert names == ["LinkedIn", "Instagram", "Facebook", "X", "Threads"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://www.youtube.com/@BrightBistro",
            "https://www.youtube.com/@brightbistro",
        ),
        (
            "https://youtube.com/channel/UC" + "a" * 22 + "/videos",
            "https://www.youtube.com/channel/UC" + "a" * 22,
        ),
        ("https://www.youtube.com/watch?v=abc123", None),
        ("https://www.youtube.com/results?search_query=chess", None),
        ("https://youtube.com.evil.example/@someone", None),
    ],
)
def test_youtube_channel_links_recognised(url, expected):
    assert youtube.profile_url(url) == expected
