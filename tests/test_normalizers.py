"""Tests for app/utils/normalizers.py."""

import pytest

from app.utils.normalizers import clean_display_name, content_words


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Jane Doe - Owner - Bright Agency | LinkedIn", "Jane Doe"),
        ("Jane Doe \u2013 Founder \u2014 Startup | LinkedIn", "Jane Doe"),
        ("Jane Doe (@janedoe) • Instagram photos and videos", "Jane Doe"),
        ("Jane Doe (@janedoe) / X", "Jane Doe"),
        ("Bright Bakery | Springfield - Facebook", "Bright Bakery"),
        ("Mary-Jane Smith - Coach", "Mary-Jane Smith"),
        ("  Extra   Spaces  ", "Extra Spaces"),
        ('Jane Doe on X: "Great match tonight" / X', "Jane Doe"),
        ("Jane Doe (@janedoe) on Threads", "Jane Doe"),
        ("\u201cJane Doe\u201d - Owner | LinkedIn", "Jane Doe"),
        ("\u200eJane Doe\u200e | LinkedIn", "Jane Doe"),
        ("BRIGHT BISTRO DUBAI\u2728\U0001f451", "BRIGHT BISTRO DUBAI"),
        ("\u2764\ufe0f Jane Doe | Instagram", "Jane Doe"),
        ("Jos\u00e9 Garc\u00eda \U0001f44b\U0001f3fd - Coach", "Jos\u00e9 Garc\u00eda"),
        ("Bright Bistro\u00ae | Facebook", "Bright Bistro"),
    ],
)
def test_names_cleaned_from_titles(title, expected):
    assert clean_display_name(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "",
        "Instagram",
        "Log In | Facebook",
        "12345 - 678",
        "Contact Us",
        "About",
        "x" * 150,
        "\u2728\U0001f451",
    ],
)
def test_unusable_titles_give_none(title):
    assert clean_display_name(title) is None


def test_content_words_handle_plurals_and_small_words():
    assert content_words("Restaurants in the City of Dubai") == {
        "restaurant",
        "city",
        "dubai",
    }
    assert content_words("Chess business owners") == {"chess", "business", "owner"}
