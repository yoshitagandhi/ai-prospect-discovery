"""Tests for app/utils/normalizers.py."""

import pytest

from app.utils.normalizers import clean_display_name


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
    ],
)
def test_names_cleaned_from_titles(title, expected):
    assert clean_display_name(title) == expected


@pytest.mark.parametrize(
    "title", ["", "Instagram", "Log In | Facebook", "12345 - 678", "x" * 150]
)
def test_unusable_titles_give_none(title):
    assert clean_display_name(title) is None
