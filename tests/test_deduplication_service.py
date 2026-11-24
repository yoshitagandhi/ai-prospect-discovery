"""Tests for app/services/deduplication_service.py. Pure functions: no network."""

from app.schemas import Prospect, SocialLink
from app.services.deduplication_service import MAX_LINKS, merge_duplicates

INSTAGRAM = SocialLink(
    platform="Instagram", link="https://www.instagram.com/brightbistro/"
)
EMAIL = SocialLink(platform="Email", link="mailto:hi@brightbistro.example")
PHONE = SocialLink(platform="Phone", link="tel:+97142345678")
X_LINK = SocialLink(platform="X", link="https://x.com/brightbistro")


def prospect(name, *links, score=None, evidence=None, sources=None):
    return Prospect(
        name=name,
        social_link=list(links),
        evidence=evidence,
        source_urls=sources,
        relevance_score=score,
    )


def linkedin(slug):
    return SocialLink(platform="LinkedIn", link=f"https://www.linkedin.com/in/{slug}")


def test_no_prospects_gives_no_prospects():
    assert merge_duplicates([]) == ([], 0)


def test_same_profile_found_twice_becomes_one():
    first = prospect("Bright Bistro", INSTAGRAM)
    second = prospect("Bright Bistro Dubai", INSTAGRAM)
    merged, count = merge_duplicates([first, second])
    assert count == 1
    assert [p.name for p in merged] == ["Bright Bistro"]
    assert merged[0].social_link == [INSTAGRAM]


def test_website_that_links_the_profile_is_merged_with_it():
    profile = prospect("Bright Bistro", INSTAGRAM)
    website = prospect("Bright Bistro | Home", EMAIL, PHONE, INSTAGRAM)
    merged, count = merge_duplicates([profile, website])
    assert count == 1
    assert merged[0].social_link == [INSTAGRAM, EMAIL, PHONE]


def test_same_email_on_two_websites_is_merged():
    merged, count = merge_duplicates(
        [prospect("Bright Bistro", EMAIL), prospect("Bright Cafe", EMAIL, PHONE)]
    )
    assert count == 1
    assert merged[0].social_link == [EMAIL, PHONE]


def test_same_name_with_different_links_is_not_merged():
    first = prospect("Jane Doe", linkedin("jane-doe"))
    second = prospect("Jane Doe", linkedin("jane-doe-42"))
    merged, count = merge_duplicates([first, second])
    assert count == 0
    assert merged == [first, second]


def test_chain_of_shared_links_becomes_one_group():
    a = prospect("Bright Bistro", INSTAGRAM)
    unrelated = prospect("Sam Lee", linkedin("sam-lee"))
    c = prospect("Bright Bistro Cafe", X_LINK)
    b = prospect("Bright Bistro Website", INSTAGRAM, X_LINK)
    merged, count = merge_duplicates([a, unrelated, c, b])
    assert count == 2
    assert [p.name for p in merged] == ["Bright Bistro", "Sam Lee"]
    assert merged[0].social_link == [INSTAGRAM, X_LINK]


def test_trailing_slash_does_not_hide_a_duplicate():
    with_slash = SocialLink(platform="X", link="https://x.com/brightbistro/")
    merged, count = merge_duplicates(
        [prospect("Bright Bistro", X_LINK), prospect("Bright", with_slash)]
    )
    assert count == 1
    assert merged[0].social_link == [X_LINK]


def test_best_ranked_name_and_location_win_and_highest_score_kept():
    best = Prospect(
        name="Bright Bistro",
        location="Dubai, UAE",
        social_link=[INSTAGRAM],
        relevance_score=0.9,
    )
    other = prospect("Bright Bistro Home", INSTAGRAM, score=1.0)
    merged, _ = merge_duplicates([best, other])
    assert merged[0].name == "Bright Bistro"
    assert merged[0].location == "Dubai, UAE"
    assert merged[0].relevance_score == 1.0


def test_evidence_and_sources_combined_without_repeats():
    first = prospect(
        "Bright Bistro",
        INSTAGRAM,
        evidence=["Restaurant in Dubai", "Dubai"],
        sources=["https://www.instagram.com/brightbistro"],
    )
    second = prospect(
        "Bright Bistro",
        INSTAGRAM,
        EMAIL,
        evidence=["Dubai", "Family restaurant"],
        sources=["https://brightbistro.example/contact"],
    )
    merged, _ = merge_duplicates([first, second])
    assert merged[0].evidence == ["Restaurant in Dubai", "Dubai", "Family restaurant"]
    assert merged[0].source_urls == [
        "https://www.instagram.com/brightbistro",
        "https://brightbistro.example/contact",
    ]


def test_evidence_stays_none_outside_evidence_mode():
    merged, _ = merge_duplicates(
        [prospect("Bright Bistro", INSTAGRAM), prospect("Bright", INSTAGRAM, EMAIL)]
    )
    assert merged[0].evidence is None
    assert merged[0].source_urls is None
    assert merged[0].relevance_score is None


def test_merged_links_capped_at_the_schema_limit():
    first = prospect("Many Links", *[linkedin(f"person-{n}") for n in range(15)])
    second = prospect(
        "Many Links", linkedin("person-0"), *[linkedin(f"other-{n}") for n in range(15)]
    )
    merged, count = merge_duplicates([first, second])
    assert count == 1
    assert len(merged[0].social_link) == MAX_LINKS
    assert merged[0].social_link[:15] == first.social_link
