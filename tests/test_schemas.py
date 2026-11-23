"""Tests for app/schemas.py. Pure validation: no network, no keys."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    ErrorResponse,
    Location,
    ParsedQuery,
    Prospect,
    ProspectRequest,
    ProspectResponse,
    SocialLink,
)

GOOD_LINK = {
    "platform": "LinkedIn",
    "link": "https://www.linkedin.com/in/example-person",
}


def test_valid_query_is_stripped():
    request = ProspectRequest(query="  chess players in a small town  ")
    assert request.query == "chess players in a small town"


@pytest.mark.parametrize(
    "bad_query",
    ["", "   ", "ab", "x" * 301, "!!! ??? 123", "chess\x00club", "line one\nline two"],
)
def test_invalid_queries_rejected(bad_query):
    with pytest.raises(ValidationError):
        ProspectRequest(query=bad_query)


def test_unknown_request_field_rejected():
    with pytest.raises(ValidationError):
        ProspectRequest.model_validate({"query": "chess players", "limit": 5})


@pytest.mark.parametrize(
    ("platform", "link"),
    [
        ("LinkedIn", "https://www.linkedin.com/in/example-person"),
        ("X", "https://x.com/example"),
        ("X", "https://twitter.com/example"),
        ("YouTube", "https://www.youtube.com/@example"),
        ("Instagram", "https://www.instagram.com/example/"),
        ("Facebook", "https://www.facebook.com/example"),
        ("Threads", "https://www.threads.com/@example"),
        ("Threads", "https://www.threads.net/@example"),
        ("WhatsApp", "https://wa.me/15551234567"),
        ("Email", "mailto:person@examplecompany.com"),
        ("Phone", "tel:+15551234567"),
    ],
)
def test_valid_links_accepted(platform, link):
    assert SocialLink(platform=platform, link=link).link == link


@pytest.mark.parametrize(
    ("platform", "link"),
    [
        ("LinkedIn", "http://www.linkedin.com/in/example"),
        ("LinkedIn", "https://linkedin.com.evil.io/in/example"),
        ("LinkedIn", "https://evil.io/linkedin.com/in/example"),
        ("LinkedIn", "https://evil.io@www.linkedin.com/in/example"),
        ("LinkedIn", "https://www.linkedin.com:8443/in/example"),
        ("X", "javascript:alert(1)"),
        ("Instagram", "https://www.google.com/search?q=instagram"),
        ("WhatsApp", "https://whatsapp.evil.io/15551234567"),
        ("Email", "person@examplecompany.com"),
        ("Email", "mailto:not-an-email"),
        ("Phone", "tel:12345"),
        ("Phone", "tel:+0123456789"),
        ("Phone", "+15551234567"),
        ("Google", "https://www.google.com/search?q=example"),
    ],
)
def test_invalid_links_rejected(platform, link):
    with pytest.raises(ValidationError):
        SocialLink(platform=platform, link=link)


def test_duplicate_links_removed():
    prospect = Prospect(name="Example Person", social_link=[GOOD_LINK, GOOD_LINK])
    assert len(prospect.social_link) == 1


def test_prospect_without_links_rejected():
    with pytest.raises(ValidationError):
        Prospect(name="Example Person", social_link=[])


def test_blank_location_becomes_none():
    prospect = Prospect(name="Example Person", location="  ", social_link=[GOOD_LINK])
    assert prospect.location is None


def test_total_must_match_prospects():
    prospect = Prospect(name="Example Person", social_link=[GOOD_LINK])
    with pytest.raises(ValidationError):
        ProspectResponse(
            success=True, query="q", total_prospects=2, prospects=[prospect]
        )


def test_empty_result_is_valid():
    response = ProspectResponse(
        success=True, query="q", total_prospects=0, prospects=[]
    )
    assert response.prospects == []


def test_default_output_has_only_spec_fields():
    prospect = Prospect(name="Example Person", social_link=[GOOD_LINK])
    response = ProspectResponse(
        success=True, query="q", total_prospects=1, prospects=[prospect]
    )
    data = response.model_dump(exclude_none=True)
    assert set(data) == {"success", "query", "total_prospects", "prospects"}
    assert set(data["prospects"][0]) == {"name", "social_link"}


def test_parsed_query_accepts_partial_location_and_ignores_extra_keys():
    parsed = ParsedQuery.model_validate(
        {
            "is_prospect_query": True,
            "interest": "Chess",
            "audience": "Chess players",
            "location": {"city": "Springfield", "state": "", "country": "USA"},
            "target_type": "individual",
            "unexpected": "ignored",
        }
    )
    assert parsed.location.state is None
    assert parsed.related_terms == []


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"target_type": "robot"},
        {"interest": ""},
        {"related_terms": ["a", "b", "c", "d", "e", "f"]},
        {"audience": None},
        {"is_prospect_query": "maybe"},
    ],
)
def test_parsed_query_rejects_bad_values(bad_fields):
    data = {
        "is_prospect_query": True,
        "interest": "Chess",
        "audience": "Chess players",
        "location": {"country": "USA"},
        "target_type": "individual",
    }
    data.update(bad_fields)
    with pytest.raises(ValidationError):
        ParsedQuery.model_validate(data)


def test_error_response_is_always_unsuccessful():
    assert ErrorResponse(error="Invalid query").model_dump() == {
        "success": False,
        "error": "Invalid query",
    }


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("in", "IN"),
        (" us ", "US"),
        ("UK", "GB"),
        ("ZZ", None),
        ("Narnia", None),
        (None, None),
    ],
)
def test_country_code_normalised_or_dropped(given, expected):
    location = Location.model_validate({"city": "Somewhere", "country_code": given})
    assert location.country_code == expected


def test_non_prospect_query_may_leave_interest_empty():
    parsed = ParsedQuery.model_validate(
        {"is_prospect_query": False, "interest": "", "target_type": "individual"}
    )
    assert parsed.interest is None
    assert parsed.location.city is None
