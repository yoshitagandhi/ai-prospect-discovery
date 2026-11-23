"""Tests for app/platforms/youtube.py. respx fakes the YouTube API: no network."""

import asyncio

import httpx
import pytest
import respx

from app.config import Settings
from app.platforms import youtube
from app.platforms.youtube import YouTubeError, channel_link, find_channels

CHANNEL_A = "UC" + "a" * 22
CHANNEL_B = "UC" + "b" * 22
SEARCH_PATTERN = r"https://www\.googleapis\.com/youtube/v3/search.*"
CHANNELS_PATTERN = r"https://www\.googleapis\.com/youtube/v3/channels.*"


def make_settings(with_key=True):
    return Settings(
        tavily_api_key="tvly-test-not-a-real-key",
        gemini_api_key="gemini-test-not-a-real-key",
        youtube_api_key="yt-test-not-a-real-key" if with_key else None,
    )


@pytest.fixture(autouse=True)
def empty_cache():
    youtube.reset_youtube_cache()
    yield
    youtube.reset_youtube_cache()


def run_find(settings, query="chess coaching", country_code="US"):
    async def run():
        async with httpx.AsyncClient() as client:
            return await find_channels(client, settings, query, country_code)

    return asyncio.run(run())


def search_payload(*channel_ids):
    return {"items": [{"id": {"channelId": channel_id}} for channel_id in channel_ids]}


def channels_payload():
    return {
        "items": [
            {
                "id": CHANNEL_A,
                "snippet": {
                    "title": "Chess With Jane",
                    "description": "Weekly chess lessons from Springfield.",
                    "country": "US",
                    "customUrl": "@ChessWithJane",
                },
            },
            {"id": CHANNEL_B, "snippet": {"title": "Board Games Daily"}},
        ]
    }


def test_channels_found_with_details_and_key_in_header():
    with respx.mock() as router:
        search_route = router.get(url__regex=SEARCH_PATTERN).mock(
            return_value=httpx.Response(
                200, json=search_payload(CHANNEL_A, CHANNEL_B, "not-a-channel")
            )
        )
        router.get(url__regex=CHANNELS_PATTERN).mock(
            return_value=httpx.Response(200, json=channels_payload())
        )
        channels = run_find(make_settings())
    request = search_route.calls.last.request
    assert request.headers["x-goog-api-key"] == "yt-test-not-a-real-key"
    assert "yt-test-not-a-real-key" not in str(request.url)
    assert request.url.params["regionCode"] == "US"
    assert request.url.params["type"] == "channel"
    assert [channel.link for channel in channels] == [
        "https://www.youtube.com/@chesswithjane",
        f"https://www.youtube.com/channel/{CHANNEL_B}",
    ]
    assert channels[0].country == "US"
    assert channels[1].country is None


def test_repeated_search_served_from_cache():
    with respx.mock() as router:
        search_route = router.get(url__regex=SEARCH_PATTERN).mock(
            return_value=httpx.Response(200, json=search_payload(CHANNEL_A))
        )
        router.get(url__regex=CHANNELS_PATTERN).mock(
            return_value=httpx.Response(200, json=channels_payload())
        )
        run_find(make_settings())
        run_find(make_settings())
    assert search_route.call_count == 1


def test_no_matching_channels_skips_details_call():
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=SEARCH_PATTERN).mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        channels_route = router.get(url__regex=CHANNELS_PATTERN)
        assert run_find(make_settings()) == []
    assert not channels_route.called


def test_missing_key_raises_without_any_request():
    with pytest.raises(YouTubeError, match="no API key"):
        run_find(make_settings(with_key=False))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json={"error": {"message": "quota exceeded"}}),
        httpx.Response(200, content=b"not json"),
    ],
)
def test_bad_responses_raise_youtube_error(response):
    with respx.mock() as router:
        router.get(url__regex=SEARCH_PATTERN).mock(return_value=response)
        with pytest.raises(YouTubeError):
            run_find(make_settings())


def test_channel_link_rejects_odd_handles():
    assert channel_link(CHANNEL_A, "not a handle") == (
        f"https://www.youtube.com/channel/{CHANNEL_A}"
    )
