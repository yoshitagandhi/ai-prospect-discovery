"""YouTube: channels found through the official YouTube Data API v3.

Two calls per search: search.list (type=channel, its own quota bucket of 100 calls a
day) and channels.list (1 unit) for each channel's description, country and handle.
The API key is sent in the X-Goog-Api-Key header, never in the URL, so it cannot leak
into logs or error messages. Responses are cached in memory for cache_ttl_seconds.
regionCode only biases results towards a country; the channel's own country (when the
owner set one) is kept as location evidence.
"""

import re

import httpx
from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings

NAME = "YouTube"
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
MAX_QUERY_CHARS = 200
MAX_DESCRIPTION_CHARS = 1_000
CACHE_SIZE = 128
CHANNEL_ID_PATTERN = re.compile(r"UC[A-Za-z0-9_-]{22}")
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9._-]{3,30}")


class YouTubeError(Exception):
    """YouTube could not be used (no key, network error, quota or bad response)."""


class YouTubeChannel(BaseModel):
    """A channel with the public details used as evidence."""

    channel_id: str
    title: str
    description: str
    country: str | None = None
    link: str


class _SearchItemId(BaseModel):
    model_config = ConfigDict(extra="ignore")

    channelId: str | None = None


class _SearchItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: _SearchItemId


class _SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_SearchItem] = []


class _ChannelSnippet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    description: str = ""
    country: str | None = None
    customUrl: str | None = None


class _ChannelItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    snippet: _ChannelSnippet


class _ChannelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_ChannelItem] = []


_cache: TTLCache | None = None


def _get_cache(ttl_seconds: int) -> TTLCache:
    """Create the cache on first use, so its lifetime follows the settings."""
    global _cache
    if _cache is None:
        _cache = TTLCache(maxsize=CACHE_SIZE, ttl=max(ttl_seconds, 1))
    return _cache


def reset_youtube_cache() -> None:
    """Forget every cached result. Used by tests."""
    global _cache
    _cache = None


def channel_link(channel_id: str, custom_url: str | None) -> str:
    """Prefer the readable @handle link; fall back to the stable channel-ID link."""
    if custom_url and HANDLE_PATTERN.fullmatch(custom_url):
        return f"https://www.youtube.com/{custom_url.lower()}"
    return f"https://www.youtube.com/channel/{channel_id}"


async def _get(
    client: httpx.AsyncClient, settings: Settings, url: str, params: dict
) -> bytes:
    """GET one YouTube API endpoint and return the raw JSON body."""
    try:
        response = await client.get(
            url,
            params=params,
            headers={"X-Goog-Api-Key": settings.youtube_api_key.get_secret_value()},
            timeout=settings.http_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise YouTubeError("YouTube request timed out") from None
    except httpx.HTTPError:
        raise YouTubeError("YouTube request failed") from None
    if response.status_code != 200:
        raise YouTubeError(f"YouTube API returned HTTP {response.status_code}")
    return response.content


async def find_channels(
    client: httpx.AsyncClient,
    settings: Settings,
    query: str,
    country_code: str | None,
) -> list[YouTubeChannel]:
    """Search for channels matching the query, then fetch their public details."""
    if settings.youtube_api_key is None:
        raise YouTubeError("YouTube skipped: no API key configured")

    cache = _get_cache(settings.cache_ttl_seconds)
    cache_key = (query.lower(), country_code)
    if settings.cache_ttl_seconds > 0 and cache_key in cache:
        return cache[cache_key]

    search_params = {
        "part": "snippet",
        "type": "channel",
        "q": query[:MAX_QUERY_CHARS],
        "maxResults": settings.max_results_per_platform,
    }
    if country_code:
        search_params["regionCode"] = country_code

    try:
        search = _SearchResponse.model_validate_json(
            await _get(client, settings, SEARCH_URL, search_params)
        )
        channel_ids = []
        for item in search.items:
            channel_id = item.id.channelId
            if (
                channel_id
                and CHANNEL_ID_PATTERN.fullmatch(channel_id)
                and channel_id not in channel_ids
            ):
                channel_ids.append(channel_id)
        if not channel_ids:
            return []

        details = _ChannelsResponse.model_validate_json(
            await _get(
                client,
                settings,
                CHANNELS_URL,
                {"part": "snippet", "id": ",".join(channel_ids)},
            )
        )
    except ValidationError:
        raise YouTubeError("YouTube API returned an unexpected response") from None

    channels = [
        YouTubeChannel(
            channel_id=item.id,
            title=item.snippet.title,
            description=item.snippet.description[:MAX_DESCRIPTION_CHARS],
            country=item.snippet.country,
            link=channel_link(item.id, item.snippet.customUrl),
        )
        for item in details.items
        if CHANNEL_ID_PATTERN.fullmatch(item.id)
    ]
    if settings.cache_ttl_seconds > 0:
        cache[cache_key] = channels
    return channels
