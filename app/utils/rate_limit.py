"""In-memory sliding-window rate limiting for the public demo.

Each key (one per client IP, a shared per-minute "global" key and a shared "daily"
key) remembers the times of its requests in its own window. A request is allowed while
fewer than `limit` requests were made in that window.

Known limits (documented in the README): the counts live in memory, so they reset when
the server restarts and assume a single server process, which is how the free Render
instance runs.
"""

import math
import time
from collections import deque

MAX_TRACKED_KEYS = 10_000

_request_times: dict[str, deque[float]] = {}
_windows: dict[str, float] = {}


def _drop_expired(times: deque[float], now: float, window_seconds: float) -> None:
    """Remove request times that are older than the window."""
    while times and now - times[0] >= window_seconds:
        times.popleft()


def _forget_idle_keys(now: float) -> None:
    """Delete keys with no recent requests so memory cannot grow without limit.

    Each key is checked against its own window, so a flood of new per-minute keys can
    never erase the history of the daily key.
    """
    for key in list(_request_times):
        _drop_expired(_request_times[key], now, _windows[key])
        if not _request_times[key]:
            del _request_times[key]
            del _windows[key]


def check_rate_limit(
    key: str, limit: int, window_seconds: float = 60.0, now: float | None = None
) -> int | None:
    """Record one request for `key` if it is allowed.

    Returns None when the request is allowed, otherwise the number of whole seconds
    to wait before trying again. `now` can be passed in tests instead of the real clock.
    """
    current_time = time.monotonic() if now is None else now
    if len(_request_times) >= MAX_TRACKED_KEYS:
        _forget_idle_keys(current_time)

    times = _request_times.setdefault(key, deque())
    _windows[key] = window_seconds
    _drop_expired(times, current_time, window_seconds)
    if len(times) >= limit:
        seconds_left = window_seconds - (current_time - times[0])
        return max(1, math.ceil(seconds_left))
    times.append(current_time)
    return None


def reset_rate_limits() -> None:
    """Forget every recorded request. Used by tests."""
    _request_times.clear()
    _windows.clear()
