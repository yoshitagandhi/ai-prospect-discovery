"""Tests for app/utils/rate_limit.py. A fake clock is passed in, so nothing waits."""

import pytest

from app.utils import rate_limit
from app.utils.rate_limit import check_rate_limit, reset_rate_limits


@pytest.fixture(autouse=True)
def empty_limiter():
    reset_rate_limits()
    yield
    reset_rate_limits()


def test_requests_within_limit_allowed_then_blocked():
    for second in range(5):
        assert check_rate_limit("ip:a", limit=5, now=100.0 + second) is None
    assert check_rate_limit("ip:a", limit=5, now=105.0) == 55


def test_allowed_again_after_window_passes():
    for _ in range(5):
        check_rate_limit("ip:a", limit=5, now=100.0)
    assert check_rate_limit("ip:a", limit=5, now=130.0) is not None
    assert check_rate_limit("ip:a", limit=5, now=160.0) is None


def test_keys_are_counted_separately():
    for _ in range(5):
        check_rate_limit("ip:a", limit=5, now=100.0)
    assert check_rate_limit("ip:a", limit=5, now=101.0) is not None
    assert check_rate_limit("ip:b", limit=5, now=101.0) is None


def test_blocked_requests_are_not_counted():
    for _ in range(5):
        check_rate_limit("ip:a", limit=5, now=100.0)
    for _ in range(10):
        check_rate_limit("ip:a", limit=5, now=110.0)
    assert check_rate_limit("ip:a", limit=5, now=160.0) is None


def test_idle_keys_are_forgotten(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_KEYS", 3)
    for name in ("a", "b", "c"):
        check_rate_limit("ip:" + name, limit=5, now=0.0)
    check_rate_limit("ip:new", limit=5, now=1000.0)
    assert set(rate_limit._request_times) == {"ip:new"}


def test_cleanup_uses_each_keys_own_window(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_KEYS", 2)
    check_rate_limit("daily", limit=1, window_seconds=86_400, now=0.0)
    check_rate_limit("ip:a", limit=5, now=0.0)
    check_rate_limit("ip:b", limit=5, now=120.0)
    assert (
        check_rate_limit("daily", limit=1, window_seconds=86_400, now=130.0) == 86_270
    )
