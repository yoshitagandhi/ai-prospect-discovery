"""Tests for scripts/run_test_queries.py.

httpx.MockTransport plays the API, so there is no network, and sleep is replaced so
the tests run instantly.
"""

import json

import httpx
import pytest

from scripts import run_test_queries as runner

BASE_URL = "http://api.test"
OK_BODY = {
    "success": True,
    "query": "q",
    "total_prospects": 2,
    "prospects": [
        {
            "name": "Jane Doe",
            "social_link": [
                {"platform": "LinkedIn", "link": "https://www.linkedin.com/in/jane"},
                {"platform": "X", "link": "https://x.com/jane"},
            ],
        },
        {
            "name": "Sam Lee",
            "social_link": [
                {"platform": "LinkedIn", "link": "https://www.linkedin.com/in/sam"}
            ],
        },
    ],
}


def fake_api(responses):
    """A client whose answers come from the given list, recording every request."""
    seen = []
    answers = iter(responses)

    def handler(request):
        seen.append(request)
        return next(answers)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, seen


def test_read_queries_skips_blank_lines(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("Chess players in Springfield\n\n  \nBakers in Paris \n", "utf-8")
    assert runner.read_queries(path) == [
        "Chess players in Springfield",
        "Bakers in Paris",
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Restaurant owners in Dubai UAE", "restaurant-owners-in-dubai-uae"),
        ("../../etc/passwd", "etc-passwd"),
        ("C:\\Windows\\x", "c-windows-x"),
        ("???", "query"),
        ("a" * 100, "a" * 60),
    ],
)
def test_slugs_are_safe_file_names(query, expected):
    assert runner.slugify(query) == expected


@pytest.mark.parametrize(
    "base_url", ["ftp://api.test", "file:///etc/passwd", "api.test", "http://"]
)
def test_only_http_base_urls_allowed(base_url):
    with pytest.raises(ValueError):
        runner.check_base_url(base_url)


def test_one_file_saved_per_query_with_the_real_response(tmp_path):
    client, seen = fake_api(
        [httpx.Response(200, json=OK_BODY), httpx.Response(200, json=OK_BODY)]
    )
    sleeps = []
    records = runner.run_all(
        client, BASE_URL, ["Chess players", "Bakers"], tmp_path, sleeps.append
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "1-chess-players.json",
        "2-bakers.json",
    ]
    saved = json.loads((tmp_path / "1-chess-players.json").read_text("utf-8"))
    assert saved["status_code"] == 200
    assert saved["response"] == OK_BODY
    assert saved["request"]["body"] == {"query": "Chess players"}
    assert saved == records[0]
    assert (
        str(seen[0].url) == "http://api.test/prospect-discovery?include_evidence=true"
    )
    assert json.loads(seen[0].content) == {"query": "Chess players"}
    assert sleeps == [runner.SECONDS_BETWEEN_QUERIES]


def test_429_waits_for_retry_after_and_retries_once(tmp_path):
    client, seen = fake_api(
        [
            httpx.Response(429, headers={"Retry-After": "7"}, json={"success": False}),
            httpx.Response(200, json=OK_BODY),
        ]
    )
    sleeps = []
    records = runner.run_all(client, BASE_URL, ["Chess"], tmp_path, sleeps.append)
    assert len(seen) == 2
    assert sleeps == [7]
    assert records[0]["status_code"] == 200


def test_second_429_is_saved_not_retried_again(tmp_path):
    too_many = httpx.Response(429, json={"success": False, "error": "Too many"})
    client, seen = fake_api([too_many, too_many])
    records = runner.run_all(client, BASE_URL, ["Chess"], tmp_path, lambda _: None)
    assert len(seen) == 2
    assert records[0]["status_code"] == 429


def test_errors_are_saved_not_hidden(tmp_path):
    error = {
        "success": False,
        "error": "Query understanding is temporarily unavailable",
    }
    client, _ = fake_api([httpx.Response(503, json=error)])
    records = runner.run_all(client, BASE_URL, ["Chess"], tmp_path, lambda _: None)
    assert records[0]["status_code"] == 503
    assert records[0]["response"] == error


def test_connection_failure_recorded(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = runner.run_all(client, BASE_URL, ["Chess"], tmp_path, lambda _: None)
    assert records[0]["status_code"] is None
    assert records[0]["response"] == {"note": "request failed: ConnectError"}


def test_retry_after_is_kept_in_a_safe_range():
    assert runner.retry_seconds(httpx.Response(429, headers={"Retry-After": "0"})) == 1
    assert (
        runner.retry_seconds(httpx.Response(429, headers={"Retry-After": "999"})) == 60
    )
    assert runner.retry_seconds(httpx.Response(429)) == runner.DEFAULT_RETRY_SECONDS


def test_summary_counts_match_the_records(tmp_path):
    client, _ = fake_api(
        [httpx.Response(200, json=OK_BODY), httpx.Response(503, json={"x": 1})]
    )
    records = runner.run_all(
        client, BASE_URL, ["Chess | players", "Bakers"], tmp_path, lambda _: None
    )
    summary = runner.summary_markdown(records, BASE_URL, "2026-11-25T10:00:00+00:00")
    rows = [line for line in summary.splitlines() if line.startswith(("| 1", "| 2"))]
    assert rows[0].startswith("| 1 | Chess \\| players | 200 | 2 | LinkedIn 2, X 1 |")
    assert rows[1].startswith("| 2 | Bakers | 503 |  |  |")
    assert "2026-11-25T10:00:00+00:00" in summary
