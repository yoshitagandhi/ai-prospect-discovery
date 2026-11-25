"""Run the spec's test queries against the running API and save the real responses.

Usage, with the API already running:
    python -m scripts.run_test_queries
    python -m scripts.run_test_queries --base-url https://your-demo.example.com

Reads samples/test_queries.txt (one query per line) and sends each query to
POST <base-url>/prospect-discovery?include_evidence=true. For every query it saves
samples/responses/<number>-<slug>.json with the request, the status code, the
measured duration and the API's own response body. samples/summary.md is then built
from those records.

Queries are 60 seconds apart because the free Groq tier allows about 8,000 tokens
per minute and one query can use most of that.

Nothing is invented: every saved value comes from a real response or the clock.
No API keys are used, because the script only talks to our own API.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = ROOT / "samples" / "test_queries.txt"
RESPONSES_DIR = ROOT / "samples" / "responses"
SUMMARY_FILE = ROOT / "samples" / "summary.md"
DEFAULT_BASE_URL = "http://127.0.0.1:4000"
ENDPOINT = "/prospect-discovery"
PARAMS = {"include_evidence": "true"}
SECONDS_BETWEEN_QUERIES = 60
REQUEST_TIMEOUT_SECONDS = 90
DEFAULT_RETRY_SECONDS = 60
MAX_RETRY_SECONDS = 60
SLUG_MAX_LENGTH = 60
NOT_ALLOWED_IN_SLUG = re.compile(r"[^a-z0-9]+")


def read_queries(path: Path) -> list[str]:
    """The queries in the file, one per line, with blank lines skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def slugify(text: str) -> str:
    """'Restaurant owners in Dubai UAE' -> 'restaurant-owners-in-dubai-uae'.

    Only a-z, 0-9 and '-' are kept, so a query can never create an odd file path.
    """
    slug = NOT_ALLOWED_IN_SLUG.sub("-", text.lower()).strip("-")
    return slug[:SLUG_MAX_LENGTH].strip("-") or "query"


def check_base_url(base_url: str) -> str:
    """The base URL without a trailing '/'. Only http and https are allowed."""
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("base URL must start with http:// or https:// and name a host")
    return base_url.rstrip("/")


def retry_seconds(response: httpx.Response) -> int:
    """Seconds to wait after a 429, from Retry-After, kept between 1 and 60."""
    value = response.headers.get("Retry-After", "")
    seconds = int(value) if value.isdigit() else DEFAULT_RETRY_SECONDS
    return min(max(seconds, 1), MAX_RETRY_SECONDS)


def _response_body(response: httpx.Response) -> object:
    """The API's JSON body, or a short note when the body is not JSON."""
    try:
        return response.json()
    except ValueError:
        return {"note": "response body was not JSON"}


def run_query(
    client: httpx.Client,
    base_url: str,
    query: str,
    sleep: Callable[[float], None],
) -> dict:
    """Send one query (retrying once after a 429) and return the record to save."""
    url = base_url + ENDPOINT
    record: dict = {
        "request": {
            "method": "POST",
            "url": f"{url}?include_evidence=true",
            "body": {"query": query},
        },
        "status_code": None,
        "duration_seconds": None,
        "response": None,
    }
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            response = client.post(url, params=PARAMS, json={"query": query})
        except httpx.HTTPError as error:
            record["response"] = {"note": f"request failed: {type(error).__name__}"}
            return record
        record["duration_seconds"] = round(time.perf_counter() - started, 2)
        if response.status_code == 429 and attempt == 1:
            sleep(retry_seconds(response))
            continue
        record["status_code"] = response.status_code
        record["response"] = _response_body(response)
        break
    return record


def platform_counts(record: dict) -> str:
    """'LinkedIn 2, X 1' from the links of the prospects in one response."""
    response = record.get("response")
    if not isinstance(response, dict):
        return ""
    counts = Counter(
        link.get("platform", "?")
        for prospect in response.get("prospects", [])
        for link in prospect.get("social_link", [])
    )
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))


def summary_markdown(records: list[dict], base_url: str, run_at: str) -> str:
    """A markdown table with one row per query, built only from the saved records."""
    lines = [
        "# Test query results",
        "",
        f"Run at {run_at} (UTC) against {base_url}{ENDPOINT}?include_evidence=true",
        "",
        "| # | Query | Status | Prospects | Links by platform | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for number, record in enumerate(records, start=1):
        query = record["request"]["body"]["query"].replace("|", "\\|")
        response = record.get("response")
        total = (
            response.get("total_prospects", "") if isinstance(response, dict) else ""
        )
        status = record["status_code"] or "no response"
        seconds = record["duration_seconds"] or ""
        lines.append(
            f"| {number} | {query} | {status} | {total} | "
            f"{platform_counts(record)} | {seconds} |"
        )
    return "\n".join(lines) + "\n"


def run_all(
    client: httpx.Client,
    base_url: str,
    queries: list[str],
    responses_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Run every query in order, save one JSON file each and return the records."""
    responses_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, query in enumerate(queries, start=1):
        if number > 1:
            sleep(SECONDS_BETWEEN_QUERIES)
        record = run_query(client, base_url, query, sleep)
        path = responses_dir / f"{number}-{slugify(query)}.json"
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        response = record["response"]
        total = (
            response.get("total_prospects", "-") if isinstance(response, dict) else "-"
        )
        print(
            f"[{number}/{len(queries)}] {query}: status {record['status_code']}, "
            f"{total} prospects, {record['duration_seconds']} s"
        )
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    """Run the queries and write the summary. Exit code 0 only if every query got 200."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    try:
        base_url = check_base_url(args.base_url)
    except ValueError as error:
        print(f"Error: {error}")
        return 2

    queries = read_queries(QUERIES_FILE)
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        records = run_all(client, base_url, queries, RESPONSES_DIR)
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    SUMMARY_FILE.write_text(
        summary_markdown(records, base_url, run_at), encoding="utf-8"
    )
    print(f"Saved {len(records)} responses and {SUMMARY_FILE.name}")
    return 0 if all(record["status_code"] == 200 for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
