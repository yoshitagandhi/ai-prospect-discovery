AI Prospect Discovery
=====================

A FastAPI service that turns a plain-English request such as
"Restaurant owners in Dubai UAE" into a list of **real, public prospects** with their
public profile links, and, for businesses, the public email, phone and WhatsApp details
found on their own website.

Every prospect comes with evidence: quotes from public pages that prove the interest and
the location, plus the URL each one came from. Nothing is generated or guessed. Fewer
correct prospects are returned rather than many weak ones.

**Live demo:** https://ai-prospect-discovery-pl6l.onrender.com/docs
(free instance: the first request after 15 idle minutes takes about a minute to wake
up; limited to 5 requests per minute per visitor and 150 per day in total).


How it works
------------

`POST /prospect-discovery` runs six steps:

1. **Understand the query.** One small LLM call turns the text into structured JSON
   (interest, audience, city, state, country, individual or business, related terms).
   The reply is validated with Pydantic and checked against the query words. Results
   are cached for an hour.
2. **Discover candidates.** In parallel: one Tavily web search per platform
   (LinkedIn, Instagram, Facebook, X, Threads), the official YouTube Data API for
   channels, and, for business queries, a search for business websites. Social pages
   are never fetched; only search titles and snippets are used.
3. **Check relevance.** One batched LLM call judges every candidate and must quote the
   exact words that prove the interest and the place. Our code then checks that each
   quote really appears in that candidate's text and proves what it claims. A candidate
   is kept only if the interest, the audience, a specific person or business, and (when
   the query names a place) the most specific place named are all verified.
4. **Read business websites.** Only kept websites: robots.txt first, then one page,
   through an SSRF-safe fetcher. Emails, phones, WhatsApp links and linked profiles are
   extracted and validated. Pages that look like lists of several businesses are skipped.
5. **Merge duplicates.** Prospects are merged only when they share an exact link
   (profile URL, email or phone). Never on name alone.
6. **Cap and return** at most 25 prospects, best first.

```
app/
  main.py               FastAPI app, error handlers, rate limits, body size limit
  config.py             settings from environment variables (frozen, keys as SecretStr)
  schemas.py            request and response models with per-platform link checks
  platforms/            one module per platform (URL rules), Tavily and YouTube clients
  services/             query, search, relevance, website contacts, dedup, pipeline, LLM
  utils/                SSRF-safe fetching, rate limiting, validators, name clean-up
scripts/run_test_queries.py   runs the sample queries and saves the real responses
samples/                      saved responses, summary and LLM usage log
tests/                        362 tests, all external APIs replaced (no network, no keys)
```


Setup
-----

Requirements: Python 3.14 and four free API keys (no credit card needed for any of them).

| Variable | Required | Where to get it (free) |
|---|---|---|
| `TAVILY_API_KEY` | yes | [app.tavily.com](https://app.tavily.com), 1,000 credits a month |
| `GEMINI_API_KEY` | yes | [Google AI Studio](https://aistudio.google.com), free tier |
| `GROQ_API_KEY` | recommended | [console.groq.com](https://console.groq.com), LLM fallback |
| `YOUTUBE_API_KEY` | recommended | Google Cloud console, YouTube Data API v3 |

Optional: `LLM_PRIMARY_MODEL` (default `gemini-3.5-flash-lite`), `LLM_FALLBACK_MODEL`
(default `openai/gpt-oss-20b`), `LOG_LEVEL` (default `INFO`; `DEBUG` logs one line per
candidate explaining why it was kept or dropped).

Windows PowerShell:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then put your keys in `.env`. It is listed in `.gitignore` and is never committed.
The service refuses to start if a required key is missing, and the error names the
missing variable without printing any value.


Run
---

```
uvicorn app.main:app --port 4000
```

* Swagger UI: http://localhost:4000/docs (the root URL redirects there)
* Health check: `GET /health`


Example request and response
----------------------------

curl:

```
curl -X POST http://localhost:4000/prospect-discovery \
  -H "Content-Type: application/json" \
  -d '{"query": "Restaurant owners in Dubai UAE"}'
```

PowerShell:

```
Invoke-RestMethod -Method Post -Uri "http://localhost:4000/prospect-discovery" -ContentType "application/json" -Body '{"query": "Restaurant owners in Dubai UAE"}' | ConvertTo-Json -Depth 8
```

Response (real, from the saved run of 25 Nov 2026; two of the six prospects shown, full
file in `samples/responses/5-restaurant-owners-in-dubai-uae.json`):

```json
{
  "success": true,
  "query": "Restaurant owners in Dubai UAE",
  "total_prospects": 6,
  "prospects": [
    {
      "name": "Shahbaz Shahid Khan",
      "location": "Dubai, UAE",
      "social_link": [
        {"platform": "LinkedIn", "link": "https://www.linkedin.com/in/shahbaz-shahid-khan-4b10a6a0"}
      ]
    },
    {
      "name": "BHATKAL RESTAURANT",
      "location": "Dubai, UAE",
      "social_link": [
        {"platform": "Email", "link": "mailto:hello@haiuae.com"},
        {"platform": "Phone", "link": "tel:+97142267545"}
      ]
    }
  ]
}
```

**Evidence mode.** Add `?include_evidence=true` to see, for each prospect, the verified
`evidence` quotes, `source_urls` and `relevance_score`, plus `warnings` (what each step
found) and `parsed_query`. The default response contains only `success`, `query`,
`total_prospects` and `prospects`.

**Errors** always return `{"success": false, "error": "..."}` without stack traces:
422 (invalid input, or a query that does not ask for people or businesses), 413/411
(body too large or missing length), 429 (rate limit, with `Retry-After`), 503 (no LLM
provider available, every search source down, or the request took longer than 60
seconds; the service then returns no results rather than guessed or empty ones) and
500 (unexpected error, logged).


Discovery sources and supported platforms
-----------------------------------------

| Platform | How it is found | What is returned |
|---|---|---|
| LinkedIn | Tavily search limited to linkedin.com | `/in/` and `/company/` profile links |
| Instagram | Tavily search limited to instagram.com | profile links (post links are dropped) |
| Facebook | Tavily search limited to facebook.com | page and profile links |
| X | Tavily search limited to x.com and twitter.com | author profile (post links become the author) |
| Threads | Tavily search limited to threads.com and threads.net | author profile |
| YouTube | official YouTube Data API v3 | channel links, channel country |
| Email, Phone, WhatsApp | read from the prospect's own business website | `mailto:`, `tel:` (E.164), `wa.me` links |

Rules that apply to every source:

* Only official APIs and public pages. Nothing behind a login; no CAPTCHA or rate-limit
  bypassing. Social network pages are never fetched (their terms forbid scraping);
  their search titles and snippets are enough to judge relevance.
* Contact details are never generated. An email or phone appears only if it was found
  on a public page, and that page's URL is recorded in `source_urls`.
* Websites are fetched only after robots.txt allows it.

**Not supported, and why**

| Source | Reason |
|---|---|
| TikTok, Pinterest | terms forbid scraping; no free official search API for profiles |
| Reddit, Quora | forums: posts are anonymous discussions, not prospects |
| Google Search | the Custom Search JSON API is closed to new customers (since 20 Jan 2026) and is discontinued on 1 Jan 2027 |
| Directories (Yelp, Justdial, Tripadvisor, Clutch and similar) | list many businesses; contacts cannot be tied to one prospect |


Test results on five sample queries
-----------------------------------

`python -m scripts.run_test_queries` runs the queries in `samples/test_queries.txt`
against the running API and saves each full response in `samples/responses/`, plus
`samples/summary.md`. The LLM usage of that run is in `samples/llm_usage.txt`.

Final run, 25 Nov 2026 (local, evidence mode):

| Query | Prospects | Hand check |
|---|---|---|
| Football lovers in Washington DC in USA | 2 | both correct: a fan-club member and a DC football channel |
| Digital marketing agency owners in New York USA | 17 | mostly correct agencies and owners; 2 borderline (New York State, not the city) |
| Startup founders in London UK | 0 | search found mainly lists, communities and events; all correctly rejected |
| Cricket lovers in Ahmedabad India | 0 | the fallback LLM judged only 4 of 27 candidates and kept none (the response warns about this) |
| Restaurant owners in Dubai UAE | 6 | all 6 are restaurants or their owners |

The zero results are honest: when public sources do not give enough evidence, the
service returns fewer results rather than padding the list.

How the results improved during development (all from real runs, kept in git history):
the first runs kept news agencies, software vendors and directory pages; fixes added
quote verification, "is the audience, not serves it", "specific person or business",
list-page detection and "prove the most specific place".


LLM comparison and recommendation
---------------------------------

The LLM does two jobs per request: query understanding (small: about 330 to 700 input
tokens) and the relevance check (one batched call: about 2,700 to 5,000 input tokens).

**MEASURED** = from our own logs (`samples/llm_usage.txt` and earlier runs).
**ESTIMATED** = from the provider's published price list, using our measured token
counts; not run by us. Prices checked on 26 Nov 2026.

| Model | Status | Query call | Relevance call | Price per 1M tokens (input / output) |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (primary) | MEASURED | 5 of 5 succeeded (some after one retry); 332 / 106-127 tokens; 3.8-11 s | 1 of 5 succeeded, 4 got 503 "high demand"; 3,101 / 1,978 tokens; 5.6 s | free tier; paid $0.30 / $2.50 |
| Groq gpt-oss-20b (fallback) | MEASURED | about 692 / 87-184 tokens; 0.7-1.4 s | 4 of 4 succeeded; 2,764-4,841 / 307-1,760 tokens; 1.4-3.6 s | free tier (8,000 tokens per minute); paid $0.075 / $0.30 |
| Gemini 3.1 Flash-Lite (first choice, replaced) | MEASURED | 0 of 10 on 25 Nov (503 or timeout) | 400 "schema too complex", then 503 | free tier; paid $0.25 / $1.50 |
| OpenAI GPT-6 Luna | ESTIMATED | not run | not run | $0.10 / $0.50 |
| Anthropic Claude Haiku 5.5 | ESTIMATED | not run | not run | $0.10 / $0.50 (prompts under 100K tokens) |

Quality, from the hand checks:

* **Gemini 3.5 Flash-Lite** followed the rules best: its relevance runs kept only real
  restaurant owners and real fans.
* **Groq gpt-oss-20b** is fast and cheap but less reliable on this judgement task: in
  some runs it kept an embassy, a sports team and an accelerator, and twice it returned
  judgements for only some candidates (the response then says so in `warnings`).
  At default reasoning it also used about 6,500 output tokens per relevance call, which
  exceeded its free 8,000 tokens per minute; low reasoning fixed that.

**Cost per query** (typical: 4,300 input and 2,100 output tokens in total)

| Item | Cost per query |
|---|---|
| Gemini 3.5 Flash-Lite, paid tier | about $0.0066 (ESTIMATED from measured tokens) |
| Groq gpt-oss-20b, paid tier | about $0.0010 |
| GPT-6 Luna or Claude Haiku 5.5 | about $0.0015 each (ESTIMATED) |
| Tavily, 5-6 basic searches (1 credit each) | $0 on the free 1,000 credits (about 170 queries a month); $0.040-0.048 pay as you go |
| YouTube, search.list + channels.list | $0; 1 of the 100 free channel searches a day, plus 1 of 10,000 daily quota units |

Search, not the LLM, is the main cost.

**Recommendation**

* **For this demo (free):** Gemini 3.5 Flash-Lite as primary with Groq gpt-oss-20b as
  fallback, as built. The fallback keeps the service answering when Gemini is overloaded.
* **For production:** a paid small model with a service-level agreement. GPT-6 Luna or
  Claude Haiku 5.5 cost about $0.0015 per query at our token counts, less than a third
  of one Tavily search. The relevance step is where quality matters, so it is worth
  testing both on the five saved queries before choosing; the code only needs a new
  `_call_*` function in `llm_service.py`.

Price sources (checked 26 Nov 2026):
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) (Gemini 3.5
Flash-Lite price confirmed through [FavTray](https://favtray.com/blog/gemini-api-pricing)
and [BenchLM](https://benchlm.ai/google/api-pricing), because our tool could not read
that row of Google's page),
[Groq gpt-oss-20b](https://console.groq.com/docs/model/openai/gpt-oss-20b),
[Groq rate limits](https://console.groq.com/docs/rate-limits),
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[Tavily credits](https://docs.tavily.com/documentation/api-credits).


Cost and quota optimisations already built in
---------------------------------------------

* Query understanding is cached for an hour, so a repeated query costs no LLM call.
* Relevance is **one** batched call for up to 40 candidates, not one call per candidate.
* Evidence per candidate is cut to 300 characters before it is sent to the LLM.
* Groq gpt-oss runs with `reasoning_effort="low"` and a 3,500-token output cap.
* The schema sent to the LLM has no length or number limits (they made Gemini reject it);
  Pydantic still enforces every limit on the reply.
* Only one retry, only on Gemini 503; quota errors are never retried.
* Social pages are never fetched; at most 10 websites per request, 5 at a time.
* Per-IP, global per-minute and global daily limits protect the free quotas on the
  public demo.


Security
--------

* **Secrets:** only in environment variables, loaded as `SecretStr`; `.env` is git-ignored,
  `.env.example` has empty values, and API errors are logged with keys redacted.
* **SSRF:** fetched URLs must be http or https; every DNS answer must be a public
  address (private, loopback and link-local are blocked); redirects are followed by hand
  and re-checked (at most 3); 10-second timeout; 1 MB size limit.
* **Prompt injection:** web text is sent to the LLM only inside `<untrusted_data>` tags
  (look-alike tags are stripped), the prompt says it is data, and every reply is
  validated with Pydantic. Quotes are then checked against the real text by our code.
* **Untrusted data:** never passed to `eval`, `exec`, shell commands or file paths.
  Names are logged with `repr()` so they cannot fake log lines.
* **Input limits:** 10,000-byte bodies, 3-300 character queries, control characters
  rejected, extra fields forbidden.
* **Regex safety:** emails are found by splitting words, then a bounded full match, then
  email-validator; phones use the phonenumbers matcher. No pattern can backtrack badly.
* **Abuse:** 5 requests per minute per IP, 30 per minute in total and 150 per day in
  total. Each request uses about 6 Tavily credits, so the daily cap keeps the free
  1,000 monthly credits from being used up in minutes.
* **Time limit:** a whole request stops after 60 seconds (`request_deadline_seconds`).
* **Errors:** clear message and status code; no stack traces, paths or keys.
* **Personal data:** nothing is stored except the parsed-query cache (one hour); search
  results are not cached.


Testing and quality checks
--------------------------

```
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
bandit -r app scripts
pip-audit -r requirements.txt
```

362 tests. Every external API (Tavily, YouTube, Gemini, Groq, websites) is replaced
with fakes, so the tests need no network and no keys.


Known limits
------------

* **Free-tier LLMs.** Gemini 3.5 Flash-Lite often answers 503 for the larger relevance
  call; the Groq fallback is less reliable at the relevance rules (see above). Results
  for the same query can differ between runs.
* **Search results vary.** Tavily returns different pages from run to run, so a real
  restaurant can be a candidate in one run and missing in the next.
* **Individuals rarely state their city.** Fans and enthusiasts seldom write
  "Ahmedabad" in a public bio, so queries about individuals return few prospects.
* **Single-business pages on directory sites.** A directory page about one restaurant
  passes the list-page check, and its email can belong to the directory
  (seen once: `hello@haiuae.com` for Bhatkal Restaurant).
* **City words match one at a time.** "New York" is matched word by word, so a quote
  with only "New York" State places (such as Cazenovia) also passes.
* **Websites linking several profiles are skipped,** including a real business that
  links both a personal and a company profile: the safer mistake.
* **Rate limits and caches are in memory,** so they reset on restart and are not
  shared between several server processes.
* **DNS rebinding:** the address is checked before the request, and the HTTP client
  resolves it again; a hostile DNS server could answer differently the second time.
* **Gemini free tier:** Google may use free-tier prompts to improve its products. The
  prompts contain only public search snippets.


Incidents during development (all real)
---------------------------------------

* Gemini 3.1 Flash-Lite returned 503 "high demand" on most calls for two days, then was
  replaced by Gemini 3.5 Flash-Lite.
* Gemini rejected the relevance schema with 400 "too many states"; fixed by sending a
  schema without length and number limits (Pydantic still checks them).
* The Gemini SDK retried silently; now set to one attempt, with our own single retry on 503.
* Groq gpt-oss-120b was blocked at organisation and project level in the Groq console,
  so the fallback is gpt-oss-20b.
* Groq returned 429 and 413 (8,000 tokens per minute) at default reasoning; fixed with
  low reasoning and a smaller output cap.
* An early keyword-matching fallback returned vendors and news articles when the LLM was
  down; it was removed, and the service now answers 503 instead.


Deployment
----------

Live on Render (free tier, Oregon):

* API docs: https://ai-prospect-discovery-pl6l.onrender.com/docs
* Health: https://ai-prospect-discovery-pl6l.onrender.com/health

Settings used:

* Build command: `pip install -r requirements.txt`
* Start command:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"`
* Health check path: `/health`
* Keys: set as environment variables in the Render dashboard, never in the repository.
* Python: 3.14, from `.python-version`.

`--proxy-headers` lets the per-IP limit see the visitor's address behind Render's
proxy. A visitor can fake that header, which is why the global per-minute and daily
limits exist as well.

Checked after deployment (26 Nov 2026): `/health` returned `ok`, `/docs` loaded, and
"Restaurant owners in Dubai UAE" returned 5 prospects, including a restaurant whose
public email, phone and WhatsApp number were read from its own contact page.
