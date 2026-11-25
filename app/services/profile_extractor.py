"""Public contact details from business websites.

Only websites that passed relevance scoring are read, at most max_pages_to_fetch per
request and MAX_CONCURRENT_FETCHES at a time. For each one:
1. robots.txt is checked, then the single page found by the search is fetched through
   safe_http (SSRF checks, size limit, timeout, redirects re-checked). No crawling.
2. The HTML is parsed with BeautifulSoup. Contacts come from mailto:, tel: and WhatsApp
   links and from the visible text; social profiles come from links to the platforms.
   Every value is validated (validators.py, platform profile_url rules, SocialLink).
3. Script and style content is ignored, so contact details hidden in code do not count.

A site that is blocked, unsafe, too large or failing is skipped with a warning; a site
with no valid contact or profile link is skipped too. Values are never guessed.

A page that looks like a list of several businesses is skipped, because its contacts
cannot be trusted to belong to one prospect: more than one profile on the same social
platform (for example two LinkedIn profiles), or more than MAX_DIRECT_CONTACTS emails,
phone numbers or WhatsApp numbers. A real business with a personal and a company
profile, or with many branch numbers, is skipped too; that is the safer mistake.
"""

import asyncio
from collections import Counter
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.platforms import SEARCH_PLATFORMS, youtube
from app.platforms.web_search import SearchResult
from app.schemas import SocialLink
from app.services.search_service import Candidate
from app.utils import safe_http
from app.utils.normalizers import clean_display_name
from app.utils.safe_http import FetchError, UnsafeUrlError
from app.utils.validators import (
    email_to_link,
    extract_emails,
    extract_phones,
    parse_phone,
    parse_whatsapp_link,
    phone_to_link,
)

WEBSITE_PLATFORM = "Website"
MAX_CONCURRENT_FETCHES = 5
MAX_LINKS = 20
PROFILE_MODULES = (*SEARCH_PLATFORMS, youtube)
HIDDEN_TAGS = ("script", "style", "noscript", "template")
DIRECT_CONTACT_PLATFORMS = {"Email", "Phone", "WhatsApp"}
MAX_DIRECT_CONTACTS = 3


class SiteContacts(BaseModel):
    """Validated links found on one business website."""

    candidate: Candidate
    page_url: str
    links: list[SocialLink]


def site_name(url: str) -> str | None:
    """'https://www.coyarestaurant.com/dubai/contact' -> 'coyarestaurant.com'."""
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.") or None


def website_candidate(result: SearchResult) -> Candidate | None:
    """Turn a business-website search hit into a candidate for relevance scoring.

    The name comes from the page title. When the title is only a generic word such as
    "Contact" or "Home", the site's own domain is used as the name instead.
    """
    name = clean_display_name(result.title) or site_name(result.url)
    if name is None:
        return None
    return Candidate(
        platform=WEBSITE_PLATFORM,
        name=name,
        profile_url=result.url,
        evidence=result.snippet,
        source_url=result.url,
        found_via="web_search",
    )


def _add(found: list[tuple[str, str]], platform: str, link: str) -> None:
    """Keep the first occurrence of each (platform, link) pair, in page order."""
    if (platform, link) not in found:
        found.append((platform, link))


def parse_contacts(
    html: str, page_url: str, country_code: str | None
) -> list[SocialLink]:
    """Extract validated Email, Phone, WhatsApp and social profile links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(HIDDEN_TAGS):
        tag.decompose()

    found: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        lowered = href.lower()
        if lowered.startswith("mailto:"):
            address = href[len("mailto:") :].split("?")[0]
            for email in extract_emails(address):
                _add(found, "Email", email_to_link(email))
        elif lowered.startswith("tel:"):
            phone = parse_phone(href, country_code)
            if phone:
                _add(found, "Phone", phone_to_link(phone))
        elif lowered.startswith(("http://", "https://", "/")):
            absolute = urljoin(page_url, href)
            whatsapp = parse_whatsapp_link(absolute)
            if whatsapp:
                _add(found, "WhatsApp", whatsapp)
                continue
            for module in PROFILE_MODULES:
                profile = module.profile_url(absolute)
                if profile:
                    _add(found, module.NAME, profile)
                    break

    text = soup.get_text(" ")
    for email in extract_emails(text):
        _add(found, "Email", email_to_link(email))
    for phone in extract_phones(text, country_code):
        _add(found, "Phone", phone_to_link(phone))

    links: list[SocialLink] = []
    for platform, link in found:
        try:
            links.append(SocialLink(platform=platform, link=link))
        except ValidationError:
            continue
    return links[:MAX_LINKS]


def looks_like_list_page(links: list[SocialLink]) -> bool:
    """True if the links look like they belong to several businesses.

    That is: more than one profile on the same social platform, or more than
    MAX_DIRECT_CONTACTS emails, phones or WhatsApp numbers.
    """
    for platform, count in Counter(link.platform for link in links).items():
        limit = MAX_DIRECT_CONTACTS if platform in DIRECT_CONTACT_PLATFORMS else 1
        if count > limit:
            return True
    return False


async def _read_site(
    client: httpx.AsyncClient,
    settings: Settings,
    candidate: Candidate,
    country_code: str | None,
    semaphore: asyncio.Semaphore,
) -> SiteContacts:
    """Check robots.txt, fetch the page safely and parse its contacts."""
    async with semaphore:
        if not await safe_http.is_allowed_by_robots(
            client, candidate.profile_url, settings
        ):
            raise FetchError("blocked by robots.txt")
        page = await safe_http.fetch_page(client, candidate.profile_url, settings)
    links = parse_contacts(page.text, page.final_url, country_code)
    return SiteContacts(candidate=candidate, page_url=page.final_url, links=links)


async def extract_contacts(
    client: httpx.AsyncClient,
    settings: Settings,
    candidates: list[Candidate],
    country_code: str | None,
) -> tuple[list[SiteContacts], list[str]]:
    """Read the kept business websites. Returns sites with links, plus warnings."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    selected = candidates[: settings.max_pages_to_fetch]
    results = await asyncio.gather(
        *(
            _read_site(client, settings, candidate, country_code, semaphore)
            for candidate in selected
        ),
        return_exceptions=True,
    )

    sites: list[SiteContacts] = []
    warnings: list[str] = []
    for candidate, result in zip(selected, results, strict=True):
        host = urlsplit(candidate.profile_url).hostname or "website"
        if isinstance(result, (FetchError, UnsafeUrlError)):
            warnings.append(f"Website {host} skipped: {result}")
        elif isinstance(result, BaseException):
            raise result
        elif not result.links:
            warnings.append(f"Website {host}: no public contact details found")
        elif looks_like_list_page(result.links):
            warnings.append(
                f"Website {host} skipped: looks like a list of several businesses"
            )
        else:
            sites.append(result)
    return sites, warnings
