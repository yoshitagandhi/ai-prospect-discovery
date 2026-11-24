"""Merge prospects that are provably the same person or business.

Two prospects are merged only when they share an exact link: the same profile URL,
the same email address or the same phone number. For example, an Instagram prospect
and a business website that links to that same Instagram profile.

Names, locations and similar-looking handles are never used to merge, because two
different people can share a name. The trade-off is that the same person found on
two platforms with no shared link stays as two prospects.

Links can be compared as plain text because every profile link (including the ones
found on websites) already went through the platform profile_url() functions, and
emails and phones are already in one standard form (mailto: lower-case, tel: E.164).
"""

from collections.abc import Iterable

from app.schemas import Prospect, SocialLink

MAX_LINKS = 20


def link_key(link: SocialLink) -> str:
    """The text used to compare links. A trailing '/' is ignored as a safety check."""
    return link.link.rstrip("/")


def _unique_links(links: Iterable[SocialLink]) -> list[SocialLink]:
    """Links in their original order with repeated links removed."""
    unique: list[SocialLink] = []
    seen: set[str] = set()
    for link in links:
        key = link_key(link)
        if key not in seen:
            seen.add(key)
            unique.append(link)
    return unique


def _combine(values: Iterable[list[str] | None]) -> list[str] | None:
    """Join evidence or source lists in order without repeats. None if all are None."""
    combined: list[str] = []
    found_any = False
    for value in values:
        if value is None:
            continue
        found_any = True
        for item in value:
            if item not in combined:
                combined.append(item)
    return combined if found_any else None


def _merge_group(members: list[Prospect]) -> Prospect:
    """One prospect from a group. The first member is the best ranked one."""
    best = members[0]
    if len(members) == 1:
        return best
    links = _unique_links(link for member in members for link in member.social_link)
    scores = [m.relevance_score for m in members if m.relevance_score is not None]
    return Prospect(
        name=best.name,
        location=best.location,
        social_link=links[:MAX_LINKS],
        evidence=_combine(member.evidence for member in members),
        source_urls=_combine(member.source_urls for member in members),
        relevance_score=max(scores) if scores else None,
    )


def merge_duplicates(prospects: list[Prospect]) -> tuple[list[Prospect], int]:
    """Merge prospects that share a link. Returns the prospects and how many merged.

    The input must be in ranking order (best first); the output keeps that order.
    Each group remembers the positions of its members and the set of their links.
    A new prospect joins every group it shares a link with, so chains work too:
    if A and B share a link and B and C share another, all three become one.
    """
    group_members: list[list[int]] = []
    group_keys: list[set[str]] = []
    for position, prospect in enumerate(prospects):
        keys = {link_key(link) for link in prospect.social_link}
        matches = [index for index, found in enumerate(group_keys) if found & keys]
        if not matches:
            group_members.append([position])
            group_keys.append(keys)
            continue
        first = matches[0]
        for index in reversed(matches[1:]):
            group_members[first].extend(group_members.pop(index))
            group_keys[first] |= group_keys.pop(index)
        group_members[first].append(position)
        group_keys[first] |= keys

    merged = [
        _merge_group([prospects[position] for position in sorted(members)])
        for members in group_members
    ]
    return merged, len(prospects) - len(merged)
