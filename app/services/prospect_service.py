"""Prospect discovery pipeline.

Temporary placeholder: it returns no prospects until the discovery steps are built.
The function signature is final, so app/main.py will not need to change.
"""

from app.schemas import Prospect


async def find_prospects(
    query: str, include_evidence: bool = False
) -> tuple[list[Prospect], list[str]]:
    """Return (prospects, warnings) for an already validated query."""
    return [], []
