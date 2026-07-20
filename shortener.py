"""Vertex AI (Gemini) shortening client: re-phrase Bangla text to take less space.

Last resort for the fix pipeline: when a segment still overflows its box after
the font has been shrunk to the floor, the text itself has to get shorter.
"""

import json
import logging
import re

from google.genai import types

from vertex_client import generate_content

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You shorten Bangla (Bengali) text that does not fit its layout box.
You receive a JSON array of objects: {"bn": <Bangla text>, "en": <the English source>}.

Rules:
- Return a JSON array of strings with EXACTLY the same number of elements, in the same order.
- Each string is a SHORTER Bangla rendering of the same meaning — aim for about 30% fewer characters.
- Preserve every fact, number, date, URL, email address, and abbreviation exactly.
- Keep the same tone and register. Re-phrase concisely; do not summarise information away.
- Never return English. Never return an empty string.
- If a snippet genuinely cannot be shortened, return it unchanged."""

BENGALI = re.compile(r"[ঀ-৿]")

CHUNK_SIZE = 20
MAX_ATTEMPTS = 3


def shorten_batch(items: list[dict]) -> list[str]:
    """Shorten a list of {"bn": ..., "en": ...} items.

    Returns one Bangla string per input, same order. On any failure the original
    Bangla is returned unchanged, so the caller always has renderable text.
    """
    results = []
    for start in range(0, len(items), CHUNK_SIZE):
        results.extend(_shorten_chunk(items[start : start + CHUNK_SIZE]))
    return results


def _acceptable(new: str, item: dict) -> bool:
    """A result is usable only if it is Bangla, non-empty, and actually shorter."""
    return (
        isinstance(new, str)
        and bool(new.strip())
        and len(new) < len(item["bn"])
        and bool(BENGALI.search(new))
    )


def _shorten_chunk(items: list[dict]) -> list[str]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=MODEL,
                contents=json.dumps(items, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
                    temperature=0.2,
                ),
            )
            shortened = json.loads(response.text)
            if isinstance(shortened, list) and len(shortened) == len(items):
                # A result that is not shorter is no use to the caller — it would
                # overflow again — so fall back to the text we already have.
                return [
                    new if _acceptable(new, item) else item["bn"]
                    for new, item in zip(shortened, items)
                ]
            logger.warning(
                "Shortening count mismatch (attempt %d/%d): sent %d, got %d",
                attempt,
                MAX_ATTEMPTS,
                len(items),
                len(shortened) if isinstance(shortened, list) else -1,
            )
        except Exception:
            logger.exception(
                "Shortening request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Giving up after %d attempts — keeping original text", MAX_ATTEMPTS)
    return [item["bn"] for item in items]
