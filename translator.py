"""Vertex AI (Gemini) translation client: batch-translate English strings to Bangla."""

import logging

from google.genai import types

from vertex_client import get_client

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are a professional English-to-Bangla (Bengali) translator.
You receive a JSON array of English text snippets extracted from a PDF document.
Translate each snippet to natural, fluent Bangla.

Rules:
- Return a JSON array of strings with EXACTLY the same number of elements, in the same order.
- Keep numbers, dates, email addresses, URLs, code, and abbreviations unchanged.
- Keep proper nouns (names of people, companies, products) in English unless a common Bangla form exists.
- Preserve the tone and register of the original.
- If a snippet is not translatable (e.g. it is only a number or symbol), return it unchanged.
- Prefer concise phrasing: the translation must fit in the same space as the original."""


# Large requests raise the odds of a count mismatch; translate in chunks.
CHUNK_SIZE = 40
MAX_ATTEMPTS = 3


def translate_batch(texts: list[str]) -> list[str]:
    """Translate a list of English strings to Bangla.

    Returns one translation per input, same order. On any failure the original
    texts are returned unchanged so the PDF is never corrupted.
    """
    translations, _ = translate_batch_status(texts)
    return translations


def translate_batch_status(texts: list[str]) -> tuple[list[str], list[bool]]:
    """Translate, and report per text whether its request actually succeeded.

    Returned English can mean two very different things — the request failed, or
    the text was untranslatable to begin with (an acronym, a URL) — and they are
    indistinguishable from the text alone. The flag tells them apart, so callers
    can retry the first without churning on the second.
    """
    results = []
    status = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start : start + CHUNK_SIZE]
        translations, ok = _translate_chunk(chunk)
        results.extend(translations)
        status.extend([ok] * len(chunk))
    return results, status


def _translate_chunk(texts: list[str]) -> tuple[list[str], bool]:
    import json

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_client().models.generate_content(
                model=MODEL,
                contents=json.dumps(texts, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
                    temperature=0.1,
                ),
            )
            translations = json.loads(response.text)
            if isinstance(translations, list) and len(translations) == len(texts):
                return [
                    t if isinstance(t, str) and t.strip() else orig
                    for t, orig in zip(translations, texts)
                ], True
            logger.warning(
                "Translation count mismatch (attempt %d/%d): sent %d, got %d",
                attempt,
                MAX_ATTEMPTS,
                len(texts),
                len(translations) if isinstance(translations, list) else -1,
            )
        except Exception:
            logger.exception(
                "Translation request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Giving up after %d attempts — keeping original text", MAX_ATTEMPTS)
    return texts, False
