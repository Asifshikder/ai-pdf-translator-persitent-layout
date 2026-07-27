"""Vertex AI (Gemini) translation client: batch-translate English strings to Bangla."""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types

from vertex_client import generate_content

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"


# Naming a term here is what keeps it stable across the document. Every chunk is
# an independent request, so nothing but the prompt carries a decision from one
# page to the next: left to itself the model rendered "symptom" as লক্ষণ 28 times
# and উপসর্গ 20 times in a single 40-page manual, and split "stress" three ways.
# Only terms measured to drift are pinned — ব্যায়াম and লবণ came back consistent
# on their own and need no help.
SYSTEM_PROMPT = """You are a Bangla (Bengali) translator working on a patient-facing health manual.
You receive a JSON array of English text snippets extracted from a PDF, and return their Bangla.

The document is a self-help manual about heart failure and recovering from a heart attack. It is
read by patients and by the family members looking after them — ordinary people, many of them
elderly. It is not written for doctors. Your Bangla must sound like a person explaining something
to them kindly, not like a translated document.

Language and register:
- The reader may have little formal schooling. Any ordinary person must be able to read
  each sentence once and understand it completely. Never use a transliterated English
  abbreviation (like "জিপি" for GP) or a technical or bookish word a general reader would
  not know — always choose the plainest everyday word that carries the meaning. Keep it
  clear and natural, never long-winded.
- Write Bangladeshi Bangla in the modern colloquial standard (চলিত). Never সাধু ভাষা.
- Always address the reader as আপনি. Never তুমি or তোমরা.
- Say it the way a Bangla speaker would say it, not word by word after the English. English idioms
  almost never survive a literal rendering: "slow and steady" is not "ধীর এবং স্থির".
- Choose the everyday word over the bookish one: খেয়াল রাখা not নিরীক্ষণ করা; বুঝতে পারা not সনাক্ত করা;
  দেওয়া হয়েছে not সরবরাহ করা হয়েছে; বুকলেট not পুস্তিকা; কাজ not কার্যকলাপ.
- Prefer active sentences. An English passive usually turns into stiff Bangla.
- Breaking one long English sentence into two shorter Bangla ones is good if it reads better.
- Watch for English words used figuratively, and translate the sense rather than the dictionary
  entry: the "space" you need to recover is mental room, not স্থান; a "tool" that is really a
  worksheet is not সরঞ্জাম; "some people find that..." means they experience it, not that they
  মনে করেন it.

Vocabulary:
- Keep the English term, written in Bangla script, where that is what people actually say. Do not
  invent purist coinages for words patients already know.
- Use these renderings every time they appear:
  heart failure → হার্ট ফেইলিওর (never হৃদরোগ, which means heart disease in general)
  symptom → উপসর্গ
  stress → স্ট্রেস
  medication, medicine → ওষুধ (this spelling)
  GP → ডাক্তার (never জিপি, which is meaningless to most readers)
  water → পানি (never জল)
  exercise → ব্যায়াম
  salt → লবণ
- For any other term that recurs, pick one Bangla rendering and use that one throughout.

Leave unchanged:
- Numbers, dates, email addresses, URLs, code, and abbreviations.
- Drug names (Bisoprolol, Varenicline) and proper nouns — names of people, companies, products — in
  Latin script, unless a common Bangla form exists.
- Any snippet with nothing to translate, such as one that is only a number or a symbol.

Output:
- Return a JSON array of strings with EXACTLY the same number of elements, in the same order.
- A snippet may start or end mid-sentence, because the page break split it there. Translate the
  fragment as a fragment; do not complete it into a whole sentence.
- Match the tone of the original: where it is warm and encouraging, stay warm and encouraging.
- Keep each translation close to its English in length, and do not pad it. But never make the Bangla
  awkward to save a few characters."""


# Large requests raise the odds of a count mismatch; translate in chunks.
CHUNK_SIZE = 40
MAX_ATTEMPTS = 3

# Retrying a rate limit without pausing just spends the attempts: the quota is
# still exhausted a millisecond later. Wait 2s, then 4s, with jitter so that
# requests which were throttled together do not all come back in lockstep.
RETRY_BACKOFF = 2.0
RETRY_JITTER = 0.5


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
    # Split into chunks and send them concurrently (up to 3 at a time) to the API.
    # Sequential chunk processing was the bottleneck: waiting for each API call
    # before sending the next. Now chunks from multiple pages can be in-flight.
    chunks = [texts[start : start + CHUNK_SIZE] for start in range(0, len(texts), CHUNK_SIZE)]

    if not chunks:
        return [], []

    # For a single chunk, no benefit to threading overhead.
    if len(chunks) == 1:
        translations, ok = _translate_chunk(chunks[0])
        return translations, ok

    results = [None] * len(texts)
    status = [False] * len(texts)

    # Send up to 2 chunks concurrently to API (reduced from 3 to avoid rate limits).
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, chunk in enumerate(chunks):
            start = i * CHUNK_SIZE
            futures[executor.submit(_translate_chunk, chunk)] = (i, start, len(chunk))

        for future in as_completed(futures):
            i, start, chunk_len = futures[future]
            translations, ok = future.result()
            for j, (trans, is_ok) in enumerate(zip(translations, ok)):
                results[start + j] = trans
                status[start + j] = is_ok

    return results, status


def _translate_chunk(texts: list[str]) -> tuple[list[str], list[bool]]:
    """Translate one chunk, halving it if the model answers unusably.

    A page is usually a single chunk, so writing one off leaves a whole page in
    English. When the model replies but the reply is unusable — a count mismatch,
    or JSON truncated because the Bangla ran past the output limit — the batch
    itself is the problem, and each half stands on its own. Splitting isolates
    the text that actually breaks it instead of discarding dozens of good
    translations alongside the one bad one.
    """
    translations, splittable = _request(texts)
    if translations is not None:
        return translations, [True] * len(texts)
    if not splittable or len(texts) == 1:
        logger.warning("Keeping %d text(s) in English — translation failed", len(texts))
        return list(texts), [False] * len(texts)

    mid = len(texts) // 2
    logger.info("Splitting a %d-text batch after an unusable reply", len(texts))
    left_text, left_ok = _translate_chunk(texts[:mid])
    right_text, right_ok = _translate_chunk(texts[mid:])
    return left_text + right_text, left_ok + right_ok


def _request(texts: list[str]) -> tuple[list[str] | None, bool]:
    """One translation request, retried with backoff when it never lands.

    Returns the translations, or None with a flag saying whether splitting the
    batch is worth trying.

    Only a request that failed to reach the model is retried: a rate limit
    clears, a socket resets, and the same batch can go again. An unusable
    *reply* is not asked for a second time: what makes a reply unusable is the
    batch — too many texts to count, too much Bangla to fit in one response —
    and asking again changes neither. So it returns straight away and lets the
    caller split the batch, which both isolates the offending text and halves
    the output the model has to produce in one reply.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=MODEL,
                contents=json.dumps(texts, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
                    # Near-greedy decoding takes the most probable token, which
                    # in translation is reliably the most literal one — the
                    # source of the word-for-word Bangla the prompt above works
                    # to avoid. Kept moderate on purpose: this is medical text
                    # and the facts have to survive the phrasing.
                    temperature=0.4,
                ),
            )
            translations = json.loads(response.text)
            if isinstance(translations, list) and len(translations) == len(texts):
                return [
                    t if isinstance(t, str) and t.strip() else orig
                    for t, orig in zip(translations, texts)
                ], False
            logger.warning(
                "Translation count mismatch: sent %d, got %d",
                len(texts),
                len(translations) if isinstance(translations, list) else -1,
            )
            return None, True
        # Truncated or malformed JSON: the model answered, so the batch is the
        # problem. Must be caught before the transport case below — retrying it
        # would only spend the attempts that a rate limit needs.
        except ValueError as exc:
            logger.warning("Unusable translation reply: %s", exc)
            return None, True
        except Exception as exc:
            logger.warning(
                "Translation request failed (attempt %d/%d): %s",
                attempt,
                MAX_ATTEMPTS,
                exc,
            )
        if attempt < MAX_ATTEMPTS:
            delay = RETRY_BACKOFF**attempt * (1 + random.random() * RETRY_JITTER)
            logger.info("Retrying in %.1fs", delay)
            time.sleep(delay)
    return None, False
