"""Vertex AI (Gemini) translation client: batch-translate English strings to Bangla."""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types

from vertex_client import generate_content

logger = logging.getLogger(__name__)

MODEL = "gemini-3-flash-preview"


# Naming a term here is what keeps it stable across the document. Every chunk is
# an independent request, so nothing but the prompt carries a decision from one
# page to the next: left to itself the model rendered "symptom" as লক্ষণ 28 times
# and উপসর্গ 20 times in a single 40-page manual, and split "stress" three ways.
# Only terms measured to drift are pinned — ব্যায়াম and লবণ came back consistent
# on their own and need no help.
#
# Written for a reasoning model (gemini-3-flash-preview), which differs from how
# 2.5-flash was prompted in three ways that matter:
#   * It is given a procedure — understand, say it, read it back — because the
#     thinking pass is where literal word-order Bangla gets caught, and a rule it
#     can check its own draft against is worth more than another adjective.
#   * The naturalness rules are demonstrated, not just asserted. A ✗/✓ pair fixes
#     what "say it the way a Bangla speaker would" cannot: the model agrees with
#     the instruction and then writes the ✗ anyway.
#   * It follows instructions literally and infers less, so anything left implied
#     gets dropped. Latin digits are now stated outright — the older prompt only
#     said "leave numbers unchanged" and 999 still came back as ৯৯৯.
SYSTEM_PROMPT = """You translate a patient-facing health manual from English into Bangla (Bengali).

You receive a JSON array of English snippets taken from a PDF. Return a JSON array of their Bangla:
same number of elements, same order.

WHO IS READING THIS
Someone in Bangladesh living with heart failure or recovering from a heart attack, and the family
looking after them. Many are elderly; some have little formal schooling; none of them are doctors.
Your Bangla has to sound like a kind, plain-spoken person explaining something to them — not like a
document that has been translated. Any ordinary reader must understand each sentence on one reading.

HOW TO TRANSLATE EACH SNIPPET
1. Work out what the English is actually telling the reader.
2. Write that in Bangla the way a Bangla speaker would say it out loud. Do not walk through the
   English word by word.
3. Read your Bangla back. If nobody would say it that way, write it again.

REGISTER
- Bangladeshi Bangla, modern colloquial চলিত. Never সাধু ভাষা.
- Address the reader as আপনি. Never তুমি or তোমরা.
- The plainest everyday word that carries the meaning, every time. Never a bookish or technical word
  a general reader would not know, and never a transliterated English abbreviation.
- Everyday over bookish: খেয়াল রাখা not নিরীক্ষণ করা; বুঝতে পারা not সনাক্ত করা; দেওয়া হয়েছে not
  সরবরাহ করা হয়েছে; বুকলেট not পুস্তিকা; কাজ not কার্যকলাপ.
- Prefer active sentences. An English passive usually turns into stiff Bangla.
- Follow Bangla word order, not English. Bangla puts the condition first and the verb last, and
  drops possessives that English repeats. Splitting one long English sentence into two Bangla ones
  is good if it reads better.
- Match the tone of the original: where it is warm and encouraging, stay warm and encouraging.

NATURAL, NOT LITERAL — this is what separates a good translation here from a bad one
- "Speak to your GP if your ankles start to swell."
    ✗ আপনার গোড়ালি ফুলতে শুরু করলে আপনার জিপির সাথে কথা বলুন।   (জিপি means nothing to the reader;
      the second আপনার is English grammar showing through)
    ✓ পায়ের গোড়ালি ফুলতে শুরু করলে ডাক্তারের সাথে কথা বলুন।
- "Take your medicines every day, even on days when you feel well."
    ✗ প্রতিদিন আপনার ঔষধসমূহ সেবন করুন, এমনকি সুস্থ অনুভবের দিনগুলিতেও।   (সেবন করা and অনুভব are
      bookish; the noun phrase at the end is not how anyone speaks)
    ✓ প্রতিদিন আপনার ওষুধ খান, এমনকি যেদিন আপনি সুস্থ বোধ করছেন সেদিনও।
- English idioms almost never survive a literal rendering: "slow and steady" is not ধীর এবং স্থির.
- Words used figuratively take their sense, not their dictionary entry: the "space" you need to
  recover is mental room, not স্থান; a "tool" that is really a worksheet is not সরঞ্জাম; "some
  people find that..." means they experience it, not that they মনে করেন it.

VOCABULARY
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

CARRY THROUGH UNCHANGED
- Digits stay in Latin form, exactly as printed: 999, 2.5mg, 30%, 6 weeks. Never Bengali numerals —
  the rest of the booklet uses Latin digits and the two must not be mixed.
- Dates, email addresses, URLs, code, and abbreviations.
- Drug names (Bisoprolol, Varenicline) and proper nouns — people, companies, products — in Latin
  script, unless a common Bangla form exists.
- A snippet with nothing to translate, such as one that is only a number or a symbol, comes back
  exactly as it went in.

OUTPUT
- A JSON array of strings, EXACTLY as many as you were given, in the same order. One snippet in,
  one string out — never merge two, never split one, never leave one out.
- A snippet may start or end mid-sentence, because a line or page break split it there. Translate
  the fragment as a fragment; do not complete it into a whole sentence.
- Keep each translation close to its English in length and do not pad it — but never make the
  Bangla awkward to save a few characters.
- Return only the translations. No English, no notes, no explanation."""


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
                    # Left at the Gemini 3 default of 1.0 deliberately. Google's
                    # guidance for this generation is not to tune temperature —
                    # the reasoning pass is calibrated for the default and a
                    # lowered one degrades it (looping, flattened output). It
                    # also happens to be the right value for this task: greedy
                    # decoding takes the most probable token, which in
                    # translation is reliably the most literal one, and literal
                    # is exactly the Bangla the prompt above works to avoid. The
                    # facts are protected by the prompt, not by the temperature.
                    temperature=1.0,
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
