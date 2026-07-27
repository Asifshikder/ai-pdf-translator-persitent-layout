"""Vertex AI (Gemini) image localizer: decide whether a PDF image needs cultural
adaptation for a Bangladeshi audience, and if so, edit it in place.

Two model calls per image:
  1. classify_image  — cheap multimodal Gemini call, structured JSON decision.
  2. localize_image  — image-editing model that returns an edited image.

Both fail safe: on any error the caller keeps the original image, so the PDF is
never corrupted (mirrors translator._translate_chunk)."""

import json
import logging

from google.genai import types

from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

# Use available models only in this Vertex AI project
CLASSIFY_MODEL = "gemini-2.5-flash"  # For classification and text translation
TEXT_TRANSLATE_MODEL = "gemini-2.5-flash"  # For translating text to Bangla

# Image generation/editing models, tried in order. If the primary is rate-limited (429) or
# unavailable, the next model is used — an independent path when nano-banana's quota is exhausted.
EDIT_MODEL = "gemini-2.5-flash-image"
EDIT_MODEL_FALLBACKS = ["gemini-3.1-flash-lite-image", "gemini-3.1-flash-image"]
EDIT_MODELS = [EDIT_MODEL, *EDIT_MODEL_FALLBACKS]
# Attempts per model before moving to the next one (transient 429/503 backoff is handled inside
# vertex_client.generate_content, so a couple of attempts per model is plenty).
EDIT_ATTEMPTS_PER_MODEL = 2

MAX_ATTEMPTS = 5

# The four culturally-specific categories the user asked to localize. Anything
# that fits none of these (logos, charts, diagrams, icons, decorative graphics,
# UI screenshots, or already-Bangladeshi content) is left untouched.
CATEGORIES = ["people_attire", "scenes_settings", "food_objects", "signage_text"]

CLASSIFY_SYSTEM_PROMPT = """You decide whether an image inside a document should be \
adapted for a Bangladeshi audience, so the document feels local to Bangladeshi readers.

Return needs_localization=true if the image depicts culturally-adaptable content in one \
or more of these categories:
- people_attire: ANY people or faces (photo, illustration, cartoon, or icon of a person). \
Their appearance and clothing can always be made Bangladeshi (saree, salwar kameez, \
panjabi, hijab), so an image containing a person should almost always be localized.
- scenes_settings: streets, buildings, landscapes, rooms, or environments (including \
diagrams, charts showing Western settings, or infographics with non-Bangladeshi visual context)
- food_objects: meals, produce, or everyday cultural objects (including food plates, \
beverage photos, eating utensils, or food-related diagrams/charts)
- signage_text: readable signs, labels, or text baked into the image (render in Bangla)

Be decisive, not cautious: if any person is visible, return true even for a simple or \
stylized illustration. A generic "Western/international" look is exactly what should be \
localized — do not keep an image just because it looks neutral. Include diagrams and \
infographics if they show Western contexts, food, or objects that should be made local. \
Even text-heavy images of food, scenes, or objects should be localized — the text will be \
handled explicitly during image editing.

Return needs_localization=false ONLY for:
- Pure medical/clinical images: x-rays, CT scans, anatomical diagrams, clinical charts, \
and medical reference images where the image's primary purpose is technical/diagnostic.
- Functional technical elements that must remain intact: QR codes, UI chrome, buttons, \
decorative rules, flowcharts of pure abstract concepts.
- Images already looking authentically Bangladeshi.

List only the categories that apply.

FIRST, before anything else, answer is_logo. A logo is the identity of a real organisation \
and is never ours to change — not its drawing, not its colours, and above all not its \
wording. Return is_logo=true for:
- Any logo, wordmark, lettermark, emblem, crest, seal, coat of arms, badge or roundel.
- Any brand or product mark, and any charity, hospital, trust, university, government, \
ministry, NHS, WHO or other institutional mark.
- A name or initials set as an organisation's identity — a stylised wordmark, a name locked \
up with a symbol, a masthead — even when it is only lettering.
- A strapline or tagline printed as part of such a mark.
- Copyright lines, registration numbers, ISBNs and publisher imprints.
- Any of the above even when it also contains people, a building, a plant or a landscape: a \
crest with a lion in it is a crest, not a picture of a lion.
When is_logo=true, needs_localization MUST be false, and its text must NOT be translated. \
If you are unsure whether a mark is a logo, answer true — a redrawn or translated logo \
misrepresents a real organisation, while a logo left alone costs the document nothing.

Separately, judge information_role — what the picture is FOR. This is not the same \
question as whether it can be localized, and you must answer it independently:
- "referential": the specific thing shown IS the information. Redrawing it as something \
else would make the surrounding document factually wrong. This covers a drink pictured to \
define a measure or dose, a food pictured to place it in a food group or show a portion \
size, a labelled specimen or product shown so the reader can recognise it, a picture \
carrying a number/quantity/percentage that refers to what is depicted, and any single tile \
of a chart, key, grid or comparison series.
- "decorative": the picture illustrates, sets a scene, or shows people doing something. \
Someone talking to a nurse, a family at a meal, a person walking, a figure holding a sign. \
Changing what is depicted costs the document nothing factual.
When the two readings are both arguable, answer "referential" — a picture redrawn when it \
should not have been is a factual error in the document, while one left alone is merely \
un-localized."""

# Two ways to regenerate a picture, chosen per image by image_processor._regeneration_mode:
#
#   "context" — the page's own words go to the model with the picture, so what comes back
#   still illustrates the paragraph it sits beside. A figure printed next to "walk for 20
#   minutes every day" comes back walking rather than sitting, and a plate beside a section
#   on portion sizes keeps being a plate.
#
#   "simple"  — a straight cultural swap with no page text. Used where there is no real
#   context to give (a margin icon, a cover ornament, a picture on an otherwise blank
#   divider): a context clause assembled from three stray words is worse than none, because
#   the model reads whatever it is handed as a brief and draws to it.
LOCALIZE_MODES = ("context", "simple")

CONTEXT_CLAUSE = """CONTEXT — the page this picture is printed on says:
"{context}"
Use it to keep the picture's MEANING intact: the same activity, the same kind of objects, \
the same number of people doing the same thing. It tells you what the picture is FOR — it \
is not a list of things to add, and not one word of it may be written into the image.
"""

EDIT_INSTRUCTION = """Edit this image to reflect Bangladeshi culture and context. Keep the \
EXACT composition, framing, camera angle, and same aspect ratio and dimensions.
{context_clause}CRITICAL — NOTHING MOVES AND NOTHING RESIZES:
  - Every element stays at exactly the same position and exactly the same size as in the \
original. Do not shift, rotate, rescale, crop, re-centre, or re-compose anything.
  - This matters most for blank surfaces: a sign, board, placard, card or panel must keep \
its edges, its corners and its size to the pixel. Real text is printed onto those surfaces \
afterwards at fixed positions, so a board that moves or shrinks leaves that text hanging \
off it and overlapping the artwork.
  - Do not zoom in or out, and do not change how much of the subject is visible or how much \
empty space surrounds it.
CRITICAL — MATCH THE ORIGINAL'S COLOURS:
  - This picture is printed inside a document, surrounded by the page it sits on. It must \
still look like it belongs to that page, so the palette is not yours to change.
  - Reproduce the SAME colour palette as the original: the same hues, the same lightness, \
the same level of saturation, the same overall tone.{palette}
  - The background must stay the EXACT same colour as in the original — if it is plain, \
keep that plain colour; if it is white, keep it white. Never replace a plain background \
with a scene, gradient, texture, or a different colour.
  - Do NOT boost saturation, warm the image up, add new accent colours, or restyle it. \
A recoloured picture is a failure even if it looks nice on its own.
  - If the original is a flat line drawing or a limited two- or three-colour illustration, \
keep exactly that style — do not turn it into a photo, a painting, or a shaded 3D render.
- Clarity: the output must be CLEARER than the original — cleaner and steadier lines, \
sharper edges, better-resolved detail, no blur or compression artifacts. Clarity comes \
from draughtsmanship, not from stronger colour: sharpen the drawing, not the palette.
- People: Make any visible people Bangladeshi with appropriate attire (saree, salwar \
kameez, panjabi, hijab, lungi, or traditional wear as fitting for age/gender/context). \
Use skin tones and features consistent with Bangladeshi people. Keep poses, gestures and \
expressions as engaged and natural as the original's, and rendered in the original's style.
- Settings & objects: Adapt architecture, vehicles, streets, buildings, furniture, and \
household items to look Bangladeshi — but drawn in the original's palette and style.
- Food & drinks: Replace Western foods with Bangladeshi equivalents (rice, dal, fish \
curry, vegetables, tea). Adapt serving dishes and utensils to Bangladeshi style.
- Text & signs: {text_instruction}
CRITICAL — NO TEXT:
  - Do NOT draw, write, render, or hallucinate ANY text, letters, words, numbers, or symbols.
  - Every sign, placard, board, label, poster, or text surface must be reproduced BLANK and CLEAN \
(empty paper/board of the same shape, color, and material) — no glyphs of any kind.
  - Text is added back separately after this step, so leaving it out is required, not a mistake.
CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate, or invent any logo, wordmark, emblem, crest, \
badge, or institutional mark. Never substitute a different organisation's mark for the one there.
  - Leave the area a mark occupies as clean, empty background of the surrounding colour. The \
original marks are stamped back on top afterwards, unchanged.
Do not add or remove objects, change the layout, or add new elements beyond cultural \
adaptation. Keep all text surfaces present but empty. Focus especially on: {focus}."""

# The compact prompt used in "simple" mode. Not a trimmed copy of the one above for its own
# sake: a small picture — a margin icon, one tile of a row — given thirty lines of art
# direction comes back elaborated, with detail invented to satisfy clauses that were never
# about it. The rules that actually matter at that size are the palette, the blank text
# surfaces, and the framing.
SIMPLE_EDIT_INSTRUCTION = """Redraw this image so it depicts Bangladeshi people, clothing, \
food, and surroundings instead of Western ones.
- Keep the EXACT composition, framing, aspect ratio and dimensions. Same number of subjects, \
same poses, same layout.
- NOTHING moves and NOTHING resizes: every element keeps its exact position and size. Blank \
signs, boards and panels keep their edges to the pixel — text is printed onto them afterwards \
at fixed positions, so one that moves or shrinks leaves that text overlapping the artwork.
- Keep the SAME colour palette, the same style, and the same background colour as the \
original.{palette} Do not brighten, restyle, or turn a flat drawing into a photo.
- People become Bangladeshi: skin tones, features, and dress (saree, salwar kameez, panjabi, \
hijab, lungi) suited to their age and role.
- Food becomes Bangladeshi (rice, dal, fish curry, vegetables, tea), served in Bangladeshi dishes.
- Draw NO text of any kind. Every sign, label or lettered surface comes back blank and clean \
— the text is restored separately afterwards.
- Draw NO logo, wordmark, emblem, crest or institutional mark, and never invent or substitute \
one. Leave its area clean and empty; the original mark is stamped back afterwards, unchanged.
Focus especially on: {focus}."""

# The cover is a whole printed page, not a picture inside one, and that changes what has to
# be protected. There is no surrounding page for it to match, so the palette instruction is
# about the booklet's identity rather than about blending in; and the title, subtitle and
# publisher lines are real PDF text that will be laid back over this raster, so the areas
# they occupy have to come back as clean, empty background of the original's colour.
COVER_INSTRUCTION = """This is the front cover of a printed health booklet. Redraw it as the \
cover of the same booklet published in Bangladesh, for Bangladeshi readers.
{context_clause}- Keep the EXACT page layout: the same panels, bands and blocks of colour in \
the same places, the same aspect ratio, the same overall design. This must still be \
recognisably the same booklet, not a new design.
- Keep the SAME colour palette as the original — the same background colours, the same \
accent colours, the same tone.{palette} Do not restyle or rebrand it.
- Any people become Bangladeshi in appearance and dress (saree, salwar kameez, panjabi, \
hijab, lungi), keeping their poses, positions and scale.
- Any setting, building, vehicle, food or object becomes its Bangladeshi equivalent, drawn \
in the original's style.
CRITICAL — NO TEXT ANYWHERE:
  - Draw NO letters, words, numbers, logos or symbols. Not in the title area, not on the \
artwork, not in the footer.
  - Every area that held text must come back as CLEAN, EMPTY background in exactly the \
colour it had — flat and unmarked, ready to be printed over.
  - The booklet's real title is placed back on top of your image afterwards, so leaving \
those areas blank is required, not an omission.
CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate, or invent any logo, wordmark, emblem, \
crest, badge, or the mark of any hospital, trust, charity, ministry or publisher. Never put \
a different organisation's mark in place of the one that was there.
  - Leave every such area clean and empty in its original background colour. The real marks \
are stamped back over your image afterwards, exactly as they were printed."""

INFORMATION_ROLES = ["decorative", "referential"]

_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_logo": {"type": "BOOLEAN"},
        "needs_localization": {"type": "BOOLEAN"},
        "categories": {"type": "ARRAY", "items": {"type": "STRING", "enum": CATEGORIES}},
        "information_role": {"type": "STRING", "enum": INFORMATION_ROLES},
        "reason": {"type": "STRING"},
    },
    "required": ["is_logo", "needs_localization", "categories", "information_role", "reason"],
}


def classify_image(image_bytes: bytes, mime: str) -> dict:
    """Return {"is_logo": bool, "needs_localization": bool, "categories": [...],
    "information_role": str, "reason": str}.

    On any failure returns needs_localization=False so the image is kept as-is, and
    information_role="referential" so a classifier that answered nothing can never be
    read as permission to redraw the picture.

    `is_logo` is authoritative over everything else: a logo is a real organisation's
    identity, so it is never redrawn and its wording is never translated. The model is asked
    to keep the two answers consistent, but they are reconciled here as well rather than
    trusted — a mark that came back is_logo=true *and* needs_localization=true would
    otherwise be redrawn on the strength of the second answer alone.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[part],
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_CLASSIFY_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "needs_localization" in data:
                data.setdefault("categories", [])
                data.setdefault("reason", "")
                role = str(data.get("information_role") or "").strip().lower()
                data["information_role"] = (
                    role if role in INFORMATION_ROLES else "referential"
                )
                data["is_logo"] = bool(data.get("is_logo"))
                if data["is_logo"]:
                    data["needs_localization"] = False
                    data["categories"] = []
                return data
            logger.warning(
                "Classify: unexpected response shape (attempt %d/%d): %r",
                attempt,
                MAX_ATTEMPTS,
                response.text[:200],
            )
        except Exception:
            logger.exception(
                "Classify request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Classify: giving up — treating image as no-localization")
    return {
        "is_logo": False,
        "needs_localization": False,
        "categories": [],
        "information_role": "referential",
        "reason": "classify failed",
    }


_TEXT_BLOCKS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "blocks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "lang": {"type": "STRING"},
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["text", "lang", "bbox"],
            },
        }
    },
    "required": ["blocks"],
}

_TEXT_BLOCKS_PROMPT = (
    "Find every distinct block of visible text in this image (signs, placards, labels, headings, "
    "captions, words). For each block return: 'text' (the exact text as it appears, one block's "
    "lines joined with spaces), 'lang' (ISO code of its language — 'en' for English, 'bn' for "
    "Bangla, 'hi' for Hindi, etc.), and 'bbox' as [x0, y0, x1, y1] with each value a fraction "
    "between 0 and 1 of the image width/height (x0,y0 = top-left corner, x1,y1 = bottom-right). "
    "Return an empty list if there is no text."
)


def _extract_text_blocks(image_bytes: bytes, mime: str) -> list[dict]:
    """Structured OCR: return a list of {"text", "lang", "bbox": [x0,y0,x1,y1]} blocks, with bbox
    normalized to 0..1 of image dimensions. Returns [] on no-text or any failure (text overlay is
    then simply skipped and the original/blank surface is kept).
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    try:
        response = generate_content(
            model=CLASSIFY_MODEL,
            contents=[_TEXT_BLOCKS_PROMPT, part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_TEXT_BLOCKS_SCHEMA,
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        blocks = data.get("blocks", []) if isinstance(data, dict) else []
        cleaned: list[dict] = []
        for b in blocks:
            text = (b.get("text") or "").strip()
            bbox = b.get("bbox") or []
            if not text or len(bbox) != 4:
                continue
            # Clamp to [0,1] and ensure a valid, non-empty rect.
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in bbox)
            if x1 <= x0 or y1 <= y0:
                continue
            cleaned.append({
                "text": text,
                "lang": (b.get("lang") or "").strip().lower(),
                "bbox": [x0, y0, x1, y1],
            })
        return cleaned
    except Exception:
        logger.debug("Could not extract text blocks from image", exc_info=True)
    return []


_LOGO_REGIONS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "logos": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["label", "bbox"],
            },
        }
    },
    "required": ["logos"],
}

_LOGO_REGIONS_PROMPT = (
    "Find every logo or brand mark in this image. That means: logos, wordmarks, lettermarks, "
    "emblems, crests, seals, coats of arms, badges and roundels; charity, hospital, trust, "
    "university, government, ministry, NHS, WHO and other institutional marks; an "
    "organisation's name or initials set as its identity, including a plain stylised wordmark; "
    "and any strapline printed as part of such a mark. For each, return 'label' (the "
    "organisation or brand, or a short description if you cannot name it) and 'bbox' as "
    "[x0, y0, x1, y1], each value a fraction between 0 and 1 of the image width/height "
    "(x0,y0 = top-left, x1,y1 = bottom-right). Draw the box tightly around the mark itself, "
    "including its wording, and nothing else. Return an empty list if there are none. Do not "
    "report ordinary headings, captions, body text or page furniture as logos."
)


def detect_logo_regions(image_bytes: bytes, mime: str) -> list[dict]:
    """Locate logos and brand marks in an image: [{"label": str, "bbox": [x0,y0,x1,y1]}].

    bbox values are fractions of the image's width/height, so they map onto any rendering of
    the same picture. Returns [] on no-logos or any failure — a caller that finds no regions
    simply leaves the generated image alone, which is the same outcome as before this
    existed.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    try:
        response = generate_content(
            model=CLASSIFY_MODEL,
            contents=[_LOGO_REGIONS_PROMPT, part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LOGO_REGIONS_SCHEMA,
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        found = data.get("logos", []) if isinstance(data, dict) else []
    except Exception:
        logger.debug("Could not detect logo regions", exc_info=True)
        return []

    cleaned: list[dict] = []
    for entry in found:
        bbox = entry.get("bbox") or []
        if len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        cleaned.append({"label": (entry.get("label") or "").strip(), "bbox": [x0, y0, x1, y1]})
    return cleaned


def _is_valid_bangla(text: str) -> bool:
    """Validate that text contains proper Bangla characters (not hallucinated)."""
    if not text:
        return False
    # Bangla Unicode range: U+0980 to U+09FF
    bangla_chars = sum(1 for c in text if 'ঀ' <= c <= '৿')
    # At least 30% of text should be Bangla characters
    return bangla_chars > 0 and (bangla_chars / len(text)) > 0.3


def _translate_to_bangla(text: str) -> str:
    """Translate English text to Bangla using Gemini 1.5 Pro with retries."""
    if not text or not text.strip():
        return text

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Adjust temperature for each retry
            temperature = 0.1 + (attempt - 1) * 0.1

            response = generate_content(
                model=TEXT_TRANSLATE_MODEL,
                contents=[
                    f"Translate this English text to natural, proper, clear Bangla. "
                    f"Use correct Bangla spelling and pronunciation. "
                    f"Preserve meaning, style, and tone. "
                    f"Return ONLY the Bangla translation, nothing else. "
                    f"Do not include English text or explanations.\n\n"
                    f"English: {text}"
                ],
                config=types.GenerateContentConfig(temperature=temperature),
            )
            bangla_text = response.text.strip()

            # Validate the translation
            if _is_valid_bangla(bangla_text):
                logger.info("Translated (attempt %d): %s → %s", attempt, text[:50], bangla_text[:50])
                return bangla_text
            else:
                logger.warning(
                    "Translation attempt %d produced invalid Bangla (not enough Bangla chars): %s",
                    attempt, bangla_text[:100]
                )
        except Exception as e:
            logger.warning("Translation attempt %d failed: %s", attempt, str(e)[:200])

    logger.warning("All translation attempts failed for: %s", text[:50])
    return text


def resolve_block_text(block: dict) -> str:
    """Return the text to render for an OCR block: keep it as-is if it is already Bangla,
    otherwise translate it. Returns "" only for an empty block.

    A failed translation falls back to the original words rather than to nothing. By the
    time this is called the baked text has already been painted out of the image, so
    returning "" does not leave the English standing — it deletes it. Untranslated English
    is a shortcoming; a blank where a caption or a dosage figure used to be is a defect.
    """
    text = (block.get("text") or "").strip()
    if not text:
        return ""
    lang = (block.get("lang") or "").lower()
    if lang.startswith("bn") or _is_valid_bangla(text):
        return text  # already Bangla — keep exactly
    translated = _translate_to_bangla(text)
    if _is_valid_bangla(translated):
        return translated
    logger.warning("Could not translate %r — keeping the original text rather than a blank", text[:60])
    return text


def localize_image(
    image_bytes: bytes,
    mime: str,
    categories: list[str],
    page_context: str = "",
    palette: str = "",
    mode: str = "context",
) -> bytes | None:
    """Return edited image bytes adapted to Bangladeshi culture, or None on failure.

    Adapts visual content (people, settings, objects) to Bangladeshi culture.
    Keeps all text from the original image as-is to avoid hallucination.

    `palette` is a measured description of the source's colours (see
    image_processor._palette_summary). Naming the actual hex values holds the model to the
    page's palette far better than asking it to "match the original" — left to itself it
    returns a warmer, more saturated picture that reads as pasted in from another book.

    `mode` is "context" (the page's own words steer what is drawn) or "simple" (a compact
    cultural swap with no page text). See LOCALIZE_MODES for when each applies.

    None means the caller keeps the original image untouched.
    """
    focus = ", ".join(categories) if categories else "any culturally-specific content"
    palette_note = f"\n  - {palette}" if palette.strip() else ""
    context = page_context.strip()

    if mode == "simple" or not context:
        instruction = SIMPLE_EDIT_INSTRUCTION.format(focus=focus, palette=palette_note)
    else:
        # The image model must NOT draw any text — it garbles glyphs (especially Bangla) and
        # fights the real text layer. Leave every text surface blank; text is restored
        # afterwards as a Noto overlay (see image_processor._decide / localize_pdf).
        text_instruction = (
            "Leave every sign, placard, label, and text surface completely BLANK — an empty board "
            "or paper of the same shape and color, with no letters, words, numbers, or symbols at "
            "all. Do not translate or re-draw any text; just clear it."
        )
        instruction = EDIT_INSTRUCTION.format(
            focus=focus,
            text_instruction=text_instruction,
            palette=palette_note,
            context_clause=CONTEXT_CLAUSE.format(context=context),
        )

    return _run_edit_models(instruction, image_bytes, mime)


def localize_cover(
    image_bytes: bytes,
    mime: str,
    page_context: str = "",
    palette: str = "",
) -> bytes | None:
    """Return a Bangladeshi-localized rendering of a whole cover page, or None on failure.

    Separate from `localize_image` because a cover is the page, not a picture on it: there
    is no surrounding layout for it to blend into, and the areas its title and publisher
    lines occupy have to come back blank so the real text layer can be printed back over
    them. See COVER_INSTRUCTION.
    """
    context = page_context.strip()
    instruction = COVER_INSTRUCTION.format(
        palette=f"\n  - {palette}" if palette.strip() else "",
        context_clause=CONTEXT_CLAUSE.format(context=context) if context else "",
    )
    return _run_edit_models(instruction, image_bytes, mime)


def _run_edit_models(instruction: str, image_bytes: bytes, mime: str) -> bytes | None:
    """Send one edit instruction down the model fallback chain; return image bytes or None."""
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    contents = [instruction, part]

    # Walk the model fallback chain: try each edit model a couple of times before moving on. This
    # is the recovery path when the primary image model is rate-limited (429) or unavailable —
    # transient backoff is already handled inside vertex_client.generate_content.
    for model in EDIT_MODELS:
        for attempt in range(1, EDIT_ATTEMPTS_PER_MODEL + 1):
            temperature = 0.2 + (attempt - 1) * 0.3  # 0.2, 0.5, ...
            try:
                logger.info("Image generation: model=%s attempt %d/%d (temp=%.1f)",
                            model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature)
                response = generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=temperature,
                    ),
                    # Image generation is legitimately slower than text; give it a wider
                    # ceiling than the client-level default so a real render isn't aborted.
                    timeout_ms=IMAGE_TIMEOUT_MS,
                )
                out = _first_image_bytes(response)
                if out:
                    logger.info("Image generation succeeded with %s on attempt %d", model, attempt)
                    return out
                logger.warning(
                    "Localize: no image from %s (attempt %d/%d, temp=%.1f)",
                    model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature,
                )
                _log_response_diagnostics(response, attempt)
            except Exception as e:
                logger.warning(
                    "Localize request failed on %s (attempt %d/%d, temp=%.1f): %s",
                    model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature, str(e)[:200],
                )
        logger.info("Localize: model %s exhausted; trying next fallback model", model)

    logger.warning("Localize: all edit models failed — keeping original image")
    return None


def _first_image_bytes(response) -> bytes | None:
    """Pull the first inline image payload out of a generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def _log_response_diagnostics(response, attempt: int) -> None:
    """Log diagnostics when a response doesn't contain an image."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        logger.debug("Attempt %d: no candidates in response", attempt)
        return

    for i, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason and finish_reason != "STOP":
            logger.warning("Attempt %d: candidate %d finish_reason=%s", attempt, i, finish_reason)

    feedback = getattr(response, "prompt_feedback", None)
    if feedback:
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            logger.warning("Attempt %d: prompt_feedback.block_reason=%s", attempt, block_reason)
        safety_ratings = getattr(feedback, "safety_ratings", None)
        if safety_ratings:
            for rating in safety_ratings:
                category = getattr(rating, "category", None)
                probability = getattr(rating, "probability", None)
                blocked = getattr(rating, "blocked", None)
                if blocked or probability == "HIGH":
                    logger.warning(
                        "Attempt %d: safety rating %s=%s, blocked=%s",
                        attempt,
                        category,
                        probability,
                        blocked,
                    )
