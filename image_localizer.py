"""Vertex AI (Gemini) image localizer: decide whether a PDF image needs cultural
adaptation for a Bangladeshi audience, and if so, edit it in place.

Two model calls per image:
  1. classify_image  — cheap multimodal Gemini call, structured JSON decision.
  2. localize_image  — image-editing model that returns an edited image.

Both fail safe: on any error the caller keeps the original image, so the PDF is
never corrupted (mirrors translator._translate_chunk)."""

import json
import logging
import re
import time

from google.genai import types

from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

# Use available models only in this Vertex AI project
CLASSIFY_MODEL = "gemini-2.5-flash"  # For classification and text translation
TEXT_TRANSLATE_MODEL = "gemini-2.5-flash"  # For translating text to Bangla

# Image generation/editing models, tried in order. Ordered by measured framing fidelity: text is
# restored onto blank surfaces at positions measured before the edit, so a model that shifts or
# rescales a placard leaves that text hanging off it. See the table in image_regen.EDIT_MODELS.
#
# The chain is NOT a quota escape: the image quota is per-project and shared across all image
# models, so a 429 on one is a 429 on every one of them. It helps only when a model refuses.
EDIT_MODEL = "gemini-3.1-flash-lite-image"
EDIT_MODEL_FALLBACKS = ["gemini-2.5-flash-image", "gemini-3.1-flash-image"]
EDIT_MODELS = [EDIT_MODEL, *EDIT_MODEL_FALLBACKS]
# Attempts per model before moving to the next one (transient 429/503 backoff is handled inside
# vertex_client.generate_content, so a couple of attempts per model is plenty).
EDIT_ATTEMPTS_PER_MODEL = 2

# Quality over speed: an image that failed only because the project was out of quota must not
# ship untouched. The image quota is per-MINUTE and shared by every image model, so waiting
# out a minute is the only thing that clears it — the model chain cannot. Two extra passes
# costs at most ~2.5 minutes on a picture that would otherwise be lost.
EDIT_QUOTA_RETRY_PASSES = 2
EDIT_QUOTA_COOLDOWN_SEC = 70.0

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

A cartoon, line drawing, sketch or simplified illustration of a REAL thing is not \
"abstract" — it is a picture of that thing, and it counts. A cartoon car is a car (a \
vehicle: an object). A cartoon person is a person. A hand-drawn house is a building. A \
simple sketch of a plate of food is food. Style has nothing to do with it: only a diagram \
of a pure concept — a flowchart of ideas, an arrow, a graph of quantities, a gradient or \
an ornamental shape — is abstract. If you can name the real-world thing the drawing shows, \
and that thing is a person, a place, a vehicle, a building, a garment, a utensil, a piece \
of furniture, a household object or food, return needs_localization=true.

Return needs_localization=false ONLY for:
- Pure medical/clinical images: x-rays, CT scans, anatomical diagrams, clinical charts, \
and medical reference images where the image's primary purpose is technical/diagnostic.
- Functional technical elements that must remain intact: QR codes, barcodes, UI chrome, \
buttons, decorative rules, arrows, ornamental shapes, colour gradients, and flowcharts or \
graphs of pure abstract concepts. These are things with no real-world subject at all — do \
not put a drawing of a real object in this group because it is drawn simply.
- Images already looking authentically Bangladeshi.

List only the categories that apply.

FIRST, before anything else, answer is_logo. A logo is the identity of a real organisation \
and is never ours to change — not its drawing, not its colours, and above all not its \
wording.

Answer is_logo=true whenever a mark is present anywhere in the image, then answer \
logo_fills_image to say WHICH of the two situations it is:
- logo_fills_image=true — the image IS the mark. The mark and its immediate lockup are \
essentially the whole picture; there is nothing else in the frame but the mark and its \
background. This image will be left exactly as printed.
- logo_fills_image=false — the image CONTAINS a mark. It is a photograph, illustration, \
diagram or chart with an organisation's mark somewhere in it, typically in a corner or along \
an edge, and the rest of the frame is a real subject: people, food, a place, a chart. The \
mark is found and protected separately by its own detector — its pixels stay as printed and \
its wording is never translated — while the rest of the picture is localized normally.
Getting this wrong in the "fills" direction is expensive: a whole photograph of a meal was \
left in English because a food agency's crest sat in its top corner. If the frame holds a \
real subject as well as the mark, answer false.

Return is_logo=true for:
- Any logo, wordmark, lettermark, emblem, crest, seal, coat of arms, badge or roundel.
- Any brand or product mark, and any charity, hospital, trust, university, government, \
ministry, NHS, WHO or other institutional mark.
- A name or initials set as an organisation's identity — a stylised wordmark, a name locked \
up with a symbol, a masthead — even when it is only lettering.
- A strapline or tagline printed as part of such a mark.
- Copyright lines, registration numbers, ISBNs and publisher imprints.
- Any of the above even when it also contains people, a building, a plant or a landscape: a \
crest with a lion in it is a crest, not a picture of a lion.
When is_logo=true AND logo_fills_image=true, needs_localization MUST be false, and the \
mark's text must NOT be translated. When is_logo=true but logo_fills_image=false, judge \
needs_localization on the rest of the picture as usual — the mark itself is protected \
elsewhere. If you are unsure whether a mark is a logo, answer is_logo=true; if you are \
unsure whether it fills the frame, answer logo_fills_image=false, because a picture wrongly \
called a whole logo is deleted from the localization entirely while a mark inside a picture \
is still protected.

Separately, judge information_role — what the picture is FOR. This is not the same \
question as whether it can be localized, and you must answer it independently:
- "referential": the specific thing shown IS a datum the page states in words. Redrawing it \
as something else would make the page factually WRONG, not merely less local. This is: a \
drink or a food pictured to define a measure, a unit or a dose ("1.5 units", "one portion = \
80g"); a labelled specimen, product, tablet or piece of equipment the reader is meant to \
recognise; one tile of a chart, key, grid or comparison series whose tiles are being \
contrasted with each other; and any picture printed beside a number, percentage or quantity \
that describes what is in the picture.
- "decorative": everything else, INCLUDING ordinary pictures of food and meals. A plate of \
food illustrating what balanced eating looks like, a family at a meal, someone talking to a \
nurse, a person walking, a figure holding a sign. These are decorative because the document \
states nothing factual about the particular dish or the particular person shown. A picture \
is not referential merely because it shows food, or because the page it sits on is about \
health.
Worked examples: a photograph of a plate divided into food groups, printed to show what a \
balanced diet looks like -> decorative (the groups are what matter, and the redraw is \
separately required to keep the same food groups and the same portions). One card in a row \
of eight, each showing a drink beside the number of alcohol units in it -> referential.
When the two readings are genuinely both arguable, answer "referential" — a picture redrawn \
when it should not have been is a factual error in the document, while one left alone is \
merely un-localized."""

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
  - Return the picture at the SAME aspect ratio you were given. Do not pad it, do not crop \
it, and do not fit it into a square or a 2:3 frame.
  - If you cannot keep a board, card or placard exactly where it is, leave that part of the \
picture unchanged rather than moving it. A blank surface that has shifted is worse than one \
that was never adapted: real text is printed onto it afterwards at fixed positions.
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
CRITICAL — FOOD RULES. All three apply, in this order:
  1. HALAL ONLY. Never depict pork, ham, bacon, lard, alcohol, beer, wine, or a wine glass. \
If the original shows one, replace it with a halal food filling the same role.
  2. KEEP THE NUTRITIONAL MEANING. This picture is printed in a health booklet, where a food \
is very often shown to represent a food group, a portion size, a measure or a dose. The \
replacement MUST be in the SAME food group and show the SAME portion: oily fish -> ilish or \
rui (never dal); wholegrain -> lal chal or atta ruti (never white rice); leafy vegetable -> \
lal shak or palong shak; pulse -> dal; dairy -> doi or milk; fruit -> a fruit. Never swap \
across food groups, and never change how much food is shown or how many items are on the plate.
  3. MAKE IT BANGLADESHI. Subject to rules 1 and 2, replace Western dishes with everyday \
Bangladeshi food — bhat, dal, machher jhol, shobji, cha — served on Bangladeshi plates and \
eaten with Bangladeshi utensils.
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
- Food becomes Bangladeshi and halal — never pork, alcohol, beer or wine — and stays in the \
SAME food group and the SAME portion as the original (oily fish -> ilish or rui, wholegrain \
-> lal chal or atta ruti, pulse -> dal, dairy -> doi), served in Bangladeshi dishes.
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
        "logo_fills_image": {"type": "BOOLEAN"},
        "needs_localization": {"type": "BOOLEAN"},
        "categories": {"type": "ARRAY", "items": {"type": "STRING", "enum": CATEGORIES}},
        "information_role": {"type": "STRING", "enum": INFORMATION_ROLES},
        "reason": {"type": "STRING"},
    },
    "required": [
        "is_logo", "logo_fills_image", "needs_localization", "categories",
        "information_role", "reason",
    ],
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
                # Missing means "the whole image is the mark", which is the safe reading:
                # a picture wrongly kept whole is un-localized, while one wrongly redrawn
                # misrepresents a real organisation.
                data["logo_fills_image"] = bool(data.get("logo_fills_image", True))
                # Only a picture that IS a mark is taken off the table entirely. One that
                # merely contains a mark is localized normally, with the mark protected by
                # detect_logo_regions — see image_processor._decide_from_png.
                if data["is_logo"] and data["logo_fills_image"]:
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
        "logo_fills_image": True,
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


# OCR retry budget. Deliberately small and time-boxed: vertex_client already backs off five
# times inside every one of these attempts, and the failure being retried is a 504 on a large
# payload, which more identical requests cannot fix. Three attempts at a 90s ceiling bounds a
# hopeless picture at ~5 minutes instead of the hour that five attempts at the 180s default
# would have cost — on the eatwell plate, measured.
OCR_ATTEMPTS = 3
OCR_TIMEOUT_MS = 90_000
# Longest edge of the copy sent on each successive attempt.
OCR_RETRY_DIMS = (None, 1024, 768)


def _shrunk_for_retry(image_bytes: bytes, mime: str, attempt: int) -> tuple[bytes, str]:
    """The copy to send on `attempt`: the original first, then progressively smaller ones.

    Returns the bytes unchanged if no resize is wanted or possible, so a failure to shrink
    costs an ordinary retry rather than the whole OCR.
    """
    target = OCR_RETRY_DIMS[min(attempt, len(OCR_RETRY_DIMS)) - 1]
    if not target:
        return image_bytes, mime
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if max(img.size) <= target:
                return image_bytes, mime
            img.thumbnail((target, target), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
        logger.info("OCR retry %d: resending at %dpx", attempt, target)
        return buf.getvalue(), "image/png"
    except Exception:
        logger.debug("Could not shrink the image for an OCR retry", exc_info=True)
        return image_bytes, mime


def _extract_text_blocks(image_bytes: bytes, mime: str, notes: dict | None = None) -> list[dict]:
    """Structured OCR: return a list of {"text", "lang", "bbox": [x0,y0,x1,y1]} blocks, with bbox
    normalized to 0..1 of image dimensions.

    Retried, because the one thing this must not do is confuse "no text" with "the request
    failed". Both used to return [] on a single attempt with the reason logged at debug level,
    so a transient error on a picture full of labels read as a picture with no labels — and
    since the edit model is separately told to blank every text surface, the words were then
    deleted rather than translated. That is what happened to the eatwell plate's food-group
    labels: regenerated correctly, captions gone, nothing in the log.

    Pass `notes` to tell the two apart: on total failure it gets "ocr_failed": True, and the
    caller can decline to blank a picture whose words it could not read.
    """
    for attempt in range(1, OCR_ATTEMPTS + 1):
        # Retrying the identical request is the one thing that does not work here: the
        # failure on the biggest pictures is a server-side 504, and the payload is why. Each
        # retry hands over a smaller copy, which is also the cheaper request to serve.
        payload, part_mime = _shrunk_for_retry(image_bytes, mime, attempt)
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[
                    _TEXT_BLOCKS_PROMPT,
                    types.Part.from_bytes(data=payload, mime_type=part_mime),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_TEXT_BLOCKS_SCHEMA,
                    temperature=0.0,
                ),
                timeout_ms=OCR_TIMEOUT_MS,
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
        except Exception as exc:
            logger.warning(
                "Text-block OCR attempt %d/%d failed: %s",
                attempt, OCR_ATTEMPTS, str(exc)[:200],
            )
    logger.error("Text-block OCR failed after %d attempts — the image's words are unknown",
                 OCR_ATTEMPTS)
    if notes is not None:
        notes["ocr_failed"] = True
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
    "report ordinary headings, captions, body text or page furniture as logos.\n"
    "A mark identifies an ORGANISATION. A slogan, motto or message lettered onto something "
    "inside a picture — words on a character's t-shirt, a hand-written sign, a placard, a "
    "poster, a banner — is not a mark unless an organisation's name or emblem is part of it. "
    "'HELP YOURSELF TO A HEALTHY FUTURE' hand-lettered on a cartoon figure's shirt is a "
    "message to the reader and must NOT be reported; the same shirt carrying 'NHS Lothian' "
    "or a charity's crest must be. Reporting a slogan as a mark takes it out of translation "
    "and deletes it from the page, so when the words name no organisation, leave them out."
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


def _keeps_numbers(source: str, translated: str) -> bool:
    """True if every run of digits in the source survives into the translation.

    The one check worth making automatically: this booklet keeps its numbers in Latin digits
    (see translator.SYSTEM_PROMPT, "Leave unchanged: numbers, dates"), so a translation that
    dropped one is detectable without knowing any Bangla. "3 units" -> "তিন" fails here, and
    that exact answer shipped on the alcohol-units grid: a card that defines a measure, with
    the measure gone.
    """
    return all(run in translated for run in re.findall(r"\d+(?:[.,]\d+)?", source))


_BLOCK_TRANSLATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "bn": {"type": "STRING"},
                },
                "required": ["index", "bn"],
            },
        }
    },
    "required": ["translations"],
}

_BLOCK_TRANSLATION_PROMPT = """This picture is printed in a health booklet that is being \
republished in Bangladesh. Below is every piece of text printed in it, numbered. Translate \
each one into natural, everyday Bangla for a Bangladeshi patient.

Look at the picture before you answer. It tells you what each piece of text is doing: a \
heading, a caption under a drawing, the name of a food group, a figure printed beside the \
thing it measures.

Rules:
- Return one entry for every index, with the same index numbers. Never merge two entries, \
never split one, never leave one out.
- KEEP EVERY NUMBER, UNIT, MEASURE AND SYMBOL. "3 units" is "3 ইউনিট", not "তিন". \
"250ml", "12%", "1.5", "80g", "(125ml, ABV 12%)" — the quantity AND its unit both come \
through, written the way they are printed, in the same Latin digits the rest of the booklet \
uses. A number that has lost its unit is a clinical error in this document.
- Translate the whole of a block, including anything inside brackets.
- These blocks all belong to ONE picture. Translate a word that appears in several of them \
the same way every time, and give sibling labels the same style and register — a row of \
tiles has to read as a row.
- An organisation's name, a brand, a drug name or a person's name stays in Latin script.
- Return only the Bangla. No English, no explanation, no quotation marks.
{context_clause}
The text blocks:
{listing}"""


def translate_blocks(
    image_bytes: bytes,
    mime: str,
    blocks: list[dict],
    page_context: str = "",
) -> int:
    """Fill each block's "bn" with its Bangla, in one call that can see the picture.

    Replaces one call per block. The per-block call could not see what it was translating:
    "3 units" came back as "তিন" because nothing in the request said the words were a
    quantity printed beside the drink it measures, and eight tiles of one grid were eight
    separate conversations, so they came back in different registers and most of them not at
    all. One call, with the image and all of the blocks, fixes both — the model sees which
    label is a heading and which is a figure, and it sees its own siblings.

    Returns how many blocks were translated. A block whose answer fails validation is left
    without a "bn", so `resolve_block_text` falls back to the per-block call and then to the
    English: nothing depends on this succeeding.
    """
    pending = [
        (i, (b.get("text") or "").strip())
        for i, b in enumerate(blocks)
        # Nothing to do for a block that is already Bangla, or that a previous call has
        # already answered — the discard path reaches this twice for the same blocks.
        if not _is_valid_bangla((b.get("bn") or "").strip())
    ]
    pending = [(i, t) for i, t in pending if t and not _is_valid_bangla(t)]
    if not pending:
        return 0

    context_clause = ""
    if page_context.strip():
        context_clause = (
            f'\nThe page this picture sits on says: "{page_context.strip()}"\n'
            "Use it only to understand what the words mean. Do not translate it, and do not "
            "add any of it to your answers.\n"
        )

    done = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        listing = "\n".join(f"{i}. {text}" for i, text in pending)
        try:
            response = generate_content(
                model=TEXT_TRANSLATE_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    _BLOCK_TRANSLATION_PROMPT.format(
                        context_clause=context_clause, listing=listing
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0 + 0.2 * (attempt - 1),
                    response_mime_type="application/json",
                    response_schema=_BLOCK_TRANSLATION_SCHEMA,
                ),
            )
            answers = json.loads(response.text).get("translations", [])
        except Exception as exc:
            logger.warning(
                "Batch block translation attempt %d failed: %s", attempt, str(exc)[:200]
            )
            continue

        by_index = {int(a.get("index", -1)): (a.get("bn") or "").strip() for a in answers}
        still: list[tuple[int, str]] = []
        for i, text in pending:
            bn = by_index.get(i, "")
            # A block with no letters at all ("12%", "1.5") has nothing to translate into
            # Bangla script, so for those the digits surviving IS the whole test.
            has_letters = any(ch.isalpha() for ch in text)
            ok = bool(bn) and _keeps_numbers(text, bn) and (_is_valid_bangla(bn) or not has_letters)
            if ok:
                blocks[i]["bn"] = bn
                done += 1
            else:
                still.append((i, text))
        if len(still) == len(pending):
            logger.warning(
                "Batch block translation attempt %d validated nothing (%d block(s))",
                attempt, len(pending),
            )
        pending = still

    if pending:
        logger.info(
            "%d of %d text block(s) fell back to per-block translation: %s",
            len(pending), len(pending) + done,
            "; ".join(t[:30] for _, t in pending[:3]),
        )
    return done


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
    # Already done by the image-aware batch call, which is the better answer because it saw
    # the picture and the block's siblings. image_regen never sets this, so it is unaffected.
    pre = (block.get("bn") or "").strip()
    if pre and _is_valid_bangla(pre):
        return pre
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
    notes: dict | None = None,
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

    `notes` is a dict the failure reason is written into — pass the caller's audit record so
    that a picture lost to quota is distinguishable from one the model refused.

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

    return _run_edit_models(instruction, image_bytes, mime, notes)


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


def _run_edit_models(
    instruction: str, image_bytes: bytes, mime: str, notes: dict | None = None
) -> bytes | None:
    """Send one edit instruction down the model fallback chain; return image bytes or None.

    `notes` is written into rather than returned — the caller passes its own audit record, so
    why an edit failed lands in the audit with no extra plumbing. Sets "edit_error" to
    "quota", "refused" or "error", and "edit_attempts" to the number of requests made.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    contents = [instruction, part]
    attempts = 0
    outcome = "error"

    # Walk the model fallback chain: try each edit model a couple of times before moving on. This
    # is the recovery path when the primary image model is rate-limited (429) or unavailable —
    # transient backoff is already handled inside vertex_client.generate_content.
    #
    # Then walk the whole chain again, because a picture that failed only because the project
    # was out of quota is not a picture that cannot be localized, and shipping it untouched is
    # exactly the failure this pipeline exists to prevent. By this point vertex_client has
    # already backed off five times and tried both projects, so the only thing left to change
    # is the wait: the image quota is per-MINUTE and shared by every image model, so a pause
    # longer than a minute is the one thing that can actually clear it. Only retried when
    # every failure in the pass was transient — a model that refused will refuse again.
    for extra_pass in range(EDIT_QUOTA_RETRY_PASSES + 1):
        if extra_pass:
            logger.info(
                "Localize: every model was rate-limited; waiting %.0fs for the per-minute "
                "image quota to refill (pass %d of %d)",
                EDIT_QUOTA_COOLDOWN_SEC, extra_pass, EDIT_QUOTA_RETRY_PASSES,
            )
            time.sleep(EDIT_QUOTA_COOLDOWN_SEC)
        all_transient = True
        for model in EDIT_MODELS:
            for attempt in range(1, EDIT_ATTEMPTS_PER_MODEL + 1):
                temperature = 0.2 + (attempt - 1) * 0.3  # 0.2, 0.5, ...
                attempts += 1
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
                        logger.info(
                            "Image generation succeeded with %s on attempt %d", model, attempt
                        )
                        if notes is not None:
                            notes["edit_attempts"] = attempts
                        return out
                    logger.warning(
                        "Localize: no image from %s (attempt %d/%d, temp=%.1f)",
                        model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature,
                    )
                    _log_response_diagnostics(response, attempt)
                    # A response that came back without an image is a refusal, not a queue.
                    all_transient = False
                    outcome = "refused"
                except Exception as e:
                    logger.warning(
                        "Localize request failed on %s (attempt %d/%d, temp=%.1f): %s",
                        model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature, str(e)[:200],
                    )
                    if _is_quota_error(e):
                        if outcome != "refused":
                            outcome = "quota"
                    else:
                        all_transient = False
                        outcome = "refused" if outcome == "refused" else "error"
            logger.info("Localize: model %s exhausted; trying next fallback model", model)
        if not all_transient:
            break  # a refusal will not become an acceptance by waiting

    logger.warning(
        "Localize: all edit models failed after %d request(s) (%s) — keeping original image",
        attempts, outcome,
    )
    if notes is not None:
        notes["edit_error"] = outcome
        notes["edit_attempts"] = attempts
    return None


# Markers of a "come back later" failure, as opposed to a refusal. Mirrors
# vertex_client._TRANSIENT_MARKERS, kept short here because only the quota case earns a wait.
_QUOTA_MARKERS = ("resource_exhausted", "429", "quota", "rate limit", "unavailable", "503")


def _is_quota_error(exc: Exception) -> bool:
    """True if an edit failure is a rate limit rather than a refusal."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


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
