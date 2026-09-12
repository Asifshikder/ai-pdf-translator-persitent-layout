"""Standalone image localization: bytes of ONE picture in, localized bytes out.

Its own pipeline, deliberately. Nothing here is imported from `image_processor`,
`image_localizer` or `image_regen`, and nothing there imports this — the only shared module is
`vertex_client`, which is credentials, retry and project failover rather than any decision
about pictures. The helpers below are therefore copies rather than imports, and the reasoning
in their docstrings comes with them.

Why it is separate: those pipelines localize pictures found *inside a PDF*, and every choice
they make follows from that. They can read the page around a picture for context, they measure
reserved rectangles because the page's own text is printed over the artwork, and they restore
translated text as a PDF overlay on the page. Here there is no page. What arrives is whatever
the user dropped on the "Localize Images" tile — most often a screenshot — and the translated
text has nowhere to live except the picture's own pixels.

What a picture goes through:

  1. classify — is it a mark, does it hold text, does its layout have to survive, and is
     there anything culturally adaptable in it.
  2. logo regions — found on the ORIGINAL, before anything is redrawn. Their pixels are put
     back afterwards and their wording is never translated.
  3. OCR — every block of text, with its box, read off the original while the words are still
     there to read.
  4. translate — one image-aware call for all the blocks, so sibling labels match.
  5. redraw — only when the classifier asked for it AND the layout is free. A screenshot, a
     chart or a form keeps its own pixels: what makes it useful is the layout, and a redraw
     is exactly what destroys that.
  6. bake — the Bangla is drawn into the picture at the boxes measured in step 3, and the
     logos from step 2 are stamped back over it.

Step 3 is what was missing, and its absence is the whole bug this module was rewritten for:
the edit model is told to hand every text surface back BLANK because a generative model
garbles glyphs, and with no OCR list to restore afterwards the words were simply gone. The
user saw the drawing change while the writing turned to mush.

Fails safe throughout: any step that fails leaves the picture with whatever pixels it had, so
the worst outcome is an un-localized image rather than a damaged one.
"""

import html
import io
import json
import logging
import os
import re

import fitz  # PyMuPDF
from google.genai import types
from PIL import Image, ImageDraw

from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

TEXT_MODEL = "gemini-3-flash-preview"   # classify, OCR, translate
EDIT_MODEL = "gemini-3.1-flash-image"   # the redraw
EDIT_ATTEMPTS = 3

MODEL_MAX_DIM = 1536  # longest side of the copy sent to any model

# Retry budgets. The text calls are cheap and worth repeating; the OCR one is the expensive
# request on a big screenshot, so it retries at progressively smaller sizes instead (see
# _ocr_blocks) — resending an identical payload that timed out does not help.
TEXT_ATTEMPTS = 3
OCR_ATTEMPTS = 3
OCR_TIMEOUT_MS = 90_000
OCR_RETRY_DIMS = (None, 1024, 768)

# Above this many text blocks a picture is treated as layout-critical whatever the classifier
# said about it. The classifier answers `layout_locked` directly and is usually right, but a
# picture holding this many separate strings is a screenshot, a form, a table or an
# infographic in every case seen so far, and the cost of the two mistakes is not symmetric: a
# redraw that should not have happened destroys the thing the user uploaded, while a redraw
# skipped leaves a correctly translated picture that simply still looks Western.
LAYOUT_LOCKED_MIN_BLOCKS = 6

# Bangla needs far more pixels than Latin to stay legible — matras, conjuncts and the
# headstroke all have to resolve — so a small picture is enlarged before any text is drawn
# into it. The text is drawn AFTER the enlargement, so only it gets genuinely sharper; the
# artwork is no worse for a LANCZOS upscale than it was at its own size.
TEXT_BAKE_MIN_DIM = 900
TEXT_BAKE_MAX_UPSCALE = 4.0
# The scratch page the Bangla is rendered on is one point per pixel, so a very large picture
# would allocate a very large pixmap. Above this the overlay is rendered smaller and scaled.
TEXT_RENDER_MAX_DIM = 3000
# An OCR box is drawn tight around Latin text; Bangla reaches above the ascender and below the
# baseline, so rendering into that exact box shrinks the type hard. Kept small — a box grown
# past the coloured bar its text sits on puts pale type on white paper.
TEXT_BOX_GROW = 0.08  # fraction of the box's own height, added top and bottom

# How far a box may then grow into blank surface, as a multiple of its own height.
#
# The fixed grow above is a fudge factor; this is measured. A line of Bangla is longer than the
# English it replaces and needs 1.45x line height to itself, so a paragraph rendered into the
# exact rectangle its English occupied gets shrunk until it fits — on a panel with empty space
# below the text, that produced body copy at half the size of the identical panel next to it,
# for no reason but the shape of the box the OCR drew. So the box is walked outwards while the
# pixels it is walking into are the same colour as the surface the text already sits on, and it
# stops at anything else: a rule, an edge, the panel's border, or another block's box.
TEXT_BOX_MAX_GROWTH = 1.2
TEXT_BOX_GROW_STEP = 0.12   # of the box's own height, per step
TEXT_SURFACE_TOLERANCE = 24  # per-channel distance still counted as the same surface
# The share of a text box sampled to decide what colour its surface is. Measured from the
# MIDDLE rather than the border ring: an OCR box is drawn around the text, so its edges
# straddle whatever the text sits next to, not what it sits on.
TEXT_SURFACE_SAMPLE = 0.6
TEXT_LIGHT_BG = 140  # mean luminance above which a box falls back to black type, not white
# A block wider than this share of the picture is set left rather than centred. Screenshot
# text is a left-aligned paragraph or list item; a caption inside an illustration is a short
# centred label. Getting this wrong is cosmetic either way, which is why it is a ratio and
# not a model call.
TEXT_LEFT_ALIGN_MIN_WIDTH = 0.40

# A logo region wider or taller than this share of the picture is not a mark *inside* the
# picture — the detector has boxed the whole thing, which is the `logo_fills_image` case and
# is handled by returning the upload untouched. Stamping it back here would paste the entire
# original over the redraw and quietly undo the localization.
LOGO_REGION_MAX_SHARE = 0.55
LOGO_REGION_PAD = 0.01   # fraction of width/height; the detector's boxes clip outer strokes
LOGO_BLOCK_OVERLAP = 0.30  # share of a text block inside a mark's box before it counts as its wording

CSS_TEMPLATE = """
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Regular.ttf); }}
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Bold.ttf); font-weight: bold; }}
* {{
    font-family: bengali, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.45;
    font-size: {size:.1f}px;
    color: {color};
    text-align: {align};
}}
"""


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are preparing an image for a Bangladeshi audience. Decide what should \
happen to it. Answer every field.

FIRST answer is_logo and logo_fills_image. A logo is a real organisation's identity and is \
never ours to change — not its drawing, not its colours, and above all not its wording. \
Answer is_logo=true whenever a mark is present anywhere in the picture: any logo, wordmark, \
lettermark, emblem, crest, seal, badge or roundel; any brand or product mark; any hospital, \
charity, trust, university, government, ministry, NHS or WHO mark; an organisation's name or \
initials set as its identity; a strapline printed as part of such a mark; and copyright lines, \
registration numbers and publisher imprints. A crest containing a lion is a crest, not a \
picture of a lion. If unsure, answer true.
Then say WHICH situation it is:
- logo_fills_image=true — the picture IS the mark. The mark and its lockup are essentially \
the whole frame, with nothing else in it but background. The upload is returned untouched.
- logo_fills_image=false — the picture CONTAINS a mark: a screenshot, photograph, chart or \
illustration with an organisation's mark somewhere in it, usually in a corner, while the rest \
of the frame is a real subject. The mark is protected separately, by position, and the rest of \
the picture is handled normally. If the frame holds a real subject as well as the mark, this \
is the answer — a whole screenshot was once thrown away because a hospital's crest sat in its \
header.

Then answer has_text: true if any readable words, letters or numbers are printed anywhere in \
the picture, in any language.

Then answer layout_locked. This asks whether the picture's VALUE IS ITS LAYOUT — whether \
redrawing it would destroy what it is for, even if every pixel came back beautiful. Answer \
true for: a screenshot of an app, a website, a phone or a desktop; any user interface, menu, \
dialog, form or settings page; a table, a spreadsheet, a receipt, a form, a certificate or a \
document scan; a chart, a graph, a timetable, a map or a wiring/flow diagram; a technical or \
clinical figure whose parts are labelled. Anything where a reader is meant to read positions \
and labels rather than look at a scene. Answer false for photographs, illustrations, cartoons, \
posters and scenes — pictures whose subject, not whose arrangement, is the point.

Finally answer needs_localization: true if the picture depicts culturally-adaptable content \
that could be Bangladeshi instead — people or faces of any kind (photo, illustration, cartoon \
or icon), streets, buildings, rooms, landscapes, furniture, vehicles, household objects, \
clothing, or food and drink. Be decisive: a generic "Western" or "international" look is \
exactly what should be adapted, so do not keep a picture merely because it looks neutral. A \
cartoon of a real thing is a picture of that thing and counts.
Answer needs_localization=false for: pure medical or clinical imagery whose purpose is \
technical (x-rays, CT scans, ECG traces, anatomical diagrams); functional elements that must \
survive intact (QR codes, barcodes, buttons, arrows, plain charts of abstract quantities); and \
pictures that already look authentically Bangladeshi. When is_logo and logo_fills_image are \
both true, needs_localization MUST be false.

needs_localization and layout_locked are INDEPENDENT questions and both are used. A screenshot \
showing photographs of Western people is layout_locked=true and needs_localization=true: its \
words are translated and its layout is kept exactly. Answer each on its own merits.

List the categories that apply and give a one-line reason."""

_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_logo": {"type": "BOOLEAN"},
        "logo_fills_image": {"type": "BOOLEAN"},
        "has_text": {"type": "BOOLEAN"},
        "layout_locked": {"type": "BOOLEAN"},
        "needs_localization": {"type": "BOOLEAN"},
        "categories": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": ["people_attire", "scenes_settings", "food_objects", "signage_text"],
            },
        },
        "reason": {"type": "STRING"},
    },
    "required": [
        "is_logo", "logo_fills_image", "has_text", "layout_locked",
        "needs_localization", "categories", "reason",
    ],
}

# Boxes are asked for as `box_2d` — [y0, x0, y1, x1] on a 0-1000 grid — rather than as the
# fractions this module works in, because that is the convention the model was trained to
# answer in and it is markedly better at it: asked for fractions it misses blocks and places
# the ones it finds low, and two runs of the identical request disagree with each other.
_OCR_PROMPT = (
    "Detect every distinct block of visible text in this image — headings, labels, buttons, "
    "menu items, captions, body text, signs, numbers. For each block return: 'text' (the exact "
    "text as it appears, one block's lines joined with spaces), 'lang' (ISO code of its "
    "language — 'en' for English, 'bn' for Bangla, 'hi' for Hindi), and 'box_2d' as "
    "[y0, x0, y1, x1] integers on a 0-1000 grid of the image height/width (y0,x0 = top-left, "
    "y1,x1 = bottom-right). Draw each box tightly around that block's own text and nothing "
    "else. Keep separate lines that are separated by a gap, a rule, a colour change or a "
    "different purpose as separate blocks — do not merge a heading with the paragraph under "
    "it, or two menu items into one. Return an empty list if there is no text."
)

_OCR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "blocks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "lang": {"type": "STRING"},
                    "box_2d": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["text", "lang", "box_2d"],
            },
        }
    },
    "required": ["blocks"],
}

_LOGO_PROMPT = (
    "Find every logo or brand mark in this image. That means: logos, wordmarks, lettermarks, "
    "emblems, crests, seals, badges and roundels; charity, hospital, trust, university, "
    "government, ministry, NHS, WHO and other institutional marks; an organisation's name or "
    "initials set as its identity, including a plain stylised wordmark; and any strapline "
    "printed as part of such a mark. For each, return 'label' (the organisation or brand, or a "
    "short description if you cannot name it) and 'bbox' as [x0, y0, x1, y1], each value a "
    "fraction between 0 and 1 of the image width/height (x0,y0 = top-left, x1,y1 = "
    "bottom-right). Draw the box tightly around the mark itself, including its wording, and "
    "nothing else. Return an empty list if there are none.\n"
    "A mark identifies an ORGANISATION. Ordinary headings, captions, body text, buttons, menu "
    "items and page furniture are NOT marks. A slogan, motto or message lettered onto "
    "something inside the picture — words on a t-shirt, a hand-written sign, a placard, a "
    "banner — is not a mark unless an organisation's name or emblem is part of it. Reporting "
    "ordinary text as a mark takes it out of translation and leaves it in English, so when the "
    "words name no organisation, leave them out."
)

_LOGO_SCHEMA = {
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

_BATCH_TRANSLATE_PROMPT = """Below is every piece of text printed in this picture, numbered. \
Translate each one into natural, everyday Bangla for a Bangladeshi reader.

Look at the picture before you answer. It tells you what each piece of text is doing: a \
heading, a button, a menu item, a caption under a drawing, a figure printed beside the thing \
it measures.

Write Bangladeshi modern colloquial চলিত, never সাধু ভাষা, and address the reader as আপনি. Use \
the plainest everyday word rather than a bookish one, and say each label the way a Bangla \
speaker would say it out loud rather than word by word after the English. Never use a \
transliterated English abbreviation — GP is ডাক্তার, never জিপি.

Rules:
- Return one entry for every index, with the same index numbers. Never merge two entries, \
never split one, never leave one out.
- KEEP EVERY NUMBER, UNIT, MEASURE AND SYMBOL. "3 units" is "3 ইউনিট", not "তিন". "250ml", \
"12%", "1.5", "80g" — the quantity AND its unit both come through, in the same Latin digits \
they are printed in. A number that has lost its unit is an error.
- Translate the whole of a block, including anything inside brackets.
- These blocks all belong to ONE picture. Translate a word that appears in several of them the \
same way every time, and give sibling labels the same style and register — a row of tiles, a \
menu or a list of buttons has to read as one.
- The text is redrawn into the same box it was read from, so keep each translation about as \
long as its English. A label that doubles in length is set in type too small to read.
- An organisation's name, a brand, a product name, a drug name, a person's name, a file name, \
a URL, an email address, a keyboard shortcut or a piece of code stays exactly as printed, in \
Latin script.
- Return only the Bangla. No English, no explanation, no quotation marks.

The text blocks:
{listing}"""

_BATCH_TRANSLATE_SCHEMA = {
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

# The per-block fallback, used only for a block the batch call could not answer for. It is the
# one translation path with no picture to look at, so the register it has to match is stated
# outright — left to "clear Bangla" it writes something more formal than its siblings.
_SINGLE_TRANSLATE_PROMPT = """Translate this line of text, printed inside a picture, into \
Bangla for an ordinary Bangladeshi reader.

- Bangladeshi Bangla, modern colloquial চলিত. Never সাধু ভাষা. Address the reader as আপনি.
- The plainest everyday word, never a bookish one and never a transliterated abbreviation.
- Say it the way a Bangla speaker would say it, not word by word after the English.
- It is a label, so keep it short — about as long as the English.
- Keep every number, unit and symbol exactly as printed, in Latin digits: "3 units" is
  "3 ইউনিট", never "তিন".
- A brand, an organisation, a product, a drug name, a person's name, a file name, a URL or a
  piece of code stays in Latin script.
- Return only the Bangla. No English, no explanation, no quotation marks.

English: {text}"""

# The redraw brief. Shorter than the PDF pipeline's, and that is deliberate: what arrives here
# is a single picture with no page around it, so there is no context clause, no reserved
# rectangle geometry and no surrounding layout to blend into. The two rules that carry all the
# weight are that text surfaces come back blank (the words are baked back afterwards, at boxes
# measured before this call) and that marks are not touched.
EDIT_INSTRUCTION = """Redraw this image so it depicts Bangladeshi people, clothing, food and \
surroundings instead of Western ones.

CRITICAL — KEEP THE SAME PICTURE:
  - Keep the EXACT composition, framing, camera angle, aspect ratio and dimensions. The same
    number of subjects, in the same positions, at the same scale, doing the same thing.
  - NOTHING moves and NOTHING resizes. This matters most for lettered surfaces: a sign, board,
    placard, panel, button or label keeps its edges, corners and size to the pixel. Real text
    is printed back onto those surfaces afterwards at fixed positions, so one that has moved or
    shrunk leaves that text hanging off it and across the artwork.
  - Do not zoom, crop, pad, re-centre, or fit the picture into a square or a 2:3 frame. Return
    it at the aspect ratio you were given.

CRITICAL — NO TEXT:
  - Draw NO text, letters, words, numbers or symbols anywhere, in any script.
  - Every sign, placard, board, label, button, poster or lettered surface comes back BLANK and
    CLEAN — empty paper, board or panel of the same shape, colour and material, with no glyphs
    of any kind.
  - The real text is translated and drawn back separately afterwards, so leaving it out is
    required, not a mistake. Inventing lettering is the worst thing you can do here: it lands
    under the real text and shows through it.

CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate or invent any logo, wordmark, emblem, crest,
    badge or institutional mark, and never substitute one organisation's mark for another's.
  - Leave the area a mark occupies as clean, empty background of the surrounding colour. The
    original marks are stamped back on top afterwards, exactly as they were.

CRITICAL — MATCH THE ORIGINAL'S COLOURS:
  - Reproduce the SAME hues, the same lightness, the same saturation, the same tone.{palette}
  - The background stays the EXACT colour it is. If it is plain, it stays plain; if it is
    white, it stays white. Never replace a plain background with a scene, gradient or texture.
  - Do NOT boost saturation, warm the picture up, add accent colours or restyle it. A
    recoloured picture is a failure even if it looks good on its own.
  - If the original is a flat line drawing or a two- or three-colour illustration, keep exactly
    that style — do not turn it into a photo, a painting or a shaded 3D render.
  - It must look PRINTED, not generated: no soft glow, no bloom, no vignette, no drop shadows,
    no glossy highlights, no depth-of-field blur, no sparkles, no over-rendered 3D lighting.
  - The output must be CLEARER than the original: cleaner lines, sharper edges, no blur and no
    compression artifacts. Clarity comes from draughtsmanship, not from stronger colour.

CRITICAL — PEOPLE ARE REDRAWN, NOT RE-DRESSED:
  - Every visible person must BE Bangladeshi, not a Western person wearing Bangladeshi clothes.
    Changing only the clothing is the most common failure and is not acceptable.
  - Redraw the person: face shape, nose, lips, eyes, brow and jaw as a Bangladeshi person's;
    warm brown South Asian skin; black or dark brown hair with South Asian hair texture. No
    blond, red or light-brown hair, no blue or green eyes, no pale complexion.
  - THEN dress them: saree, salwar kameez with the orna, panjabi, kurta, hijab, lungi, or
    ordinary modern Bangladeshi clothing as fits their age, gender and role.
  - Keep the same number of people, the same poses, gestures, expressions and positions, at the
    same size and in the original's drawing style.

CRITICAL — BEHAVIOUR MUST FIT BANGLADESH, NOT ONLY APPEARANCE:
  - MODESTY — everyone is covered. Women in a saree, a salwar kameez with the orna over the
    chest, or other loose full-length clothing: shoulders, arms, chest, midriff and legs
    covered. Men in a shirt with full trousers, pyjama or lungi, never bare-chested. No shorts,
    vests, sleeveless or low-cut tops, clinging fits, swimwear or gym-wear, and no bare legs —
    this holds during exercise, sport and at home. Draw the SAME activity in modest clothing
    rather than dropping the activity.
  - CONTACT BETWEEN MEN AND WOMEN — none. No hugging, kissing, hand-holding, an arm around a
    shoulder or waist, or leaning on one another; they stand or sit side by side at a
    respectful distance. Same-gender contact and a parent holding their own young child stay.
  - MANNERS — eat, give and receive with the right hand; feet stay off tables and chairs and
    soles are not turned towards anyone; people greet with salam or a nod.
  - A pub, bar, nightclub, dance floor or beach scene becomes the Bangladeshi setting that
    serves the same purpose — a tea stall, a home sitting room, a park or riverside walk, a
    community hall — with the same activity and the same number of people. No dogs indoors.

CRITICAL — FOOD RULES. All three apply, in this order:
  1. HALAL ONLY. Never depict pork, ham, bacon, lard, alcohol, beer, wine or a wine glass. If
     the original shows one, replace it with a halal food filling the same role.
  2. KEEP THE MEANING. A food is very often shown to represent a food group, a portion or a
     measure. The replacement must be in the SAME food group and show the SAME portion: oily
     fish -> ilish or rui (never dal); wholegrain -> lal chal or atta ruti (never white rice);
     leafy vegetable -> lal shak or palong shak; pulse -> dal; dairy -> doi or milk; fruit -> a
     fruit. Never swap across food groups and never change how much food is shown.
  3. MAKE IT BANGLADESHI. Subject to 1 and 2, everyday Bangladeshi food — bhat, dal, machher
     jhol, shobji, cha — on steel or melamine plates, eaten the way it is eaten there.

CRITICAL — THE WHOLE FRAME IS LOCALIZED, NOT ONLY THE PEOPLE:
  - Changing the faces and the clothing while the room, the street, the furniture, the crockery
    and the props stay exactly as drawn is the single most common failure of this task. A
    Bangladeshi family in a Western kitchen is not localized.
  - Every object a Bangladeshi household or street would not contain is replaced by what fills
    the same role there — in the same position, at the same size, in the original's palette and
    style. Walls, floors, roofing, windows, doors, furniture, crockery, utensils, appliances,
    vehicles, shopfronts, trees and plants are all in scope.
  - THIS INCLUDES WHAT PEOPLE WEAR AND HOLD. Hi-vis jackets, tabards, work gloves, mittens,
    helmets, hard hats, beanies, caps, coats, fleeces, scarves, ties, suits, jeans, trainers,
    boots and Western bags belong to the original's world, not to its message. Keeping the
    original's gloves or jacket on a Bangladeshi face is the commonest way this task is failed.

WHAT BANGLADESHI LOOKS LIKE — draw from this, do not stop at skin and clothing:
  - PEOPLE: Bengali faces, rounded jaw, broad nose, dark brown eyes, thick black hair; warm
    brown skin across a real range from fair-wheatish to deep brown. Not East Asian, not Arab,
    not a tanned European.
  - HOMES: brick or plastered walls painted pale green, blue or cream; a corrugated tin roof or
    a flat concrete one; cement or red-oxide floors; a ceiling fan; a mosquito net over a
    wooden khat; plastic moulded chairs, a wooden almirah, a wall calendar and clock; steel or
    melamine crockery; a jug and glass on a tray.
  - STREETS: cycle rickshaws with painted hoods, green CNG auto-rickshaws, crowded buses,
    hawker carts; a tea stall with a kettle and small glass cups; shops behind corrugated
    shutters; tangled overhead cables; brick-paved lanes; a mosque minaret; a pukur with steps
    down to it, paddy fields, banana, coconut and betel-nut palms, a river with a wooden nouka.
  - HEALTH SETTINGS: a community clinic or upazila health complex — pale green or white walls,
    a metal bed with a plain sheet, a wooden desk; a doctor in a white coat over a saree or a
    panjabi; not a Western hospital corridor.
  - FOOD: bhat, dal, machher jhol, shobji bhaji, ruti, khichuri, doi, muri; cha in a small
    glass cup; aam, kathal, kola, peyara.
Draw Bangladesh specifically, not a generic "South Asian" or "Middle Eastern" stand-in: no
desert, no pagoda, no Gulf skyline, no Western suburban kitchen or lawn. And nothing from this
list survives if it was in the original: snow, autumn leaves, bare deciduous trees, a grey
European sky, fitted kitchens, sofas with scatter cushions, carpets, radiators, wheelie bins,
mown lawns, zebra crossings, cars, supermarket trolleys, a knife and fork beside a plate.

Focus especially on: {focus}."""


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------


def _to_png(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """Normalize any upload to PNG bytes. Returns (png, width, height), or None if the bytes
    cannot be decoded — in which case the upload is handed back untouched."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), width, height
    except Exception:
        logger.debug("Could not decode the uploaded image", exc_info=True)
        return None


def _downscale_for_model(png_bytes: bytes, max_dim: int = MODEL_MAX_DIM) -> bytes:
    """A copy scaled so its longest side is <= max_dim, for cheaper and faster model calls.

    Every box this module receives back is normalized to 0..1, so downscaling the copy sent to
    the model does not affect how those boxes map onto the full-size original.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            if max(img.size) <= max_dim:
                return png_bytes
            scale = max_dim / max(img.size)
            resized = img.convert("RGB").resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS
            )
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.debug("Could not downscale for the model call; using the original", exc_info=True)
        return png_bytes


def _resize_to(image_bytes: bytes, width: int, height: int) -> bytes:
    """Bring a generated picture back to the upload's aspect ratio, keeping extra resolution.

    Extra pixels are free — they simply print and display sharper — so a generation that came
    back larger in the right proportions is left alone. A different aspect ratio would display
    stretched, so that case is resampled to the original dimensions.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            source_aspect = width / max(height, 1)
            new_aspect = img.width / max(img.height, 1)
            matches = abs(source_aspect - new_aspect) / source_aspect < 0.01
            if not (matches and img.width * img.height >= width * height):
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.exception("Could not resize the generated image; using it as-is")
        return image_bytes


_PIL_FORMATS = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/bmp": "BMP",
    "image/tiff": "TIFF",
}


def _encode_as(png_bytes: bytes, mime: str) -> bytes:
    """Re-encode the result in the format the upload arrived in.

    The endpoint hands the response back under the uploaded file's content type and extension,
    so returning PNG bytes labelled image/jpeg would be a lie about the payload. Everything
    inside this module works in PNG because that is lossless between steps; only the last hop
    converts.
    """
    fmt = _PIL_FORMATS.get(mime.lower().split(";")[0].strip(), "PNG")
    if fmt == "PNG":
        return png_bytes
    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=95)
        return buf.getvalue()
    except Exception:
        logger.debug("Could not re-encode as %s; returning PNG bytes", fmt, exc_info=True)
        return png_bytes


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _median_color(pixels: list) -> tuple[int, int, int] | None:
    """The per-channel median of a list of RGB pixels. Median, not mean: a box that catches a
    few pixels of something else returns the surface's colour rather than a blend of the two."""
    if not pixels:
        return None
    channels = []
    for i in range(3):
        values = sorted(p[i] for p in pixels)
        channels.append(values[len(values) // 2])
    return (channels[0], channels[1], channels[2])


def _palette_summary(png_bytes: bytes, max_colors: int = 6) -> str:
    """Describe the upload's colours for the edit model, as measured hex values.

    "Match the original's palette" on its own does not survive a generative edit; naming the
    colours does. Returns "" when the picture cannot be read, in which case the prompt falls
    back to its qualitative instruction.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
            # Quantize first: a photograph has thousands of near-identical shades, and the
            # handful that survive quantization are the ones a reader would name.
            reduced = img.convert("P", palette=Image.ADAPTIVE, colors=max_colors).convert("RGB")
            counts = reduced.getcolors(maxcolors=max_colors * 4) or []
    except Exception:
        logger.debug("Could not measure the source palette", exc_info=True)
        return ""
    if not counts:
        return ""
    counts.sort(reverse=True)
    total = sum(n for n, _ in counts) or 1
    named = ", ".join(f"{_hex(rgb)} ({n / total:.0%})" for n, rgb in counts[:max_colors])
    return (
        f"\n    The original's colours, measured: {named}. Reuse these, not brighter "
        "versions of them."
    )


# --------------------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------------------


def _classify(png_bytes: bytes) -> dict:
    """What kind of picture this is. Fails safe in every direction: an unanswerable image is
    treated as a mark that fills the frame, which returns the upload untouched."""
    part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    for attempt in range(1, TEXT_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=TEXT_MODEL,
                contents=[CLASSIFY_PROMPT, part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_CLASSIFY_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if not isinstance(data, dict) or "needs_localization" not in data:
                logger.warning("Classify: unexpected shape (attempt %d)", attempt)
                continue
            decision = {
                "is_logo": bool(data.get("is_logo")),
                # Missing means "the whole picture is the mark", the safe reading: a picture
                # wrongly kept whole is merely un-localized, while one wrongly redrawn
                # misrepresents a real organisation.
                "logo_fills_image": bool(data.get("logo_fills_image", True)),
                "has_text": bool(data.get("has_text", True)),
                "layout_locked": bool(data.get("layout_locked", False)),
                "needs_localization": bool(data.get("needs_localization")),
                "categories": data.get("categories") or [],
                "reason": (data.get("reason") or "")[:200],
            }
            # Reconciled here as well as asked for in the prompt: a mark that came back
            # is_logo=true *and* needs_localization=true would otherwise be redrawn on the
            # strength of the second answer alone.
            if decision["is_logo"] and decision["logo_fills_image"]:
                decision["needs_localization"] = False
                decision["categories"] = []
            return decision
        except Exception:
            logger.exception("Classify failed (attempt %d/%d)", attempt, TEXT_ATTEMPTS)
    logger.warning("Classify: giving up — the upload is returned untouched")
    return {
        "is_logo": True,
        "logo_fills_image": True,
        "has_text": False,
        "layout_locked": True,
        "needs_localization": False,
        "categories": [],
        "reason": "classify failed",
    }


def _detect_logos(png_bytes: bytes) -> list[dict]:
    """Locate the marks inside a picture: [{"label", "bbox": [x0,y0,x1,y1]}] in fractions.

    Returns [] on no-marks or on any failure. A failure is not silent-but-harmless here — it
    means a mark may be redrawn and its wording translated — so it is logged at warning.
    """
    try:
        response = generate_content(
            model=TEXT_MODEL,
            contents=[_LOGO_PROMPT, types.Part.from_bytes(data=png_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LOGO_SCHEMA,
                temperature=0.0,
            ),
        )
        found = json.loads(response.text).get("logos", [])
    except Exception as exc:
        logger.warning("Logo detection failed (%s) — no mark can be protected by position",
                       str(exc)[:160])
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


def _shrunk_for_retry(png_bytes: bytes, attempt: int) -> bytes:
    """The copy to send on `attempt`: the original first, then progressively smaller ones."""
    target = OCR_RETRY_DIMS[min(attempt, len(OCR_RETRY_DIMS)) - 1]
    if not target:
        return png_bytes
    logger.info("OCR retry %d: resending at %dpx", attempt, target)
    return _downscale_for_model(png_bytes, target)


def _ocr_blocks(png_bytes: bytes) -> list[dict] | None:
    """Read the picture's text: [{"text", "lang", "bbox": [x0,y0,x1,y1]}], boxes in fractions.

    None when every attempt failed, [] when the picture genuinely has no text — a distinction
    the caller depends on completely. If the two were the same value, a transient error on a
    screenshot full of labels would read as "no text", and since the edit model is separately
    told to hand back every lettered surface blank, the words would be deleted rather than
    translated. That is the failure this module was rewritten to remove; it must not come back
    disguised as an empty list.
    """
    for attempt in range(1, OCR_ATTEMPTS + 1):
        # Retrying the identical request is the one thing that does not work: the failure on a
        # large screenshot is a server-side timeout and the payload is why. Each retry hands
        # over a smaller copy, which is also the cheaper request to serve.
        payload = _shrunk_for_retry(png_bytes, attempt)
        try:
            response = generate_content(
                model=TEXT_MODEL,
                contents=[
                    _OCR_PROMPT,
                    types.Part.from_bytes(data=payload, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_OCR_SCHEMA,
                    temperature=0.0,
                ),
                timeout_ms=OCR_TIMEOUT_MS,
            )
            data = json.loads(response.text)
            blocks = data.get("blocks", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("OCR attempt %d/%d failed: %s", attempt, OCR_ATTEMPTS, str(exc)[:200])
            continue

        cleaned: list[dict] = []
        for block in blocks:
            text = (block.get("text") or "").strip()
            box_2d = block.get("box_2d") or []
            if not text or len(box_2d) != 4:
                continue
            try:
                # box_2d is [y0, x0, y1, x1] on a 0-1000 grid; everything downstream works in
                # [x0, y0, x1, y1] fractions.
                top, left, bottom, right = (
                    min(max(float(v) / 1000.0, 0.0), 1.0) for v in box_2d
                )
            except (TypeError, ValueError):
                continue
            if right <= left or bottom <= top:
                continue
            cleaned.append({
                "text": text,
                "lang": (block.get("lang") or "").strip().lower(),
                "bbox": [left, top, right, bottom],
            })
        return cleaned

    logger.error("OCR failed after %d attempts — the picture's words are unknown", OCR_ATTEMPTS)
    return None


def _is_valid_bangla(text: str) -> bool:
    """True if a string is really Bangla rather than an English echo or an apology."""
    if not text:
        return False
    bangla = sum(1 for c in text if "ঀ" <= c <= "৿")
    return bangla > 0 and (bangla / len(text)) > 0.3


def _keeps_numbers(source: str, translated: str) -> bool:
    """True if every run of digits in the source survives into the translation.

    The one check worth making automatically: numbers stay in Latin digits, so a translation
    that dropped one is detectable without knowing any Bangla. "3 units" -> "তিন" fails here.
    """
    return all(run in translated for run in re.findall(r"\d+(?:[.,]\d+)?", source))


def _translate_blocks(png_bytes: bytes, blocks: list[dict]) -> None:
    """Fill each block's "bn" in one call that can SEE the picture.

    One call rather than one per block, and with the image attached, because both matter:
    without the picture the model cannot tell a heading from a button from a figure printed
    beside the thing it measures, and block-by-block the labels come back in different
    registers because each one was a separate conversation.

    A block whose answer fails validation is simply left without a "bn"; `_resolve` then falls
    back to a per-block call and finally to the original words. Nothing depends on this
    succeeding.
    """
    pending = [
        (i, (b.get("text") or "").strip())
        for i, b in enumerate(blocks)
        if (b.get("text") or "").strip() and not _is_valid_bangla((b.get("text") or "").strip())
    ]
    if not pending:
        return

    for attempt in range(1, TEXT_ATTEMPTS + 1):
        if not pending:
            return
        listing = "\n".join(f"{i}. {text}" for i, text in pending)
        try:
            response = generate_content(
                model=TEXT_MODEL,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    _BATCH_TRANSLATE_PROMPT.format(listing=listing),
                ],
                config=types.GenerateContentConfig(
                    # Gemini 3's default. A retry does not need a temperature ladder to come
                    # back with something different, and tuning it down degrades the reasoning.
                    temperature=1.0,
                    response_mime_type="application/json",
                    response_schema=_BATCH_TRANSLATE_SCHEMA,
                ),
            )
            answers = json.loads(response.text).get("translations", [])
        except Exception as exc:
            logger.warning("Batch translation attempt %d failed: %s", attempt, str(exc)[:200])
            continue

        by_index = {int(a.get("index", -1)): (a.get("bn") or "").strip() for a in answers}
        still: list[tuple[int, str]] = []
        for i, text in pending:
            bn = by_index.get(i, "")
            # A block with no letters at all ("12%", "1.5") has nothing to translate into
            # Bangla script, so for those the digits surviving IS the whole test.
            has_letters = any(ch.isalpha() for ch in text)
            if bn and _keeps_numbers(text, bn) and (_is_valid_bangla(bn) or not has_letters):
                blocks[i]["bn"] = bn
            else:
                still.append((i, text))
        pending = still

    if pending:
        logger.info(
            "%d block(s) fell back to per-block translation: %s",
            len(pending), "; ".join(t[:30] for _, t in pending[:3]),
        )


def _translate_one(text: str) -> str:
    """Translate a single block. Returns the original text if nothing valid came back."""
    for attempt in range(1, TEXT_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=TEXT_MODEL,
                contents=[_SINGLE_TRANSLATE_PROMPT.format(text=text)],
                config=types.GenerateContentConfig(temperature=1.0),
            )
            bn = response.text.strip()
            if _is_valid_bangla(bn) and _keeps_numbers(text, bn):
                return bn
        except Exception as exc:
            logger.warning("Single-block translation attempt %d failed: %s",
                           attempt, str(exc)[:160])
    return text


def _resolve(block: dict) -> str:
    """The text to draw for one block: its Bangla if there is any, otherwise its own words.

    A failed translation falls back to the original words rather than to nothing. By the time
    this is drawn the box has been painted out, so returning "" would not leave the English
    standing — it would delete it. Untranslated English is a shortcoming; a blank where a label
    used to be is a defect.
    """
    text = (block.get("text") or "").strip()
    if not text:
        return ""
    bn = (block.get("bn") or "").strip()
    if bn and _is_valid_bangla(bn):
        return bn
    if (block.get("lang") or "").startswith("bn") or _is_valid_bangla(text):
        return text  # already Bangla — keep it exactly
    if not any(ch.isalpha() for ch in text):
        # A page number, a price, a version string: there is nothing here to translate INTO
        # Bangla, and the digits stay Latin either way. Asking anyway costs a model call and
        # comes back with the same characters.
        return text
    translated = _translate_one(text)
    if translated != text:
        return translated
    logger.warning("Could not translate %r — keeping the original words rather than a blank",
                   text[:60])
    return text


def _redraw(png_bytes: bytes, categories: list[str], palette: str) -> bytes | None:
    """Send the picture to the image model. Returns new bytes, or None to keep the original."""
    instruction = EDIT_INSTRUCTION.format(
        focus=", ".join(categories) if categories else "any culturally-specific content",
        palette=palette,
    )
    contents = [instruction, types.Part.from_bytes(data=png_bytes, mime_type="image/png")]
    for attempt in range(1, EDIT_ATTEMPTS + 1):
        temperature = 0.2 + (attempt - 1) * 0.3
        try:
            response = generate_content(
                model=EDIT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"], temperature=temperature
                ),
                # Image generation is legitimately slower than a text call.
                timeout_ms=IMAGE_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.warning("Redraw attempt %d/%d failed: %s", attempt, EDIT_ATTEMPTS, str(exc)[:200])
            continue
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    logger.info("Redraw succeeded on attempt %d", attempt)
                    return inline.data
        logger.warning("Redraw attempt %d/%d returned no image (finish_reason=%s)",
                       attempt, EDIT_ATTEMPTS,
                       getattr((getattr(response, "candidates", None) or [None])[0],
                               "finish_reason", None))
    return None


# --------------------------------------------------------------------------------------
# Protecting the marks
# --------------------------------------------------------------------------------------


def _logo_named(text: str, logos: list[dict]) -> bool:
    """True if a text block simply IS the name of a detected mark.

    Geometry alone is not enough: the OCR and the logo detector are two independent model
    calls that do not have to agree on where a mark is, and a wordmark boxed in two different
    places by the two of them would sail through the overlap test below. The names match when
    the boxes do not, so both tests are applied and either one protects.
    """
    name = " ".join(text.lower().split())
    if len(name) < 4:
        return False  # too short to match a label with any confidence
    for logo in logos:
        label = " ".join((logo.get("label") or "").lower().split())
        if len(label) >= 4 and (label in name or name in label):
            return True
    return False


def _outside_logos(blocks: list[dict], logos: list[dict]) -> list[dict]:
    """Drop the text blocks that belong to a mark, so its wording is never touched.

    An organisation's name is the one string here that must survive in its own language and
    its own lettering. Both things that would otherwise happen to it — the erase and the Bangla
    drawn over it — work from this list, so removing the mark's blocks keeps both off it.
    """
    if not logos or not blocks:
        return blocks
    kept: list[dict] = []
    for block in blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        area = max((bx1 - bx0) * (by1 - by0), 1e-9)
        inside = _logo_named(block.get("text") or "", logos)
        for logo in logos:
            if inside:
                break
            lx0, ly0, lx1, ly1 = logo["bbox"]
            overlap = (
                max(0.0, min(bx1, lx1) - max(bx0, lx0))
                * max(0.0, min(by1, ly1) - max(by0, ly0))
            )
            if overlap / area >= LOGO_BLOCK_OVERLAP:
                inside = True
        if inside:
            logger.info("Leaving %r as printed — it is part of a mark",
                        (block.get("text") or "")[:40])
        else:
            kept.append(block)
    return kept


def _restamp_logos(edited_png: bytes, source_png: bytes, logos: list[dict]) -> tuple[bytes, int]:
    """Paste each mark's region from the original back over the redraw.

    A logo is a real organisation's identity: it may not be redrawn, restyled, recoloured or
    translated. The edit model cannot reproduce one faithfully — asked to keep a mark it
    invents a similar-looking one, and asked to blank the lettering it deletes the mark — so
    the only sound answer is to put the original pixels back.

    Returns (png_bytes, stamped_count). The boxes are fractions, so the two pictures do not
    have to be the same size.
    """
    if not logos:
        return edited_png, 0
    try:
        with Image.open(io.BytesIO(edited_png)) as raw:
            edited = raw.convert("RGB")
        with Image.open(io.BytesIO(source_png)) as raw:
            source = raw.convert("RGB")
    except Exception:
        logger.exception("Could not open the images to restamp marks; keeping the redraw")
        return edited_png, 0

    stamped = 0
    for logo in logos:
        x0, y0, x1, y1 = logo["bbox"]
        if (x1 - x0) > LOGO_REGION_MAX_SHARE and (y1 - y0) > LOGO_REGION_MAX_SHARE:
            logger.info(
                "Not restamping %r: its box covers the whole picture, which is the "
                "logo_fills_image case rather than a mark inside a picture",
                logo.get("label", "")[:60],
            )
            continue
        x0, y0 = max(0.0, x0 - LOGO_REGION_PAD), max(0.0, y0 - LOGO_REGION_PAD)
        x1, y1 = min(1.0, x1 + LOGO_REGION_PAD), min(1.0, y1 + LOGO_REGION_PAD)

        def box(img: Image.Image) -> tuple[int, int, int, int]:
            return (
                int(x0 * img.width), int(y0 * img.height),
                int(round(x1 * img.width)), int(round(y1 * img.height)),
            )

        src_box, dst_box = box(source), box(edited)
        if src_box[2] - src_box[0] < 2 or src_box[3] - src_box[1] < 2:
            continue
        target = (dst_box[2] - dst_box[0], dst_box[3] - dst_box[1])
        if target[0] < 1 or target[1] < 1:
            continue
        try:
            patch = source.crop(src_box)
            if patch.size != target:
                patch = patch.resize(target, Image.LANCZOS)
            edited.paste(patch, dst_box[:2])
            stamped += 1
        except Exception:
            logger.exception("Could not restamp %r", logo.get("label", "")[:60])

    if not stamped:
        return edited_png, 0
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    return buf.getvalue(), stamped


# --------------------------------------------------------------------------------------
# Drawing the Bangla back into the picture
# --------------------------------------------------------------------------------------


def _denorm(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box = (
        max(0, min(width - 1, int(x0 * width))),
        max(0, min(height - 1, int(y0 * height))),
        max(1, min(width, int(round(x1 * width)))),
        max(1, min(height, int(round(y1 * height)))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return (box[0], box[1], box[0] + 1, box[1] + 1)
    return box


def _surface_color(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The colour of the surface a text box sits on, sampled from the middle of the box.

    See TEXT_SURFACE_SAMPLE: the middle is the surface, the edges straddle its neighbours.
    Sampled with the text still present, so the median has to survive the ink — which is what
    a median is for, since the ink is the minority of the pixels in a text box.
    """
    x0, y0, x1, y1 = box
    inset_x = int((x1 - x0) * (1 - TEXT_SURFACE_SAMPLE) / 2)
    inset_y = int((y1 - y0) * (1 - TEXT_SURFACE_SAMPLE) / 2)
    middle = (
        x0 + inset_x, y0 + inset_y,
        max(x0 + inset_x + 1, x1 - inset_x), max(y0 + inset_y + 1, y1 - inset_y),
    )
    return _median_color(list(img.crop(middle).getdata())) or (255, 255, 255)


def _same_surface(img: Image.Image, strip: tuple[int, int, int, int],
                  surface: tuple[int, int, int]) -> bool:
    """True if a strip of pixels is still the surface the text sits on, within tolerance."""
    if strip[2] <= strip[0] or strip[3] <= strip[1]:
        return False
    colour = _median_color(list(img.crop(strip).getdata()))
    if colour is None:
        return False
    return all(abs(colour[i] - surface[i]) <= TEXT_SURFACE_TOLERANCE for i in range(3))


def _grow_into_surface(
    img: Image.Image,
    box: tuple[int, int, int, int],
    surface: tuple[int, int, int],
    blocked: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Extend a text box up and down through blank surface, so the Bangla can be set larger.

    `img` is the picture the box was measured on (the upload), `surface` the colour that box's
    text sits on, `blocked` every other block's box — grown into, they would overlap.

    Vertical only. Line height is what squeezes Bangla, and growing sideways is where the
    collisions are: two labels side by side on one bar share no gap worth taking.
    """
    x0, y0, x1, y1 = box
    height = max(y1 - y0, 1)
    step = max(1, int(height * TEXT_BOX_GROW_STEP))
    budget = int(height * TEXT_BOX_MAX_GROWTH)
    top, bottom = y0, y1

    def free(a: int, b: int) -> bool:
        return not any(
            bx0 < x1 and x0 < bx1 and by0 < b and a < by1 for bx0, by0, bx1, by1 in blocked
        )

    gained = 0
    while gained < budget:
        moved = False
        if bottom + step <= img.height and _same_surface(
            img, (x0, bottom, x1, bottom + step), surface
        ) and free(bottom, bottom + step):
            bottom += step
            gained += step
            moved = True
        if gained < budget and top - step >= 0 and _same_surface(
            img, (x0, top - step, x1, top), surface
        ) and free(top - step, top):
            top -= step
            gained += step
            moved = True
        if not moved:
            break
    return (x0, top, x1, bottom)


def _ink_color(img: Image.Image, box: tuple[int, int, int, int],
               surface: tuple[int, int, int]) -> str:
    """The colour the original text was written in, so the Bangla is written in it too.

    Black-on-white is the easy case and a luminance test would handle it. This exists for the
    other ones: a blue link, a red warning, a white button label on a coloured chip. Falls back
    to black or white by the surface's luminance when nothing in the box is far enough from the
    surface to be ink — which is what happens after a redraw, where the surface came back blank.
    """
    fallback = "#000000" if (
        0.299 * surface[0] + 0.587 * surface[1] + 0.114 * surface[2]
    ) > TEXT_LIGHT_BG else "#ffffff"
    try:
        crop = img.crop(box)
        # Quantize before hunting for the ink: antialiasing puts a smear of intermediate
        # shades around every glyph, and the true ink colour is a small population among them.
        reduced = crop.convert("P", palette=Image.ADAPTIVE, colors=8).convert("RGB")
        counts = reduced.getcolors(maxcolors=64) or []
    except Exception:
        return fallback
    if not counts:
        return fallback

    total = sum(n for n, _ in counts) or 1
    best, best_distance = None, 0
    for n, rgb in counts:
        # Ink is the minority of a text box by definition. A colour covering most of the box is
        # the surface or a neighbouring panel, not the writing.
        if n / total > 0.45:
            continue
        distance = sum(abs(rgb[i] - surface[i]) for i in range(3))
        if distance > best_distance:
            best, best_distance = rgb, distance
    # Well clear of the surface, or it is a shadow, an edge or a compression artifact rather
    # than the writing.
    if best is None or best_distance < 120:
        return fallback
    return _hex(best)


def _bake_text(png_bytes: bytes, source_png: bytes, blocks: list[dict]) -> bytes:
    """Paint out each block's box and draw its Bangla in the same place.

    `png_bytes` is the picture as it stands (redrawn or original); `source_png` is always the
    upload, and is read only for colours — the surface a block sits on and the colour its
    writing was in. After a redraw those surfaces have come back blank, so the original is the
    only place that information still exists.

    The Bangla is rendered by `insert_htmlbox` on a scratch PDF page rather than by PIL:
    Bengali needs conjunct shaping and pre-base matra reordering, which PIL only does when it
    was built against libraqm and otherwise gets silently wrong. `insert_htmlbox` goes through
    MuPDF's HarfBuzz.
    """
    blocks = [b for b in blocks if (b.get("bn_final") or "").strip()]
    if not blocks:
        return png_bytes

    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
        with Image.open(io.BytesIO(source_png)) as raw:
            source = raw.convert("RGB")
    except Exception:
        logger.exception("Could not open the picture to draw text into; keeping it as it is")
        return png_bytes

    # Enlarge before anything is drawn, so the Bangla is rendered at the larger size rather
    # than scaled up with the picture afterwards. See TEXT_BAKE_MIN_DIM.
    upscale = min(TEXT_BAKE_MAX_UPSCALE, TEXT_BAKE_MIN_DIM / max(max(img.size), 1))
    if upscale > 1.0:
        img = img.resize((round(img.width * upscale), round(img.height * upscale)), Image.LANCZOS)

    width, height = img.size
    draw = ImageDraw.Draw(img)
    placements: list[tuple[tuple[int, int, int, int], str, str, str]] = []

    # Measured on the upload, once, because each block's growth has to know where all the
    # others are: a box that grows into its neighbour's is worse than one set small.
    source_boxes = [_denorm(b["bbox"], source.width, source.height) for b in blocks]

    for index, (block, source_box) in enumerate(zip(blocks, source_boxes)):
        text = block["bn_final"].strip()
        box = _denorm(block["bbox"], width, height)

        fill = _surface_color(source, source_box)
        colour = _ink_color(source, source_box, fill)
        # Erase whatever is in the box. On the redraw path the model was told to hand text
        # surfaces back blank and mostly does, but models leak lettering; on the keep-the-
        # pixels path the English is still there and this is the only thing that removes it.
        # Either way it is deterministic, where the model is not.
        draw.rectangle(box, fill=fill)

        roomy = _grow_into_surface(
            source, source_box, fill,
            [b for i, b in enumerate(source_boxes) if i != index],
        )
        if roomy == source_box:
            # Nothing to grow into, so fall back to the fixed allowance for Bangla's ascenders
            # and descenders.
            grow = int((box[3] - box[1]) * TEXT_BOX_GROW)
            target = (
                max(0, box[0] - grow), max(0, box[1] - grow),
                min(width, box[2] + grow), min(height, box[3] + grow),
            )
        else:
            target = _denorm(
                [roomy[0] / source.width, roomy[1] / source.height,
                 roomy[2] / source.width, roomy[3] / source.height],
                width, height,
            )
        wide = (box[2] - box[0]) / max(width, 1) >= TEXT_LEFT_ALIGN_MIN_WIDTH
        placements.append((target, text, colour, "left" if wide else "center"))

    overlay = _render_text_overlay(placements, width, height)
    if overlay is not None:
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_text_overlay(
    placements: list[tuple[tuple[int, int, int, int], str, str, str]], width: int, height: int
) -> Image.Image | None:
    """Render the Bangla as a transparent RGBA layer the size of the picture.

    One scratch page at one point per pixel, so a box in image pixels is the same rectangle in
    page points. A fresh page has no content, so `get_pixmap(alpha=True)` returns everything
    except the glyphs fully transparent.
    """
    scale = min(1.0, TEXT_RENDER_MAX_DIM / max(width, height, 1))
    page_w, page_h = max(1, int(width * scale)), max(1, int(height * scale))

    scratch = None
    try:
        scratch = fitz.open()
        page = scratch.new_page(width=page_w, height=page_h)
        archive = fitz.Archive(FONTS_DIR)
        for (x0, y0, x1, y1), text, colour, align in placements:
            rect = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
            css = CSS_TEMPLATE.format(
                # A starting size; scale_low=0.0 lets insert_htmlbox shrink from here as far
                # as it needs to. Given no floor it always finds a scale that fits, which is
                # the guarantee that matters — at a floor it cannot meet it draws nothing.
                size=max(4.0, rect.height * 0.62),
                color=colour,
                align=align,
            )
            page.insert_htmlbox(rect, html.escape(text), css=css, scale_low=0.0, archive=archive)

        pixmap = page.get_pixmap(alpha=True)
        overlay = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception:
        logger.exception("Could not render the Bangla; leaving the text surfaces as they are")
        return None
    finally:
        if scratch is not None:
            scratch.close()

    if overlay.size != (width, height):
        overlay = overlay.resize((width, height), Image.LANCZOS)
    return overlay


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------


def localize_single_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> bytes:
    """Localize one uploaded picture: its words translated into Bangla, its marks untouched,
    and its artwork redrawn for a Bangladeshi audience where that is safe to do.

    Returns the localized bytes in the format they arrived in, or the upload unchanged if
    there was nothing to do or nothing could be done.
    """
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    normalized = _to_png(image_bytes)
    if normalized is None:
        logger.warning("Could not decode the upload — returning it untouched")
        return image_bytes
    png, width, height = normalized
    model_png = _downscale_for_model(png)

    decision = _classify(model_png)

    # A picture that IS a mark is never ours to change — checked before OCR, deliberately, so
    # a wordmark cannot have its lettering read and translated on the way to deciding to keep it.
    if decision["is_logo"] and decision["logo_fills_image"]:
        logger.info("Upload is a logo or brand mark — returned exactly as it arrived (%s)",
                    decision["reason"])
        return image_bytes

    # Marks INSIDE the picture: found on the original, before anything is redrawn. Two things
    # depend on this list — their pixels go back over the redraw, and their wording is dropped
    # from the translation.
    logos = _detect_logos(model_png) if (decision["has_text"] or decision["needs_localization"]) else []
    if logos:
        logger.info("Upload carries %d mark(s) (%s) — left exactly as printed",
                    len(logos), "; ".join((lg.get("label") or "?")[:30] for lg in logos[:3]))

    # Read the words while they are still there to read.
    blocks: list[dict] = []
    ocr_failed = False
    if decision["has_text"]:
        found = _ocr_blocks(model_png)
        if found is None:
            ocr_failed = True
        else:
            blocks = _outside_logos(found, logos)

    if blocks:
        _translate_blocks(model_png, blocks)
        for block in blocks:
            block["bn_final"] = _resolve(block)
        blocks = [b for b in blocks if (b.get("bn_final") or "").strip()]

    # Whether the artwork may be redrawn. Three separate reasons not to, and each of them on
    # its own is enough:
    #   - the classifier found nothing culturally adaptable;
    #   - the picture's value is its layout (a screenshot, a form, a chart), so a redraw
    #     destroys the thing that was uploaded;
    #   - the OCR failed, so nobody knows what the words were. The edit model returns every
    #     lettered surface BLANK and the words are restored from a list this run does not
    #     have, so redrawing now would delete them. Checked before the generation is paid for.
    layout_locked = decision["layout_locked"] or len(blocks) >= LAYOUT_LOCKED_MIN_BLOCKS
    redraw = decision["needs_localization"] and not layout_locked and not ocr_failed
    if decision["needs_localization"] and layout_locked:
        logger.info(
            "Keeping the artwork as uploaded (%d text block(s), layout_locked=%s): its layout "
            "is what it is for. Its text is still translated.",
            len(blocks), decision["layout_locked"],
        )
    if decision["needs_localization"] and ocr_failed:
        logger.error(
            "Not redrawing: the picture's text could not be read, and a redraw returns every "
            "lettered surface blank — it would delete words nothing can put back."
        )

    result = png
    redrawn = False
    if redraw:
        generated = _redraw(model_png, decision["categories"], _palette_summary(model_png))
        if generated is None:
            logger.warning("Redraw failed — keeping the original artwork; its text is still "
                           "translated")
        else:
            result = _resize_to(generated, width, height)
            redrawn = True
            if logos:
                result, stamped = _restamp_logos(result, png, logos)
                logger.info("%d of %d mark(s) stamped back from the original", stamped, len(logos))

    # Which blocks get drawn. After a redraw every one of them must: the model handed those
    # surfaces back blank, so a block skipped here is a word deleted. Without a redraw the
    # pixels are the upload's own, so only blocks whose text actually CHANGED are touched —
    # re-rendering a line as itself would erase real type and set it again in a different
    # face, which is a loss and never a gain.
    if redrawn:
        to_draw = blocks
    else:
        to_draw = [b for b in blocks if b["bn_final"].strip() != (b.get("text") or "").strip()]

    if to_draw:
        result = _bake_text(result, png, to_draw)

    if not redrawn and not to_draw:
        logger.info("Nothing to change — returning the upload untouched (%s)", decision["reason"])
        return image_bytes

    logger.info(
        "Localized: redrawn=%s, %d/%d text block(s) rewritten in Bangla, %d mark(s) protected",
        redrawn, len(to_draw), len(blocks), len(logos),
    )
    return _encode_as(result, mime_type)
