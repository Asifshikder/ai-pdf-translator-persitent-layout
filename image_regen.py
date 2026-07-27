"""Image regeneration pipeline: redraw every picture in a PDF for a Bangladeshi audience.

A deliberately small, second pipeline, independent of `image_processor.localize_pdf`.
Its whole design follows from one property of PDF:

    An image XObject is always painted into a *unit square* which the placement matrix
    in the page's content stream maps onto the page. `doc.update_stream(xref, ...)`
    rewrites the object's pixels and never touches that matrix.

So swapping an image's bytes cannot move it, cannot resize it on the page, cannot
reorder it, and cannot disturb the text layer — no matter how many pixels the new image
has. Every raster image in the document, including a full-bleed one fixed to the page
layout, is handled by that one mechanism and nothing else. No redaction, no insertion,
no overlay, therefore no possible overlap.

Vector-drawn artwork is the one thing that cannot be done that way — it is not an image
object, so there are no pixels to swap and the page content *must* be edited. That path
is guarded instead (see `_accept_vector_regions`): a region is only taken when it holds
no page text and touches no other picture, which is what makes a foreground insert safe.

Text baked into a picture is translated and drawn back *into the image*, never onto the
page, so the document's own text layer is never involved.

Every image ends up with a status in the audit log, so "nothing was skipped" is a fact
that can be checked rather than a claim.
"""

import html
import io
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from google.genai import types
from PIL import Image, ImageDraw

import image_regions
from image_localizer import _extract_text_blocks, resolve_block_text
from pdf_processor import CSS_TEMPLATE, FONTS_DIR, _panels, _rules, _vector_marks
from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

REGEN_SUFFIX = "_regen.pdf"  # keep in sync with MODES.regenerate.suffix in static/index.html

CLASSIFY_MODEL = "gemini-2.5-flash"
# Image models tried in order. If the primary is rate-limited or refuses, the next one is an
# independent path — transient 429/503 backoff itself is handled inside vertex_client.
EDIT_MODELS = ["gemini-2.5-flash-image", "gemini-3.1-flash-lite-image", "gemini-3.1-flash-image"]
EDIT_ATTEMPTS_PER_MODEL = 2

# 2, not more: the image model's requests-per-minute limit is what produces 429s, and more
# workers simply spend the retry budget faster.
CONCURRENT_IMAGES = 2

MODEL_MAX_DIM = 1536  # longest side of the copy sent to the model

# Beyond this ratio of long side to short side, an image is a decorative strip, a bleed
# fragment or a rule — not a picture of a person, a meal or a place, because none of those is
# ever drawn that thin. It is also the shape a regeneration handles worst: measured on page 39
# of the Heart Failure manual, a 114x1381 sliver holding one green leaf came back with the leaf
# shifted a few percent to the right, and since the page crops all but the leftmost quarter of
# that image, the shift moved the leaf out of view and left a white gap down the page edge.
# Nothing cultural was on offer in exchange.
MAX_PICTURE_ASPECT = 6.0
CONTEXT_MARGIN = 60.0  # pt of page around a picture read as its context
CONTEXT_MAX_CHARS = 900
CONTEXT_MIN_WORDS = 8  # below this the tight read is abandoned for the whole page

# The scratch page used to render Bangla is one point per pixel, so a very large picture
# would allocate a very large pixmap. Above this the overlay is rendered smaller and scaled.
TEXT_RENDER_MAX_DIM = 3000

# A picture is enlarged to at least this before text is baked into it. Bangla needs far more
# pixels than Latin to stay legible — matras, conjuncts and the headstroke all have to resolve
# — and the alcohol-unit tiles in this manual are only 180px square, where a caption lands
# about seven pixels tall and comes out as mush. Enlarging costs nothing on the page: the
# placement rect is fixed, so more pixels simply print sharper, and the artwork itself is no
# worse for a LANCZOS upscale than it was at its own size. Only the text gets genuinely
# sharper, because it is drawn *after* the enlargement rather than scaled up with the picture.
TEXT_BAKE_MIN_DIM = 800
TEXT_BAKE_MAX_UPSCALE = 4.0
# An OCR box is drawn tight around Latin text; Bangla matras reach above the ascender and
# conjuncts below the baseline, so rendering into that exact box shrinks the type hard. Kept
# small: a box grown past the coloured bar its text sits on puts pale type on white paper.
TEXT_BOX_GROW = 0.08  # fraction of the box's own height, added top and bottom
# The share of a text box sampled to decide what colour its surface is. Measured from the
# middle rather than the border ring: an OCR box is drawn around the *text*, so its edges
# straddle whatever the text sits on. On the alcohol-units page every caption is white type on
# a blue bar barely taller than the words, and the ring caught as much white paper as blue —
# the median came back white, which painted the bar out and left the Bangla on a pale patch.
# The model is told to return text surfaces blank, so the middle of the box is that surface.
TEXT_SURFACE_SAMPLE = 0.6
TEXT_LIGHT_BG = 140  # mean luminance above which a box gets black text rather than white

CATEGORIES = ["people_attire", "scenes_settings", "food_objects", "signage_text"]


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are preparing a health booklet for Bangladeshi readers. Decide what \
should happen to this image.

FIRST answer is_logo. A logo is a real organisation's identity and is never ours to change — \
not its drawing, not its colours, and above all not its wording. Answer true for any logo, \
wordmark, lettermark, emblem, crest, seal, coat of arms, badge or roundel; any brand or product \
mark; any hospital, charity, trust, university, government, ministry, NHS or WHO mark; a name or \
initials set as an organisation's identity; a strapline printed as part of such a mark; and \
copyright lines, registration numbers, ISBNs and publisher imprints. A crest containing a lion \
is a crest, not a picture of a lion. If you are unsure, answer true — a redrawn logo \
misrepresents a real organisation, while a logo left alone costs the document nothing.

Then answer needs_regeneration. Answer true if the image shows anything culturally specific that \
could be Bangladeshi instead: people or faces of any kind (photo, illustration, cartoon or icon), \
streets, buildings, rooms, landscapes, furniture, vehicles, household objects, or food and drink. \
Be decisive: a generic "Western" or "international" look is exactly what should be adapted, so do \
not keep an image merely because it looks neutral.

Answer needs_regeneration=false ONLY for:
- Pure medical or clinical imagery whose purpose is technical: x-rays, CT scans, ECG traces, \
anatomical diagrams, clinical reference figures.
- Functional elements that must survive intact: QR codes, barcodes, UI chrome, buttons, arrows, \
plain charts and graphs of abstract quantities.
- Images that already look authentically Bangladeshi.
When is_logo is true, needs_regeneration MUST be false.

Then answer has_text: true if any readable words, letters or numbers are printed in the image.

Finally list the categories that apply, and give a one-line reason."""

# Written fresh rather than shared with image_localizer, so the food rules below are explicit
# and the existing /localize prompts keep behaving exactly as they do today.
EDIT_INSTRUCTION = """Redraw this image so it belongs in a health booklet published in \
Bangladesh, for Bangladeshi readers.
{context_clause}
CRITICAL — KEEP THE SAME PICTURE:
  - Keep the EXACT composition, framing, camera angle, aspect ratio and dimensions. The same
    number of subjects, in the same positions, at the same scale, doing the same thing.
  - Do not add or remove objects, do not change the layout, do not introduce new elements
    beyond the cultural adaptation itself.

CRITICAL — MATCH THE ORIGINAL'S COLOURS:
  - This picture is printed inside a document and has to keep looking like it belongs to the
    page around it, so the palette is not yours to change.
  - Reproduce the SAME hues, the same lightness, the same saturation, the same tone.{palette}
  - Do NOT boost saturation, warm the image up, add accent colours, or restyle it. A
    recoloured picture is a failure even if it looks good on its own.
  - If the original is a flat line drawing or a two- or three-colour illustration, keep exactly
    that style — do not turn it into a photo, a painting, or a shaded 3D render.
  - The output must be CLEARER than the original: cleaner lines, sharper edges, no blur or
    compression artifacts. Clarity comes from draughtsmanship, not from stronger colour.

CRITICAL — FOOD RULES. All three apply, in this order:
  1. HALAL ONLY. Never depict pork, ham, bacon, lard, alcohol, beer, wine, or a wine glass.
     If the original shows one, replace it with a halal food filling the same role.
  2. KEEP THE NUTRITIONAL MEANING. This picture is printed in a health booklet, where a food
     is very often shown to represent a food group, a portion size, a measure or a dose. The
     replacement MUST be in the SAME food group and show the SAME portion:
       oily fish -> ilish or rui (never dal); wholegrain -> lal chal or atta ruti (never white
       rice); leafy vegetable -> lal shak or palong shak; pulse -> dal; dairy -> doi or milk;
       fruit -> a fruit. Never swap across food groups, and never change how much food is
       shown or how many items are on the plate.
  3. MAKE IT BANGLADESHI. Subject to rules 1 and 2, replace Western dishes with everyday
     Bangladeshi food — bhat, dal, machher jhol, shobji, cha — served on Bangladeshi plates
     and eaten with Bangladeshi utensils.

PEOPLE: any visible person becomes Bangladeshi in skin tone and features, dressed as fits their
age, gender and role (saree, salwar kameez, panjabi, hijab, lungi, or ordinary modern clothes).
Keep their poses, gestures and expressions exactly as engaged and natural as the original's.

SETTINGS & OBJECTS: architecture, streets, vehicles, rooms, furniture and household items become
Bangladeshi — drawn in the original's palette and style, not photographed anew.

CRITICAL — NO TEXT:
  - Do NOT draw, write, render or hallucinate ANY text, letters, words, numbers or symbols.
  - Every sign, placard, board, label, poster or lettered surface must come back BLANK and
    CLEAN — empty paper or board of the same shape, colour and material.
  - The text is translated and put back separately afterwards, so leaving it out is required,
    not a mistake.

CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate or invent any logo, wordmark, emblem, crest,
    badge or institutional mark, and never substitute one organisation's mark for another's.
  - Leave the area a mark occupies as clean, empty background of the surrounding colour.

Focus especially on: {focus}."""

CONTEXT_CLAUSE = """
CONTEXT — the page this picture is printed on says:
"{context}"
Use it to keep the picture's MEANING intact: the same activity, the same kind of objects, the
same number of people doing the same thing. It tells you what the picture is FOR — it is not a
list of things to add, and not one word of it may be written into the image.
"""

_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_logo": {"type": "BOOLEAN"},
        "needs_regeneration": {"type": "BOOLEAN"},
        "has_text": {"type": "BOOLEAN"},
        "categories": {"type": "ARRAY", "items": {"type": "STRING", "enum": CATEGORIES}},
        "reason": {"type": "STRING"},
    },
    "required": ["is_logo", "needs_regeneration", "has_text", "categories", "reason"],
}


# --------------------------------------------------------------------------------------
# Image helpers
#
# Copied from image_processor rather than imported, so this module stands alone and the old
# pipeline can be deleted without touching it. The docstrings come with them: they record why
# each is written the way it is, and that reasoning is the expensive part.
# --------------------------------------------------------------------------------------


def _to_png(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """Normalize any extracted image to PNG bytes. Returns (png_bytes, w, h), or None if the
    bytes cannot be decoded (in which case the original image is kept)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), width, height
    except Exception:
        logger.debug("Could not decode extracted image; keeping the original", exc_info=True)
        return None


def _downscale_for_model(png_bytes: bytes, max_dim: int = MODEL_MAX_DIM) -> bytes:
    """A copy scaled so its longest side is <= max_dim, for cheaper and faster model calls.

    OCR bboxes come back normalized (0..1), so downscaling the copy sent to the model does not
    affect how the resulting text maps back onto the full-size original.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            longest = max(img.size)
            if longest <= max_dim:
                return png_bytes
            scale = max_dim / longest
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
    """Fit the generated image to the original's aspect ratio, keeping extra resolution.

    The placement rect on the page is fixed, so only the *aspect ratio* has to match — a
    higher-resolution image in the same proportions simply prints sharper. A different aspect
    ratio, though, would render stretched, so that case is resampled to the original dims.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            source_aspect = width / max(height, 1)
            new_aspect = img.width / max(img.height, 1)
            aspect_matches = abs(source_aspect - new_aspect) / source_aspect < 0.01

            if not (aspect_matches and img.width * img.height >= width * height):
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.exception("Could not resize the generated image; using it as-is")
        return image_bytes


def _swap_image_in_place(doc: fitz.Document, xref: int, png_bytes: bytes) -> bool:
    """Rewrite an image XObject's pixels, leaving the page content stream untouched.

    This is what keeps the replacement visible *and* in place. A page's layout routinely
    draws an opaque matte behind a picture — every illustration in these manuals sits on a
    white rectangle of its own size — so the real z-order is `matte, image, text`.
    `insert_image` cannot express that: its `overlay` flag only chooses "before all existing
    content" or "after" it. Inserting in the background puts the new image *under* the matte,
    which paints straight over it; inserting in the foreground buries any text drawn over it.

    Rewriting the object itself sidesteps the choice entirely: the drawing operator stays
    exactly where the layout put it, so the new pixels land in the old picture's place in the
    z-order. Every placement of the xref updates at once, rotated and sheared ones included,
    and no redaction is needed anywhere.

    Not `page.replace_image`: that leaves a duplicate resource reference which Adobe and
    Chrome resolve back to the *original* image, so the edit is in the file but invisible in
    a viewer. Rewriting the object in place cannot produce a duplicate.

    Returns True if the swap was applied; on failure the original image is left alone.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            img = img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88, optimize=True)

        doc.update_stream(xref, buf.getvalue(), new=True, compress=False)
        doc.xref_set_key(xref, "Width", str(width))
        doc.xref_set_key(xref, "Height", str(height))
        doc.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
        doc.xref_set_key(xref, "BitsPerComponent", "8")
        doc.xref_set_key(xref, "Filter", "/DCTDecode")
        # Keys inherited from the old stream that no longer describe this one. /ImageMask and
        # /Decode must go or a former 1-bit stencil keeps being painted as a stencil.
        #
        # /SMask is deliberately NOT cleared. A soft mask is scaled to the base image by the
        # viewer (PDF 32000-1, 8.9.6.4), so the original one still fits — and it is what keeps
        # a cut-out figure a cut-out. Dropping it turns the picture into an opaque rectangle
        # that covers whatever panel it was floating over, which reads as overlap on the page.
        # The trade is that the old silhouette also clips the new picture, so a redrawn figure
        # whose outline moved can lose an edge. An occasional clipped elbow is the cheaper
        # failure than a white box printed over the layout.
        for key in ("DecodeParms", "Decode", "Mask", "ImageMask", "Interpolate"):
            doc.xref_set_key(xref, key, "null")
        return True
    except Exception:
        logger.exception("xref %d: in-place image swap failed — keeping the original", xref)
        return False


# --------------------------------------------------------------------------------------
# Colour measurement
# --------------------------------------------------------------------------------------


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _ring_pixels(img: Image.Image, box: tuple[int, int, int, int] | None = None) -> list:
    """The outermost ring of pixels of the image, or of a box inside it.

    Sampled from the border rather than the whole area: a document illustration is drawn on a
    field (white paper, a painted wash, a coloured panel) and it is that field, not the
    subject, which has to keep matching the page around it.
    """
    if box is not None:
        img = img.crop(box)
    w, h = img.size
    if w < 4 or h < 4:
        return list(img.getdata())
    ring = list(img.crop((0, 0, w, 1)).getdata())
    ring += list(img.crop((0, h - 1, w, h)).getdata())
    ring += list(img.crop((0, 0, 1, h)).getdata())
    ring += list(img.crop((w - 1, 0, w, h)).getdata())
    return ring


def _median_color(pixels: list) -> tuple[int, int, int] | None:
    if not pixels:
        return None
    channels = list(zip(*pixels))
    return tuple(round(statistics.median(c)) for c in channels[:3])  # type: ignore[return-value]


# A background does not have to be perfectly flat to be a background: a painted wash or a JPEG
# gradient is obviously one colour to a reader while failing any strict test. The question that
# actually separates a background from a picture is looser — is most of the border near one
# colour? A photograph that bleeds to its edges has no such colour and answers None.
BACKGROUND_MIN_SHARE = 0.55
BACKGROUND_SPREAD = 60


def _background_color(img: Image.Image) -> tuple[int, int, int] | None:
    """The image's background colour, or None if it has no background to speak of."""
    ring = _ring_pixels(img)
    median = _median_color(ring)
    if median is None:
        return None
    near = sum(
        1 for px in ring if max(abs(a - b) for a, b in zip(px[:3], median)) <= BACKGROUND_SPREAD
    )
    if near / len(ring) < BACKGROUND_MIN_SHARE:
        return None
    return median


def _palette_summary(png_bytes: bytes, max_colors: int = 6) -> str:
    """Describe the source's colours to the edit model, as measured hex values.

    "Match the original's palette" does not survive a generative edit on its own; naming the
    colours does. Returns "" when the image cannot be read, and the prompt then falls back to
    its qualitative instruction.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
            background = _background_color(img)
            # Quantize first: a photograph has thousands of near-identical shades, and the
            # handful surviving quantization are the ones a reader would actually name.
            reduced = img.convert("P", palette=Image.ADAPTIVE, colors=max_colors).convert("RGB")
            counts = reduced.getcolors(maxcolors=max_colors * 4) or []
    except Exception:
        logger.debug("Could not measure the source palette", exc_info=True)
        return ""

    counts.sort(reverse=True)
    total = sum(n for n, _ in counts) or 1
    named = ", ".join(f"{_hex(rgb)} ({n / total:.0%})" for n, rgb in counts[:max_colors])
    if not named:
        return ""
    summary = (
        f"\n  - The original's colours, measured: {named}. Reuse these, not brighter versions."
    )
    if background is not None:
        # Stated as an absolute because the model treats a *described* background as a
        # suggestion and returns a "nicer" one — a warmer white, a gradient, a scene — which
        # reads as a picture borrowed from another book.
        summary += (
            f"\n  - Its background is EXACTLY {_hex(background)}: reproduce that hex value "
            f"across the whole background, flat and unchanged. Do not tint, shade, gradient "
            f"or replace it."
        )
    return summary


# --------------------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------------------


def _classify(png_bytes: bytes) -> dict:
    """Decide what to do with one image.

    Fails safe in the direction that cannot damage the document: on any error the image is
    reported as a logo-free, no-change-needed picture, so it is kept exactly as it is.
    """
    part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    for attempt in range(1, 4):
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[part],
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_CLASSIFY_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "needs_regeneration" in data:
                data["is_logo"] = bool(data.get("is_logo"))
                data["has_text"] = bool(data.get("has_text"))
                data["needs_regeneration"] = bool(data.get("needs_regeneration"))
                data.setdefault("categories", [])
                data.setdefault("reason", "")
                # Reconciled here rather than trusted: a mark answered is_logo=true *and*
                # needs_regeneration=true would otherwise be redrawn on the second answer alone.
                if data["is_logo"]:
                    data["needs_regeneration"] = False
                    data["categories"] = []
                return data
            logger.warning("Classify: unexpected response shape (%d/3): %r", attempt, response.text[:200])
        except Exception:
            logger.exception("Classify request failed (%d/3)", attempt)
    return {
        "is_logo": False,
        "needs_regeneration": False,
        "has_text": False,
        "categories": [],
        "reason": "classify failed",
    }


def _first_image_bytes(response) -> bytes | None:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def _regenerate(png_bytes: bytes, categories: list[str], context: str) -> bytes | None:
    """Ask an image model to redraw the picture. Returns None if every model refuses."""
    instruction = EDIT_INSTRUCTION.format(
        context_clause=CONTEXT_CLAUSE.format(context=context) if context else "",
        palette=_palette_summary(png_bytes),
        focus=", ".join(categories) or "any culturally-specific content",
    )
    part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")

    for model in EDIT_MODELS:
        for attempt in range(1, EDIT_ATTEMPTS_PER_MODEL + 1):
            try:
                response = generate_content(
                    model=model,
                    contents=[instruction, part],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        # Nudged up on a retry: a model that declined once at low temperature
                        # tends to decline again at exactly the same setting.
                        temperature=0.2 + 0.3 * (attempt - 1),
                    ),
                    timeout_ms=IMAGE_TIMEOUT_MS,
                )
            except Exception:
                logger.warning("%s: image request failed (%d/%d)", model, attempt, EDIT_ATTEMPTS_PER_MODEL)
                continue
            data = _first_image_bytes(response)
            if data:
                return data
            feedback = getattr(response, "prompt_feedback", None)
            logger.warning(
                "%s: returned no image (%d/%d); block_reason=%s",
                model, attempt, EDIT_ATTEMPTS_PER_MODEL, getattr(feedback, "block_reason", None),
            )
    return None


# --------------------------------------------------------------------------------------
# Baking translated text back into a picture
# --------------------------------------------------------------------------------------


def _denorm(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box = (
        max(0, min(width - 1, int(x0 * width))),
        max(0, min(height - 1, int(y0 * height))),
        max(1, min(width, int(x1 * width))),
        max(1, min(height, int(y1 * height))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return (box[0], box[1], box[0] + 1, box[1] + 1)
    return box


def _surface_color(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The colour of the surface a text box sits on, sampled from the middle of the box.

    See TEXT_SURFACE_SAMPLE: the middle is the surface, the edges are the neighbours.
    """
    x0, y0, x1, y1 = box
    inset_x = int((x1 - x0) * (1 - TEXT_SURFACE_SAMPLE) / 2)
    inset_y = int((y1 - y0) * (1 - TEXT_SURFACE_SAMPLE) / 2)
    middle = (x0 + inset_x, y0 + inset_y, max(x0 + inset_x + 1, x1 - inset_x),
              max(y0 + inset_y + 1, y1 - inset_y))
    return _median_color(list(img.crop(middle).getdata())) or (255, 255, 255)


def _bake_text(png_bytes: bytes, blocks: list[dict]) -> bytes:
    """Paint out the baked-in English and draw the Bangla in its place, inside the image.

    The Bangla is rendered by `insert_htmlbox` on a scratch PDF page rather than by PIL:
    Bengali needs conjunct shaping and pre-base matra reordering, which PIL only does when it
    was built against libraqm and otherwise gets silently wrong. `insert_htmlbox` goes through
    MuPDF's HarfBuzz, the same path that renders the translated document itself.

    Nothing here touches the PDF being processed — the text ends up in the picture's own
    pixels, so the document's text layer is not involved and cannot be overlapped.
    """
    # Nothing to draw means nothing to do: return the bytes untouched rather than enlarging and
    # re-encoding a picture for no reason.
    blocks = [b for b in blocks if (b.get("bn") or "").strip()]
    if not blocks:
        return png_bytes

    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
    except Exception:
        logger.exception("Could not open the image to bake text into; keeping it as generated")
        return png_bytes

    # Enlarge before anything is drawn, so the Bangla is rendered at the larger size rather
    # than scaled up with the picture. See TEXT_BAKE_MIN_DIM.
    upscale = min(TEXT_BAKE_MAX_UPSCALE, TEXT_BAKE_MIN_DIM / max(max(img.size), 1))
    if upscale > 1.0:
        img = img.resize(
            (round(img.width * upscale), round(img.height * upscale)), Image.LANCZOS
        )

    width, height = img.size
    draw = ImageDraw.Draw(img)
    placements: list[tuple[tuple[int, int, int, int], str, str]] = []

    for block in blocks:
        text = block["bn"].strip()
        box = _denorm(block["bbox"], width, height)

        # Erase whatever is there. The model is told to return text surfaces blank, but models
        # leak text, so this runs either way and is deterministic where the model is not.
        fill = _surface_color(img, box)
        draw.rectangle(box, fill=fill)

        grow = int((box[3] - box[1]) * TEXT_BOX_GROW)
        grown = (
            max(0, box[0] - grow),
            max(0, box[1] - grow),
            min(width, box[2] + grow),
            min(height, box[3] + grow),
        )
        luminance = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        placements.append((grown, text, "#000000" if luminance > TEXT_LIGHT_BG else "#ffffff"))

    overlay = _render_text_overlay(placements, width, height)
    if overlay is not None:
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_text_overlay(
    placements: list[tuple[tuple[int, int, int, int], str, str]], width: int, height: int
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
        for (x0, y0, x1, y1), text, color in placements:
            rect = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
            css = CSS_TEMPLATE.format(
                # A starting size; scale_low=0.0 lets insert_htmlbox shrink from here as far as
                # it needs to. Given no floor it always finds a scale that fits, which is the
                # guarantee that matters — at a floor it cannot meet, it draws nothing at all.
                size=max(4.0, rect.height * 0.62),
                color=color,
                align="center",
                bold="",
            )
            page.insert_htmlbox(rect, html.escape(text), css=css, scale_low=0.0, archive=archive)

        pixmap = page.get_pixmap(alpha=True)
        overlay = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception:
        logger.exception("Could not render the Bangla overlay; leaving the text surfaces blank")
        return None
    finally:
        if scratch is not None:
            scratch.close()

    if overlay.size != (width, height):
        overlay = overlay.resize((width, height), Image.LANCZOS)
    return overlay


# --------------------------------------------------------------------------------------
# Collecting the work
# --------------------------------------------------------------------------------------


@dataclass
class Job:
    """One picture to regenerate. All PyMuPDF reading is done before a job exists, so the
    worker threads touch nothing but bytes."""

    key: str
    kind: str  # "raster" | "vector"
    page_num: int
    png: bytes
    width: int
    height: int
    context: str
    xref: int = 0  # raster only
    rect: fitz.Rect | None = None  # vector only
    dedup_key: str = ""  # vector only: identical artwork on other pages reuses the result
    status: str = ""
    result: bytes | None = None
    record: dict = field(default_factory=dict)


def _page_context(page: fitz.Page, rect: fitz.Rect | None) -> str:
    """The text printed around a picture, as context for regenerating it.

    Read from the picture's own placement rect rather than from the page as a whole: what tells
    the model what a figure illustrates is the heading over it and the caption under it, and on
    a two-column page the first 300 characters are as likely to be about something else. Falls
    back to the whole page when the tight read comes back empty — a picture on a divider page
    has nothing beside it but still sits under a section title.
    """
    if page is None:
        return ""

    def read(target: fitz.Rect) -> str:
        try:
            return " ".join((page.get_textbox(target) or "").split())
        except Exception:
            return ""

    text = ""
    if rect is not None:
        margin = (-CONTEXT_MARGIN, -CONTEXT_MARGIN, CONTEXT_MARGIN, CONTEXT_MARGIN)
        text = read((rect + margin) & page.rect)
    if len(text.split()) < CONTEXT_MIN_WORDS:
        text = read(page.rect) or text
    return text[:CONTEXT_MAX_CHARS].strip()


def _collect_rasters(doc: fitz.Document, statuses: dict[str, str]) -> list[Job]:
    """One job per image XObject in the document.

    Deduped by xref: an image placed on ten pages is regenerated once, and the in-place swap
    updates all ten placements together — which is also why repeated artwork cannot come back
    looking different on different pages.
    """
    jobs: dict[int, Job] = {}
    for page_num, page in enumerate(doc, start=1):
        for entry in page.get_images(full=True):
            xref = entry[0]
            if xref in jobs or f"xref:{xref}" in statuses:
                continue
            key = f"xref:{xref}"

            # A stencil is a 1-bit mask painted with the page's fill colour, not a picture:
            # extract_image hands back a bare black-and-white mask with no colour context, and
            # a classifier reads that as a black square. Reported, never regenerated.
            if image_regions._is_image_mask(doc, xref):
                statuses[key] = "stencil_skipped"
                continue

            try:
                raw = doc.extract_image(xref)
            except Exception:
                logger.debug("xref %d: could not extract", xref, exc_info=True)
                statuses[key] = "extract_failed"
                continue

            normalized = _to_png(raw.get("image") or b"")
            if normalized is None:
                statuses[key] = "extract_failed"
                continue
            png, width, height = normalized

            rects = page.get_image_rects(xref)
            jobs[xref] = Job(
                key=key,
                kind="raster",
                page_num=page_num,
                png=png,
                width=width,
                height=height,
                context=_page_context(page, rects[0] if rects else None),
                xref=xref,
            )
    return list(jobs.values())


def _accept_vector_regions(
    page: fitz.Page, regions: list, raster_rects: list[fitz.Rect], statuses: dict[str, str]
) -> list:
    """Filter detected vector regions down to the ones it is provably safe to replace.

    Vector artwork is the one thing this pipeline cannot do by rewriting an image object, so
    the page content has to be edited: the drawn paths are redacted away and a raster is
    inserted in their place. That insert must go in the *foreground*, because a background
    panel excluded from the cluster would otherwise paint straight over it — and a foreground
    insert covers anything drawn under it.

    So the guards below are not caution, they are what makes the foreground insert safe:
    nothing may be under it that matters. A region holding page text is dropped, because an
    insert would bury that text; a region touching a picture or another region is dropped,
    because that is the one way this design could produce an overlap. Each rejection is
    recorded, so a dropped region is visible in the summary rather than silently missing.
    """
    accepted: list = []
    for region in sorted(regions, key=lambda r: r.rect.get_area(), reverse=True):
        if (page.get_text("text", clip=region.rect) or "").strip():
            statuses[region.key] = "vector_has_text"
            continue
        if any(region.rect.intersects(r) for r in raster_rects):
            statuses[region.key] = "vector_over_raster"
            continue
        if any(region.rect.intersects(other.rect) for other in accepted):
            statuses[region.key] = "vector_overlap"
            continue
        accepted.append(region)
    return accepted


def _collect_vectors(doc: fitz.Document, statuses: dict[str, str]) -> list[Job]:
    """One job per vector-drawn illustration that passes the guards.

    Detection is `image_regions._illustration_clusters` unchanged — it is the only module here
    with real test coverage, it is pure PyMuPDF, and it already refuses to run on a page with
    more than a dozen text lines (a body page's tables and flow diagrams are line art too, and
    an image model regenerates those as pictures, i.e. destroys them).
    """
    jobs: list[Job] = []
    text_free = image_regions.TextFreePages(doc)
    try:
        for page_num, page in enumerate(doc, start=1):
            raster_rects = [
                rect
                for entry in page.get_images(full=True)
                for rect in page.get_image_rects(entry[0])
            ]
            protected = _rules(page) + _vector_marks(page) + _panels(page)
            try:
                regions = image_regions._illustration_clusters(
                    page, page_num, protected, raster_rects
                )
            except Exception:
                logger.debug("Page %d: vector detection failed", page_num, exc_info=True)
                continue

            for region in _accept_vector_regions(page, regions, raster_rects, statuses):
                # Rendered from a copy of the page with its text layer stripped: the model
                # treats baked-in words as part of the picture and redraws them, so a region
                # rasterized with live text comes back with English burnt into it.
                rendered = image_regions._rasterize_rect(text_free.page(page_num), region.rect)
                if rendered is None:
                    statuses[region.key] = "extract_failed"
                    continue
                png, width, height = rendered
                jobs.append(
                    Job(
                        key=region.key,
                        kind="vector",
                        page_num=page_num,
                        png=png,
                        width=width,
                        height=height,
                        context=_page_context(page, region.rect),
                        rect=region.rect,
                        dedup_key=region.content_key,
                    )
                )
    finally:
        text_free.close()

    if len(jobs) > image_regions.MAX_ILLUSTRATION_REGIONS_PER_DOC:
        for job in jobs[image_regions.MAX_ILLUSTRATION_REGIONS_PER_DOC:]:
            statuses[job.key] = "vector_capped"
        jobs = jobs[: image_regions.MAX_ILLUSTRATION_REGIONS_PER_DOC]
    return jobs


# --------------------------------------------------------------------------------------
# Per-image work (runs in worker threads — API calls and PIL only, never fitz)
# --------------------------------------------------------------------------------------


def _process(job: Job) -> Job:
    """Classify, translate and regenerate one picture. Never raises: a failed image keeps its
    original pixels, so the worst outcome is an un-localized picture rather than a broken PDF."""
    try:
        job.record = {"key": job.key, "page": job.page_num, "size": f"{job.width}x{job.height}"}
        if max(job.width, job.height) / max(min(job.width, job.height), 1) > MAX_PICTURE_ASPECT:
            job.status = "decorative_strip"
            return job

        model_png = _downscale_for_model(job.png)
        decision = _classify(model_png)
        job.record = {
            "key": job.key,
            "page": job.page_num,
            "size": f"{job.width}x{job.height}",
            "is_logo": decision["is_logo"],
            "reason": decision.get("reason", "")[:120],
        }

        # Checked before OCR, deliberately: a wordmark must never have its lettering erased or
        # translated on the way to deciding not to redraw it.
        if decision["is_logo"]:
            job.status = "logo_kept"
            return job

        if not decision["needs_regeneration"]:
            job.status = "no_change_needed"
            return job

        blocks = []
        if decision["has_text"]:
            for block in _extract_text_blocks(model_png, "image/png"):
                block["bn"] = resolve_block_text(block)
                if block["bn"]:
                    blocks.append(block)

        generated = _regenerate(model_png, decision.get("categories", []), job.context)
        if generated is None:
            job.status = "edit_failed"
            return job

        job.result = _resize_to(generated, job.width, job.height)
        job.status = "regenerated"
        if blocks:
            job.result = _bake_text(job.result, blocks)
            job.status = "text_translated"
            job.record["text_blocks"] = len(blocks)
    except Exception:
        logger.exception("%s: regeneration failed — keeping the original", job.key)
        job.status = job.status or "edit_failed"
        job.result = None
    return job


# --------------------------------------------------------------------------------------
# Applying the results (main thread only — PyMuPDF is not thread-safe)
# --------------------------------------------------------------------------------------


def _apply_vectors(doc: fitz.Document, jobs: list[Job]) -> None:
    """Replace vector artwork: redact the drawn paths, then insert the new raster over them.

    Every redaction on a page is applied before any image is inserted into it. The order is not
    stylistic — `apply_redactions` re-flushes the page's content stream, so a redaction pass run
    after an insert drops the image that was just placed.
    """
    by_page: dict[int, list[Job]] = {}
    for job in jobs:
        by_page.setdefault(job.page_num, []).append(job)

    for page_num, page_jobs in by_page.items():
        page = doc[page_num - 1]
        try:
            for job in page_jobs:
                page.add_redact_annot(job.rect)
            page.apply_redactions(
                # Nothing but line art is ever removed: no text and no picture is at risk here.
                images=fitz.PDF_REDACT_IMAGE_NONE,
                # REMOVE_IF_COVERED, not IF_TOUCHED: only paths lying wholly inside the region
                # go, so a panel or rule extending past it cannot be damaged.
                graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                text=fitz.PDF_REDACT_TEXT_NONE,
            )
            for job in page_jobs:
                page.insert_image(
                    job.rect, stream=job.result, keep_proportion=False, overlay=True
                )
        except Exception:
            logger.exception("Page %d: could not apply vector regions", page_num)


def _summarize(statuses: dict[str, str]) -> str:
    tally: dict[str, int] = {}
    for status in statuses.values():
        tally[status] = tally.get(status, 0) + 1
    ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    return f"{len(statuses)} images: " + ", ".join(f"{n} {name}" for name, n in ordered)


def regenerate_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Regenerate every image in the PDF for a Bangladeshi audience.

    Returns (new_pdf_bytes, summary). The summary is a one-line tally of what happened to each
    image, and every image in the document appears in it exactly once.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    statuses: dict[str, str] = {}

    # Every PyMuPDF read happens here, on this thread, before any concurrency starts.
    raster_jobs = _collect_rasters(doc, statuses)
    vector_jobs = _collect_vectors(doc, statuses)
    jobs = raster_jobs + vector_jobs
    logger.info(
        "Image regeneration: %d rasters, %d vector regions, %d already resolved",
        len(raster_jobs), len(vector_jobs), len(statuses),
    )

    # Identical vector artwork repeated across pages (a divider illustration, a cover cartoon)
    # is regenerated once and reused, so it cannot come back looking different on each page.
    # Rasters get this for free — they share one xref.
    seen: dict[str, Job] = {}
    to_run: list[Job] = []
    for job in jobs:
        if job.kind == "vector" and job.dedup_key:
            twin = seen.get(job.dedup_key)
            if twin is not None:
                continue
            seen[job.dedup_key] = job
        to_run.append(job)

    if to_run:
        with ThreadPoolExecutor(max_workers=CONCURRENT_IMAGES) as pool:
            for job in pool.map(_process, to_run):
                statuses[job.key] = job.status

    # Copy each deduped twin's result onto the jobs that were skipped above.
    for job in jobs:
        if job.status:
            continue
        twin = seen.get(job.dedup_key)
        if twin is not None:
            job.result, job.status = twin.result, twin.status
        statuses[job.key] = job.status or "edit_failed"

    applied_vectors = [j for j in jobs if j.kind == "vector" and j.result and j.rect is not None]
    if applied_vectors:
        _apply_vectors(doc, applied_vectors)

    for job in jobs:
        if job.kind == "raster" and job.result:
            if not _swap_image_in_place(doc, job.xref, job.result):
                statuses[job.key] = "swap_failed"

    logger.info("Image regeneration audit: %s", json.dumps([j.record for j in jobs if j.record]))
    summary = _summarize(statuses)
    logger.info("Image regeneration complete: %s", summary)

    # No subset_fonts(): it corrupts the HarfBuzz-shaped Bangla glyphs an already-translated
    # document carries, and this pipeline has no fonts of its own to shrink anyway.
    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return out, summary
