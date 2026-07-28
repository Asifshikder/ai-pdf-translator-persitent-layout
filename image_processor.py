"""Image localization pipeline: find images in a PDF, and for the ones that depict
culturally-specific content, replace them with a Bangladeshi-adapted version in place.

The existing text-translation pipeline (pdf_processor.py) is untouched; this is a
fully separate flow. Layout is preserved for free: PyMuPDF's page.replace_image swaps
the pixel content of an existing image object, so every placement of that image keeps
its exact position and size."""

import html
import io
import json
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime

import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageDraw, ImageStat

from image_localizer import (
    classify_image,
    detect_logo_regions,
    localize_cover,
    localize_image,
    resolve_block_text,
    translate_blocks,
    _extract_text_blocks,
)
from image_regions import (
    ENABLE_VECTOR_ILLUSTRATION_LOCALIZATION,
    MAX_ILLUSTRATION_REGIONS_PER_DOC,
    IllustrationRegion,
    TextFreePages,
    _illustration_clusters,
    _is_image_mask,
    _rasterize_rect,
    _text_line_count,
)
from pdf_processor import CSS_TEMPLATE, FONTS_DIR, SCALE_LADDER, _panels, _rules, _vector_marks

logger = logging.getLogger(__name__)

# Number of concurrent image processing tasks (classify + edit). The image-edit model
# (gemini-2.5-flash-image) has a low requests-per-minute limit, so running many edits in
# parallel is what triggers 429s. Keep this small; vertex_client backs off on transient 429s.
CONCURRENT_IMAGES = 2

# Large, text-dense images (nutrition charts, infographics, diagrams) are handled "text-only":
# the original image is kept (its precise layout would be wrecked by generative regeneration, and
# such images are slow/429-prone), and only its baked text is erased + retranslated. An image is
# treated this way when it is both big and text-heavy.
TEXT_ONLY_MIN_PIXELS = 1_500_000
TEXT_ONLY_MIN_BLOCKS = 5

# Below this, a picture cannot survive a round trip through the edit model. It is sent at its
# own resolution, redrawn on the model's much larger canvas, and squeezed back down: framing,
# line weights and the position of small elements all drift, and on a tile that is part of a
# grid the drift is visible as the grid stops lining up. The alcohol-units cards on p.109 of
# the Heart Failure manual are 180x177 = 32k pixels, and came back reframed with their number
# tabs resized and displaced.
REGEN_MIN_PIXELS = 250_000

# A page that repeats the same-shaped picture several times is showing a set — a key, a grid,
# a comparison series — and the pictures are being contrasted with each other. Regenerating
# them one at a time is the worst case for the model: each is redrawn independently, so the
# things that made them a set (identical framing, a common scale) are gone. Any image that
# belongs to such a group is handled text-only whatever the classifier says about it.
SERIES_MIN_MEMBERS = 3
SERIES_SIZE_TOLERANCE = 0.15  # fractional difference in width and height allowed within a set

# Cap the resolution of the copy sent to the Gemini models (classify, OCR, edit). Big images are
# slow and 429-prone; the models don't need full resolution, and OCR bboxes are normalized so they
# still map back onto the full-size original. The full-res image is kept for text-only erasing and
# is the target _resize_to maps a regenerated image back up to.
MODEL_MAX_DIM = 1536

# Beyond this ratio a picture is a decorative strip, a bleed fragment or a rule — never a
# person, a meal or a place — and it is the shape a regeneration handles worst. Ported from
# image_regen.MAX_PICTURE_ASPECT: a 114x1381 sliver came back with its one leaf shifted a few
# percent, and since the page crops all but its left quarter the leaf moved out of view and
# left a white gap down the page edge. That image is real — xref 43 on page 1 of the [41-80]
# slice — and it carries an /SMask this pipeline was also dropping.
MAX_PICTURE_ASPECT = 6.0

# Every status an image can end a run with, in the order the summary reports them. Unknown
# statuses are appended rather than dropped, so adding one cannot make an image disappear
# from the count — which is how "regeneration_discarded" spent its life listed here while
# nothing was able to return it.
STATUS_VOCABULARY = [
    "edit_ok", "text_only", "text_translated", "logo_kept", "classify_kept", "aspect_kept",
    "extract_failed", "decode_failed", "classify_failed", "edit_failed", "ocr_failed",
    "regeneration_discarded", "swap_failed", "stencil_skipped",
]

# Where the per-image audit is written. The endpoint hands back bytes rather than a file, so
# there is no "next to the output"; Output/ is where this project's results already live.
# One file per run, and a failure to write it is logged and swallowed — an audit is not worth
# losing a localized PDF for.
AUDIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
WRITE_AUDIT_FILE = True


def _write_audit(records: list[dict], summary: str) -> str | None:
    """Write the run's audit to Output/localize_audit_<YYYYmmdd-HHMMSS>.json.

    Returns the path, or None if it could not be written (in which case the caller falls back
    to logging the records inline). Until this existed the audit was a single log line on
    stdout, so the only way to find out why a picture had shipped untouched was to have been
    watching the server at the time.
    """
    if not WRITE_AUDIT_FILE:
        return None
    try:
        os.makedirs(AUDIT_DIR, exist_ok=True)
        path = os.path.join(
            AUDIT_DIR, f"localize_audit_{datetime.now():%Y%m%d-%H%M%S}.json"
        )
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "records": records,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        return path
    except Exception:
        logger.exception("Could not write the localization audit; logging it instead")
        return None

# ---------------------------------------------------------------------------------------
# How much of the document is actually redrawn
# ---------------------------------------------------------------------------------------
# Which of the four "do not redraw this" guards are in force. This was one boolean that
# turned all of them off together, which had the effect of making `information_role` — the
# classifier's own answer to what a picture is FOR — dead code, and let the alcohol-units
# cards be redrawn: the edit model, told to "replace Western foods with Bangladeshi
# equivalents", returned a brass cocktail cup where a 125ml glass of wine had been, beside a
# "1.5 units" label that then described nothing.
#
# Only the referential guard is on, and it is the one no prompt can substitute for: a picture
# whose identity IS the datum cannot be redrawn correctly at any temperature. The other three
# are off deliberately — with them on almost nothing was regenerated, which is the failure
# this pipeline exists to prevent. `_series_xrefs` and REGEN_MIN_PIXELS are kept rather than
# deleted: they are the switches to reach for if a grid starts coming back reframed.
#
# The referential guard is only as good as the classifier's definition of the word, so it is
# paired with the narrowed one in image_localizer.CLASSIFY_SYSTEM_PROMPT. Turning this on
# against a definition that counts every plate of food as referential would stop the pipeline
# localizing food at all.
VETO_REFERENTIAL_REGENERATION = True
VETO_SERIES_REGENERATION = False
VETO_SMALL_REGENERATION = False
VETO_LARGE_TEXT_DENSE_REGENERATION = False

# When a regeneration is kept. Both are measured on the placed blocks rather than asked of a
# model. Below the first, too few of the picture's words found anywhere to go and the boxes
# must be wrong; above the second, so many words needed a drawn plate that the model has
# moved its blank surfaces out from under them — the original is the better picture to work
# from, because there every box is still over its own text.
KEEP_REGENERATION_MIN_RESTORED = 0.6
MAX_SCRIM_SHARE = 0.5

# Words of surrounding page text needed before "context" mode is worth using. Below this the
# picture is regenerated in "simple" mode instead — see image_localizer.LOCALIZE_MODES for
# why a scrap of context is worse than none.
CONTEXT_MIN_WORDS = 8

# Pictures at or below this size take "simple" mode whatever context is available. A margin
# icon or one tile of a row cannot absorb a paragraph of art direction; asked to illustrate
# the page's sentence it comes back with detail invented to satisfy it.
SIMPLE_MODE_MAX_PIXELS = 120_000

# ---------------------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------------------
# The front cover is a page, not a picture on a page: its artwork is usually a mix of raster
# photos, vector panels and background fills that no per-image or per-cluster rule assembles
# into "the cover". It is therefore handled whole — rendered, redrawn, and put back as a
# single full-page raster underneath the (preserved, already-translated) text layer.
ENABLE_COVER_LOCALIZATION = True
COVER_PAGE_NUM = 1  # 1-based; only the first page is ever treated as a cover
COVER_MAX_TEXT_LINES = 30  # a cover is a title and a few lines, not a page of prose
COVER_MIN_INK_COVERAGE = 0.12  # fraction of the page covered by images/drawings
COVER_DPI = 150  # a full A4 page at 150dpi is ~1240x1754 — plenty for the edit model
COVER_ID = "cover"  # key for the cover in the results/statuses/audit maps


def _to_png(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """Normalize any extracted image to PNG bytes. Returns (png_bytes, w, h) or None
    if the bytes can't be decoded (in which case the original image is kept)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), width, height
    except Exception:
        logger.exception("Could not decode extracted image; keeping original")
        return None


def _downscale_for_model(png_bytes: bytes, max_dim: int = MODEL_MAX_DIM) -> bytes:
    """Return a copy scaled so its longest side is <= max_dim, for cheaper/faster model calls.

    Returns the original bytes unchanged if it is already within the cap or on any error. OCR
    bboxes are normalized (0..1), so downscaling the copy sent to the model does not affect how
    the resulting text maps back onto the full-size original.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            w, h = img.size
            longest = max(w, h)
            if longest <= max_dim:
                return png_bytes
            scale = max_dim / longest
            resized = img.convert("RGB").resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
            )
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.debug("Could not downscale image for model call; using original", exc_info=True)
        return png_bytes


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# How far inside the edge to read a picture's background, as a fraction of its size. Not the
# outermost row: a picture rendered from a placement rect does not land on an exact pixel
# boundary, so its edge row carries a sliver of the page behind it. Measured on p.27 of the
# Heart Failure manual — a yellow-fielded illustration whose outermost ring is page white —
# reading the edge reported the background as #ffffff, and `_match_background` then held the
# regenerated picture *to* white, which is the opposite of its job. Everything downstream
# depends on this ring being the picture's own field.
BORDER_INSET = 0.01  # 1% of the dimension
BORDER_INSET_MAX = 6  # px: enough for a clip sliver plus anti-aliasing, never a real margin


def _border_ring(img: Image.Image) -> list[tuple[int, int, int]]:
    """The ring of pixels just inside the image's edge — what its background is made of.

    Sampled from the border rather than the whole picture: a document illustration is drawn
    on a field (white paper, the cover's lavender panel, a painted wash) and it is that
    field, not the subject, which has to keep matching the page around it.
    """
    width, height = img.size
    if width < 8 or height < 8:
        return []
    inset = max(1, min(BORDER_INSET_MAX, int(BORDER_INSET * min(width, height))))
    x0, y0 = inset, inset
    x1, y1 = width - inset, height - inset
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    ring = list(img.crop((x0, y0, x1, y0 + 1)).getdata())
    ring += list(img.crop((x0, y1 - 1, x1, y1)).getdata())
    ring += list(img.crop((x0, y0, x0 + 1, y1)).getdata())
    ring += list(img.crop((x1 - 1, y0, x1, y1)).getdata())
    return ring


def _border_color(img: Image.Image, tolerance: int = 22) -> tuple[int, int, int] | None:
    """The image's background colour, or None if its border is not one flat colour.

    The strict reading, kept for the callers that need certainty that a border is plain
    (`_punch_text_holes`'s flatness logic and the tests). `_background_color` is the one to
    use when the question is "what colour is this picture's background".
    """
    ring = _border_ring(img)
    if not ring:
        return None
    channels = list(zip(*ring))
    if any(max(c) - min(c) > tolerance for c in channels):
        return None
    return tuple(round(statistics.mean(c)) for c in channels)  # type: ignore[return-value]


# A background does not have to be flat to be a background. `_border_color` demands every
# edge pixel agree within 22 levels and `_dominant_ring_color` demands one quantized colour
# hold 70% of the ring — neither is met by a *painted* field, which is what this document's
# illustrations sit on. Measured on p.27 of the Heart Failure manual: a hand-painted yellow
# wash, obviously one colour to a reader, failed both tests, so no background was enforced
# and the model returned the picture on a yellow-green field instead of a yellow-tan one.
#
# The question that actually separates a background from a picture is looser: is most of the
# border near one colour, allowing for brush texture and JPEG noise?
BACKGROUND_MIN_SHARE = 0.55
BACKGROUND_SPREAD = 60  # per-channel distance from the median still counted as the same field


def _background_color(img: Image.Image) -> tuple[int, int, int] | None:
    """The image's background colour, or None if it has no background to speak of.

    The median of the border ring, accepted only when most of the ring sits near it. A
    picture that bleeds to its edges — a photograph, a full-bleed scene — has no colour
    holding that share of its border, so it answers None and is left alone.
    """
    ring = _border_ring(img)
    if not ring:
        return None
    channels = list(zip(*ring))
    median = tuple(round(statistics.median(c)) for c in channels)
    near = sum(
        1 for px in ring if max(abs(a - b) for a, b in zip(px, median)) <= BACKGROUND_SPREAD
    )
    if near / len(ring) < BACKGROUND_MIN_SHARE:
        return None
    return median  # type: ignore[return-value]


def _palette_summary(png_bytes: bytes, max_colors: int = 6) -> str:
    """Describe the source's colours for the edit model, as measured hex values.

    "Match the original's palette" on its own does not survive a generative edit; naming
    the colours does. Returns "" when the image can't be read, in which case the prompt
    simply falls back to its qualitative instruction.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as raw:
            img = raw.convert("RGB")
            background = _background_color(img)
            # Quantize first: a photograph has thousands of near-identical shades, and the
            # handful that survive quantization are the ones a reader would name.
            reduced = img.convert("P", palette=Image.ADAPTIVE, colors=max_colors).convert("RGB")
            counts = reduced.getcolors(maxcolors=max_colors * 4) or []
    except Exception:
        logger.debug("Could not measure source palette", exc_info=True)
        return ""

    counts.sort(reverse=True)
    total = sum(n for n, _ in counts) or 1
    named = ", ".join(f"{_hex(rgb)} ({n / total:.0%})" for n, rgb in counts[:max_colors])
    if not named:
        return ""
    summary = f"The original's colours, measured: {named}. Reuse these, not brighter versions of them."
    if background is not None:
        # Stated as an absolute, and repaired by measurement afterwards in _match_background.
        # The model treats a described background as a suggestion and returns a "nicer" one —
        # a warmer white, a gradient, a scene — which reads as a picture from another book.
        summary += (
            f" Its background is EXACTLY {_hex(background)}: reproduce that hex value across "
            f"the whole background, flat and unchanged. Do not invent, tint, shade, gradient "
            f"or replace the background — it must come back the same colour it went in."
        )
    return summary


# Above this share of the image, what the border sampled is not a background — see
# _match_background. A picture on a plain field runs to maybe 70% field; a full-bleed cover
# whose panel really is most of the page is the case this has to stay clear of, so the line
# sits close to "all of it".
MATCH_BACKGROUND_MAX_SHARE = 0.92


def _mask_share(mask: Image.Image) -> float:
    """Fraction of a 1-bit mask that is set."""
    try:
        histogram = mask.convert("L").histogram()
        total = sum(histogram) or 1
        return histogram[255] / total
    except Exception:
        logger.debug("Could not measure mask coverage", exc_info=True)
        return 0.0


def _match_background(
    edited_png: bytes, source_png: bytes, tolerance: int = BACKGROUND_SPREAD
) -> bytes:
    """Repaint the generated image's flat background to the source's exact colour.

    The prompt asks for this and mostly gets it, but "mostly" is visible: an illustration
    whose white field comes back faintly cream, or the cover's lavender panel coming back
    a different lavender, reads as a patch stuck onto the page. The background is therefore
    not left to the model at all — it is measured on both sides and forced. Only pixels
    close to the generated background are touched, so the artwork itself is never
    recoloured, and a picture that bleeds to its edges (no background to speak of) is left
    alone.

    Returns the original bytes unchanged whenever the correction does not apply.
    """
    try:
        with Image.open(io.BytesIO(source_png)) as raw:
            want = _background_color(raw.convert("RGB"))
        with Image.open(io.BytesIO(edited_png)) as raw:
            edited = raw.convert("RGB")
        have = _background_color(edited)
    except Exception:
        logger.debug("Could not match generated background to source", exc_info=True)
        return edited_png

    if want is None or have is None:
        return edited_png
    drift = max(abs(a - b) for a, b in zip(want, have))
    # Exact, not "close enough": the source's hex is what the page around it was printed in,
    # and a 3-level drift across a large flat field is visible as a seam at the picture's edge.
    if drift == 0:
        return edited_png

    # Select every pixel near the generated background.
    band = Image.new("RGB", edited.size, have)
    difference = ImageChops.difference(edited, band).convert("L")
    mask = difference.point(lambda v: 255 if v <= tolerance else 0, mode="1")

    # If nearly the whole picture matches its own border colour then the border sample did
    # not find a background — it found the picture. Repainting it flattens the image, which
    # is silent and total, and it is a whole cover page at stake when the caller is
    # _localize_cover_render. Leave such an image exactly as generated.
    share = _mask_share(mask)
    if share > MATCH_BACKGROUND_MAX_SHARE:
        logger.info(
            "Skipping background correction: %.0f%% of the generated image is its own "
            "border colour %s — that is the picture, not a background",
            share * 100, _hex(have),
        )
        return edited_png

    # Shift those pixels by the difference rather than flooding them with a flat colour.
    # For a plain field the two are the same thing; for a painted or textured one, flooding
    # would iron the brushwork flat and leave a conspicuously smooth patch where the
    # original had grain. Shifting moves the field onto the source's exact colour and keeps
    # everything else about it.
    delta = tuple(w - h for w, h in zip(want, have))
    shifted = Image.merge(
        "RGB",
        [
            channel.point(lambda v, d=delta[i]: max(0, min(255, v + d)))
            for i, channel in enumerate(edited.split())
        ],
    )
    corrected = edited.copy()
    corrected.paste(shifted, mask=mask)
    logger.info(
        "Recoloured generated background %s -> %s (drift %d, %.0f%% of the image)",
        _hex(have), _hex(want), drift, share * 100,
    )
    buf = io.BytesIO()
    corrected.save(buf, format="PNG")
    return buf.getvalue()


# A logo region wider or taller than this share of the picture is not a logo *inside* the
# picture — the detector has boxed the whole thing. That case belongs to the classifier's
# is_logo, which keeps the original image untouched; restamping it here would paste the
# entire original back over the regeneration and quietly undo the localization.
LOGO_REGION_MAX_SHARE = 0.55
# Grow each stamp slightly: the detector's boxes are as tight as the OCR ones and clip a
# mark's outer stroke or its registered-trademark glyph often enough to be visible.
LOGO_REGION_PAD = 0.01  # fraction of the image's width/height


def _restamp_logos(edited_png: bytes, source_png: bytes, logos: list[dict]) -> tuple[bytes, int]:
    """Paste each logo region from the original picture back over the regenerated one.

    A logo is a real organisation's identity: it may not be redrawn, restyled, recoloured or
    translated. The edit model cannot reproduce one faithfully — asked to keep a mark it
    invents a similar-looking one, and asked to blank the lettering it deletes the mark — so
    the only sound answer is to put the original pixels back.

    This is what covers the marks no object scan can find: a crest drawn as vector paths, or
    a wordmark that is part of a larger photo, has no xref of its own to protect. Both are
    just pixels in a rendering, and both come back here.

    Returns (png_bytes, stamped_count). The images are expected to be the same size (the
    caller resizes the regeneration to the source first); any mismatch is handled by mapping
    through the normalized bbox, which is why the boxes are fractions.
    """
    if not logos:
        return edited_png, 0
    try:
        with Image.open(io.BytesIO(edited_png)) as raw:
            edited = raw.convert("RGB")
        with Image.open(io.BytesIO(source_png)) as raw:
            source = raw.convert("RGB")
    except Exception:
        logger.exception("Could not open images to restamp logos; keeping the generated one")
        return edited_png, 0

    stamped = 0
    for logo in logos:
        x0, y0, x1, y1 = logo["bbox"]
        if (x1 - x0) > LOGO_REGION_MAX_SHARE and (y1 - y0) > LOGO_REGION_MAX_SHARE:
            logger.info(
                "Not restamping %r: its box covers the whole picture, which is the "
                "classifier's is_logo case rather than a mark inside a picture",
                logo.get("label", "")[:60],
            )
            continue
        x0, y0 = max(0.0, x0 - LOGO_REGION_PAD), max(0.0, y0 - LOGO_REGION_PAD)
        x1, y1 = min(1.0, x1 + LOGO_REGION_PAD), min(1.0, y1 + LOGO_REGION_PAD)

        def box(img: Image.Image) -> tuple[int, int, int, int]:
            return (
                int(x0 * img.width),
                int(y0 * img.height),
                int(round(x1 * img.width)),
                int(round(y1 * img.height)),
            )

        src_box, dst_box = box(source), box(edited)
        if src_box[2] - src_box[0] < 2 or src_box[3] - src_box[1] < 2:
            continue
        try:
            patch = source.crop(src_box)
            target = (dst_box[2] - dst_box[0], dst_box[3] - dst_box[1])
            if target[0] < 1 or target[1] < 1:
                continue
            if patch.size != target:
                patch = patch.resize(target, Image.LANCZOS)
            edited.paste(patch, dst_box[:2])
            stamped += 1
        except Exception:
            logger.exception("Could not restamp logo %r", logo.get("label", "")[:60])

    if not stamped:
        return edited_png, 0
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    return buf.getvalue(), stamped


def _preserve_logos(
    corrected: bytes, source_png: bytes, record: dict, ident: str, page_num: int
) -> tuple[bytes, list[dict]]:
    """Put back any logo the regeneration redrew, blanked or restyled. See _restamp_logos.

    Returns (image_bytes, logo_regions). The regions matter to the caller as much as the
    pixels do: the text pass that runs next would otherwise OCR the mark we have just
    restored, paint its wording out and write Bangla over it — see _outside_logos.
    """
    logos = detect_logo_regions(_downscale_for_model(source_png), "image/png")
    if not logos:
        return corrected, []
    restamped, stamped = _restamp_logos(corrected, source_png, logos)
    record["logo_regions"] = [logo.get("label", "") for logo in logos]
    record["logos_restamped"] = stamped
    logger.info(
        "Page %d: %s carries %d logo/brand mark(s) (%s); %d restamped from the original",
        page_num, ident, len(logos),
        "; ".join((logo.get("label") or "?")[:30] for logo in logos[:3]), stamped,
    )
    return restamped, logos


# How much of a text block has to fall inside a logo's box before the block is treated as
# part of the mark. Well under half, because the detector's box is drawn tightly around the
# mark while an OCR box around its wordmark routinely runs wider than the artwork does.
LOGO_BLOCK_OVERLAP = 0.30


def _logo_named(text: str, logos: list[dict]) -> bool:
    """True if a text block simply *is* the name of a detected mark.

    Geometry alone is not enough, because the two detectors are two independent model calls
    that do not have to agree on where a mark is: on the eatwell plate the OCR boxed
    "FOOD STANDARDS AGENCY" at the top right and the logo detector boxed the same agency's
    mark somewhere with **zero** overlap, so the block sailed through the box test and the
    pipeline was about to draw a plate and English lettering over the crest. The names match
    exactly when the boxes do not, so both tests are applied and either one protects.
    """
    name = " ".join(text.lower().split())
    if len(name) < 4:
        return False  # too short to be a confident match against a label
    for logo in logos:
        label = " ".join((logo.get("label") or "").lower().split())
        if len(label) >= 4 and (label in name or name in label):
            return True
    return False


def _outside_logos(blocks: list[dict], logos: list[dict]) -> list[dict]:
    """Drop the text blocks that belong to a logo, so its wording is never touched.

    An organisation's name is the one string in the document that must survive in its own
    language and its own lettering. Everything downstream of this — the erase pass, the
    residual-text pass, the Bangla overlay — works from block lists, so removing the mark's
    blocks here is what keeps all three off it.

    A block is dropped if it overlaps a mark's box *or* reads as that mark's name; see
    _logo_named for why the box test alone was not enough.
    """
    if not logos or not blocks:
        return blocks
    kept: list[dict] = []
    for block in blocks:
        named = _logo_named(block.get("text") or "", logos)
        bbox = block.get("bbox") or []
        if len(bbox) != 4:
            if not named:
                kept.append(block)
                continue
            bx0, by0, bx1, by1 = 0.0, 0.0, 0.0, 0.0
        else:
            bx0, by0, bx1, by1 = bbox
        area = max((bx1 - bx0) * (by1 - by0), 1e-9)
        inside = named
        for logo in logos:
            if inside:
                break
            lx0, ly0, lx1, ly1 = logo["bbox"]
            overlap = max(0.0, min(bx1, lx1) - max(bx0, lx0)) * max(
                0.0, min(by1, ly1) - max(by0, ly0)
            )
            if overlap / area >= LOGO_BLOCK_OVERLAP:
                inside = True
        if inside:
            logger.debug(
                "Leaving %r alone — it is part of a logo",
                (block.get("text") or "")[:40],
            )
        else:
            kept.append(block)
    return kept


def _series_xrefs(doc: fitz.Document) -> set[int]:
    """Xrefs that belong to a repeated set of same-shaped pictures on one page.

    Such a group is a key or a comparison grid: the alcohol-units chart's eight drink
    cards, a portion-size series, a before/after pair repeated across a row. What makes
    it readable is that the tiles match each other, and regenerating them one by one
    destroys exactly that — each comes back with its own framing, so the grid stops
    lining up even when every individual tile looks fine.

    Grouped by placed size rather than by pixel size: two tiles can be stored at
    different resolutions and still be printed as a matching pair.

    Once a page is found to carry such a set, *every* image on it is protected, not only
    the tiles that matched. A grid is not always cut into equal pieces — on p.109 the
    first two drink cards share one wider image, so size alone leaves them out, and
    regenerating just those two is the worst of both worlds. A page built around a
    comparison is a chart page; nothing on it should be redrawn.
    """
    series: set[int] = set()
    for page in doc:
        placements: list[tuple[int, fitz.Rect]] = []
        for img in page.get_images(full=True):
            xref = img[0]
            for rect in page.get_image_rects(xref):
                if not rect.is_empty:
                    placements.append((xref, rect))

        for i, (_, ref) in enumerate(placements):
            matched = sum(
                1
                for j, (_, other) in enumerate(placements)
                if j != i
                and abs(other.width - ref.width) <= SERIES_SIZE_TOLERANCE * ref.width
                and abs(other.height - ref.height) <= SERIES_SIZE_TOLERANCE * ref.height
            )
            if matched + 1 >= SERIES_MIN_MEMBERS:
                series.update(xref for xref, _ in placements)
                break
    if series:
        logger.info(
            "%d image(s) sit on a page built around a repeated series%s: %s",
            len(series),
            " and will not be regenerated" if VETO_SERIES_REGENERATION else "",
            sorted(series),
        )
    return series


# Colour spaces whose stored pixels PIL converts the way a PDF viewer would. Anything else —
# DeviceCMYK above all, but also Separation and DeviceN — has to be rendered instead of
# extracted. Measured on the Heart Failure manual, whose images are all DeviceCMYK: the
# picture on p.27 sits on a yellow-tan field, `extract_image` + PIL returns it yellow-green,
# and `_swap_image_in_place` then writes that back as DeviceRGB. The whole picture, its
# background included, came out a different colour from the one it went in — with no help
# from the edit model at all. Rendering the placement is what a reader sees, and DeviceRGB
# reproduces a render exactly.
RGB_NATIVE_COLORSPACES = ("/DeviceRGB", "/DeviceGray")
RENDER_MIN_DPI = 72
RENDER_MAX_DPI = 600


def _stored_out_of_rgb(doc: fitz.Document, xref: int) -> bool:
    """True if this image's pixels must be rendered rather than extracted to keep its colour."""
    try:
        colorspace = doc.xref_get_key(xref, "ColorSpace")[1] or ""
    except Exception:
        logger.debug("xref %d: could not read ColorSpace", xref, exc_info=True)
        return False
    return colorspace not in RGB_NATIVE_COLORSPACES


def _render_dpi_for(doc: fitz.Document, xref: int, rect: fitz.Rect) -> int:
    """A dpi that renders `rect` at no less than the image's own pixel width.

    Rendering to the placement's size would throw away resolution on a picture stored larger
    than it is printed, which is most of them.
    """
    try:
        native_width = int(doc.xref_get_key(xref, "Width")[1])
    except Exception:
        logger.debug("xref %d: could not read Width", xref, exc_info=True)
        return 200
    if rect.width <= 0 or native_width <= 0:
        return 200
    return max(RENDER_MIN_DPI, min(RENDER_MAX_DPI, int(72.0 * native_width / rect.width) + 1))


def _page_ink_coverage(page: fitz.Page) -> float:
    """Fraction of the page that is not blank paper, measured from a thumbnail render.

    Measured rather than derived from image placements and drawing rects: a cover's artwork
    is a mix of rasters, vector panels and background fills, and summing their rects
    double-counts everything that overlaps (which on a cover is everything). Rendering it
    and counting what is not white answers the question directly and costs one 72dpi pixmap.
    """
    try:
        pix = page.get_pixmap(dpi=36, alpha=False)
        with Image.open(io.BytesIO(pix.tobytes("png"))) as raw:
            img = raw.convert("L")
        pixels = list(img.getdata())
    except Exception:
        logger.debug("Could not measure page ink coverage", exc_info=True)
        return 0.0
    if not pixels:
        return 0.0
    return sum(1 for p in pixels if p < 245) / len(pixels)


def _detect_cover_page(doc: fitz.Document) -> int | None:
    """The 1-based page number of the front cover, or None if this document has no cover.

    A cover is the first page, and it is a picture with a title on it: little text, and most
    of the paper carrying ink. Both tests are needed. A slice of a manual (this project's
    usual input is an extracted page range) opens on a body page, which has the ink but also
    seventy lines of prose; a plain typographic title page has the sparse text but almost no
    ink, and regenerating it would replace a white page with an invented illustration.
    """
    if not ENABLE_COVER_LOCALIZATION or len(doc) < COVER_PAGE_NUM:
        return None
    page = doc[COVER_PAGE_NUM - 1]
    lines = _text_line_count(page)
    coverage = _page_ink_coverage(page)
    if lines > COVER_MAX_TEXT_LINES or coverage < COVER_MIN_INK_COVERAGE:
        logger.info(
            "Page %d is not being treated as a cover: %d text lines (max %d), "
            "%.0f%% ink coverage (min %.0f%%)",
            COVER_PAGE_NUM, lines, COVER_MAX_TEXT_LINES, coverage * 100,
            COVER_MIN_INK_COVERAGE * 100,
        )
        return None
    logger.info(
        "Page %d looks like a front cover (%d text lines, %.0f%% ink) — localizing it whole",
        COVER_PAGE_NUM, lines, coverage * 100,
    )
    return COVER_PAGE_NUM


def _localize_cover_render(
    png_bytes: bytes, context: str, record: dict
) -> tuple[bytes | None, str, dict]:
    """Redraw a rendered cover page for a Bangladeshi audience.

    Same (bytes, status, record) shape as `_decide_from_png` so it can share the worker pool
    and the audit trail. None keeps the original cover.
    """
    model_png = _downscale_for_model(png_bytes)
    start = time.time()
    new_cover = localize_cover(model_png, "image/png", context, _palette_summary(model_png))
    record["edit_time_sec"] = round(time.time() - start, 2)
    if not new_cover:
        logger.warning("Cover regeneration failed — keeping the original cover")
        record["status"] = "edit_failed"
        return None, "edit_failed", record
    # The cover is put back over the whole page rect, so it has to come back at the page's
    # proportions; and its background is what the preserved title text sits on, so the same
    # measured colour correction the pictures get applies here too.
    corrected = _match_background(_resize_to(new_cover, *_png_size(png_bytes)), png_bytes)
    # A cover is where an institution's marks live — the publisher, the trust, the charity
    # that funded it — and the cover prompt asks for every lettered surface to come back
    # blank, which would erase them. Put the originals back.
    corrected, logos = _preserve_logos(
        corrected, png_bytes, record, COVER_ID, record.get("page", 0)
    )

    # The cover's words come back the same way every other picture's do. Until this, the
    # cover was the one path that OCR'd nothing: the prompt asks for every lettered surface
    # to come back blank and the model obliges, and only the *live* PDF text layer — the
    # title and the caption — was restored on top. Anything lettered into the artwork was
    # simply deleted, which is what removed "HELP YOURSELF TO A HEALTHY FUTURE" from the
    # cover figure's shirt and put nothing in its place.
    #
    # Boxes that land on the preserved text layer are dropped by _overlay_text_blocks' own
    # get_textbox check, so this cannot double up on the title.
    width, height = _png_size(png_bytes)
    blocks = _outside_logos(_extract_text_blocks(model_png, "image/png", notes=record), logos)
    if blocks:
        _translate_blocks(model_png, blocks, context, record)
        residual = _outside_logos(
            _extract_text_blocks(_downscale_for_model(corrected), "image/png"), logos
        )
        corrected, overlay_blocks = _prepare_text_only_image(
            corrected, width, height, blocks, also_erase=residual, anchor=True
        )
        record["text_blocks"] = overlay_blocks
        record["scrim_blocks"] = sum(1 for b in overlay_blocks if b.get("scrim"))
        logger.info(
            "Cover: %d of %d lettered element(s) restored in Bangla (%d on a drawn plate)",
            len(overlay_blocks), len(blocks), record["scrim_blocks"],
        )
    record["status"] = "edit_ok"
    return corrected, "edit_ok", record


def _png_size(png_bytes: bytes) -> tuple[int, int]:
    """(width, height) of PNG bytes, or (1, 1) if they cannot be read."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            return img.size
    except Exception:
        logger.debug("Could not read PNG size", exc_info=True)
        return (1, 1)


# How far the colour behind a preserved line of text may move before the regenerated cover
# is refused. Generous, because a redrawn panel is never the same hex: what this is looking
# for is the panel having gone somewhere else entirely.
COVER_TEXT_BG_TOLERANCE = 90


def _cover_keeps_text_backgrounds(new_cover: bytes, page: fitz.Page) -> bool:
    """True if the regenerated cover still has the right colour behind the page's own text.

    The cover's real text layer is not redrawn — it is preserved and printed back on top of
    the regeneration, at fixed positions. So the model is free to move the artwork out from
    under it, and when it does the result is invisible text: on one run the teal panel came
    back covering only the top half of the page, and the white caption that had been sitting
    on it was printed onto bare white paper.

    Nothing in the prompt can guarantee this, so it is measured. The check is per text line
    rather than global because a cover whose panel merely shifted a little is still fine.
    """
    try:
        pix = page.get_pixmap(dpi=COVER_DPI)
        before = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        with Image.open(io.BytesIO(new_cover)) as img:
            after = img.convert("RGB").resize(before.size, Image.LANCZOS)
        scale_x = before.width / max(page.rect.width, 1)
        scale_y = before.height / max(page.rect.height, 1)

        def backdrop(img: Image.Image, box: tuple[int, int, int, int]):
            """The colour behind a line of text: the median over its own box.

            The median rather than the centre pixel, because the centre pixel is as likely
            to land on a glyph as on the paper behind it — and comparing glyph to glyph
            reports no change however far the background has moved. Lettering is a minority
            of its own box, so the median is the backdrop.
            """
            crop = img.crop(box)
            crop.thumbnail((24, 24))
            data = list(crop.getdata())
            return tuple(statistics.median(c[i] for c in data) for i in range(3))

        moved = 0
        checked = 0
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                x0, y0, x1, y1 = line["bbox"]
                box = (
                    max(0, int(x0 * scale_x)), max(0, int(y0 * scale_y)),
                    min(before.width, int(x1 * scale_x) + 1),
                    min(before.height, int(y1 * scale_y) + 1),
                )
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                checked += 1
                was, now = backdrop(before, box), backdrop(after, box)
                if max(abs(a - b) for a, b in zip(was, now)) > COVER_TEXT_BG_TOLERANCE:
                    moved += 1
        if checked and moved / checked > 0.25:
            logger.warning(
                "Cover regeneration moved the background under %d of %d preserved text "
                "lines — keeping the original cover rather than printing text onto the "
                "wrong colour",
                moved, checked,
            )
            return False
    except Exception:
        logger.debug("Could not verify the cover's text backgrounds", exc_info=True)
    return True


def _apply_cover(page: fitz.Page, new_cover: bytes) -> bool:
    """Replace everything drawn on the cover page with `new_cover`, keeping its text layer.

    Redaction is what makes this safe to do as a background insert: with every image and
    every drawn path removed from the page, nothing survives that could be painted over the
    new raster (the z-order trap that `_swap_image_in_place` exists to avoid does not arise,
    because there is no longer a matte). The text is spared and stays on top, which is the
    point — the title has already been translated by the text pipeline and the generated
    cover was asked to leave its area blank.
    """
    try:
        page.add_redact_annot(page.rect)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            # REMOVE_IF_TOUCHED, not the REMOVE_IF_COVERED used for illustration regions:
            # the redaction rect is the whole page, and a cover's background panel usually
            # bleeds past the trim, so "covered" leaves exactly the full-bleed fills that
            # would then hide the new raster.
            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=fitz.PDF_REDACT_TEXT_NONE,
        )
        page.insert_image(page.rect, stream=new_cover, keep_proportion=False, overlay=False)
        return True
    except Exception:
        logger.exception("Cover page: could not apply the regenerated cover")
        return False


def _rotation_degrees(matrix: fitz.Matrix) -> int | None:
    """Return the placement's rotation as one of 0/90/180/270 if the transformation is a pure
    scale+rotation (orthogonal), or None for a genuinely sheared placement we can't reproduce.

    PyMuPDF matrices are [a,b,c,d,e,f]: [a,b] is where the X-axis maps, [c,d] the Y-axis.
    A pure rotation+scale keeps the two axes perpendicular (a*c + b*d ≈ 0). We snap the angle
    to the nearest right angle and reject anything not close to it. This lets rotated images be
    re-placed via insert_image(rotate=...) instead of being skipped and left un-localized.
    """
    angle = math.degrees(math.atan2(matrix.b, matrix.a)) % 360
    nearest = round(angle / 90) * 90 % 360
    # Reject if the angle isn't within ~1° of a right angle (arbitrary rotation / shear).
    if abs((angle - nearest + 180) % 360 - 180) > 1.0:
        return None
    # Reject sheared placements: axes must stay perpendicular.
    dot = matrix.a * matrix.c + matrix.b * matrix.d
    scale = math.hypot(matrix.a, matrix.b) * math.hypot(matrix.c, matrix.d) or 1.0
    if abs(dot / scale) > 1e-3:
        return None
    return int(nearest)


def _resize_to(image_bytes: bytes, original_width: int, original_height: int) -> bytes:
    """Keep the edited image at its native resolution if it matches the original aspect ratio
    and has at least as many pixels as the original. Otherwise, upscale or fit to original dims.

    This preserves clarity from high-res model output instead of downscaling it.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            new_width, new_height = img.size

            # Check aspect ratio match within 1% tolerance
            original_aspect = original_width / max(original_height, 1)
            new_aspect = new_width / max(new_height, 1)
            aspect_tolerance = 0.01
            aspect_match = abs(original_aspect - new_aspect) / original_aspect < aspect_tolerance

            original_pixels = original_width * original_height
            new_pixels = new_width * new_height

            # If aspect ratio matches and generated image has more pixels, keep native resolution
            if aspect_match and new_pixels >= original_pixels:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            # Otherwise, upscale or downscale to original dimensions
            if img.size != (original_width, original_height):
                img = img.resize((original_width, original_height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.exception("Could not resize edited image; using it as-is")
        return image_bytes


def _swap_image_in_place(doc: fitz.Document, xref: int, png_bytes: bytes) -> bool:
    """Rewrite an image XObject's pixels, leaving the page content stream untouched.

    This is what keeps the replacement visible. A page's layout routinely draws an
    opaque matte behind a picture — every illustration in the Heart Manual sits on a
    white rectangle of its own size, and the cover on a full-bleed lavender panel — so
    the original z-order is `matte, image, text`. `insert_image` cannot express that:
    its `overlay` flag only chooses "before all existing content" or "after" it.
    Inserting in the background put the new image *under* the matte, which painted
    straight over it (the picture vanished, or survived only as a hairline sliver where
    the matte was a point narrower than the placement). Inserting in the foreground
    shows the picture but buries any text drawn over it.

    Rewriting the object itself sidesteps the choice: the drawing operator stays exactly
    where the layout put it, so the new pixels land in the old picture's place in the
    z-order. Every placement of the xref updates at once, rotated and sheared ones
    included, and no redaction is needed.

    Not `page.replace_image`: that leaves a duplicate resource reference which Adobe and
    Chrome resolve back to the *original* image, so the edit is in the file but invisible
    in a viewer. Rewriting the object in place cannot produce a duplicate.

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
        #
        # It matters more here than in image_regen: _stored_out_of_rgb makes this pipeline
        # *render* a CMYK image from its placement, so the replacement already contains the
        # page background — nulling the SMask then prints that background over the layout.
        for key in ("DecodeParms", "Decode", "Mask", "ImageMask", "Interpolate"):
            doc.xref_set_key(xref, key, "null")
        return True
    except Exception:
        logger.exception("xref %d: in-place image swap failed — keeping the original", xref)
        return False


def _is_flat_area(img: Image.Image, box: tuple[int, int, int, int]) -> bool:
    """True if a box of the image is a plain background rather than artwork.

    Used to decide whether a piece of page text that overlaps an illustration region is
    floating over blank space (safe to let show through the replacement) or sitting on
    the picture itself (in which case knocking a hole would put a gap in the artwork).
    """
    x0, y0, x1, y1 = box
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    try:
        crop = img.crop(box).convert("RGB")
        # extrema is per band: ((min,max), (min,max), (min,max))
        return all(hi - lo <= 24 for lo, hi in crop.getextrema())
    except Exception:
        logger.debug("Could not sample region flatness", exc_info=True)
        return False


def _punch_text_holes(
    edited_png: bytes, source_png: bytes, rect: fitz.Rect, text_rects: list[fitz.Rect]
) -> bytes:
    """Make the replacement transparent wherever real page text crosses the region.

    A vector illustration has to be painted in the foreground to clear the backdrop it
    was drawn on, which would otherwise cover any text inside its bounding box. On the
    cover that is the "Put your heart attack behind you…" blurb: the cartoon's raised arm
    pushes the bounding box out under it, though there is no ink there. Turning those
    patches transparent lets the preserved text show through unharmed.

    A hole is only punched where the *source* render is flat — text that genuinely sits
    on top of artwork is covered instead, which loses the text but keeps the picture
    whole.
    """
    if not text_rects:
        return edited_png
    try:
        with Image.open(io.BytesIO(edited_png)) as raw:
            edited = raw.convert("RGBA")
        with Image.open(io.BytesIO(source_png)) as raw:
            source = raw.convert("RGB")
    except Exception:
        logger.exception("Could not open region images to punch text holes")
        return edited_png

    pad = 1.0  # pt: text ink can graze just outside its reported line box
    punched = 0
    for text_rect in text_rects:
        overlap = (text_rect + (-pad, -pad, pad, pad)) & rect
        if overlap.is_empty:
            continue
        fx0 = (overlap.x0 - rect.x0) / rect.width
        fy0 = (overlap.y0 - rect.y0) / rect.height
        fx1 = (overlap.x1 - rect.x0) / rect.width
        fy1 = (overlap.y1 - rect.y0) / rect.height

        def to_box(img: Image.Image) -> tuple[int, int, int, int]:
            return (
                max(0, int(fx0 * img.width)),
                max(0, int(fy0 * img.height)),
                min(img.width, int(round(fx1 * img.width))),
                min(img.height, int(round(fy1 * img.height))),
            )

        if not _is_flat_area(source, to_box(source)):
            continue
        box = to_box(edited)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        edited.paste((0, 0, 0, 0), box)
        punched += 1

    if not punched:
        return edited_png
    logger.debug("Punched %d text hole(s) in a localized illustration region", punched)
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    return buf.getvalue()


def _text_rects_in(page: fitz.Page, rect: fitz.Rect) -> list[fitz.Rect]:
    """Bounding boxes of the page's text lines that fall inside `rect`."""
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        logger.debug("Could not read text lines for region", exc_info=True)
        return []
    found = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            line_rect = fitz.Rect(line["bbox"])
            if not (line_rect & rect).is_empty:
                found.append(line_rect)
    return found


def localize_pdf(pdf_bytes: bytes) -> bytes:
    """Return the PDF with culturally-specific images adapted to Bangladeshi culture.

    Localization covers:
    - Raster images: standard images found via page.get_images() (e.g. photos, charts)
    - ImageMask images: PDF stencil masks (1-bit, no color) with baked text
    - Vector illustrations: clusters of vector drawing paths (e.g. cover cartoons)

    Each unique image (by xref or content hash) is decided once and cached, so an image
    reused across pages costs a single classify/edit. The localized image is then placed
    on every page/region that uses it. Any failure keeps the original content.

    Also logs a structured audit record of every image xref/region analyzed.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Anything rendered for the edit model is rendered from here: a copy of the page with
    # the live text layer stripped, so the model is not handed English words it will try
    # to redraw into the picture (see TextFreePages).
    text_free = TextFreePages(doc)

    # The front cover is redrawn whole rather than picture by picture — see
    # _detect_cover_page. Detected before anything else, because everything printed on that
    # page is about to be replaced by a single raster: sending its photos through the edit
    # model individually would buy pictures that are then redacted away.
    cover_page_num = _detect_cover_page(doc)
    cover_source: tuple[bytes, int, int] | None = None
    cover_context = ""
    if cover_page_num is not None:
        cover_page = doc[cover_page_num - 1]
        cover_context = _get_page_context(cover_page, cover_page.rect)
        cover_source = _rasterize_rect(
            text_free.page(cover_page_num), cover_page.rect, target_dpi=COVER_DPI
        )
        if cover_source is None:
            logger.warning(
                "Cover page %d could not be rendered; leaving it as-is", cover_page_num
            )
            cover_page_num = None

    # Collect all unique xrefs across all pages, and their first-seen page number.
    pages_by_xref: dict[int, list[int]] = {}
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            pages_by_xref.setdefault(img[0], []).append(page_num)

    unique_xrefs: dict[int, int] = {}
    for xref, pages in pages_by_xref.items():
        # A picture used only on the cover is the cover's business. One used elsewhere too
        # (a logo, a repeated motif) is still processed, but decided from a page where it
        # will survive — the cover's copy is about to be overwritten either way.
        elsewhere = [p for p in pages if p != cover_page_num]
        if elsewhere:
            unique_xrefs[xref] = elsewhere[0]
    if cover_page_num is not None:
        skipped = len(pages_by_xref) - len(unique_xrefs)
        if skipped:
            logger.info(
                "%d image(s) appear only on the cover page and are covered by the "
                "whole-page cover regeneration", skipped,
            )

    # Tiles of a repeated set are never regenerated — the set is the information.
    series_xrefs = _series_xrefs(doc)

    # Extract page context strings in the main thread (PyMuPDF is not thread-safe).
    # Build maps for xref contexts and rasterized images (for ImageMask and vector clusters).
    xref_contexts: dict[int, str] = {}
    raster_by_xref: dict[int, tuple[bytes, int, int]] = {}  # xref -> (png_bytes, w, h)
    regions_by_page: dict[int, list[IllustrationRegion]] = {}  # page_num -> regions
    unique_regions_by_content_key: dict[str, IllustrationRegion] = {}  # content_key -> region (for dedup)

    for xref, page_num in unique_xrefs.items():
        page = doc[page_num - 1]
        # Read around where the picture is actually printed. `get_image_rects` can come back
        # empty (an image drawn inside a Form XObject), in which case the whole page is the
        # only honest answer.
        placements = page.get_image_rects(xref)
        anchor = placements[0] if placements else page.rect
        if not isinstance(anchor, fitz.Rect):
            anchor = anchor[0]
        xref_contexts[xref] = _get_page_context(page, anchor)

        # Two kinds of image are rendered rather than extracted, both here in the main thread
        # (PyMuPDF is not thread-safe) so the worker receives the pixels a reader sees:
        # ImageMask stencils, which extract as bare 1-bit masks with no colour at all, and
        # anything stored outside RGB, which extracts with its colours shifted.
        stencil = _is_image_mask(doc, xref)
        if stencil or _stored_out_of_rgb(doc, xref):
            # Take the first placement; if an image is used in several places, each gets the
            # same rasterized result (same xref = same content).
            rect = placements[0] if placements else None
            if rect is not None and not isinstance(rect, fitz.Rect):
                rect = rect[0]
            if rect is not None and not rect.is_empty:
                raster_result = _rasterize_rect(
                    text_free.page(page_num), rect, target_dpi=_render_dpi_for(doc, xref, rect)
                )
                if raster_result:
                    raster_by_xref[xref] = raster_result
                    logger.debug(
                        "Page %d: xref %d rendered rather than extracted (%s)",
                        page_num, xref, "ImageMask stencil" if stencil else "not stored in RGB",
                    )

    # Detect vector illustration clusters (if enabled).
    if ENABLE_VECTOR_ILLUSTRATION_LOCALIZATION:
        for page_num, page in enumerate(doc, start=1):
            if page_num == cover_page_num:
                continue  # the whole page is being regenerated; clusters on it are moot
            # Get image placement rects on this page for exclusion checks.
            image_rects: list[fitz.Rect] = []
            for xref_on_page in unique_xrefs:
                if unique_xrefs[xref_on_page] <= page_num:  # rough approximation; more precise check below
                    rects = page.get_image_rects(xref_on_page)
                    image_rects.extend(r if isinstance(r, fitz.Rect) else r[0] for r in rects)

            # Get protected regions (checkboxes, rules, panels) that must be preserved.
            protected_rects = (
                _rules(page) + _vector_marks(page) + _panels(page)
            )

            clusters = _illustration_clusters(page, page_num, protected_rects, image_rects)
            # IllustrationRegion is frozen and is built without context (image_regions has no
            # business reading page text); fill it in here, where the page is in hand.
            clusters = [
                replace(c, page_context=_get_page_context(page, c.rect)) for c in clusters
            ]
            if clusters:
                regions_by_page[page_num] = clusters
                # Build dedup map: each distinct content_key -> the first region with that signature
                for region in clusters:
                    if region.content_key not in unique_regions_by_content_key:
                        unique_regions_by_content_key[region.content_key] = region

    # Log and cap the total regions found.
    total_regions = sum(len(r) for r in regions_by_page.values())
    distinct_regions = len(unique_regions_by_content_key)
    if distinct_regions > MAX_ILLUSTRATION_REGIONS_PER_DOC:
        logger.warning(
            "Document has %d distinct illustration signatures (capped at %d); keeping largest by area per page",
            distinct_regions,
            MAX_ILLUSTRATION_REGIONS_PER_DOC,
        )

    # Process all unique images (xrefs and vector regions) concurrently.
    results: dict[int | str, bytes | None] = {}  # xref (int) or content_key (str) -> bytes
    statuses: dict[int | str, str] = {}
    blocks_by_id: dict[int | str, list[dict]] = {}  # xref or content_key -> text blocks
    audit_records: list[dict] = []

    logger.info(
        "Found %d unique raster images + %d unique vector illustration signatures to process",
        len(unique_xrefs),
        distinct_regions,
    )

    with ThreadPoolExecutor(max_workers=CONCURRENT_IMAGES) as executor:
        futures: dict = {}

        # Schedule xref-based images.
        for xref, page_num in unique_xrefs.items():
            futures[
                executor.submit(
                    _decide,
                    doc,
                    xref,
                    page_num,
                    xref_contexts[xref],
                    raster_by_xref.get(xref),
                    xref in series_xrefs,
                )
            ] = ("xref", xref)

        # Schedule vector regions (one per distinct content_key).
        region_sources: dict[str, bytes] = {}  # content_key -> the render sent to the model
        for content_key, region in unique_regions_by_content_key.items():
            raster_result = _rasterize_rect(text_free.page(region.page_num), region.rect)
            if raster_result is None:
                logger.warning("Page %d: region %s failed to rasterize", region.page_num, content_key)
                results[content_key] = None
                statuses[content_key] = "extract_failed"
                continue
            png_bytes, w, h = raster_result
            region_sources[content_key] = png_bytes
            futures[
                executor.submit(_decide_from_png, png_bytes, w, h, f"region:{content_key}", region.page_num, region.page_context)
            ] = ("region", content_key)

        # Schedule the cover as a single whole-page job.
        if cover_source is not None:
            futures[
                executor.submit(
                    _localize_cover_render,
                    cover_source[0],
                    cover_context,
                    {
                        "ident": COVER_ID,
                        "page": cover_page_num,
                        "width": cover_source[1],
                        "height": cover_source[2],
                        "mode": "cover",
                    },
                )
            ] = ("cover", COVER_ID)

        # Collect results as they complete.
        for future in as_completed(futures):
            id_type, id_val = futures[future]
            try:
                result = future.result()
                image_bytes, status, record = result
                results[id_val] = image_bytes
                statuses[id_val] = status
                blocks_by_id[id_val] = record.get("text_blocks", []) or []
                audit_records.append(record)
            except Exception as exc:
                logger.exception("%s %s processing failed", id_type, id_val)
                results[id_val] = None
                statuses[id_val] = "edit_failed"

    # Shared font archive for the Bangla text overlay (same Noto fonts the translator uses).
    archive = fitz.Archive(FONTS_DIR)

    # Raster images are swapped inside their XObject, once per xref rather than once per
    # placement: the drawing operator never moves, so every page that uses the image keeps
    # the picture's original position, size, rotation and — crucially — z-order.
    swapped_xrefs = {
        xref
        for xref in unique_xrefs
        if results.get(xref) is not None and _swap_image_in_place(doc, xref, results[xref])
    }
    logger.debug("Swapped %d raster xref(s) in place", len(swapped_xrefs))

    # Vector illustrations have no XObject to rewrite, so they still go through
    # redact-then-insert. Per page: redact every region first, then insert — applying
    # redactions after content exists would re-flush the inserted images.
    for page_num, page in enumerate(doc, start=1):
        if page_num == cover_page_num:
            # Everything drawn here is about to be redacted and replaced by the regenerated
            # cover; a Bangla overlay written now would survive that and float over it.
            continue
        region_placements: list[tuple[IllustrationRegion, bytes, list[dict], list[fitz.Rect]]] = []
        for region in regions_by_page.get(page_num, []):
            new_image = results.get(region.content_key)
            if new_image is None:
                continue
            region_placements.append(
                (
                    region,
                    new_image,
                    blocks_by_id.get(region.content_key, []),
                    _text_rects_in(page, region.rect),
                )
            )

        # The Bangla overlay for a swapped raster is drawn per placement, since it is page
        # content rather than image content.
        overlay_targets: list[tuple[fitz.Rect, list[dict]]] = []
        for img in page.get_images(full=True):
            xref = img[0]
            blocks = blocks_by_id.get(xref, [])
            if xref not in swapped_xrefs or not blocks:
                continue
            for item in page.get_image_rects(xref, transform=True):
                rect, matrix = item if isinstance(item, tuple) else (item, None)
                # A rotated or sheared placement gets the new pixels either way — only the
                # flat text overlay, whose boxes are axis-aligned, has to sit this one out.
                if matrix is not None and _rotation_degrees(matrix) != 0:
                    logger.debug(
                        "Page %d: xref %d text overlay skipped — placement is not upright",
                        page_num,
                        xref,
                    )
                    continue
                overlay_targets.append((rect, blocks))

        if not region_placements and not overlay_targets:
            continue

        logger.debug(
            "Page %d: %d vector region(s), %d raster text overlay(s)",
            page_num,
            len(region_placements),
            len(overlay_targets),
        )

        if region_placements and ENABLE_VECTOR_ILLUSTRATION_LOCALIZATION:
            # Clears the illustration's own paths. REMOVE_IF_COVERED spares the backdrop
            # it was drawn on (which the redaction rect only partly covers) and
            # TEXT_NONE spares every word on the page.
            try:
                for region, *_ in region_placements:
                    page.add_redact_annot(region.rect)
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                    text=fitz.PDF_REDACT_TEXT_NONE,
                )
            except Exception:
                logger.exception("Page %d: vector redaction pass failed — keeping original drawings", page_num)
                region_placements = []  # skip insertions for this page

        for i, (region, new_image, blocks, text_rects) in enumerate(region_placements):
            try:
                # Foreground: the backdrop the illustration was drawn on survives the
                # redaction, so a background insert would be hidden behind it. Punching
                # the text out is what stops a foreground image eating page text.
                page.insert_image(
                    region.rect,
                    stream=_punch_text_holes(
                        new_image,
                        region_sources.get(region.content_key, new_image),
                        region.rect,
                        text_rects,
                    ),
                    keep_proportion=False,
                    overlay=True,
                    rotate=0,  # regions are always axis-aligned (no rotation detection needed)
                )
                logger.debug(
                    "Page %d: region placement %d/%d inserted at (%.0f,%.0f,%.0f,%.0f)",
                    page_num, i + 1, len(region_placements), region.rect.x0, region.rect.y0, region.rect.x1, region.rect.y1,
                )
            except Exception:
                logger.exception(
                    "Page %d: failed to insert region at placement %d/%d", page_num, i + 1, len(region_placements)
                )
                continue

            if blocks:
                _overlay_text_blocks(page, region.rect, blocks, archive)

        for rect, blocks in overlay_targets:
            _overlay_text_blocks(page, rect, blocks, archive)

    # The cover goes on last: it redacts its whole page, and a redaction pass run after
    # content has been inserted re-flushes that content.
    if cover_page_num is not None and results.get(COVER_ID):
        cover_page = doc[cover_page_num - 1]
        if not _cover_keeps_text_backgrounds(results[COVER_ID], cover_page):
            statuses[COVER_ID] = "regeneration_discarded"
        elif _apply_cover(cover_page, results[COVER_ID]):
            logger.info("Cover page %d replaced with its localized rendering", cover_page_num)
            # After the redaction, not before: _apply_cover re-flushes the content stream,
            # which would drop anything inserted ahead of it.
            cover_blocks = blocks_by_id.get(COVER_ID, [])
            if cover_blocks:
                _overlay_text_blocks(cover_page, cover_page.rect, cover_blocks, archive)
        else:
            statuses[COVER_ID] = "edit_failed"

    status_counts = {}
    for status in statuses.values():
        status_counts[status] = status_counts.get(status, 0) + 1

    summary_parts = []
    for status in STATUS_VOCABULARY:
        count = status_counts.get(status, 0)
        if count > 0:
            summary_parts.append(f"{count} {status}")
    # Anything the vocabulary has not been taught yet, rather than silently omitted.
    for status, count in sorted(status_counts.items()):
        if status not in STATUS_VOCABULARY:
            summary_parts.append(f"{count} {status}")

    summary = ", ".join(summary_parts) if summary_parts else "no images"
    logger.info("Image localization complete: %s (rasters + vectors)", summary)

    path = _write_audit(audit_records, summary)
    if path:
        logger.info("Image localization audit (%d records) written to %s",
                    len(audit_records), path)
    else:
        logger.info("Image localization audit (%d records): %s",
                    len(audit_records), json.dumps(audit_records, default=str))

    out = doc.tobytes(garbage=3, deflate=True)
    text_free.close()
    doc.close()
    return out


def _overlay_text_blocks(
    page: fitz.Page, rect: fitz.Rect, blocks: list[dict], archive: fitz.Archive
) -> None:
    """Redraw each OCR text block as legible Bangla using insert_htmlbox + the Noto archive.

    The edited image is text-free by design; this restores the text. bbox values are fractions
    (0..1) of the image and map linearly into the placement rect. A block is skipped when a real
    PDF text layer already covers its spot (that vector text is the clean source, so we avoid
    doubling), when its text can't be resolved to valid Bangla, or when it would land on top of
    a block already drawn.

    Overlap has to be policed here because the boxes come from a model, not a layout: OCR
    routinely returns the same words twice — once per line and once as the whole caption —
    and drawing both stacks two runs of Bangla in the same place, which is illegible in a
    way that neither box alone would be.
    """
    # Biggest first, so that when a heading and its individual lines are both reported, the
    # box that is kept is the one that can actually hold the translated text.
    ordered = sorted(
        (b for b in blocks if len(b.get("bbox") or []) == 4),
        key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]),
        reverse=True,
    )
    placed: list[fitz.Rect] = []
    for block in ordered:
        bx0, by0, bx1, by1 = block["bbox"]
        sub = fitz.Rect(
            rect.x0 + bx0 * rect.width,
            rect.y0 + by0 * rect.height,
            rect.x0 + bx1 * rect.width,
            rect.y0 + by1 * rect.height,
        )
        # A model bbox can run a hair past the edge it was measured in; keep the text inside
        # the picture it belongs to rather than letting it spill onto the page.
        sub &= rect
        if sub.width < MIN_OVERLAY_WIDTH or sub.height < MIN_OVERLAY_HEIGHT:
            continue
        # What this block will actually occupy on the page. A plated block covers its own
        # padding too, so collisions are measured against the plate rather than the words
        # inside it — otherwise two plates a point apart overlap while both boxes "fit".
        footprint = sub
        if block.get("scrim"):
            footprint = (sub + (-SCRIM_PAD, -SCRIM_PAD, SCRIM_PAD, SCRIM_PAD)) & rect
        overlap = next(
            (
                p
                for p in placed
                if (p & footprint).get_area()
                > 0.25 * min(p.get_area(), footprint.get_area())
            ),
            None,
        )
        if overlap is not None:
            logger.debug(
                "Skipping overlapping text block %r at %s (collides with %s)",
                (block.get("text") or "")[:30], sub, overlap,
            )
            continue
        # Don't double up on an existing vector text layer preserved by the redaction step.
        #
        # Checked at BOTH ends. The footprint is where the words would land — read over the
        # footprint rather than the text box because an opaque plate is drawn on top of
        # everything and would hide a caption the redaction deliberately kept. `source` is
        # where the block was read from, and it is the one that catches the real failure:
        # anchoring can slide a box onto clean background a little away from the live text,
        # and the destination then looks empty while the words are already on the page a
        # few points up. On the cover that drew the whole caption a second time, offset and
        # illegible. A block whose own position already carries live text is on the page
        # already, wherever we would otherwise put it.
        source = sub
        src_bbox = block.get("source_bbox") or []
        if len(src_bbox) == 4:
            source = fitz.Rect(
                rect.x0 + src_bbox[0] * rect.width,
                rect.y0 + src_bbox[1] * rect.height,
                rect.x0 + src_bbox[2] * rect.width,
                rect.y0 + src_bbox[3] * rect.height,
            ) & rect
        try:
            if page.get_textbox(footprint).strip() or (
                not source.is_empty and page.get_textbox(source).strip()
            ):
                continue
        except Exception:
            pass
        target = resolve_block_text(block)
        if not target:
            continue
        color = block.get("color", "#000000")  # set by the text-only path to contrast the bg
        body = html.escape(target)
        # A plate under the words, where the picture gave them nowhere clean to sit. Drawn
        # here rather than painted into the pixels so that plate and text are one operation:
        # this loop skips rotated placements and blocks already covered by a real text layer,
        # and a plate painted into the image would survive those skips as a bare coloured
        # patch where the English used to be. Vector-crisp and no JPEG re-encode, too.
        plate = block.get("scrim")
        if plate:
            try:
                page.draw_rect(
                    footprint,
                    color=None,
                    fill=_rgb01(plate),
                    radius=SCRIM_RADIUS,
                    overlay=True,
                )
            except Exception:
                logger.debug("Could not draw a label plate for %r", target[:40], exc_info=True)
        # Never larger than the box can hold on one line: a small badge whose box was reported
        # generously would otherwise start the ladder at a size that swamps its own tile.
        size = min(_fit_size(sub, body, color, archive), sub.height / 1.25)
        css = CSS_TEMPLATE.format(size=size, color=color, align="center", bold="")
        for low in SCALE_LADDER:
            try:
                spare, _ = page.insert_htmlbox(sub, body, css=css, scale_low=low, archive=archive)
            except Exception:
                logger.debug("insert_htmlbox failed for block %r", target[:40], exc_info=True)
                break
            if spare >= 0:
                break
        placed.append(footprint)


# Font sizes tried when fitting a Bangla overlay into an erased text box, largest first.
# A caption box is a couple of lines of small print and a sign is one line of large print,
# and only the fit tells them apart, so the ladder has to span both.
OVERLAY_SIZES = (48.0, 36.0, 28.0, 22.0, 18.0, 15.0, 13.0, 11.0, 9.5, 8.0, 7.0, 6.0)

# A destination smaller than this cannot hold legible Bangla: insert_htmlbox will shrink it to
# a smear, and with a plate under it that smear is a coloured smudge on the artwork. Better to
# leave the block out than to mark the picture with something nobody can read.
MIN_OVERLAY_WIDTH = 6.0  # pt
MIN_OVERLAY_HEIGHT = 4.0  # pt


def _fit_size(rect: fitz.Rect, body: str, color: str, archive: fitz.Archive) -> float:
    """The largest size from OVERLAY_SIZES at which `body` fits `rect` without being shrunk.

    Sizing a block at `rect.height / 1.6` assumes its text is one line, which a caption
    box is not: two lines of Bangla asked for at half the box's height can only be
    delivered by insert_htmlbox scaling them down, and what comes out is a different size
    on every box. Measuring off-page instead — the only way to ask, since insert_htmlbox
    reports a fit but cannot be asked for one — keeps neighbouring captions in step and
    stops a long one from collapsing to a smear.
    """
    scratch = fitz.open()
    try:
        canvas = scratch.new_page(width=rect.width + 2, height=rect.height + 2)
        probe = fitz.Rect(1, 1, rect.width + 1, rect.height + 1)
        for size in OVERLAY_SIZES:
            css = CSS_TEMPLATE.format(size=size, color=color, align="center", bold="")
            try:
                spare, _ = canvas.insert_htmlbox(probe, body, css=css, scale_low=1.0, archive=archive)
            except Exception:
                logger.debug("Overlay fit probe failed at %.1fpt", size, exc_info=True)
                continue
            if spare >= 0:
                return size
    finally:
        scratch.close()
    return OVERLAY_SIZES[-1]


def _sample_uniform_bg(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    """Return the background color (r,g,b) if the ring just outside the bbox is fairly uniform,
    else None. A uniform ring means the text sits on a solid area we can safely paint over
    (a title/label on white). Non-uniform means it's over a photo — leave that text alone.
    """
    W, H = img.size
    pad = 2
    top, bot = max(0, y0 - pad), min(H - 1, y1 + pad)
    left, right = max(0, x0 - pad), min(W - 1, x1 + pad)
    pts = []
    for x in range(left, right + 1):
        pts.append(img.getpixel((x, top)))
        pts.append(img.getpixel((x, bot)))
    for y in range(top, bot + 1):
        pts.append(img.getpixel((left, y)))
        pts.append(img.getpixel((right, y)))
    if len(pts) < 4:
        return None
    rs, gs, bs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    if max(statistics.pstdev(rs), statistics.pstdev(gs), statistics.pstdev(bs)) > 18:
        return None  # non-uniform (photographic) background
    return (round(statistics.mean(rs)), round(statistics.mean(gs)), round(statistics.mean(bs)))


def _ring_pixels(img: Image.Image, x0: int, y0: int, x1: int, y1: int, pad: int = 2) -> list:
    """The pixels just outside a box — what the text sits against."""
    W, H = img.size
    top, bot = max(0, y0 - pad), min(H - 1, y1 + pad)
    left, right = max(0, x0 - pad), min(W - 1, x1 + pad)
    if right <= left or bot <= top:
        return []
    return (
        list(img.crop((left, top, right + 1, top + 1)).getdata())
        + list(img.crop((left, bot, right + 1, bot + 1)).getdata())
        + list(img.crop((left, top, left + 1, bot + 1)).getdata())
        + list(img.crop((right, top, right + 1, bot + 1)).getdata())
    )


def _dominant_ring_color(
    img: Image.Image, x0: int, y0: int, x1: int, y1: int, share: float = 0.70
) -> tuple[int, int, int] | None:
    """The one colour a text box mostly sits against, or None if there isn't one.

    Looser than `_sample_uniform_bg`, which needs the whole ring to be flat. The cover
    slogan sits on a shirt whose ring is one blue plus fold lines and an outline: not
    uniform, but overwhelmingly one colour, and painting that colour over the lettering is
    invisible. A photograph has no colour holding this large a share, so it is still
    refused and its text is left alone.
    """
    pixels = _ring_pixels(img, x0, y0, x1, y1)
    if len(pixels) < 8:
        return None
    strip = Image.new("RGB", (len(pixels), 1))
    strip.putdata(pixels)
    counts = strip.convert("P", palette=Image.ADAPTIVE, colors=16).convert("RGB").getcolors(64)
    if not counts:
        return None
    top_count, top_color = max(counts)
    return top_color if top_count / len(pixels) >= share else None


def _mostly_colored(
    img: Image.Image,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    tolerance: int = 32,
    share: float = 0.35,
) -> bool:
    """True if `color` fills at least `share` of the box — i.e. the box really is a text
    box sitting on that colour, not a picture that merely happens to be ringed by it."""
    try:
        crop = img.crop(box).convert("RGB")
        crop.thumbnail((64, 64))  # counting every pixel of a large box is needless
        pixels = list(crop.getdata())
    except Exception:
        logger.debug("Could not sample box fill", exc_info=True)
        return False
    if not pixels:
        return False
    hits = sum(1 for p in pixels if max(abs(a - b) for a, b in zip(p, color)) <= tolerance)
    return hits / len(pixels) >= share


# How far an erase box may be grown to reach the end of the ink it was aimed at, as a
# fraction of its own width/height. The OCR boxes come back from the model a word or two
# short often enough that a tight cap does not fix anything ("The eatwell plate" needed
# ~54% more width to reach past "plate"), and a loose one lets a label's box crawl into
# the artwork beside it. Sideways is the generous direction because that is where a box
# runs short; downwards is kept tight so a caption cannot swallow the line beneath it.
SNAP_MAX_GROW_X = 1.0
SNAP_MAX_GROW_Y = 0.35
# Clean lines that end the growth, as a fraction of the box's height. Sideways this has to
# clear a word space comfortably — measured on the eatwell title, the space between
# "eatwell" and "plate" is wider than half the cap height, so a smaller lookahead reads
# the gap as the end of the text and stops with the last word still on the page.
SNAP_GAP_X = 0.9
SNAP_GAP_Y = 0.25
SNAP_TOLERANCE = 40  # per-channel distance from the background before a pixel counts as ink

# How much of a 1px strip has to be off-background before the strip is read as something
# other than clean paper — and above which it stops being lettering at all.
#
# The old test was "any pixel differs", which cannot tell the tail of a word from the picture
# it is printed on. Its absence is what shipped: on a regenerated figure the placard had moved
# a few percent, so every strip beyond the box was "dirty", the box grew its full
# SNAP_MAX_GROW_X allowance into the artwork, and the Bangla was painted and written over a
# hand. The same growth turned the blue tab of a "3 units" badge into a blue rectangle out
# past the edge of its own tile.
#
# The upper limit is measured against the box's own ink rather than fixed, because a fixed one
# cannot be right for both cases: a strip through the stem of a letter is most of the box's
# height, so a low cap stops growth mid-word, while a high cap lets a box walk into a picture
# whose colours happen to be dense. What actually says "this is no longer my text" is a strip
# far denser than the text the box was drawn around.
SNAP_INK_MIN_SHARE = 0.02  # below this the strip is clean paper
SNAP_INK_MAX_SHARE = 0.80  # floor for the artwork test, for a box that is nearly all ink
SNAP_INK_DENSITY_FACTOR = 2.0  # …or this multiple of the box's own ink share, whichever is more
# A box grown past this multiple of the one the model reported is no longer following a word
# to its end. Fall back to what the model said rather than paint over the picture.
SNAP_MAX_AREA_GROWTH = 2.5


def _snap_to_ink(
    img: Image.Image,
    box: tuple[int, int, int, int],
    bg: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Grow `box` outward while the lines just beyond it still hold ink.

    Gemini's normalized bboxes are a hint, not a measurement, and they run short often
    enough to matter: on p.114 the box for "The eatwell plate" stopped after "eatwell", so
    the erase left the word "plate" in English with a fringe of the letters it *had*
    covered scattered around it. Painting the rectangle the model reported and never
    looking at the result is what let that ship.

    Growth stops at a run of clean lines — the gap after the last word — or the moment it
    reaches artwork, and is capped in any case so a label tucked against a picture cannot
    swallow it.
    """
    x0, y0, x1, y1 = box
    W, H = img.size

    def off_share(region: tuple[int, int, int, int]) -> float:
        """Fraction of a region that is not the background."""
        rx0, ry0, rx1, ry1 = region
        if rx1 <= rx0 or ry1 <= ry0:
            return 0.0
        pixels = list(img.crop((rx0, ry0, rx1, ry1)).convert("RGB").getdata())
        if not pixels:
            return 0.0
        off = sum(
            1 for px in pixels if max(abs(a - b) for a, b in zip(px, bg)) > SNAP_TOLERANCE
        )
        return off / len(pixels)

    # What the text this box was drawn around actually looks like, so that "denser than my
    # own text" can be the test for having left it. Measured once.
    artwork_at = max(SNAP_INK_MAX_SHARE, SNAP_INK_DENSITY_FACTOR * off_share(box))

    def strip_state(line: tuple[int, int, int, int]) -> str:
        """What a 1px strip beyond the box is: "clean" paper, "ink", or "artwork".

        Three states rather than two because the middle one is the whole question: a strip
        far denser than the box's own text is not the rest of the word, it is the picture
        beside it, and growing into it is how an erase box ends up painting over artwork.
        """
        share = off_share(line)
        if share < SNAP_INK_MIN_SHARE:
            return "clean"
        return "artwork" if share > artwork_at else "ink"

    def grow(limit: int, gap: int, line_at) -> int:
        """Walk outward from an edge, returning the last offset that still held ink."""
        edge = clean = 0
        for offset in range(1, limit + 1):
            state = strip_state(line_at(offset))
            if state == "artwork":
                break  # the picture starts here; the word ended before it
            if state == "ink":
                edge, clean = offset, 0
            else:
                clean += 1
                if clean >= gap:
                    break
        return edge

    height = y1 - y0
    h_limit = max(1, round(SNAP_MAX_GROW_X * (x1 - x0)))
    v_limit = max(1, round(SNAP_MAX_GROW_Y * height))
    h_gap = max(2, round(SNAP_GAP_X * height))
    v_gap = max(2, round(SNAP_GAP_Y * height))

    left = grow(min(h_limit, x0), h_gap, lambda d: (x0 - d, y0, x0 - d + 1, y1))
    right = grow(min(h_limit, W - x1), h_gap, lambda d: (x1 + d - 1, y0, x1 + d, y1))
    up = grow(min(v_limit, y0), v_gap, lambda d: (x0, y0 - d, x1, y0 - d + 1))
    down = grow(min(v_limit, H - y1), v_gap, lambda d: (x0, y1 + d - 1, x1, y1 + d))

    return (x0 - left, y0 - up, x1 + right, y1 + down)


# Anchoring: where a block's words go in a picture that has just been redrawn.
#
# The OCR box was measured on the original. The model is asked to keep every blank surface to
# the pixel and mostly does, but "mostly" is a placard three percent lower than it was, and
# Bangla written at the old coordinates then hangs off its bottom edge and over a hand.
#
# Dual to _snap_to_ink: that one grows a box while the lines beyond it still hold ink, this
# one grows a seed while the lines beyond it are still the same flat colour. Pixels rather
# than a second model call because the image quota is the binding constraint here, and the
# drift being corrected is a few percent — a translate-and-clip is enough to fix it.
ANCHOR_TOLERANCE = 26  # per-channel distance that still counts as the same surface
ANCHOR_LINE_PURITY = 0.90  # share of a 1px line that must match before the line joins it
ANCHOR_MAX_SHIFT = 0.75  # how far the box may travel, as a fraction of its own size
ANCHOR_MIN_COVERAGE = 0.55  # of the block's own area, or this is not that block's surface
ANCHOR_MIN_SIDE = 6  # px: a surface thinner than this cannot hold a line of Bangla


def _line_matches(img: Image.Image, line: tuple[int, int, int, int], color) -> bool:
    """True if ANCHOR_LINE_PURITY of a 1px strip is within tolerance of `color`."""
    lx0, ly0, lx1, ly1 = line
    if lx1 <= lx0 or ly1 <= ly0:
        return False
    pixels = list(img.crop((lx0, ly0, lx1, ly1)).convert("RGB").getdata())
    if not pixels:
        return False
    hits = sum(
        1 for px in pixels if max(abs(a - b) for a, b in zip(px, color)) <= ANCHOR_TOLERANCE
    )
    return hits / len(pixels) >= ANCHOR_LINE_PURITY


def _surface_rect(
    img: Image.Image,
    seed: tuple[int, int],
    color: tuple[int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """The largest rectangle of `color` around `seed`, never leaving `bounds`.

    Grown one 1px line at a time in each direction, a line joining the rectangle when
    ANCHOR_LINE_PURITY of it matches. Rectangular rather than a flood fill on purpose: what
    is wanted is somewhere to put a line of text, and a rectangle is the only shape that
    answers that question directly.
    """
    bx0, by0, bx1, by1 = bounds
    sx, sy = seed
    if not (bx0 <= sx < bx1 and by0 <= sy < by1):
        return None
    if max(abs(a - b) for a, b in zip(img.getpixel((sx, sy)), color)) > ANCHOR_TOLERANCE:
        return None

    x0, y0, x1, y1 = sx, sy, sx + 1, sy + 1
    grew = True
    while grew:
        grew = False
        if x0 > bx0 and _line_matches(img, (x0 - 1, y0, x0, y1), color):
            x0 -= 1
            grew = True
        if x1 < bx1 and _line_matches(img, (x1, y0, x1 + 1, y1), color):
            x1 += 1
            grew = True
        if y0 > by0 and _line_matches(img, (x0, y0 - 1, x1, y0), color):
            y0 -= 1
            grew = True
        if y1 < by1 and _line_matches(img, (x0, y1, x1, y1 + 1), color):
            y1 += 1
            grew = True
    return (x0, y0, x1, y1)


def _anchor_box(
    img: Image.Image, box: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]] | None:
    """Move and clip `box` onto the blank surface it belongs to in `img`.

    Nine seeds are probed inside the box (at 1/4, 1/2 and 3/4 in each axis) so that a box
    which is half off its placard still finds the placard. The surface holding the most of
    the block wins — not the largest one, which on any page with a flat background is the
    background itself: a placard that has slipped 25px leaves seeds in both, and the panel
    behind it is always the bigger field. The box is then slid by the smallest offset that
    puts it inside the winning surface — capped at ANCHOR_MAX_SHIFT of its own size — and
    clipped to it.

    Returns (destination box, surface colour), or None when no surface under the box holds
    ANCHOR_MIN_COVERAGE of it: either the picture was reframed too far, or the words sit on
    a photograph. Both are the caller's cue to draw a plate instead (see SCRIM_*).
    """
    W, H = img.size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw < ANCHOR_MIN_SIDE or bh < ANCHOR_MIN_SIDE:
        return None

    # Search no further than the box could legitimately have drifted.
    mx, my = round(ANCHOR_MAX_SHIFT * bw), round(ANCHOR_MAX_SHIFT * bh)
    bounds = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))

    def held(surface: tuple[int, int, int, int]) -> int:
        """How much of the original box this surface covers."""
        ox = max(0, min(surface[2], x1) - max(surface[0], x0))
        oy = max(0, min(surface[3], y1) - max(surface[1], y0))
        return ox * oy

    best: tuple[int, int, int, int] | None = None
    best_color: tuple[int, int, int] | None = None
    for fx in (0.25, 0.5, 0.75):
        for fy in (0.25, 0.5, 0.75):
            sx, sy = int(x0 + fx * bw), int(y0 + fy * bh)
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            color = img.getpixel((sx, sy))[:3]
            found = _surface_rect(img, (sx, sy), color, bounds)
            if found is None:
                continue
            if best is None or held(found) > held(best):
                best, best_color = found, color

    if best is None or best_color is None:
        return None
    sx0, sy0, sx1, sy1 = best
    if sx1 - sx0 < ANCHOR_MIN_SIDE or sy1 - sy0 < ANCHOR_MIN_SIDE:
        return None

    # Slide the box into the surface by the least amount that works, then clip. Sliding
    # before clipping keeps the box its original size wherever the surface can hold it,
    # which is what keeps a caption's type the same size as its neighbours'.
    dx = max(sx0 - x0, 0) + min(sx1 - x1, 0)
    dy = max(sy0 - y0, 0) + min(sy1 - y1, 0)
    moved = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    dest = (
        max(moved[0], sx0),
        max(moved[1], sy0),
        min(moved[2], sx1),
        min(moved[3], sy1),
    )
    dw, dh = dest[2] - dest[0], dest[3] - dest[1]
    if dw < ANCHOR_MIN_SIDE or dh < ANCHOR_MIN_SIDE:
        return None
    if (dw * dh) < ANCHOR_MIN_COVERAGE * (bw * bh):
        return None  # the surface cannot hold this block — plate it instead
    return dest, best_color


# Ink left inside the box the Bangla is about to be written into. Any English still standing
# there shows through the new text — on the units grid the Bangla was drawn straight over a
# still-visible "(250ml, ABV 12%)", because that phrase was a *different* OCR block whose own
# erase had been refused. Measuring the destination rather than each block's own box is what
# catches a neighbour's leftovers.
MAX_RESIDUAL_INK = 0.02


def _ink_share(
    img: Image.Image, box: tuple[int, int, int, int], bg: tuple[int, int, int]
) -> float:
    """Fraction of a box that is not `bg`, sampled through a thumbnail like _mostly_colored."""
    try:
        crop = img.crop(box).convert("RGB")
        crop.thumbnail((64, 64))
        pixels = list(crop.getdata())
    except Exception:
        logger.debug("Could not sample residual ink", exc_info=True)
        return 0.0
    if not pixels:
        return 0.0
    off = sum(1 for p in pixels if max(abs(a - b) for a, b in zip(p, bg)) > SNAP_TOLERANCE)
    return off / len(pixels)


# The label plate. A label written on a photograph has nowhere clean to go, and the
# alternative the pipeline used until now was to write nothing — which shipped the FSA
# eatwell plate a hundred percent in English, reported as though it had been considered and
# kept. So the surface is supplied: an opaque plate in a colour sampled from what the label
# was sitting on, drawn immediately under the words. It hides a little of the picture; an
# unreadable page hides all of it.
SCRIM_SAMPLE = 0.6  # share of the box sampled to choose the plate's colour
SCRIM_PAD = 1.5  # pt of plate painted around the text box
SCRIM_RADIUS = 0.20  # corner radius, as a fraction of the plate's short side
# A plate in this luminance band contrasts neither black nor white text; lighten it until it
# does, and take black.
SCRIM_MID_LUMINANCE = (110, 175)
SCRIM_LIGHTEN = 45
# An OCR box this large is not a label, it is the picture — the case _mostly_colored catches
# on the erase path. Plating it would paint the artwork out.
SCRIM_MAX_AREA_SHARE = 0.25


def _plate_color(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The colour to paint a label plate: the median of the box's own middle, pushed out of
    the mid-luminance band where neither black nor white text reads well.

    The middle rather than the ring, because the ring is what the erase path already tried
    and failed to read — by the time this is called the surroundings are known to be busy.
    """
    x0, y0, x1, y1 = box
    ix = round((1 - SCRIM_SAMPLE) * (x1 - x0) / 2)
    iy = round((1 - SCRIM_SAMPLE) * (y1 - y0) / 2)
    inner = (x0 + ix, y0 + iy, max(x0 + ix + 1, x1 - ix), max(y0 + iy + 1, y1 - iy))
    try:
        crop = img.crop(inner).convert("RGB")
        crop.thumbnail((32, 32))
        pixels = list(crop.getdata())
        rgb = tuple(round(statistics.median(c[i] for c in pixels)) for i in range(3))
    except Exception:
        logger.debug("Could not sample a plate colour; defaulting to white", exc_info=True)
        return (255, 255, 255)
    lo, hi = SCRIM_MID_LUMINANCE
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lo <= luminance <= hi:
        rgb = tuple(min(255, c + SCRIM_LIGHTEN) for c in rgb)
    return rgb  # type: ignore[return-value]


def _rgb01(hex_color: str) -> tuple[float, float, float]:
    """'#rrggbb' as the 0..1 triple fitz wants. Inverse of _hex."""
    value = hex_color.lstrip("#")
    try:
        return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return (1.0, 1.0, 1.0)


def _prepare_text_only_image(
    png_bytes: bytes,
    width: int,
    height: int,
    blocks: list[dict],
    also_erase: list[dict] | None = None,
    anchor: bool = False,
) -> tuple[bytes, list[dict]]:
    """Decide where each of a picture's words goes, and clear the way for them.
    Returns (cleaned_png_bytes, overlay_blocks).

    Every block ends in exactly one of three states, and the third is why no block is ever
    dropped for want of somewhere to put it:
      - erased:   the box sits on a paintable field, so it is painted flat and written over.
      - anchored: `anchor` mode only — the box is moved onto the blank surface as it exists
                  in this image, and nothing is painted.
      - scrim:    neither worked (a photograph, a gradient, artwork), so the block carries a
                  plate colour and `_overlay_text_blocks` draws it a surface to sit on.

    `anchor=True` is passed only from the regeneration path. There the picture has been
    redrawn, so a box's coordinates describe a placard that may have moved a few percent, and
    painting at those coordinates is how Bangla ended up over a hand. It applies to `blocks`
    only.

    `also_erase` blocks are painted out but never overlaid, and are measured on this image
    already. That is how a regenerated image is cleaned: the model sometimes redraws the
    words it was told to omit, in its own hand and in its own position, so the boxes to erase
    are not the boxes to write into.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        logger.exception("text-only: cannot open image; keeping original untouched")
        return png_bytes, []
    draw = ImageDraw.Draw(img)
    W, H = img.size

    def box_of(b: dict) -> tuple[int, int, int, int] | None:
        bbox = b.get("bbox") or []
        if len(bbox) != 4:
            return None
        x0, y0 = int(bbox[0] * W), int(bbox[1] * H)
        x1, y1 = int(bbox[2] * W), int(bbox[3] * H)
        return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None

    def erase(b: dict) -> tuple[tuple[int, int, int], tuple[int, int, int, int]] | None:
        """Paint out one block. Returns (background colour, box actually painted), or None."""
        box = box_of(b)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        # A plain ring is the clear case; failing that, a ring that is overwhelmingly one
        # colour still paints out invisibly. Only genuine photographic detail is refused.
        bg = _sample_uniform_bg(img, x0, y0, x1, y1) or _dominant_ring_color(img, x0, y0, x1, y1)
        if bg is None:
            return None  # over a photo — leave the text alone
        # The ring only says what surrounds the box. Before painting, check the box is
        # actually filled with that same colour — otherwise a mis-sized OCR box around a
        # whole illustration, ringed by page white, would be painted out as "text".
        # Lettering covers a minority of its own box, so a real text box stays mostly
        # background even with the glyphs still in it.
        if not _mostly_colored(img, box, bg):
            return None
        # The model's box is a hint and routinely stops short of the last word, so follow
        # the ink out to where it really ends before painting anything.
        sx0, sy0, sx1, sy1 = _snap_to_ink(img, box, bg)
        # Glyphs are anti-aliased and their reported box is tight, so a box painted to its
        # exact edge leaves a grey fringe of the old lettering behind.
        pad = max(1, round(0.02 * (y1 - y0)))
        # A box that grew this far is not following a word any more. Fall back to what the
        # model reported rather than paint the picture out — see SNAP_MAX_AREA_GROWTH.
        if (sx1 - sx0) * (sy1 - sy0) > SNAP_MAX_AREA_GROWTH * (x1 - x0) * (y1 - y0):
            sx0, sy0, sx1, sy1 = box
        painted = (sx0 - pad, sy0 - pad, sx1 + pad, sy1 + pad)
        draw.rectangle(list(painted), fill=bg)
        return bg, painted

    for b in also_erase or []:
        erase(b)

    overlay_blocks: list[dict] = []
    for b in blocks:
        box = box_of(b)
        if box is None:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > SCRIM_MAX_AREA_SHARE * W * H:
            # The model boxed the picture, not a label. Painting or plating this would erase
            # the artwork; writing into it would put one caption across the whole image.
            logger.debug("Ignoring an OCR box that covers the picture: %r", (b.get("text") or "")[:40])
            continue

        scrim = None
        if anchor:
            # The picture has been redrawn, so the box describes a surface that may no longer
            # be where it was. Find it in the image as it now is, or supply one.
            found = _anchor_box(img, box)
            if found is not None:
                box, bg = found
            else:
                bg = _plate_color(img, box)
                scrim = _hex(bg)
        else:
            painted = erase(b)
            if painted is not None:
                bg, box = painted
            elif _is_flat_area(img, box):
                # Nothing was painted, but the spot may still be free to write on. The ring
                # test asks whether the area *around* the box is plain, which fails wherever
                # the box is tucked into detail — the cover slogan sits between the folds and
                # outline of a shirt. What actually matters is whether the box's own interior
                # is clear, which it is once the model has left the surface blank as asked.
                bg = tuple(round(c) for c in ImageStat.Stat(img.crop(box)).mean[:3])  # type: ignore[assignment]
            else:
                # Genuine artwork under the words. This used to `continue`, i.e. ship the
                # block in English and say nothing; now it gets a surface of its own.
                bg = _plate_color(img, box)
                scrim = _hex(bg)

        # Nothing may show through the new text. A neighbour's refused erase leaves its ink
        # inside this block's destination, which is where it would be read from.
        if scrim is None and _ink_share(img, box, bg) > MAX_RESIDUAL_INK:
            draw.rectangle(list(box), fill=bg)

        # Choose an overlay text color that contrasts whatever it is being written on.
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        nb = dict(b)
        nb["color"] = "#000000" if luminance > 140 else "#ffffff"
        nb["scrim"] = scrim  # None, or the hex of the plate to draw under the words
        # The Bangla goes where the English was wiped, not where the model said it was.
        # Those differ once the box has been snapped, and writing to the unsnapped box is
        # how the eatwell plate's "Meat, fish, eggs, beans and other non-dairy sources of
        # protein" ended up rendered to the left of the English it was replacing.
        nb["source_bbox"] = list(b.get("bbox") or [])
        nb["bbox"] = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
        overlay_blocks.append(nb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), overlay_blocks


def _translate_blocks(
    model_png: bytes, blocks: list[dict], page_context: str, record: dict
) -> None:
    """One image-aware translation call for a picture's whole text, recorded so that a
    picture which fell back to the per-block path is visible in the audit.

    Always given the *original* copy of the picture: that is where the English is still
    legible. Blocks that already carry a translation cost nothing, so the discard path
    reaching this twice is free.
    """
    if not blocks:
        return
    try:
        translate_blocks(model_png, "image/png", blocks, page_context)
    except Exception:
        logger.exception("Batch block translation failed; falling back to per-block")
    record["blocks_found"] = len(blocks)
    record["blocks_translated"] = sum(1 for b in blocks if (b.get("bn") or "").strip())
    # Only blocks that actually needed translating can fail to be translated. A caption that
    # was already Bangla carries no "bn" and never will, and counting those as untranslated
    # made a clean cover read "3 untranslated" — the number the audit is checked against.
    record["blocks_untranslated"] = sum(
        1
        for b in blocks
        if not (b.get("bn") or "").strip() and _is_english_block(b)
    )


def _text_only_result(
    png_bytes: bytes,
    width: int,
    height: int,
    blocks: list[dict],
    record: dict,
    ident: str,
    page_num: int,
    status: str = "text_only",
    logos: list[dict] | None = None,
    page_context: str = "",
    empty_status: str = "classify_kept",
) -> tuple[bytes | None, str, dict]:
    """Keep the picture exactly as drawn and translate only the words baked into it.

    Every path that decides against redrawing an image ends here — the classifier keeping a
    clinical figure, a failed classify, a failed edit, a regeneration that lost text — so
    that "not regenerated" never also means "left in English".

    Logo wording is exempt: a diagram or photograph that is not itself a logo can still carry
    a hospital's or charity's mark, and this path would otherwise paint that mark's name out
    and write a Bangla translation of it. Pass `logos` if they have already been detected;
    otherwise they are detected here.

    Returns (None, `empty_status`, record) when nothing could be written, which is now rare:
    a block with nowhere clean to go gets a plate rather than being dropped (see SCRIM_*), so
    reaching this means the picture had no usable text at all. `empty_status` is what *this
    caller's* silence means — "regeneration_discarded", "edit_failed", "classify_failed", or
    "classify_kept" only where the image genuinely had nothing to do. The single
    "classify_kept" this used to return made a thrown-away regeneration indistinguishable
    from a picture the classifier had deliberately kept, and that is how the eatwell plate
    shipped in English with an audit record saying it had been considered and left alone.
    """
    if blocks:
        if logos is None:
            logos = detect_logo_regions(_downscale_for_model(png_bytes), "image/png")
        if logos:
            record["logo_regions"] = [logo.get("label", "") for logo in logos]
            logger.info(
                "Page %d: %s contains %d logo/brand mark(s) (%s) — left as printed, "
                "not translated",
                page_num, ident, len(logos),
                "; ".join((logo.get("label") or "?")[:30] for logo in logos[:3]),
            )
        blocks = _outside_logos(blocks, logos)
        _translate_blocks(_downscale_for_model(png_bytes), blocks, page_context, record)
    cleaned, overlay_blocks = _prepare_text_only_image(png_bytes, width, height, blocks)
    record["mode"] = "text_only"
    record["text_blocks"] = overlay_blocks
    record["scrim_blocks"] = sum(1 for b in overlay_blocks if b.get("scrim"))
    if not overlay_blocks:
        logger.info(
            "Page %d: %s left untouched (%s) — it has no text that could be re-rendered",
            page_num, ident, empty_status,
        )
        record["status"] = empty_status
        record["untouched"] = True
        return None, empty_status, record
    logger.info(
        "Page %d: %s kept as artwork; %d of its %d text block(s) re-rendered in Bangla "
        "(%d on a drawn plate)",
        page_num, ident, len(overlay_blocks), len(blocks), record["scrim_blocks"],
    )
    record["status"] = status
    return cleaned, status, record


def _regeneration_mode(page_context: str, width: int, height: int) -> str:
    """Whether this picture is redrawn with the page's words ("context") or without ("simple").

    See image_localizer.LOCALIZE_MODES. Both conditions are about the same failure from
    opposite ends: the model treats whatever it is handed as a brief, so three stray words
    of context, or a paragraph of art direction aimed at a 300x300 icon, produce a picture
    with detail invented to satisfy an instruction that was never about it.
    """
    if width * height <= SIMPLE_MODE_MAX_PIXELS:
        return "simple"
    if len(page_context.split()) < CONTEXT_MIN_WORDS:
        return "simple"
    return "context"


def _block_key(block: dict) -> tuple:
    """Identity of an OCR block, stable across the copy `_prepare_text_only_image` returns.

    That copy carries a *snapped* bbox — the erase box grown out to the real ink — so the
    box the block was born with is kept beside it under `source_bbox` and is what identifies
    it. Without that, a block would fail to match itself and every regeneration would look
    like text loss.
    """
    bbox = block.get("source_bbox") or block.get("bbox") or []
    return (
        (block.get("text") or "").strip(),
        tuple(round(float(v), 4) for v in bbox),
    )


def _is_english_block(block: dict) -> bool:
    """True if an OCR block is English text that a Bangla reader could not read.

    Trusts the model's own language tag where it gave one, and falls back to the glyphs:
    Latin letters with no Bangla in the string. Digits, punctuation and stray marks are
    not text worth translating, so a block needs at least two letters to count.
    """
    text = (block.get("text") or "").strip()
    if not text:
        return False
    lang = (block.get("lang") or "").strip().lower()
    if lang.startswith("bn"):
        return False
    if any("ঀ" <= ch <= "৿" for ch in text):
        return False  # already Bangla, whatever the tag says
    if lang and not lang.startswith("en"):
        return False  # some other language — out of scope, leave it alone
    return sum(1 for ch in text if ch.isascii() and ch.isalpha()) >= 2


# How far around a picture to read for context, and how much of it to keep. 90pt is about
# six lines of body text: enough to reach the heading above a figure and the caption below
# it, without pulling in the next section. The cap is on the prompt, not the reading — a
# whole page of prose handed to an image model stops being context and becomes a brief.
CONTEXT_MARGIN = 90
CONTEXT_MAX_CHARS = 600


def _get_page_context(page: fitz.Page, rect: fitz.Rect, max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """Extract the text printed around a picture, as context for regenerating it.

    Read from the picture's own placement rect rather than from the page as a whole: what
    tells the model what a figure is illustrating is the heading over it and the caption
    under it, and on a two-column page the first 300 characters of the page are just as
    likely to be about something else entirely.

    Falls back to the whole page when the tight read comes back empty — a picture on a
    divider page has nothing beside it but still sits under a section title.
    """
    if page is None:
        return ""

    def read(target: fitz.Rect) -> str:
        try:
            return " ".join((page.get_textbox(target) or "").split())
        except Exception:
            logger.debug("Could not extract page context for rect %s", target, exc_info=True)
            return ""

    text = read((rect + (-CONTEXT_MARGIN, -CONTEXT_MARGIN, CONTEXT_MARGIN, CONTEXT_MARGIN)) & page.rect)
    if len(text.split()) < CONTEXT_MIN_WORDS:
        text = read(page.rect) or text
    return text[:max_chars].strip()


def _decide_from_png(
    png_bytes: bytes,
    width: int,
    height: int,
    ident: str,
    page_num: int,
    page_context: str = "",
    in_series: bool = False,
) -> tuple[bytes | None, str, dict]:
    """Classify, OCR, and edit a PNG image. Core decision logic used by both xref-extracted
    and rasterized-region paths.

    Args:
        png_bytes: full-resolution PNG bytes
        width, height: native dimensions
        ident: identifier for logging (e.g. xref number or region content_key)
        page_num: page number for logging
        page_context: nearby text context for the edit model
        in_series: this image is one tile of a repeated set (see _series_xrefs)

    Returns: (edited_bytes, status, audit_record)
    """
    # Checked before the classify call, so an image that can never be regenerated does not
    # spend a request finding that out.
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > MAX_PICTURE_ASPECT:
        logger.info(
            "Page %d: %s is a %dx%d strip (aspect %.1f) — left exactly as printed",
            page_num, ident, width, height, aspect,
        )
        return None, "aspect_kept", {
            "ident": ident, "page": page_num, "width": width, "height": height,
            "status": "aspect_kept", "regeneration_vetoed": "extreme_aspect",
            "untouched": True,
        }

    # A resolution-capped copy for the model calls (classify/OCR/edit). The full-res png_bytes is
    # kept for text-only erasing and as _resize_to's target dimensions.
    model_png = _downscale_for_model(png_bytes)

    start = time.time()
    decision = classify_image(model_png, "image/png")
    classify_time = time.time() - start

    record = {
        "ident": ident,
        "page": page_num,
        "width": width,
        "height": height,
        "is_logo": bool(decision.get("is_logo")),
        "categories": decision.get("categories", []),
        "needs_localization": decision.get("needs_localization", False),
        "information_role": decision.get("information_role") or "referential",
        "reason": decision.get("reason", ""),
        "classify_time_sec": round(classify_time, 2),
    }

    # A logo leaves untouched, and that has to be decided here — before OCR, before the
    # text-only path, before anything reads its pixels. Every other "we are not redrawing
    # this" branch still erases the image's baked English and writes Bangla back over it,
    # which on a wordmark means translating a real organisation's name. Returning None keeps
    # the original bytes: no regeneration, no erasing, no overlay, and no re-encode.
    # Only a picture that IS a mark stops here. One that merely *contains* one goes on to be
    # localized, with the mark protected by detect_logo_regions instead — a single boolean
    # used to decide both, and answering it wrongly shipped the FSA eatwell plate whole and
    # in English because an agency crest sat in its corner. See CLASSIFY_SYSTEM_PROMPT.
    record["logo_fills_image"] = bool(decision.get("logo_fills_image", True))
    if decision.get("is_logo") and record["logo_fills_image"]:
        logger.info(
            "Page %d: %s is a logo or brand mark — left exactly as printed (%s)",
            page_num, ident, decision.get("reason", "")[:120],
        )
        record["status"] = "logo_kept"
        record["untouched"] = True  # by design, and the one case where that is the goal
        return None, "logo_kept", record

    if decision.get("reason", "").lower().startswith("classify failed"):
        # The classifier never answered, so nothing is known about what this picture is.
        # Redrawing an unknown picture in a clinical manual is the one thing that must not
        # happen on a failure, but its baked English can still be translated — so fall
        # through to the text-only path rather than returning empty-handed.
        logger.warning(
            "Page %d: %s could not be classified; translating its text only", page_num, ident
        )
        record["classify_failed"] = True
        return _text_only_result(
            png_bytes, width, height, _extract_text_blocks(model_png, "image/png"),
            record, ident, page_num, page_context=page_context,
            empty_status="classify_failed",
        )

    # Three independent reasons a picture must not be redrawn, checked before the
    # cultural question is even asked. Any one of them is enough, and none of them
    # depends on the edit model behaving: a redrawn drink in a units chart is a
    # factual error in a clinical document, so the guards are deliberately
    # over-lapping rather than minimal. Such an image still gets its baked English
    # translated — it is the pixels that are protected, not the language.
    #
    # Which of these are on is set by the VETO_* flags: see their definition.
    veto = None
    # Defaulted here as well as in classify_image: a decision that never named a role is not
    # a decision to redraw, and this branch must not be the one place that reads a missing
    # field as permission.
    if VETO_REFERENTIAL_REGENERATION and (
        decision.get("information_role", "referential") != "decorative"
    ):
        veto = "referential"
    elif VETO_SERIES_REGENERATION and in_series:
        veto = "series_member"
    elif VETO_SMALL_REGENERATION and width * height < REGEN_MIN_PIXELS:
        veto = "too_small_to_regenerate"
    if veto:
        record["regeneration_vetoed"] = veto

    if veto or not decision.get("needs_localization"):
        # Culturally the picture can stay — but English baked into it cannot: a Bangla
        # reader is still left with an English chart, diagram or label. Keep the artwork
        # exactly as drawn and translate only the words, which is also the safe move for
        # the diagrams and clinical figures the classifier is keeping on purpose.
        if veto:
            logger.info(
                "Page %d: %s will not be regenerated (%s) — %s; translating its text only",
                page_num, ident, veto, decision.get("reason", "")[:120],
            )
        blocks = _extract_text_blocks(model_png, "image/png")
        english = [b for b in blocks if _is_english_block(b)]
        if english:
            record["english_blocks"] = len(english)
            return _text_only_result(
                png_bytes, width, height, blocks, record, ident, page_num,
                status="text_translated", page_context=page_context,
            )

        logger.debug(
            "Page %d: %s kept — %s (%.1fs)",
            page_num,
            ident,
            decision.get("reason", "no localization needed"),
            classify_time,
        )
        # Recorded even though nothing happened: "kept, and it had no English in it" is a
        # different fact from "kept, and nobody looked", and only the first one is fine.
        record["blocks_found"] = len(blocks)
        record["english_blocks"] = 0
        record["status"] = "classify_kept"
        record["untouched"] = True
        return None, "classify_kept", record

    # OCR up front: it feeds the text overlay, and tells us afterwards whether the
    # regenerated picture brought all of the original's words back with it.
    # Uses the downscaled copy; bboxes are normalized so they map onto the full-res original.
    blocks = _extract_text_blocks(model_png, "image/png", notes=record)

    if record.get("ocr_failed"):
        # The edit model is told to return every text surface blank, and the words are put
        # back afterwards from this list. An empty list because the OCR call failed is
        # therefore an instruction to delete the picture's text — so the picture does not go
        # to the edit model at all. Checked here, before the generation is paid for.
        logger.error(
            "Page %d: %s — could not read the image's text, so it is left exactly as printed "
            "rather than regenerated with its words blanked",
            page_num, ident,
        )
        record["status"] = "ocr_failed"
        record["untouched"] = True
        return None, "ocr_failed", record

    # Large, text-dense images (nutrition charts, infographics) are kept as-is — generative
    # regeneration would wreck their layout and is slow/429-prone. Erase the baked text on solid
    # backgrounds and let the Bangla overlay / existing PDF text layer supply the translation.
    if (
        VETO_LARGE_TEXT_DENSE_REGENERATION
        and width * height >= TEXT_ONLY_MIN_PIXELS
        and len(blocks) >= TEXT_ONLY_MIN_BLOCKS
    ):
        record["regeneration_vetoed"] = "large_and_text_dense"
        return _text_only_result(
            png_bytes, width, height, blocks, record, ident, page_num,
            page_context=page_context,
        )

    mode = _regeneration_mode(page_context, width, height)
    record["regeneration_mode"] = mode
    start = time.time()
    new_image = localize_image(
        model_png,
        "image/png",
        decision.get("categories", []),
        page_context,
        _palette_summary(model_png),
        mode=mode,
        notes=record,
    )
    edit_time = time.time() - start
    record["edit_time_sec"] = round(edit_time, 2)

    if not new_image:
        # Every edit model refused or failed. The picture stays as drawn, but its baked
        # English is still translated — a failed regeneration must not also cost the reader
        # the words that were already there.
        logger.warning(
            "Page %d: %s could not be regenerated after %.1fs; translating its text only",
            page_num, ident, edit_time,
        )
        record["edit_failed"] = True
        return _text_only_result(
            png_bytes, width, height, blocks, record, ident, page_num,
            status="text_translated", page_context=page_context,
            empty_status="edit_failed",
        )

    logger.info(
        "Page %d: %s localized in %s mode (%s) — classify: %.1fs, edit: %.1fs",
        page_num,
        ident,
        mode,
        ", ".join(decision.get("categories", [])) or "cultural content",
        classify_time,
        edit_time,
    )
    # Hold the regenerated picture to the source's background colour before anything else
    # measures it — the prompt asks for this, but only a measurement guarantees it.
    corrected = _match_background(_resize_to(new_image, width, height), png_bytes)

    # A picture that IS a logo never reaches here — classify_image's is_logo returned it
    # untouched long before the edit. What is still possible is a mark printed inside a
    # bigger picture: a hospital's name on a sign in a photograph, a charity's crest in the
    # corner of an illustration. Its pixels are not restamped (pasting the original back
    # would leave a visible patch over the regeneration), but its *wording* is taken off the
    # table, so neither the erase pass nor the Bangla overlay touches it.
    logos = detect_logo_regions(model_png, "image/png") if blocks else []
    if logos:
        record["logo_regions"] = [logo.get("label", "") for logo in logos]
        logger.info(
            "Page %d: %s contains %d logo/brand mark(s) (%s) — their wording is left as printed",
            page_num, ident, len(logos),
            "; ".join((logo.get("label") or "?")[:30] for logo in logos[:3]),
        )
    blocks = _outside_logos(blocks, logos)
    _translate_blocks(model_png, blocks, page_context, record)

    # The prompt demands blank text surfaces, and the model does not always obey: asked to
    # clear the cover figure's shirt it redrew "HELP YOURSELF TO A HEALTHY FUTURE" in its
    # own hand, and the Bangla overlay then landed on top of English that was still there.
    # So the generated image is re-read and whatever it drew is painted out, along with the
    # boxes the Bangla is about to occupy. Where the model obeyed, this repaints flat colour
    # with the same flat colour — a no-op that still yields a contrasting text colour. A box
    # whose surroundings are not uniform is left alone AND not overlaid, so nothing is ever
    # written on top of artwork.
    #
    # The residual pass is filtered by the logo regions too, and that filter is the load-
    # bearing one: it is reading an image that now has the original marks stamped back into
    # it, so without this it would find the logo's own wording and paint it out — undoing
    # the restamp a few lines above.
    residual = _extract_text_blocks(_downscale_for_model(corrected), "image/png") if blocks else []
    residual = _outside_logos(residual, logos)
    if residual:
        logger.info(
            "Page %d: %s — model redrew %d text block(s) it was told to leave blank; erasing",
            page_num, ident, len(residual),
        )
    # anchor=True: the picture in hand is not the one the boxes were measured on, so each
    # block is placed against the surface as the model actually redrew it.
    corrected, overlay_blocks = _prepare_text_only_image(
        corrected, width, height, blocks, also_erase=residual, anchor=True
    )

    # A word the original carried has to come back, and it has to come back somewhere clean.
    # The model was told to blank the text surfaces and did, so a block the overlay declines
    # to redraw is not "left in English" — it is gone.
    #
    # This used to discard the whole regeneration if a single block was missing, which cost
    # far more than it saved: the discarded picture went to _text_only_result, which found
    # nothing erasable on a photograph either, and the image shipped a hundred percent in
    # English under the status "classify_kept" — no regeneration, no translation, and no
    # trace of either failure. Now that a block can always be given a surface, a block that
    # is still missing is a broken box rather than a lost sentence, and the two things worth
    # backing out for are measured instead: too few blocks placed at all, or so many of them
    # on drawn plates that the model must have moved its surfaces out from under them. That
    # second test is the framing-drift rejection, bought with pixels rather than another
    # model call.
    restored = {_block_key(b) for b in overlay_blocks}
    lost = [b for b in blocks if _block_key(b) not in restored]
    share = len(overlay_blocks) / max(len(blocks), 1)
    scrims = sum(1 for b in overlay_blocks if b.get("scrim"))
    scrim_share = scrims / max(len(overlay_blocks), 1)
    record["lost_text_blocks"] = [b.get("text", "") for b in lost]
    record["restored_share"] = round(share, 3)
    record["anchored_blocks"] = len(overlay_blocks) - scrims
    record["scrim_blocks"] = scrims
    if blocks and (share < KEEP_REGENERATION_MIN_RESTORED or scrim_share > MAX_SCRIM_SHARE):
        why = "text_loss" if share < KEEP_REGENERATION_MIN_RESTORED else "framing_drift"
        logger.warning(
            "Page %d: %s — discarding the regeneration (%s: %d of %d blocks restored, %d of "
            "those on a plate); translating the original's text instead",
            page_num, ident, why, len(overlay_blocks), len(blocks), scrims,
        )
        record["regeneration_discarded"] = why
        return _text_only_result(
            png_bytes, width, height, blocks, record, ident, page_num,
            status="text_translated", logos=logos, page_context=page_context,
            empty_status="regeneration_discarded",
        )

    record["residual_text_blocks"] = len(residual)
    record["text_blocks"] = overlay_blocks
    record["status"] = "edit_ok"
    return corrected, "edit_ok", record


def _decide(
    doc: fitz.Document,
    xref: int,
    page_num: int,
    page_context: str = "",
    raster_source: tuple[bytes, int, int] | None = None,
    in_series: bool = False,
) -> tuple[bytes | None, str, dict]:
    """Extract a raster image from the PDF and decide whether to localize it.

    This is the original entry point for xref-based images. For ImageMask stencils,
    `raster_source` carries bytes already rasterized by _rasterize_rect in the main-thread
    collection pass: extract_image returns such a stencil as a bare 1-bit mask with no
    colour, which the classifier reads as a black square.

    Status is one of STATUS_VOCABULARY: extract_failed, decode_failed, classify_kept,
    aspect_kept, classify_failed, edit_failed, edit_ok, text_only, text_translated,
    logo_kept, regeneration_discarded.

    Returns a tuple of (edited_bytes, status, audit_record).
    """
    if raster_source is not None:
        png_bytes, width, height = raster_source
        result_bytes, status, record = _decide_from_png(
            png_bytes, width, height, f"xref:{xref}", page_num, page_context, in_series
        )
        record["xref"] = xref
        return result_bytes, status, record

    try:
        extracted = doc.extract_image(xref)
    except Exception:
        logger.debug("Page %d: xref %d extract_image failed", page_num, xref)
        return None, "extract_failed", {"xref": xref, "page": page_num, "status": "extract_failed"}

    if not extracted or not extracted.get("image"):
        return None, "extract_failed", {"xref": xref, "page": page_num, "status": "extract_failed"}

    normalized = _to_png(extracted["image"])
    if normalized is None:
        logger.debug("Page %d: xref %d could not decode image", page_num, xref)
        return None, "decode_failed", {"xref": xref, "page": page_num, "status": "decode_failed"}

    png_bytes, width, height = normalized
    result_bytes, status, record = _decide_from_png(
        png_bytes, width, height, f"xref:{xref}", page_num, page_context, in_series
    )
    # Add xref to the record for backwards compatibility with audit logs
    record["xref"] = xref
    return result_bytes, status, record
