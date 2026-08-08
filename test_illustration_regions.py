r"""Vector-illustration and ImageMask detection tests.

Tests the image_regions module (detection, rasterization, dedup) and the placement
half of image_processor against the real document to validate that:
  * The vector-drawn cover illustration is detected, on every page that carries it
  * A localized image is actually visible once placed (not buried under the page's
    own background matte)
  * ImageMask stencil images are correctly identified
  * False positives (crop marks, backdrops, tables, checkbox pages) are rejected
  * Dedup works (same drawing signature reused on multiple pages)

Tier 1 — no API calls, run via:
  .\.venv\Scripts\python.exe test_illustration_regions.py

Optional Tier 2 (requires --dump-crops flag):
  .\.venv\Scripts\python.exe test_illustration_regions.py --dump-crops <output-dir>
"""

import io
import json
import os
import random
import sys
import tempfile
from collections import defaultdict

import fitz
from PIL import Image, ImageDraw

from image_processor import (
    CONTEXT_MIN_WORDS,
    LOCK_MAX_AREA_SHARE,
    LOCK_MAX_INK,
    LOCK_MERGE_PAD_PT,
    LOCK_SOURCE_MAX_INK,
    MAX_RESIDUAL_INK,
    MIN_OVERLAY_HEIGHT,
    SIMPLE_MODE_MAX_PIXELS,
    SNAP_MAX_AREA_GROWTH,
    STYLE_FLAT_MAX_COLORS,
    STYLE_FLAT_MAX_SOFT,
    STYLE_PHOTO_MIN_COLORS,
    STYLE_PHOTO_MIN_SOFT,
    _anchor_box,
    _apply_cover,
    _block_key,
    _ink_share,
    _plate_color,
    _text_only_result,
    _write_audit,
    _derive_smask,
    _detect_cover_page,
    _downscale_for_model,
    _flat_locks,
    _is_english_block,
    _locked_rects,
    _locked_rects_for_xref,
    _locked_share,
    _lock_summary,
    _locks_kept,
    _match_background,
    _outside_logos,
    _overlay_text_blocks,
    _background_color,
    _page_ink_coverage,
    _restamp_locks,
    _restamp_logos,
    _palette_summary,
    _prepare_text_only_image,
    _punch_text_holes,
    _regeneration_mode,
    _series_xrefs,
    _snap_to_ink,
    _style_class,
    _style_metrics,
    _style_summary,
    _swap_image_in_place,
    _text_rects_in,
    _to_png,
)
from image_regions import (
    TextFreePages,
    _illustration_clusters,
    _is_backdrop,
    _is_furniture_ink,
    _is_image_mask,
    _is_mark_shaped,
    _outlined_text_lines,
    _rasterize_rect,
)
from pdf_processor import FONTS_DIR, _panels, _rules, _vector_marks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PDF = os.path.join(BASE_DIR, "OriginalPDF", "Heart Manual_Post Myocardial Infarction (1).pdf")
# A *translated* manual, which is the case the page-level text gate got wrong: see
# test_divider_cartoons_are_found_in_a_translated_manual.
REVASC_PDF = os.path.join(
    BASE_DIR, "OriginalPDF", "Heart Manual Revascularisation (3) [77-91]_bn_merge.pdf"
)

# The cover cartoon, and every page it is reprinted on. Detection must find it on all of
# them and dedup them to a single signature (one AI edit for the whole book).
COVER_PAGES = [1, 19, 37, 61, 81, 99, 119]
COVER_RECT = fitz.Rect(47.4, 380.2, 434.5, 900.9)

# Raster illustrations the layout draws on an opaque white matte of their own size — the
# case that used to come out blank. (page, xref, placement rect)
MATTED_IMAGES = [
    (5, 14, fitz.Rect(112.5, 173.2, 293.9, 388.3)),
    (24, 85, fitz.Rect(51.5, 328.4, 206.9, 457.6)),
]

failures = []
skipped = []


def regions_on(doc: fitz.Document, page_num: int):
    """Detected illustration regions for a page, wired the way localize_pdf wires them."""
    page = doc[page_num - 1]
    protected = _rules(page) + _vector_marks(page) + _panels(page)
    image_rects = []
    for img in page.get_images(full=True):
        image_rects.extend(page.get_image_rects(img[0]))
    return _illustration_clusters(page, page_num, protected, image_rects)


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def test_imagemask_detection():
    """Tier 1: Verify ImageMask xrefs are correctly identified."""
    print("\n=== ImageMask Detection ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("imagemask_detection")
        return

    doc = fitz.open(SOURCE_PDF)
    page = doc[128]  # p.129

    # Known xrefs on this page (verified via manual inspection)
    xref_439_is_mask = _is_image_mask(doc, 439)
    xref_440_is_mask = _is_image_mask(doc, 440)
    xref_441_is_mask = _is_image_mask(doc, 441)

    check("xref 439 is ImageMask", xref_439_is_mask, "should be True")
    check("xref 440 is ImageMask", xref_440_is_mask, "should be True")
    check("xref 441 is NOT ImageMask", not xref_441_is_mask, "should be False (normal color image)")

    doc.close()


def test_imagemask_rasterization():
    """Tier 1: Verify ImageMask rasterization produces non-black output."""
    print("\n=== ImageMask Rasterization ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("imagemask_rasterization")
        return

    doc = fitz.open(SOURCE_PDF)
    page = doc[128]

    # Get placement rects for masked xrefs
    rects_439 = page.get_image_rects(439)
    rects_440 = page.get_image_rects(440)

    for xref, rects, name in [(439, rects_439, "439"), (440, rects_440, "440")]:
        if not rects:
            check(f"xref {name} has placements", False, "no rects found")
            continue

        rect = rects[0] if isinstance(rects[0], fitz.Rect) else rects[0][0]
        result = _rasterize_rect(page, rect)

        if result is None:
            check(f"xref {name} rasterizes", False, "rasterization failed")
            continue

        png_bytes, w, h = result
        # Load the PNG and check mean brightness
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            pixels = list(img.getdata())
            mean_brightness = sum(pixels) / len(pixels) if pixels else 0
            # A "mostly-dark" image (the old bug: black square) would have mean < 50/255
            # Correct compositing (white paper + dark ink) should be > 100
            is_light = mean_brightness > 100
            check(
                f"xref {name} mean brightness > 100 (not black square)",
                is_light,
                f"mean brightness: {mean_brightness:.0f}",
            )
        except Exception as e:
            check(f"xref {name} brightness check", False, f"PIL check failed: {e}")

    doc.close()


def test_backdrop_rejection():
    """Tier 1: The page's own background panels must never join an illustration cluster.

    This is what used to fuse the cover cartoon with the whole page, and it is the
    difference between "one tight region" and "a bbox covering everything".
    """
    print("\n=== Backdrop Rejection ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("backdrop_rejection")
        return

    doc = fitz.open(SOURCE_PDF)

    cover = doc[0]
    backdrops = [d for d in cover.get_drawings() if _is_backdrop(d, cover)]
    lavender = [d for d in backdrops if fitz.Rect(d["rect"]).get_area() > 300_000]
    check(
        "Cover's full-page lavender panel reads as a backdrop",
        bool(lavender),
        f"{len(backdrops)} backdrops on the cover, none of them full-page",
    )

    # The cartoon's own filled, curved paths (t-shirt, arms) must NOT read as backdrops:
    # they are the illustration.
    artwork = [
        d
        for d in cover.get_drawings()
        if d["type"] in ("f", "fs") and any(i[0] == "c" for i in d.get("items", []))
    ]
    check(
        "Cartoon's filled curved paths are not backdrops",
        artwork and not any(_is_backdrop(d, cover) for d in artwork),
        f"{sum(1 for d in artwork if _is_backdrop(d, cover))} of {len(artwork)} misread",
    )

    # The page-wide white fill body pages are printed on is a backdrop too.
    page24 = doc[23]
    matte = [
        d
        for d in page24.get_drawings()
        if _is_backdrop(d, page24) and fitz.Rect(d["rect"]).get_area() > 100_000
    ]
    check("Page 24's page-wide white fill reads as a backdrop", bool(matte))

    doc.close()


def test_cover_illustration_detected():
    """Tier 1: The cover cartoon is found, tightly, on every page that reprints it."""
    print("\n=== Cover Illustration Detection ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("cover_illustration_detected")
        return

    doc = fitz.open(SOURCE_PDF)

    for page_num in COVER_PAGES:
        regions = regions_on(doc, page_num)
        if len(regions) != 1:
            check(
                f"Page {page_num} detects exactly one region",
                False,
                f"detected {len(regions)}: {[list(r.rect) for r in regions]}",
            )
            continue
        rect = regions[0].rect
        close = all(abs(a - b) < 2.0 for a, b in zip(rect, COVER_RECT))
        check(
            f"Page {page_num}: cover cartoon at the expected rect",
            close,
            f"got {[round(v, 1) for v in rect]}, expected {[round(v, 1) for v in COVER_RECT]}",
        )
        check(
            f"Page {page_num}: region stays inside the page",
            doc[page_num - 1].rect.contains(rect),
            f"{[round(v, 1) for v in rect]} escapes {doc[page_num - 1].rect}",
        )

    doc.close()


def test_text_heavy_pages_rejected():
    """Tier 1: Body pages must yield nothing — their line art is tables and diagrams.

    Page 24's "Exercise can:" grid clusters into a perfectly plausible-looking region;
    handing it to the image model would return a picture where the table was.
    """
    print("\n=== Text-Heavy Page Rejection ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("text_heavy_pages_rejected")
        return

    doc = fitz.open(SOURCE_PDF)

    for page_num in (5, 24, 64, 129):
        regions = regions_on(doc, page_num)
        check(
            f"Page {page_num} (body page) yields no vector regions",
            not regions,
            f"detected {len(regions)}: {[[round(v, 1) for v in r.rect] for r in regions]}",
        )

    # Pages full of checkboxes are the ones with most to lose; none may be touched.
    checkbox_pages = [
        page_num
        for page_num, page in enumerate(doc, start=1)
        if len(_vector_marks(page)) >= 5
    ]
    offenders = [p for p in checkbox_pages if regions_on(doc, p) and p not in COVER_PAGES]
    check(
        f"No checkbox-bearing body page is rasterized ({len(checkbox_pages)} such pages)",
        not offenders,
        f"regions found on pages {offenders}",
    )

    doc.close()


def test_dedup_content_key():
    """Tier 1: Verify that the same drawing signature deduplicates across pages."""
    print("\n=== Dedup via Content Key ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("dedup_content_key")
        return

    doc = fitz.open(SOURCE_PDF)

    content_keys: dict[str, list[int]] = defaultdict(list)
    for page_num in range(1, len(doc) + 1):
        for cluster in regions_on(doc, page_num):
            content_keys[cluster.content_key].append(page_num)

    check(
        "The reprinted cover art dedups to a single signature",
        len(content_keys) == 1,
        f"{len(content_keys)} signatures: "
        f"{ {k[:8]: v for k, v in content_keys.items()} }",
    )
    if len(content_keys) == 1:
        pages = sorted(next(iter(content_keys.values())))
        check(
            "…covering every page that reprints it",
            pages == COVER_PAGES,
            f"got {pages}, expected {COVER_PAGES}",
        )

    doc.close()


def test_text_free_render():
    """Tier 1: The crop sent to the edit model must carry no page text."""
    print("\n=== Text-Free Region Render ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("text_free_render")
        return

    doc = fitz.open(SOURCE_PDF)
    text_free = TextFreePages(doc)

    stripped = text_free.page(1)
    check("Cover page still has its artwork after stripping text", bool(stripped.get_drawings()))
    check(
        "Cover page has no text left to bake into the crop",
        not stripped.get_text().strip(),
        f"leftover: {stripped.get_text().strip()[:80]!r}",
    )
    check("Original page is untouched", bool(doc[0].get_text().strip()))

    result = _rasterize_rect(stripped, COVER_RECT)
    check("Text-free region rasterizes", result is not None)

    text_free.close()
    doc.close()


def test_in_place_swap_is_visible():
    """Tier 1: A replaced image must actually render — the bug behind both screenshots.

    Both illustrations sit on an opaque white matte of their own size. Inserting the
    replacement into the page background put it under that matte, so the picture
    vanished. Rewriting the XObject keeps the original z-order, so it shows.
    """
    print("\n=== In-Place Swap Visibility ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("in_place_swap_is_visible")
        return

    marker = (220, 30, 30)
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), marker).save(buf, format="PNG")
    stand_in = buf.getvalue()

    doc = fitz.open(SOURCE_PDF)
    for _, xref, _ in MATTED_IMAGES:
        check(f"xref {xref} swaps in place", _swap_image_in_place(doc, xref, stand_in))

    # Round-trip through a real save: the rewritten object has to survive garbage
    # collection and re-compression, not just render in the live document. (garbage=3
    # renumbers objects, so the saved file is probed by placement rect, not by xref.)
    saved = fitz.open(stream=doc.tobytes(garbage=3, deflate=True), filetype="pdf")
    doc.close()

    for page_num, _, rect in MATTED_IMAGES:
        pix = saved[page_num - 1].get_pixmap(clip=rect, dpi=72)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        # Mean colour over the whole placement: if any of the page's white matte were
        # still painting over the replacement, this would wash out towards white.
        mean = img.resize((1, 1), Image.BOX).getpixel((0, 0))
        close = all(abs(a - b) < 24 for a, b in zip(mean, marker))
        check(
            f"Page {page_num}: replacement covers its placement (not hidden by the matte)",
            close,
            f"mean colour {mean}, expected about {marker}",
        )

    saved.close()


def test_punch_text_holes():
    """Tier 1: Page text crossing a region shows through; artwork is not perforated."""
    print("\n=== Text Hole Punching ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("punch_text_holes")
        return

    doc = fitz.open(SOURCE_PDF)
    text_free = TextFreePages(doc)
    source_png, _, _ = _rasterize_rect(text_free.page(1), COVER_RECT)

    text_rects = _text_rects_in(doc[0], COVER_RECT)
    check(
        "The cover blurb is detected as text crossing the region",
        any(r.x0 > 300 and r.y0 > 700 for r in text_rects),
        f"found {len(text_rects)} text rects: {[[round(v) for v in r] for r in text_rects]}",
    )

    buf = io.BytesIO()
    Image.new("RGB", (400, 540), (30, 60, 200)).save(buf, format="PNG")
    punched = _punch_text_holes(buf.getvalue(), source_png, COVER_RECT, text_rects)
    img = Image.open(io.BytesIO(punched)).convert("RGBA")
    transparent = img.getchannel("A").histogram()[0]
    check(
        "Holes are punched where the blurb crosses blank artwork",
        0 < transparent < 0.25 * (img.width * img.height),
        f"{transparent} of {img.width * img.height} pixels transparent",
    )

    # Text that sits on the artwork itself must be covered, not punched out: a rect over
    # the middle of the cartoon is not flat, so it must leave the image fully opaque.
    middle = fitz.Rect(
        COVER_RECT.x0 + 0.3 * COVER_RECT.width,
        COVER_RECT.y0 + 0.4 * COVER_RECT.height,
        COVER_RECT.x0 + 0.7 * COVER_RECT.width,
        COVER_RECT.y0 + 0.6 * COVER_RECT.height,
    )
    untouched = _punch_text_holes(buf.getvalue(), source_png, COVER_RECT, [middle])
    check(
        "No hole is punched through the illustration itself",
        untouched == buf.getvalue(),
        "the replacement was perforated over real artwork",
    )

    text_free.close()
    doc.close()


def test_palette_and_background_match():
    """Tier 1: The generated picture is held to the source's colours, not just asked to be."""
    print("\n=== Palette Fidelity ===")
    if not os.path.exists(SOURCE_PDF):
        print(f"  SKIP  {SOURCE_PDF} not found")
        skipped.append("palette_and_background_match")
        return

    doc = fitz.open(SOURCE_PDF)
    text_free = TextFreePages(doc)

    # The manual is printed in one restricted blue/lavender palette; the model has to be
    # told those values, or it returns a warmer, more saturated picture.
    cover_png, _, _ = _rasterize_rect(text_free.page(1), COVER_RECT)
    summary = _palette_summary(_downscale_for_model(cover_png))
    check(
        "Cover palette names the manual's lavender and blue",
        "#ae" in summary or "#af" in summary,
        f"got: {summary}",
    )

    page5 = _to_png(doc.extract_image(14)["image"])[0]
    summary5 = _palette_summary(page5)
    check(
        "A white-field illustration reports its flat background",
        "background is EXACTLY #ffffff" in summary5,
        f"got: {summary5}",
    )

    # A generated picture that came back on a cream field is pulled back to white, without
    # touching the drawing on top of it.
    drawn = Image.new("RGB", (60, 60), (250, 248, 240))
    drawn.paste((20, 20, 20), (20, 20, 40, 40))
    buf = io.BytesIO()
    drawn.save(buf, format="PNG")
    corrected = Image.open(io.BytesIO(_match_background(buf.getvalue(), page5))).convert("RGB")
    check("Off-white generated background is pulled to the source's white",
          corrected.getpixel((0, 0)) == (255, 255, 255),
          f"corner is {corrected.getpixel((0, 0))}")
    check("…and the artwork on it is untouched",
          corrected.getpixel((30, 30)) == (20, 20, 20),
          f"blob is {corrected.getpixel((30, 30))}")

    # A picture that bleeds to its edges has no background to correct, so nothing happens.
    noisy = Image.new("RGB", (60, 60))
    noisy.putdata([((x * 4) % 256, (y * 4) % 256, 128) for y in range(60) for x in range(60)])
    nbuf = io.BytesIO()
    noisy.save(nbuf, format="PNG")
    check(
        "A full-bleed source is left alone",
        _match_background(buf.getvalue(), nbuf.getvalue()) == buf.getvalue(),
    )

    text_free.close()
    doc.close()


def test_english_block_detection():
    """Tier 1: English baked into a picture is recognised; other languages are not touched."""
    print("\n=== English Text Detection ===")
    cases = [
        ({"text": "Hill walking", "lang": "en"}, True),
        ({"text": "Type II diabetes", "lang": ""}, True),   # untagged, but plainly English
        ({"text": "পাহাড়ে হাঁটা", "lang": "bn"}, False),
        ({"text": "পাহাড়ে হাঁটা", "lang": "en"}, False),    # mis-tagged; glyphs win
        ({"text": "5", "lang": "en"}, False),               # a number is not text to translate
        ({"text": "Bonjour", "lang": "fr"}, False),         # another language: out of scope
        ({"text": "", "lang": "en"}, False),
    ]
    for block, expected in cases:
        check(
            f"{block['text'][:20]!r} (lang={block['lang']!r}) -> {expected}",
            _is_english_block(block) is expected,
        )


def test_text_surface_decisions():
    """Tier 1: where Bangla may be written, and where writing it would spoil the picture.

    The cover slogan is the awkward case: a clear patch of shirt whose *surroundings* are
    all folds and outline. Judging it by its surroundings alone silently dropped the
    translation; judging the patch itself puts it back.
    """
    print("\n=== Text Surface Decisions ===")
    from image_processor import _prepare_text_only_image

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 160, 160], fill=(30, 60, 170))          # the shirt
    for x in range(40, 160, 9):                                     # folds all around
        draw.line([(x, 40), (x + 6, 160)], fill=(10, 20, 90), width=2)
    draw.rectangle([70, 85, 130, 115], fill=(30, 60, 170))          # the clear patch
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    source = buf.getvalue()

    _, overlay = _prepare_text_only_image(
        source, 200, 200, [{"text": "HELP YOURSELF", "lang": "en", "bbox": [0.36, 0.43, 0.64, 0.57]}]
    )
    check("A clear patch inside busy artwork is written on", len(overlay) == 1)
    check(
        "…in a colour that contrasts what it sits on",
        overlay and overlay[0]["color"] == "#ffffff",
        f"got {overlay[0]['color'] if overlay else None} on a dark blue shirt",
    )

    _, busy = _prepare_text_only_image(
        source, 200, 200, [{"text": "X", "lang": "en", "bbox": [0.20, 0.20, 0.80, 0.80]}]
    )
    check("Genuine artwork is never written on", not busy, f"{len(busy)} blocks accepted")

    # also_erase covers what the model redrew, without adding it to the overlay.
    dirty = img.copy()
    ImageDraw.Draw(dirty).text((75, 92), "HELP YOURSELF")
    dbuf = io.BytesIO()
    dirty.save(dbuf, format="PNG")
    cleaned, overlay2 = _prepare_text_only_image(
        dbuf.getvalue(), 200, 200, [], also_erase=[{"bbox": [0.34, 0.41, 0.66, 0.59]}]
    )
    residue = Image.open(io.BytesIO(cleaned)).convert("RGB").crop((72, 88, 128, 112))
    check(
        "Text the model redrew is painted out",
        max(hi - lo for lo, hi in residue.getextrema()) <= 8,
        f"still varies by {max(hi - lo for lo, hi in residue.getextrema())}",
    )
    check("…and an erase-only block is not overlaid", not overlay2)


def test_overlay_blocks_do_not_overlap():
    """Tier 1: Two OCR boxes for the same words must not stack two runs of Bangla."""
    print("\n=== Overlay Overlap Guard ===")
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    rect = fitz.Rect(50, 50, 250, 150)
    archive = fitz.Archive(os.path.join(BASE_DIR, "fonts"))

    # The whole caption, the same caption again, and one line of it — as OCR really reports.
    blocks = [
        {"text": "নমুনা লেখা", "lang": "bn", "bbox": [0.0, 0.0, 1.0, 0.5]},
        {"text": "নমুনা লেখা", "lang": "bn", "bbox": [0.02, 0.02, 0.98, 0.48]},
        {"text": "আরেকটি", "lang": "bn", "bbox": [0.0, 0.6, 1.0, 1.0]},
    ]
    _overlay_text_blocks(page, rect, blocks, archive)

    drawn = [fitz.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    collisions = [
        (a, b)
        for i, a in enumerate(drawn)
        for b in drawn[i + 1:]
        if (a & b).get_area() > 0.25 * min(a.get_area(), b.get_area())
    ]
    check(
        "Duplicate boxes render once, not stacked",
        not collisions,
        f"{len(collisions)} overlapping runs among {len(drawn)} drawn",
    )
    check("The non-overlapping block still renders", len(drawn) >= 2, f"{len(drawn)} drawn")

    # A bbox the model reported slightly outside its picture is clipped back inside it.
    page2 = doc.new_page(width=300, height=300)
    _overlay_text_blocks(
        page2, rect, [{"text": "প্রান্ত", "lang": "bn", "bbox": [-0.2, -0.2, 1.4, 1.4]}], archive
    )
    outside = [
        fitz.Rect(b["bbox"])
        for b in page2.get_text("dict")["blocks"]
        if b["type"] == 0 and not (rect + (-1, -1, 1, 1)).contains(fitz.Rect(b["bbox"]))
    ]
    check("An out-of-bounds bbox is clipped to the picture", not outside, f"{outside}")
    doc.close()


def test_series_pages_are_protected():
    """Tier 1: a page built around a grid of same-sized pictures is never regenerated.

    The alcohol-units chart on p.109 of the Heart Failure manual is eight cards, each
    a drink whose identity *is* the datum — this drink is 1.5 units. The edit model was
    asked to make the food Bangladeshi and did: the 125ml glass of red wine came back a
    brass cocktail cup, the shot of spirits an iced tea. The numbers beside them then
    described nothing.
    """
    print("\n=== Series Protection ===")
    doc = fitz.open()

    grid = doc.new_page(width=400, height=400)
    tile = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    tile.set_rect(tile.irect, (60, 120, 220))
    for row in range(2):
        for col in range(3):
            grid.insert_image(
                fitz.Rect(20 + col * 120, 20 + row * 120, 120 + col * 120, 120 + row * 120),
                pixmap=tile,
            )
    # One odd-sized picture on the same page: part of the chart, so also protected.
    odd = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    odd.set_rect(odd.irect, (220, 120, 60))
    grid.insert_image(fitz.Rect(20, 270, 260, 370), pixmap=odd)

    lone = doc.new_page(width=400, height=400)
    photo = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    photo.set_rect(photo.irect, (30, 190, 60))
    lone.insert_image(fitz.Rect(40, 40, 300, 300), pixmap=photo)

    # Re-fetch the pages by index: adding a page orphans the handles taken before it.
    grid_xrefs = {img[0] for img in doc[0].get_images(full=True)}
    lone_xrefs = {img[0] for img in doc[1].get_images(full=True)}
    series = _series_xrefs(doc)
    doc.close()

    check(
        "Every picture on a grid page is protected",
        grid_xrefs and grid_xrefs <= series,
        f"{sorted(grid_xrefs - series)} left unprotected",
    )
    check(
        "A page with a single picture is not a series",
        not (lone_xrefs & series),
        f"{sorted(lone_xrefs & series)} wrongly protected",
    )


def test_erase_box_snaps_to_the_ink():
    """Tier 1: an erase box that stops short of its last word is grown to reach it.

    Gemini's bboxes are a hint, not a measurement. On p.114 the box for "The eatwell
    plate" ended after "eatwell", so the erase left the word "plate" standing in English
    surrounded by fragments of the letters it had covered. Growing has to cross a word
    space but stop at the gap before the next element — the FSA logo sat 367px further
    on and must survive.
    """
    print("\n=== Erase Box Snapping ===")
    img = Image.new("RGB", (900, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 60, 300, 140], fill=(0, 150, 70))   # "The eatwell"
    draw.rectangle([336, 60, 420, 140], fill=(0, 150, 70))   # "plate", one word-space on
    draw.rectangle([760, 60, 860, 140], fill=(20, 20, 20))   # a separate logo, far right

    short = (100, 60, 300, 140)  # what the model reported: the last word left out
    x0, y0, x1, y1 = _snap_to_ink(img, short, (255, 255, 255))
    check(
        "The box grows across a word space to the end of its text",
        x1 >= 420,
        f"reached x={x1}, needed 420",
    )
    check(
        "…and stops well before the next element",
        x1 < 760,
        f"reached x={x1}, which is inside the logo at 760",
    )

    # A box that was already right must not wander into its neighbour.
    exact = _snap_to_ink(img, (100, 60, 420, 140), (255, 255, 255))
    check("A correct box stays put", exact[2] < 760, f"grew to x={exact[2]}")


def test_erased_blocks_are_identifiable():
    """Tier 1: a block that was erased must be recognisable in what comes back.

    After regeneration the pipeline compares the blocks it found against the blocks the
    overlay will redraw, and discards the regeneration if any went missing — the check
    that would have caught p.109 losing "1.5 units" and two whole captions. It only works
    if a block still matches itself once its box has been snapped.
    """
    print("\n=== Erased Block Identity ===")
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(44, 150, 12):  # glyph-like strokes: ink is a minority of its own box
        draw.rectangle([x, 46, x + 4, 74], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    blocks = [{"text": "2.1 units", "lang": "en", "bbox": [0.20, 0.33, 0.80, 0.67]}]
    _, overlay = _prepare_text_only_image(buf.getvalue(), 200, 120, blocks)

    check("The block was accepted for overlay", len(overlay) == 1, f"{len(overlay)} accepted")
    check(
        "…and still matches the block it came from",
        overlay and _block_key(overlay[0]) == _block_key(blocks[0]),
        f"{_block_key(overlay[0]) if overlay else None} != {_block_key(blocks[0])}",
    )
    check(
        "…while carrying the snapped box it will actually be drawn into",
        overlay and overlay[0]["bbox"] != overlay[0]["source_bbox"],
        "the box was not snapped at all",
    )


def _flat_drawing(size: int = 400) -> Image.Image:
    """A hard-edged illustration: solid fills, no shading. ImageDraw does not anti-alias,
    so every boundary is an exact step — which is what a screen print's edges are."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 200, 240), fill=(30, 30, 30))
    draw.ellipse((180, 120, 360, 320), fill=(200, 60, 40))
    draw.rectangle((60, 280, 340, 360), fill=(60, 140, 90))
    return img


def test_style_is_measured_not_asserted():
    """How a picture is DRAWN is measured, the way its palette is — "keep the style" alone
    comes back soft-shaded with glow, which is what "it feels generated" means."""
    print("\n=== Drawing Style ===")

    flat = _flat_drawing()
    check(
        "Solid fills with hard boundaries read as a flat illustration",
        _style_class(_style_metrics(_png(flat))) == "flat",
        str(_style_metrics(_png(flat))),
    )
    check(
        "…and the sentence says so in the original's own numbers",
        "FLAT, HARD-EDGED" in _style_summary(_png(flat))
        and "PHOTOGRAPH" not in _style_summary(_png(flat)),
        _style_summary(_png(flat))[:120],
    )

    # The same shapes under a vertical luminance ramp. Note the ramp moves only ~0.3 levels
    # per pixel: shading is detected as "neighbours differ at all", never by the SIZE of the
    # difference, because a gradient is smooth by definition. See STYLE_SOFT_BAND.
    shaded = flat.copy()
    pixels = shaded.load()
    for y in range(shaded.height):
        k = 0.55 + 0.45 * (y / shaded.height)
        for x in range(shaded.width):
            r, g, b = pixels[x, y]
            pixels[x, y] = (int(r * k), int(g * k), int(b * k))
    check(
        "The same drawing under a wash reads as shaded, not flat",
        _style_class(_style_metrics(_png(shaded))) == "shaded",
        str(_style_metrics(_png(shaded))),
    )

    # A smooth 2-D field. Do NOT build this as `(x * 4) % 256` — that is a sawtooth wrapping
    # every 64px, whose wrap edges measure as hard steps, and it classifies as FLAT.
    photo = Image.new("RGB", (400, 400))
    photo.putdata(
        [
            ((x * 255) // 400, (y * 255) // 400, ((x + y) * 255) // 800)
            for y in range(400)
            for x in range(400)
        ]
    )
    check(
        "Continuous tone reads as a photograph",
        _style_class(_style_metrics(_png(photo))) == "photograph",
        str(_style_metrics(_png(photo))),
    )

    check(
        "A blank field is flat, not shaded — nothing drawn means nothing shaded",
        _style_class(_style_metrics(_png(Image.new("RGB", (200, 200), (255, 255, 255))))) == "flat",
    )
    check(
        "Bytes that are not an image report nothing rather than guessing",
        _style_summary(b"not a png") == "" and _style_metrics(b"not a png") is None,
    )
    check(
        "The flat and photograph bands cannot silently overlap",
        STYLE_FLAT_MAX_COLORS < STYLE_PHOTO_MIN_COLORS
        and STYLE_FLAT_MAX_SOFT < STYLE_PHOTO_MIN_SOFT,
        f"{STYLE_FLAT_MAX_COLORS}/{STYLE_PHOTO_MIN_COLORS}, "
        f"{STYLE_FLAT_MAX_SOFT}/{STYLE_PHOTO_MIN_SOFT}",
    )


def _page_with_a_captioned_placard() -> fitz.Document:
    """One page: a figure's placard (a blank grey rectangle) with a live text line on it,
    a second line 2pt below it, another line 30pt away, and a line outside the picture."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(fitz.Rect(100, 60, 300, 160), color=(0.8, 0.8, 0.8), fill=(0.8, 0.8, 0.8))
    page.insert_text((110, 90), "first caption line", fontsize=9)
    page.insert_text((110, 101), "second caption line", fontsize=9)
    page.insert_text((110, 150), "a separate line further down", fontsize=9)
    page.insert_text((20, 380), "page text nowhere near the picture", fontsize=9)
    return doc


def test_live_text_pins_a_surface():
    """The page's own printed text cannot move, so whatever it sits on is reserved."""
    print("\n=== Locked Surfaces ===")
    doc = _page_with_a_captioned_placard()
    page = doc[0]
    rect = fitz.Rect(80, 40, 320, 200)  # the picture, as an illustration region would be

    locked = _locked_rects(rect, _text_rects_in(page, rect))
    check("A caption over a picture reserves the surface it is printed on", bool(locked), "no locks")
    check(
        "…and only what is inside the picture — the page's other text is not its problem",
        all(0.0 <= v <= 1.0 for box in locked for v in box) and len(locked) <= 3,
        str(locked),
    )
    check(
        "Two lines 2pt apart are one caption, not two rectangles",
        len(locked) == 2,
        f"{len(locked)} rect(s): {locked}",
    )
    top = min(locked, key=lambda b: b[1])
    check(
        "…and that one rectangle spans both of its lines",
        (top[3] - top[1]) * rect.height > 12,
        f"height {(top[3] - top[1]) * rect.height:.1f}pt",
    )
    check(
        "A line 30pt away stays its own rectangle",
        LOCK_MERGE_PAD_PT < 30,
        f"merge pad is {LOCK_MERGE_PAD_PT}",
    )
    check(
        "A picture with no text over it reserves nothing",
        _locked_rects(fitz.Rect(20, 250, 200, 340), _text_rects_in(page, fitz.Rect(20, 250, 200, 340)))
        == [],
    )
    check("Nothing locked is nothing covered", _locked_share([]) == 0.0)
    check(
        "Overlapping locks are counted once, not twice",
        abs(_locked_share([(0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)]) - 0.25) < 1e-6,
        str(_locked_share([(0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)])),
    )
    doc.close()


def test_a_raster_locks_every_page_it_is_printed_on():
    """One XObject, rewritten once, placed on many pages — so its locks are the union."""
    print("\n=== Locks Across Placements ===")
    doc = fitz.open()
    art = _png(Image.new("RGB", (200, 200), (240, 240, 240)))
    for caption_y in (60, 300):
        page = doc.new_page(width=400, height=400)
        page.insert_image(fitz.Rect(50, 50, 350, 350), stream=art)
        page.insert_text((80, caption_y), "a caption printed over the picture", fontsize=9)

    xref = doc[0].get_images(full=True)[0][0]
    one_page, _ = _locked_rects_for_xref(doc, xref, [1])
    both, mappable = _locked_rects_for_xref(doc, xref, [1, 2])
    check("A caption on page 1 reserves its surface", len(one_page) == 1, str(one_page))
    check(
        "…and a different caption on page 2 reserves a second one, on the same xref",
        len(both) == 2,
        f"{len(both)}: {both}",
    )
    check("Upright placements are mappable", mappable)
    check(
        "Skipping a page drops its locks",
        _locked_rects_for_xref(doc, xref, [1, 2], skip_page=2)[0] == one_page,
    )
    doc.close()


def test_a_moved_placard_is_caught_in_pixels():
    """The prompt asks for the reserved rectangle back; only a measurement guarantees it."""
    print("\n=== Locks Are Verified, Not Trusted ===")
    lock = [(0.25, 0.20, 0.75, 0.45)]

    def scene(placard_at, field=(250, 250, 250), placard=(215, 215, 215)):
        img = Image.new("RGB", (200, 200), field)
        ImageDraw.Draw(img).rectangle(placard_at, fill=placard)
        return _png(img)

    source = scene((50, 40, 150, 90))
    kept = scene((50, 40, 150, 90))
    check("An untouched reserved rectangle passes", _locks_kept(kept, source, lock) == [])

    moved = scene((50, 80, 150, 130))  # the placard slid 40px down, out from under the text
    check(
        "A placard that moved out from under its caption is caught",
        _locks_kept(moved, source, lock) == [0],
        str(_locks_kept(moved, source, lock)),
    )

    drawn_on = scene((50, 40, 150, 90))
    img = Image.open(io.BytesIO(drawn_on)).convert("RGB")
    ImageDraw.Draw(img).rectangle((60, 45, 140, 85), fill=(20, 20, 20))
    check(
        "…and so is one the model drew an arm across",
        _locks_kept(_png(img), source, lock) == [0],
    )

    repaired, which = _restamp_locks(moved, source, lock, [0])
    check("A lost rectangle is repaired from the source's own blank pixels", which == [0])
    check(
        "…and the repaired picture passes the same check",
        _locks_kept(repaired, source, lock) == [],
        str(_locks_kept(repaired, source, lock)),
    )

    # The model put a genuinely different field there. Pasting the old colour into it would
    # be a visible patch, so the repair declines and the regeneration is discarded instead.
    recoloured = scene((50, 80, 150, 130), field=(40, 90, 160), placard=(30, 70, 130))
    _, none_repaired = _restamp_locks(recoloured, source, lock, [0])
    check("A repair that would show is refused", none_repaired == [], str(none_repaired))
    check(
        "…leaving the lock failed, which is what discards the regeneration",
        _locks_kept(recoloured, source, lock) == [0],
    )


def test_a_pinned_picture_does_not_get_a_free_redraw():
    """Locks buy freedom for the rest of the frame — until they cover too much of it."""
    print("\n=== Pinned Is Not Locked ===")
    import image_localizer as loc
    plenty = " ".join(["word"] * (CONTEXT_MIN_WORDS + 4))
    free = dict(has_baked_text=False, information_role="decorative", contains_logo=False)

    got = _regeneration_mode(plenty, 900, 900, **free, locked_share=0.10)
    check("A small reserved area still allows a free redraw", got == "reimagine", got)
    got = _regeneration_mode(plenty, 900, 900, **free, locked_share=0.40)
    check("…a large one does not — that picture is pinned, not locked", got == "context", got)
    check(
        "The threshold sits between the two",
        0.10 <= LOCK_MAX_AREA_SHARE < 0.40,
        f"{LOCK_MAX_AREA_SHARE}",
    )
    got = _regeneration_mode(plenty, 900, 900, **free, lock_unmappable=True)
    check("A rotated placement cannot be locked, so it is not freed", got == "context", got)

    # Which rectangles are worth reserving. Both halves of this were wrong on the first run and
    # the failure was visible on the page: a white card painted into a man's jumper, while the
    # placard it was supposed to protect went unreserved.
    # The real picture: a man in a jumper holding a white placard, on a white field.
    art = Image.new("RGB", (400, 400), (255, 255, 255))
    drawing = ImageDraw.Draw(art)
    # Big enough that the `face` box AND the band sampled around it both land inside it — a
    # smaller one lets the ring catch the paper outside and the box reads as "distinct".
    drawing.ellipse((140, 20, 300, 180), fill=(200, 150, 110))            # a face
    drawing.rectangle((40, 180, 360, 380), fill=(120, 120, 120))          # a jumper
    drawing.rectangle((80, 200, 320, 300), fill=(252, 252, 252), outline=(20, 20, 20), width=4)

    # The caption's own box, padded out to the surface — which is how a real one arrives, and
    # means it clips the placard's dark outline and a little jumper. That is exactly what the
    # first version of this filter rejected.
    placard = (0.19, 0.49, 0.81, 0.76)
    jumper = (0.62, 0.85, 0.80, 0.95)    # a page footer overlapping the artwork
    face = (0.45, 0.15, 0.62, 0.32)      # text laid straight over the subject
    record: dict = {}
    kept = _flat_locks(_png(art), [placard, jumper, face], record)

    check(
        "A caption's box is reserved even though it clips the placard's own outline",
        placard in kept,
        f"kept {kept}",
    )
    check(
        "A patch of jumper is NOT reserved — same colour inside and out, so nothing can fall "
        "off, and reserving it makes the model paint a card into the artwork",
        jumper not in kept,
        f"kept {kept}",
    )
    check("Text laid over a face is not a surface either", face not in kept, f"kept {kept}")
    check("…and the audit counts what was dropped", record.get("locks_unenforceable") == 2, str(record))

    check(
        "The source test is looser than the result test — a border in the box is not artwork",
        LOCK_SOURCE_MAX_INK > LOCK_MAX_INK,
        f"{LOCK_SOURCE_MAX_INK} vs {LOCK_MAX_INK}",
    )
    check(
        "A reserved area is never described as something blank to draw",
        "blank" not in (
            loc.LOCK_CLAUSE_HEAD + loc.LOCK_ITEM + loc.LOCK_CLAUSE_COMPACT + loc.LOCK_ITEM_COMPACT
        ).lower(),
        "the word 'blank' is back in the lock clause — it makes the model draw a card",
    )
    check(
        "…and it says outright not to add one",
        "do not add a card" in loc.LOCK_CLAUSE_HEAD.lower(),
    )


def test_every_prompt_still_formats():
    """A stray {slot} raises KeyError inside a worker and surfaces as a generic edit_failed,
    which reads as a model problem for a long time. So it is checked here instead."""
    print("\n=== Prompts Format ===")
    import image_localizer as loc

    seen = []
    original = loc._run_edit_models
    loc._run_edit_models = lambda instruction, *a, **k: seen.append(instruction)
    try:
        for mode in loc.LOCALIZE_MODES:
            for style, locks in (("measured style.", "RESERVED: one rect."), ("", "")):
                loc.localize_image(
                    b"", "image/png", ["people_attire"], " ".join(["word"] * 12),
                    "#ffffff (90%)", mode=mode, style=style, locks=locks,
                )
        for style in ("measured style.", ""):
            loc.localize_cover(b"", "image/png", "some page words here", "#fff", style)
    finally:
        loc._run_edit_models = original

    check(
        f"All {len(seen)} prompt variants format with no slot left behind",
        all("{" not in text for text in seen),
        next((t[t.index("{"):][:60] for t in seen if "{" in t), ""),
    )
    check("Every mode produced a prompt", len(seen) == len(loc.LOCALIZE_MODES) * 2 + 2)


def test_the_prompts_name_the_props_that_must_go():
    """The gloves survived because three rules named skin, hair and clothing and none named
    a glove. Naming the props is the whole mechanism — see BANGLADESHI_VOCABULARY."""
    print("\n=== Prompts Name The Props ===")
    import image_localizer as loc

    vocab = loc.BANGLADESHI_VOCABULARY.lower()
    for prop in ("glove", "hi-vis", "helmet", "boots", "knife and fork"):
        check(f"The vocabulary names {prop!r} as something that does not survive", prop in vocab)
    check(
        "…and it reaches the prompts that carry it",
        all(
            "{vocabulary}" in t
            for t in (loc.EDIT_INSTRUCTION, loc.REIMAGINE_INSTRUCTION, loc.COVER_INSTRUCTION)
        ),
    )
    check(
        "The compact prompt names them inline instead, since it carries no vocabulary block",
        "glove" in loc.SIMPLE_EDIT_INSTRUCTION.lower()
        and "{vocabulary}" not in loc.SIMPLE_EDIT_INSTRUCTION,
    )
    check(
        "Only the redraw licenses a new pose",
        "POSED DIFFERENTLY" in loc.REIMAGINE_INSTRUCTION
        and "POSED DIFFERENTLY" not in loc.EDIT_INSTRUCTION,
    )
    check(
        "The redraw holds a plain field rather than filling it with a scene",
        "A PLAIN BACKGROUND IS NOT A SCENE TO FILL IN" in loc.REIMAGINE_INSTRUCTION,
    )


def test_a_redrawn_figure_leaves_no_ghost_of_the_old_one():
    """A cut-out picture carries a soft mask shaped like the figure that was in it.

    Inheriting it across a redraw does two visible things: it clips the new figure to the old
    one's outline, and wherever the old silhouette is opaque but the new picture has only its
    own background there, the page shows a pale patch shaped like the figure that used to be
    there. On a white page that is invisible; over a coloured panel it is the ghost the user
    reported. So the mask is rebuilt from the new pixels.
    """
    print("\n=== No Ghost Of The Old Figure ===")

    def cut_out(draw_figure) -> bytes:
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw_figure(ImageDraw.Draw(img))
        return _png(img)

    # The old picture: a tall figure on the left. The new one: a wider figure on the right,
    # which is what a free redraw does.
    old = cut_out(lambda d: d.ellipse((20, 20, 90, 180), fill=(230, 60, 50, 255)))
    new_rgb = Image.new("RGB", (200, 200), (255, 255, 255))
    ImageDraw.Draw(new_rgb).ellipse((110, 40, 190, 160), fill=(60, 120, 200))
    new = _png(new_rgb)

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(page.rect, color=(0.6, 0.4, 0.7), fill=(0.6, 0.4, 0.7))  # a coloured panel
    page.insert_image(page.rect, stream=old)
    doc = fitz.open(stream=doc.tobytes(), filetype="pdf")
    page = doc[0]
    xref = page.get_images(full=True)[0][0]
    check("The fixture really is a cut-out with a soft mask", page.get_images(full=True)[0][1] != 0)

    alpha = _derive_smask(new)
    check("A picture on a flat field can describe its own silhouette", alpha is not None)

    # A picture that bleeds to its edges has no background to measure against and no cut-out
    # to preserve, so the derivation declines and the inherited mask stands.
    bleed = Image.new("RGB", (120, 120))
    bleed.putdata(
        [((x * 255) // 120, (y * 255) // 120, 128) for y in range(120) for x in range(120)]
    )
    check("A full-bleed picture declines to describe a silhouette", _derive_smask(_png(bleed)) is None)

    _swap_image_in_place(doc, xref, new)
    rendered = Image.open(io.BytesIO(page.get_pixmap(dpi=72).tobytes("png"))).convert("RGB")

    def panel_at(x, y):
        """True if the page's own purple shows here — i.e. the picture is transparent."""
        return max(abs(a - b) for a, b in zip(rendered.getpixel((x, y)), (153, 102, 178))) < 40

    check(
        "The new figure is not clipped to where the old one stood",
        not panel_at(150, 100),
        f"new figure's centre renders as {rendered.getpixel((150, 100))}",
    )
    check(
        "…and no pale patch is left in the old figure's shape",
        panel_at(55, 100),
        f"the old figure's centre renders as {rendered.getpixel((55, 100))}",
    )
    check("The panel around both is untouched", panel_at(5, 5) and panel_at(195, 195))
    doc.close()

    # An enclosed area the same colour as the background is part of the figure, not a hole:
    # a white shirt on a white field must not let the panel show through.
    shirt = Image.new("RGB", (120, 120), (255, 255, 255))
    drawing = ImageDraw.Draw(shirt)
    drawing.ellipse((20, 20, 100, 100), fill=(20, 20, 20))
    drawing.ellipse((40, 40, 80, 80), fill=(255, 255, 255))  # the shirt, background-coloured
    mask = _derive_smask(_png(shirt))
    check("A white shirt on a white field is derived opaque", mask is not None)
    if mask:
        got = Image.frombytes("L", (120, 120), mask)
        check("…so the panel behind it cannot show through", got.getpixel((60, 60)) == 255,
              f"alpha at the shirt is {got.getpixel((60, 60))}")
        check("…while the field around the figure stays transparent",
              got.getpixel((5, 5)) == 0, f"alpha at the corner is {got.getpixel((5, 5))}")

    check(
        "A picture with no background to measure keeps whatever mask it had",
        _derive_smask(b"not a png") is None,
    )


def _synthetic_cover() -> fitz.Document:
    """A one-page document that looks like a front cover: a full-bleed panel, a picture,
    and a two-line title."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(page.rect, color=(0.68, 0.69, 0.92), fill=(0.68, 0.69, 0.92))
    photo = Image.new("RGB", (400, 400), (200, 120, 80))
    buf = io.BytesIO()
    photo.save(buf, format="PNG")
    page.insert_image(fitz.Rect(100, 300, 500, 700), stream=buf.getvalue())
    page.insert_text((80, 150), "Heart Failure Manual", fontsize=28)
    page.insert_text((80, 190), "A guide for patients", fontsize=14)
    return doc


def test_regeneration_mode():
    """Context mode needs both a picture big enough to take direction and words worth giving."""
    print("\n=== Regeneration Mode ===")
    plenty = " ".join(["word"] * (CONTEXT_MIN_WORDS + 4))
    big = (900, 900)
    small = (100, 100)  # 10k pixels, well under SIMPLE_MODE_MAX_PIXELS

    check(
        "A large picture with real context around it is redrawn in context mode",
        _regeneration_mode(plenty, *big) == "context",
        _regeneration_mode(plenty, *big),
    )
    check(
        "A picture with almost no text around it falls back to simple",
        _regeneration_mode("two words", *big) == "simple",
        _regeneration_mode("two words", *big),
    )
    check(
        "…as does an icon, however much context the page offers",
        _regeneration_mode(plenty, *small) == "simple",
        _regeneration_mode(plenty, *small),
    )
    check(
        "The small-mode threshold really is about size, not text",
        small[0] * small[1] <= SIMPLE_MODE_MAX_PIXELS < big[0] * big[1],
        f"{SIMPLE_MODE_MAX_PIXELS} does not sit between the two",
    )

    # "reimagine" frees the frame, so everything that has to line up with the original
    # afterwards has to be absent before it is offered.
    free = dict(has_baked_text=False, information_role="decorative", contains_logo=False)
    check(
        "A textless decorative picture is redrawn as a Bangladeshi scene, not retouched",
        _regeneration_mode(plenty, *big, **free) == "reimagine",
        _regeneration_mode(plenty, *big, **free),
    )
    for name, override in (
        ("its words are painted back at the original's coordinates", {"has_baked_text": True}),
        ("the picture IS a datum the page states", {"information_role": "referential"}),
        ("a mark inside it is protected by position", {"contains_logo": True}),
    ):
        got = _regeneration_mode(plenty, *big, **{**free, **override})
        check(f"…but not when {name}", got == "context", got)
    check(
        "A textless icon is still too small to recompose",
        _regeneration_mode(plenty, *small, **free) == "simple",
        _regeneration_mode(plenty, *small, **free),
    )
    check(
        "…and so is a textless picture with no brief to redraw from",
        _regeneration_mode("two words", *big, **free) == "simple",
        _regeneration_mode("two words", *big, **free),
    )
    check(
        "Defaults are the cautious reading — a caller that names nothing gets the in-place edit",
        _regeneration_mode(plenty, *big) == "context",
        _regeneration_mode(plenty, *big),
    )


def test_cover_detection():
    """A cover is a picture with a title on it; a body page and a plain title page are not."""
    print("\n=== Cover Detection ===")
    cover = _synthetic_cover()
    check("A full-bleed illustrated first page is detected as the cover",
          _detect_cover_page(cover) == 1, str(_detect_cover_page(cover)))
    cover.close()

    # A typographic title page: sparse text, but nothing printed on it.
    plain = fitz.open()
    page = plain.new_page(width=595, height=842)
    page.insert_text((80, 400), "Heart Failure Manual", fontsize=28)
    check(
        "A plain typographic title page is left alone",
        _detect_cover_page(plain) is None,
        f"ink coverage {_page_ink_coverage(page):.2f}",
    )
    plain.close()

    if not os.path.exists(SOURCE_PDF):
        skipped.append("cover detection on the real manual")
        print(f"  SKIP  {SOURCE_PDF} not found")
        return
    doc = fitz.open(SOURCE_PDF)
    check(
        "The real manual's cover page is detected",
        _detect_cover_page(doc) == 1,
        str(_detect_cover_page(doc)),
    )
    doc.close()


def test_apply_cover():
    """Replacing a cover clears what was drawn on it and keeps what was typed on it."""
    print("\n=== Cover Replacement ===")
    doc = _synthetic_cover()
    page = doc[0]
    before = page.get_text().strip()

    new_cover = Image.new("RGB", (595, 842), (120, 200, 160))
    ImageDraw.Draw(new_cover).ellipse([150, 300, 450, 600], fill=(20, 20, 140))
    buf = io.BytesIO()
    new_cover.save(buf, format="PNG")

    check("The cover was applied", _apply_cover(page, buf.getvalue()), "apply returned False")

    page = doc[0]
    check("The title text survives", page.get_text().strip() == before, page.get_text()[:60])
    check("The original page graphics are gone", not page.get_drawings(),
          f"{len(page.get_drawings())} drawing(s) left")
    corner = page.get_pixmap(clip=fitz.Rect(20, 760, 50, 790), dpi=36).pixel(0, 0)
    check("…and the new cover is what shows through", corner == (120, 200, 160), str(corner))
    doc.close()


def test_background_match_leaves_a_picture_alone():
    """A generated picture that is mostly one colour must not be flattened to it.

    `_match_background` samples the outermost ring and repaints everything close to it. When
    the picture *is* that colour — which a full-page cover render can be — the correction
    stops being a correction and paints out the whole image.
    """
    print("\n=== Background Match Guard ===")
    source = Image.new("RGB", (200, 200), (255, 255, 255))
    src_buf = io.BytesIO()
    source.save(src_buf, format="PNG")

    # Almost entirely one colour: only a small square differs.
    flat = Image.new("RGB", (200, 200), (240, 90, 60))
    ImageDraw.Draw(flat).rectangle([90, 90, 110, 110], fill=(20, 20, 140))
    flat_buf = io.BytesIO()
    flat.save(flat_buf, format="PNG")

    out = _match_background(flat_buf.getvalue(), src_buf.getvalue())
    with Image.open(io.BytesIO(out)) as result:
        corner = result.convert("RGB").getpixel((2, 2))
    check(
        "A picture that is its own background is returned untouched",
        corner == (240, 90, 60),
        f"corner became {corner}",
    )

    # The ordinary case still works: a subject on a plain field, field colour corrected.
    normal = Image.new("RGB", (200, 200), (250, 248, 244))  # near-white, drifted
    ImageDraw.Draw(normal).ellipse([50, 50, 150, 150], fill=(20, 20, 140))
    norm_buf = io.BytesIO()
    normal.save(norm_buf, format="PNG")
    out = _match_background(norm_buf.getvalue(), src_buf.getvalue())
    with Image.open(io.BytesIO(out)) as result:
        rgb = result.convert("RGB")
        corner, centre = rgb.getpixel((2, 2)), rgb.getpixel((100, 100))
    check("A drifted background is still corrected to the source's", corner == (255, 255, 255),
          str(corner))
    check("…without touching the subject", centre == (20, 20, 140), str(centre))


def test_background_is_measured_not_guessed():
    """A background is found even when the border is not perfectly flat, so it can be forced."""
    print("\n=== Background Measurement ===")
    # A white field with JPEG-style speckle and one dark corner pixel: `_border_color`'s
    # strict flat test fails on this, which used to mean no background was enforced at all.
    speckled = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(speckled)
    for x in range(0, 200, 17):
        draw.point((x, 0), fill=(250, 250, 252))
    draw.point((0, 0), fill=(10, 10, 10))
    draw.ellipse([60, 60, 140, 140], fill=(200, 120, 80))

    check("A near-flat border still yields a background colour",
          _background_color(speckled) is not None, "None")
    check("…and it is the field's own colour",
          _background_color(speckled) == (255, 255, 255), str(_background_color(speckled)))

    # The real defect: a picture rendered from a placement rect does not land on a whole
    # pixel, so its outermost row carries a sliver of the page behind it. Reading that row
    # reported this yellow-fielded illustration's background as white, and the correction
    # then held the regenerated picture *to* white — the exact opposite of its job.
    field = (234, 211, 131)
    bled = Image.new("RGB", (300, 300), field)
    ImageDraw.Draw(bled).ellipse([90, 90, 210, 210], fill=(40, 60, 160))  # the subject
    edge = ImageDraw.Draw(bled)
    edge.rectangle([0, 0, 299, 0], fill=(255, 255, 255))    # page white bleeding in
    edge.rectangle([0, 299, 299, 299], fill=(255, 255, 255))
    edge.rectangle([0, 0, 0, 299], fill=(255, 255, 255))
    edge.rectangle([299, 0, 299, 299], fill=(255, 255, 255))
    check(
        "A one-pixel bleed at the edge does not get mistaken for the background",
        _background_color(bled) == field,
        f"read {_background_color(bled)}, wanted {field}",
    )

    # A picture that bleeds to its edges has no background and must not get one invented.
    bleeding = Image.new("RGB", (200, 200))
    for y in range(200):
        for x in range(0, 200, 4):
            bleeding.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
            for dx in range(1, 4):
                if x + dx < 200:
                    bleeding.putpixel((x + dx, y), ((x * 3) % 256, (y * 5) % 256, 90))
    check("A picture that bleeds to its edges reports no background",
          _background_color(bleeding) is None, str(_background_color(bleeding)))

    # And the forcing itself: a drifted background is pulled to the source's exact hex.
    source = Image.new("RGB", (200, 200), (255, 255, 255))
    ImageDraw.Draw(source).ellipse([60, 60, 140, 140], fill=(200, 120, 80))
    src_buf = io.BytesIO(); source.save(src_buf, format="PNG")
    drifted = Image.new("RGB", (200, 200), (252, 249, 243))  # model returned a warmer white
    ImageDraw.Draw(drifted).ellipse([60, 60, 140, 140], fill=(40, 160, 90))
    dr_buf = io.BytesIO(); drifted.save(dr_buf, format="PNG")

    with Image.open(io.BytesIO(_match_background(dr_buf.getvalue(), src_buf.getvalue()))) as res:
        rgb = res.convert("RGB")
        corner, centre = rgb.getpixel((3, 3)), rgb.getpixel((100, 100))
    check("A drifted background is forced back to the source's exact hex",
          corner == (255, 255, 255), str(corner))
    check("…without recolouring the subject", centre == (40, 160, 90), str(centre))


def test_logos_are_never_changed():
    """A logo's pixels come back from the original, and nothing downstream touches its words."""
    print("\n=== Logo Preservation ===")
    # A picture the model "regenerated" into something entirely different, and the original
    # it was made from. The logo lives in the top-left eighth.
    source = Image.new("RGB", (400, 200), (255, 255, 255))
    ImageDraw.Draw(source).rectangle([10, 10, 110, 60], fill=(0, 90, 160))  # the mark
    ImageDraw.Draw(source).ellipse([200, 40, 380, 180], fill=(200, 120, 80))  # the subject
    src_buf = io.BytesIO()
    source.save(src_buf, format="PNG")

    edited = Image.new("RGB", (400, 200), (255, 255, 255))
    ImageDraw.Draw(edited).rectangle([10, 10, 110, 60], fill=(20, 200, 40))  # mark mangled
    ImageDraw.Draw(edited).ellipse([200, 40, 380, 180], fill=(40, 160, 90))  # subject redrawn
    ed_buf = io.BytesIO()
    edited.save(ed_buf, format="PNG")

    logos = [{"label": "Test Trust", "bbox": [0.025, 0.05, 0.275, 0.30]}]
    out, stamped = _restamp_logos(ed_buf.getvalue(), src_buf.getvalue(), logos)
    with Image.open(io.BytesIO(out)) as result:
        rgb = result.convert("RGB")
        mark, subject = rgb.getpixel((60, 35)), rgb.getpixel((290, 110))

    check("The logo was restamped", stamped == 1, f"{stamped} stamped")
    check("…in the original's colours, not the model's", mark == (0, 90, 160), str(mark))
    check("…while the localized subject is left as regenerated", subject == (40, 160, 90),
          str(subject))

    # A box covering the whole picture is the classifier's is_logo case; restamping it would
    # paste the entire original back and silently undo the localization.
    whole = [{"label": "everything", "bbox": [0.0, 0.0, 1.0, 1.0]}]
    out, stamped = _restamp_logos(ed_buf.getvalue(), src_buf.getvalue(), whole)
    check("A whole-picture 'logo' box is refused", stamped == 0, f"{stamped} stamped")
    check("…and the image comes back untouched", out == ed_buf.getvalue(), "image was rewritten")


def test_logo_text_is_never_translated():
    """Blocks belonging to a logo are dropped before any erase, overlay or residual pass."""
    print("\n=== Logo Text Exclusion ===")
    logos = [{"label": "Test Trust", "bbox": [0.0, 0.0, 0.30, 0.30]}]
    blocks = [
        {"text": "Test Trust", "lang": "en", "bbox": [0.02, 0.05, 0.28, 0.25]},   # the mark
        {"text": "Caring for you", "lang": "en", "bbox": [0.02, 0.26, 0.28, 0.34]},  # strapline
        {"text": "Being active", "lang": "en", "bbox": [0.40, 0.10, 0.90, 0.20]},  # real caption
    ]
    kept = _outside_logos(blocks, logos)
    texts = [b["text"] for b in kept]

    check("The organisation's name is not up for translation", "Test Trust" not in texts,
          str(texts))
    check("…nor is the strapline locked up with it", "Caring for you" not in texts, str(texts))
    check("The page's own caption still is", "Being active" in texts, str(texts))
    check("No logos found means nothing is dropped",
          _outside_logos(blocks, []) == blocks, "blocks were filtered without any logo")
    check("A block with no usable box is kept rather than guessed at",
          len(_outside_logos([{"text": "x", "bbox": []}], logos)) == 1, "block was dropped")

    # The box test alone is not enough: detect_logo_regions and _extract_text_blocks are two
    # independent model calls, and on the eatwell plate they disagreed completely about where
    # the Food Standards Agency mark was — zero overlap — so its wordmark sailed through and
    # the pipeline was about to draw a plate and English lettering over the crest.
    apart = [{"label": "Food Standards Agency", "bbox": [0.02, 0.02, 0.10, 0.08]}]
    elsewhere = [
        {"text": "FOOD STANDARDS AGENCY", "bbox": [0.708, 0.083, 0.80, 0.13]},
        {"text": "Fruit and vegetables", "bbox": [0.10, 0.30, 0.40, 0.35]},
    ]
    survivors = [b["text"] for b in _outside_logos(elsewhere, apart)]
    check("A mark's name is protected even where the boxes do not overlap",
          "FOOD STANDARDS AGENCY" not in survivors, f"kept {survivors}")
    check("…while an ordinary caption is still translated",
          "Fruit and vegetables" in survivors, f"kept {survivors}")
    check("…and a one-letter block is not matched against a label",
          any(b["text"] == "a" for b in
              _outside_logos([{"text": "a", "bbox": [0.5, 0.5, 0.51, 0.51]}], apart)))


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _photograph(w: int = 240, h: int = 200) -> Image.Image:
    """An image with no flat field anywhere in it."""
    rng = random.Random(7)
    img = Image.new("RGB", (w, h))
    img.putdata([
        (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))
        for _ in range(w * h)
    ])
    return img


def test_text_lands_on_the_moved_surface():
    """Tier 1: after a regeneration the words follow their surface, not their old address.

    The model is asked to keep every blank surface to the pixel and mostly does, but
    "mostly" is a placard a few percent out of place — and the Bangla, written at the
    coordinates measured before the edit, then hangs off its bottom edge and over a hand.
    """
    print("\n=== Anchoring To The Regenerated Surface ===")
    img = Image.new("RGB", (400, 300), (120, 40, 140))
    ImageDraw.Draw(img).rectangle([70, 65, 330, 145], fill=(250, 250, 245))
    placard = (70, 65, 331, 146)

    moved = _anchor_box(img, (60, 40, 340, 120))  # where it was before the redraw
    check("A box whose surface moved is found again", moved is not None)
    if moved:
        (dx0, dy0, dx1, dy1), color = moved
        check(
            "…and the destination sits inside the surface as redrawn",
            placard[0] <= dx0 and placard[1] <= dy0 and dx1 <= placard[2] and dy1 <= placard[3],
            f"destination {(dx0, dy0, dx1, dy1)} escapes the placard {placard}",
        )
        check("…in the surface's own colour", color == (250, 250, 245), f"got {color}")

    check(
        "A box that never moved is left where it is",
        _anchor_box(img, (80, 70, 320, 140))[0] == (80, 70, 320, 140),
    )

    # Anchoring must run end to end, not just as a helper.
    _, overlay = _prepare_text_only_image(
        _png(img), 400, 300,
        [{"text": "HELP YOURSELF", "lang": "en", "bbox": [0.15, 0.133, 0.85, 0.40]}],
        anchor=True,
    )
    check("The block survives the anchored path", len(overlay) == 1)
    if overlay:
        bx = overlay[0]["bbox"]
        check("…with no plate needed", overlay[0].get("scrim") is None)
        check(
            "…and a box that landed on the placard",
            bx[1] * 300 >= 65 and bx[3] * 300 <= 146,
            f"got y {bx[1] * 300:.0f}..{bx[3] * 300:.0f}, placard is 65..146",
        )


def test_text_on_a_photograph_gets_a_plate():
    """Tier 1: a label with nowhere clean to sit is given a surface, not dropped.

    This is the eatwell plate. Every one of its labels sits on photographic detail, so the
    erase path refused all of them, the overlay wrote nothing, and the picture shipped a
    hundred percent in English — recorded as though it had been considered and kept.
    """
    print("\n=== Label Plates On Photographs ===")
    photo = _photograph()
    block = {"text": "Fruit and vegetables", "lang": "en", "bbox": [0.10, 0.10, 0.55, 0.25]}

    _, overlay = _prepare_text_only_image(_png(photo), 240, 200, [dict(block)])
    check("A label on a photograph is still written", len(overlay) == 1,
          "the block was dropped, as it used to be")
    plate = overlay[0].get("scrim") if overlay else None
    check("…on a plate drawn for it", isinstance(plate, str) and plate.startswith("#"),
          f"got scrim={plate!r}")
    if plate:
        luminance = sum(
            c * w for c, w in zip(
                [int(plate[i:i + 2], 16) for i in (1, 3, 5)], (0.299, 0.587, 0.114)
            )
        )
        expected = "#000000" if luminance > 140 else "#ffffff"
        check("…in a text colour that contrasts the plate",
              overlay[0]["color"] == expected,
              f"{overlay[0]['color']} on a plate of luminance {luminance:.0f}")

    # And the plate has to actually reach the page.
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    rect = fitz.Rect(20, 20, 260, 220)
    before = len(page.get_drawings())
    _overlay_text_blocks(page, rect, overlay, fitz.Archive(FONTS_DIR))
    drawn = page.get_drawings()
    check("The plate is drawn as page content", len(drawn) > before,
          "no rectangle reached the page")
    if len(drawn) > before:
        target = fitz.Rect(
            rect.x0 + overlay[0]["bbox"][0] * rect.width,
            rect.y0 + overlay[0]["bbox"][1] * rect.height,
            rect.x0 + overlay[0]["bbox"][2] * rect.width,
            rect.y0 + overlay[0]["bbox"][3] * rect.height,
        )
        covers = any((d["rect"] & target).get_area() > 0.8 * target.get_area() for d in drawn)
        check("…covering the words it sits under", covers,
              f"no drawing covers {target}")
    doc.close()


def test_a_badge_box_never_swallows_its_tile():
    """Tier 1: an erase box stops at the artwork instead of growing 100% into it.

    A "3 units" badge is a small tab of flat colour on top of a photograph. Growing while
    "any pixel differs from the tab" meant growing until the allowance ran out, and the tab
    colour was then painted over that whole rectangle — which is how a blue block ended up
    spilling out past the edge of its own tile.
    """
    print("\n=== Badge Boxes Stay On Their Badge ===")
    tile = _photograph(180, 177)
    ImageDraw.Draw(tile).rectangle([10, 8, 80, 40], fill=(20, 90, 170))   # the tab
    ImageDraw.Draw(tile).text((16, 16), "3 units", fill=(255, 255, 255))

    box = (14, 14, 66, 32)
    grown = _snap_to_ink(tile, box, (20, 90, 170))
    check("The box does not grow past the badge", grown[3] <= 44 and grown[2] <= 84,
          f"grew to {grown}, badge ends at (80, 40)")
    check("…and stays inside the tile", grown[0] >= 0 and grown[1] >= 0, f"got {grown}")

    _, overlay = _prepare_text_only_image(
        _png(tile), 180, 177, [{"text": "3 units", "lang": "en",
                                "bbox": [14 / 180, 14 / 177, 66 / 180, 32 / 177]}]
    )
    if overlay:
        bx = overlay[0]["bbox"]
        painted = (bx[2] - bx[0]) * 180 * (bx[3] - bx[1]) * 177
        check("…and the painted area stays within its allowance",
              painted <= SNAP_MAX_AREA_GROWTH * (66 - 14) * (32 - 14) * 1.3,
              f"painted {painted:.0f}px vs source {(66 - 14) * (32 - 14)}px")


def test_no_english_survives_under_the_bangla():
    """Tier 1: the destination is cleared, not just the block's own box.

    On the units grid the Bangla was drawn straight over a still-visible "(250ml, ABV 12%)"
    — a *different* OCR block, whose own erase had been refused. Checking each block's own
    box could never catch that; checking the box the words will be read from does.
    """
    print("\n=== No English Under The Bangla ===")
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 90), "Standard glass wine", fill=(0, 0, 0))
    draw.text((40, 115), "(250ml, ABV 12%)", fill=(0, 0, 0))  # the neighbour left standing

    cleaned, overlay = _prepare_text_only_image(
        _png(img), 400, 300,
        [{"text": "Standard glass wine", "lang": "en",
          "bbox": [0.09, 0.28, 0.55, 0.43]}],
    )
    check("The block is placed", len(overlay) == 1)
    if overlay:
        out = Image.open(io.BytesIO(cleaned)).convert("RGB")
        bx = overlay[0]["bbox"]
        dest = (int(bx[0] * 400), int(bx[1] * 300), int(bx[2] * 400), int(bx[3] * 300))
        share = _ink_share(out, dest, (255, 255, 255))
        check("…and nothing is left standing inside its destination",
              share <= MAX_RESIDUAL_INK, f"{share:.1%} of the destination is still ink")


def test_a_discarded_regeneration_is_never_reported_as_kept():
    """Tier 1: an untouched image says which failure made it untouched.

    Every "we are not redrawing this" path returned "classify_kept" when it had nothing to
    write, so a thrown-away regeneration was indistinguishable in the audit from a picture
    the classifier had deliberately left alone. That is how a page of English shipped with
    a record saying it had been considered.
    """
    print("\n=== Untouched Images Say Why ===")
    import image_processor

    original = image_processor.detect_logo_regions
    image_processor.detect_logo_regions = lambda *a, **k: []
    try:
        blank = _png(Image.new("RGB", (60, 60), (255, 255, 255)))
        _, status, record = _text_only_result(
            blank, 60, 60, [], {}, "xref:1", 1, empty_status="regeneration_discarded"
        )
        check("A discarded regeneration reports itself as one",
              status == "regeneration_discarded", f"got {status!r}")
        check("…and is flagged untouched", record.get("untouched") is True)

        _, plain, _ = _text_only_result(blank, 60, 60, [], {}, "xref:2", 1)
        check("A picture with nothing to do is still classify_kept",
              plain == "classify_kept", f"got {plain!r}")
    finally:
        image_processor.detect_logo_regions = original

    check("Every status the pipeline can return is in the summary vocabulary",
          {"aspect_kept", "regeneration_discarded", "classify_failed"}
          <= set(image_processor.STATUS_VOCABULARY))


def test_numbers_survive_translation():
    """Tier 1: a measure that lost its unit is a clinical error, and is detectable.

    "3 units" came back from the per-block translator as "তিন" — the number gone from a
    card whose entire purpose is to state that number.
    """
    print("\n=== Numbers Survive Translation ===")
    from image_localizer import _keeps_numbers, resolve_block_text

    check("A dropped number is caught", not _keeps_numbers("3 units", "তিন"))
    check("A kept number passes", _keeps_numbers("3 units", "3 ইউনিট"))
    check("Decimals and percentages count too",
          _keeps_numbers("(125ml, ABV 12%)", "(125ml, ABV 12%) ওয়াইন")
          and not _keeps_numbers("1.5 units", "১.৫ ইউনিট"))

    # A block already translated by the image-aware batch call must not be re-sent.
    calls = []
    import image_localizer

    original = image_localizer._translate_to_bangla
    image_localizer._translate_to_bangla = lambda t: calls.append(t) or t
    try:
        got = resolve_block_text({"text": "3 units", "lang": "en", "bn": "3 ইউনিট"})
        check("A pre-translated block is used as-is", got == "3 ইউনিট", f"got {got!r}")
        check("…without another API call", not calls, f"{len(calls)} call(s) made")
    finally:
        image_localizer._translate_to_bangla = original


def test_smask_is_preserved():
    """Tier 1: a cut-out figure stays cut out when its pixels are swapped.

    Dropping /SMask turns a floating figure into an opaque rectangle that prints over
    whatever panel it was sitting on. It matters more here than in image_regen: CMYK images
    are *rendered* from their placement, so the replacement already carries the page
    background, and without the mask that background is painted over the layout.
    """
    print("\n=== Soft Masks Survive The Swap ===")
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(fitz.Rect(20, 20, 180, 180), pixmap=fitz.Pixmap(
        fitz.csRGB, fitz.IRect(0, 0, 80, 80)
    ))
    xref = page.get_images(full=True)[0][0]
    doc.xref_set_key(xref, "SMask", "9 0 R")

    _swap_image_in_place(doc, xref, _png(Image.new("RGB", (80, 80), (10, 120, 200))))
    check("The soft mask is still there after the swap",
          doc.xref_get_key(xref, "SMask")[0] != "null",
          f"got {doc.xref_get_key(xref, 'SMask')}")
    check("…while the stencil keys are cleared",
          doc.xref_get_key(xref, "ImageMask")[0] == "null")
    doc.close()


def test_extreme_aspect_pictures_are_left_alone():
    """Tier 1: a page-edge sliver is not worth a model call, let alone a redraw.

    A 114x1381 strip came back with its one leaf shifted a few percent, and since the page
    crops all but its left quarter the leaf moved out of view and left a white gap down the
    edge. Checked before the classify call, so it costs no quota either.
    """
    print("\n=== Extreme Aspect Strips ===")
    import image_processor

    calls = []
    original = image_processor.classify_image
    image_processor.classify_image = lambda *a, **k: calls.append(1) or {}
    try:
        sliver = _png(Image.new("RGB", (114, 1381), (200, 210, 190)))
        _, status, record = image_processor._decide_from_png(sliver, 114, 1381, "xref:43", 1)
        check("A 12:1 sliver is kept as printed", status == "aspect_kept", f"got {status!r}")
        check("…with the reason recorded",
              record.get("regeneration_vetoed") == "extreme_aspect")
        check("…and no classification was paid for", not calls, f"{len(calls)} call(s)")
    finally:
        image_processor.classify_image = original


def test_overlay_skips_an_unreadably_small_box():
    """Tier 1: a box too small to hold Bangla gets nothing, not a smudge.

    With a plate now drawn under the words, a box that insert_htmlbox would shrink to a
    smear would leave a coloured smudge on the artwork rather than merely illegible type.
    """
    print("\n=== Unreadably Small Boxes ===")
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    rect = fitz.Rect(0, 0, 300, 300)
    before = len(page.get_drawings())
    _overlay_text_blocks(
        page, rect,
        [{"text": "ছোট", "lang": "bn", "scrim": "#ffffff",
          "bbox": [0.10, 0.10, 0.90, 0.10 + (MIN_OVERLAY_HEIGHT - 1) / 300]}],
        fitz.Archive(FONTS_DIR),
    )
    check("Nothing is drawn for a box shorter than the minimum",
          len(page.get_drawings()) == before,
          f"{len(page.get_drawings()) - before} drawing(s) added")
    doc.close()


def test_a_cover_that_moves_its_panels_is_refused():
    """Tier 1: the cover's text is preserved, so the colour behind it must be too.

    The cover's real text layer is not redrawn — it is kept and printed back over the
    regeneration at fixed positions. So the model is free to move the artwork out from under
    it, and on one run the teal panel came back covering only the top half of the page: the
    white caption that had been sitting on it was printed onto bare white paper and vanished.
    No prompt can guarantee this, so it is measured.
    """
    print("\n=== A Cover That Moves Its Panels Is Refused ===")
    from image_processor import _cover_keeps_text_backgrounds, COVER_DPI

    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(fitz.Rect(0, 0, 300, 400), color=None, fill=(0.05, 0.43, 0.43))
    for y in (60, 300, 320, 340):
        page.insert_text((30, y), "caption line", fontsize=11, color=(1, 1, 1))

    original = page.get_pixmap(dpi=COVER_DPI).tobytes("png")
    check("The original cover always passes",
          _cover_keeps_text_backgrounds(original, page))

    moved = Image.open(io.BytesIO(original)).convert("RGB")
    W, H = moved.size
    for y in range(int(H * 0.4), H):
        for x in range(W):
            moved.putpixel((x, y), (255, 255, 255))
    check("A cover whose panel slid out from under the text is refused",
          not _cover_keeps_text_backgrounds(_png(moved), page),
          "the regeneration was accepted and the text would be invisible")

    paler = Image.open(io.BytesIO(original)).convert("RGB")
    px = paler.load()
    for y in range(H):
        for x in range(W):
            r, g, b = px[x, y]
            px[x, y] = (min(255, r + 18), min(255, g + 18), min(255, b + 18))
    check("…while the same layout in a slightly different shade is kept",
          _cover_keeps_text_backgrounds(_png(paler), page))
    doc.close()


def test_a_picture_containing_a_mark_is_not_a_mark():
    """Tier 1: "is a logo" and "contains a logo" are two answers, not one.

    `is_logo` short-circuits everything — no OCR, no translation, no redraw — so a picture
    wrongly called a whole logo ships untouched and in English. The FSA eatwell plate did
    exactly that: a 2455x1672 photograph with an agency crest in one corner, and at
    temperature 0 the classifier called it a logo on some runs and not on others. A coin
    flip is not a policy, so the classifier now answers `logo_fills_image` as well and only
    a picture that IS the mark is kept whole.
    """
    print("\n=== A Picture Containing A Mark Is Not A Mark ===")
    import image_processor

    original = image_processor.classify_image
    seen = {}

    def decide(fills: bool):
        def stub(*a, **k):
            return {
                "is_logo": True, "logo_fills_image": fills, "needs_localization": False,
                "categories": [], "information_role": "decorative", "reason": "mark present",
            }
        return stub

    png = _png(Image.new("RGB", (600, 500), (250, 250, 245)))
    try:
        image_processor.classify_image = decide(True)
        _, status, record = image_processor._decide_from_png(png, 600, 500, "xref:9", 1)
        check("A picture that IS a mark is kept exactly as printed",
              status == "logo_kept", f"got {status!r}")
        check("…and recorded as untouched", record.get("untouched") is True)

        image_processor.classify_image = decide(False)
        image_processor._extract_text_blocks = lambda *a, **k: []
        _, status2, record2 = image_processor._decide_from_png(png, 600, 500, "xref:10", 1)
        check("A picture that merely CONTAINS a mark is not kept as one",
              status2 != "logo_kept", f"got {status2!r}")
        check("…and the distinction is in the audit",
              record2.get("logo_fills_image") is False)
    finally:
        image_processor.classify_image = original
        seen.clear()

    # A classifier that omits the field must be read the safe way round.
    from image_localizer import classify_image as _ci  # noqa: F401
    import image_localizer
    check("A missing answer defaults to 'the image IS the mark'",
          bool({"is_logo": True}.get("logo_fills_image", True)))


def test_nothing_the_overlay_draws_overlaps():
    """Tier 1: no two things the overlay draws may sit on top of each other.

    Two guards, and both have to measure the *plate*, not the words inside it: a plate is
    opaque and drawn over everything, so two plates a point apart overlap while both text
    boxes still "fit", and a plate laid across a caption the redaction deliberately kept
    would hide it outright.
    """
    print("\n=== Nothing The Overlay Draws Overlaps ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    rect = fitz.Rect(0, 0, 400, 400)
    # Two plated blocks whose text boxes clear each other by a hair but whose plates do not.
    blocks = [
        {"text": "একটি", "lang": "bn", "scrim": "#ffffff",
         "bbox": [0.10, 0.10, 0.90, 0.20]},
        {"text": "দুইটি", "lang": "bn", "scrim": "#ffffff",
         "bbox": [0.10, 0.2025, 0.90, 0.30]},
    ]
    _overlay_text_blocks(page, rect, blocks, fitz.Archive(FONTS_DIR))
    drawn = [d["rect"] for d in page.get_drawings()]
    worst = 0.0
    for i, a in enumerate(drawn):
        for b in drawn[i + 1:]:
            small = min(a.get_area(), b.get_area())
            if small > 0:
                worst = max(worst, (a & b).get_area() / small)
    check("Two plates never overlap each other", worst <= 0.25,
          f"worst overlap {worst:.0%} between {len(drawn)} plate(s)")
    doc.close()

    # A plate must not be laid over the page's own live text.
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((60, 120), "kept caption", fontsize=11)
    before = len(page.get_drawings())
    _overlay_text_blocks(
        page, fitz.Rect(0, 0, 400, 400),
        [{"text": "কিছু", "lang": "bn", "scrim": "#ffffff",
          "bbox": [0.10, 0.26, 0.80, 0.31]}],
        fitz.Archive(FONTS_DIR),
    )
    check("A plate is never drawn over the page's own text",
          len(page.get_drawings()) == before,
          f"{len(page.get_drawings()) - before} plate(s) drawn over live text")
    doc.close()


def test_divider_cartoons_are_found_in_a_translated_manual():
    """Tier 1: the divider illustration is detected, and tables still are not.

    Eligibility used to be judged on the whole page's text-line count, which is a proxy —
    and the proxy broke on exactly the documents this pipeline exists for. Bangla wraps into
    more lines than the English the limit of 12 was measured on, so a divider page went from
    12 lines to 16 and its cartoon stopped being detected. A 91-page run of this manual
    recorded ZERO vector regions: every divider illustration in the book shipped Western,
    and nothing in the audit said so, because an image that is never detected is never
    recorded.

    The cluster's own text is the honest signal. Measured here: the cartoon carries 3-6
    lines inside its bbox, the table and diagram clusters carry 38 to 286.
    """
    print("\n=== Divider Cartoons In A Translated Manual ===")
    if not os.path.exists(REVASC_PDF):
        print(f"  SKIP  {REVASC_PDF} not found")
        skipped.append("divider_cartoons_are_found")
        return

    doc = fitz.open(REVASC_PDF)
    found = {}
    for page_num in (13, 22, 34, 44, 53, 63):
        regions = regions_on(doc, page_num)
        if regions:
            found[page_num] = regions[0]
    check("The divider cartoon is detected on every page that carries it",
          len(found) == 6, f"found on {sorted(found)} of [13, 22, 34, 44, 53, 63]")
    check("…and dedups to a single signature, so the book gets one consistent redraw",
          len({r.content_key for r in found.values()}) == 1,
          f"{len({r.content_key for r in found.values()})} signatures")
    check("…and is redrawn, not just relabelled — its lettering is one t-shirt slogan",
          all(r.redraw_safe for r in found.values()),
          f"{sum(1 for r in found.values() if not r.redraw_safe)} refused a redraw")
    if 13 in found:
        rect = found[13].rect
        check("…framed on the artwork rather than the whole page",
              rect.get_area() / doc[12].rect.get_area() < 0.5,
              f"covers {rect.get_area() / doc[12].rect.get_area():.0%} of the page")

    # The other direction, which is what the old page-level limit was protecting: a cluster
    # with the page's words running through it is a table or a diagram, not a picture.
    for page_num in (75, 87, 89, 90):
        check(f"Page {page_num}'s table/diagram line art is still refused",
              not regions_on(doc, page_num),
              f"detected {len(regions_on(doc, page_num))} region(s)")
    doc.close()


def test_a_panel_of_drawn_words_is_translated_but_not_redrawn():
    """Tier 1: page 16's "Exercise can:" panel — words translated, artwork left alone.

    The panel shipped in English. Its ten bullets are drawn as vector outlines rather than
    set in a font, so the text pipeline had nothing to translate, and the region detector —
    which would have handed them to the image pipeline — refused it: seven letters of the
    "Exercise can:" heading measure ~7x7pt each, which is exactly what `_vector_marks` calls
    a checkbox, and a checkbox inside a cluster vetoes it. Only the two words the layout did
    set in a font, "type II diabetes" and "cancers", came out in Bangla.

    Both halves are asserted here, because passing only the first is worse than failing:
    with the veto gone the classifier calls the panel "decorative" (it is full of smiley
    faces) and the edit model would redraw ten lines of clinical advice as a picture.
    """
    print("\n=== A Panel Of Drawn Words ===")
    if not os.path.exists(REVASC_PDF):
        print(f"  SKIP  {REVASC_PDF} not found")
        skipped.append("panel_of_drawn_words")
        return

    doc = fitz.open(REVASC_PDF)
    regions = regions_on(doc, 16)
    check("The exercise panel is detected, so its drawn English is translated",
          len(regions) == 1, f"detected {len(regions)} region(s)")
    if regions:
        region = regions[0]
        check("…and never regenerated, because it is a panel of words",
              not region.redraw_safe)
        check("…framed on the panel and its heading, not the spread",
              region.rect.get_area() / doc[15].rect.get_area() < 0.15,
              f"covers {region.rect.get_area() / doc[15].rect.get_area():.0%} of the page")
        check("…and reaching the heading whose letters used to veto it",
              region.rect.y0 < 120, f"top edge at {region.rect.y0:.0f}")
    doc.close()


def test_a_letter_is_not_a_checkbox():
    """Tier 1: the size-and-shape checkbox test, applied to ink instead of a bounding box.

    `_vector_marks` reads width, height and aspect, which a 7pt letter drawn as outlines
    satisfies as well as a tick box does. What separates them is the ink: a box is one "re"
    or four straight lines, a letter is dozens of curve segments.
    """
    print("\n=== A Letter Is Not A Checkbox ===")
    box_re = {"type": "s", "rect": fitz.Rect(0, 0, 9, 9), "items": [("re", None)]}
    box_lines = {"type": "s", "rect": fitz.Rect(0, 0, 9, 9),
                 "items": [("l", None, None)] * 4}
    letter = {"type": "f", "rect": fitz.Rect(0, 0, 7, 7), "items": [("c", None)] * 36}
    circle = {"type": "f", "rect": fitz.Rect(0, 0, 18, 18), "items": [("c", None)] * 4}

    check("A rectangle drawn as one 're' is furniture", _is_furniture_ink(box_re))
    check("…and so is one drawn as four straight lines", _is_furniture_ink(box_lines))
    check("A letter outline is not", not _is_furniture_ink(letter))
    check("Nor is a circle, whose corners are not square", not _is_furniture_ink(circle))
    check("A tick box is mark-shaped", _is_mark_shaped(fitz.Rect(0, 0, 9, 9)))
    check("A rule is not", not _is_mark_shaped(fitz.Rect(0, 0, 90, 2)))

    # Rows of glyphs are what make a cluster a panel rather than a picture. Three abreast
    # is a line; two shapes on a baseline are a cartoon's eyes.
    line = [dict(letter, rect=fitz.Rect(x, 0, x + 7, 7)) for x in range(0, 70, 10)]
    eyes = [dict(letter, rect=fitz.Rect(x, 0, x + 7, 7)) for x in (0, 20)]
    check("Seven letters on a baseline are a line of text",
          _outlined_text_lines(line) == 1, f"{_outlined_text_lines(line)} lines")
    check("Two shapes on a baseline are not",
          _outlined_text_lines(eyes) == 0, f"{_outlined_text_lines(eyes)} lines")
    check("Neither are plain boxes, however many",
          _outlined_text_lines([box_re] * 8) == 0,
          f"{_outlined_text_lines([box_re] * 8)} lines")


def test_ocr_boxes_arrive_in_the_models_own_convention():
    """Tier 1: box_2d ([y0,x0,y1,x1] on a 0-1000 grid) becomes [x0,y0,x1,y1] fractions.

    Asked for fractions of width and height, the model was unreliable on anything oblong:
    on the exercise panel it found ten lines of twelve and put them a line and a half low,
    and two runs of the identical request at temperature 0 disagreed. Asked in the
    convention it was trained on it found all twelve, twice, identically. The order swap is
    the whole risk of the change — a y read as an x puts every Bangla block on the diagonal
    — so it is pinned here.
    """
    print("\n=== OCR Boxes In The Model's Own Convention ===")
    import image_localizer

    class _Response:
        text = json.dumps({"blocks": [
            {"text": "Help...", "lang": "en", "box_2d": [208, 30, 264, 150]},
            {"text": "tall and thin", "lang": "en", "box_2d": [0, 0, 1000, 40]},
            {"text": "out of range", "lang": "en", "box_2d": [-50, 900, 1200, 1100]},
            {"text": "inside out", "lang": "en", "box_2d": [800, 900, 200, 100]},
            {"text": "", "lang": "en", "box_2d": [0, 0, 100, 100]},
            {"text": "short box", "lang": "en", "box_2d": [10, 20, 30]},
        ]})

    original = image_localizer.generate_content
    image_localizer.generate_content = lambda **kw: _Response()
    try:
        blocks = image_localizer._extract_text_blocks(b"x", "image/png")
    finally:
        image_localizer.generate_content = original

    by_text = {b["text"]: b["bbox"] for b in blocks}
    check("y and x are read in the order the model sends them",
          by_text.get("Help...") == [0.03, 0.208, 0.15, 0.264],
          f"{by_text.get('Help...')}")
    check("…so a tall narrow box stays tall and narrow",
          by_text.get("tall and thin") == [0.0, 0.0, 0.04, 1.0],
          f"{by_text.get('tall and thin')}")
    check("A box running off the image is clamped to it",
          by_text.get("out of range") == [0.9, 0.0, 1.0, 1.0],
          f"{by_text.get('out of range')}")
    check("An inside-out box is dropped", "inside out" not in by_text)
    check("…and so is a block with no text", "" not in by_text)
    check("…and one whose box is the wrong length", "short box" not in by_text)


def test_unreadable_text_is_never_blanked():
    """Tier 1: a picture whose words could not be read is not sent to be blanked.

    OCR fed the overlay *and* was a single attempt returning [] on any failure, logged at
    debug. So a transient error on a picture full of labels read as "no labels" — and since
    the edit model is separately told to return every text surface blank, the words were
    deleted rather than translated. The eatwell plate came back correctly localized with its
    food-group captions simply gone, and nothing in the log said so.
    """
    print("\n=== Unreadable Text Is Never Blanked ===")
    import image_localizer

    original = image_localizer.generate_content
    image_localizer.generate_content = lambda **kw: (_ for _ in ()).throw(RuntimeError("503"))
    try:
        notes = {}
        got = image_localizer._extract_text_blocks(b"x", "image/png", notes=notes)
        check("A failed OCR call reports itself", notes.get("ocr_failed") is True)
        check("…and still returns no blocks", got == [])

        clean = {}
        image_localizer._extract_text_blocks(b"x", "image/png", notes=clean)
        check("…every time it fails", clean.get("ocr_failed") is True)
    finally:
        image_localizer.generate_content = original

    import image_processor

    classify_calls, edit_calls = [], []
    orig_classify = image_processor.classify_image
    orig_ocr = image_processor._extract_text_blocks
    orig_edit = image_processor.localize_image
    image_processor.classify_image = lambda *a, **k: (
        classify_calls.append(1) or {"is_logo": False, "needs_localization": True,
                                     "categories": ["food_objects"],
                                     "information_role": "decorative", "reason": "food"}
    )

    def failing_ocr(_b, _m, notes=None):
        if notes is not None:
            notes["ocr_failed"] = True
        return []

    image_processor._extract_text_blocks = failing_ocr
    image_processor.localize_image = lambda *a, **k: edit_calls.append(1) or b""
    try:
        png = _png(Image.new("RGB", (600, 400), (240, 240, 235)))
        out, status, record = image_processor._decide_from_png(png, 600, 400, "xref:166", 36)
        check("The picture is left exactly as printed", out is None and status == "ocr_failed",
              f"got {status!r}")
        check("…flagged untouched with the reason", record.get("untouched") is True)
        check("…and no image generation was paid for", not edit_calls,
              f"{len(edit_calls)} edit call(s) made")
    finally:
        image_processor.classify_image = orig_classify
        image_processor._extract_text_blocks = orig_ocr
        image_processor.localize_image = orig_edit


def test_audit_is_written_to_disk():
    """Tier 1: the run's decisions outlive the run.

    Until this the audit was one log line on stdout, so the only way to find out why a
    picture had shipped untouched was to have been watching the server at the time.
    """
    print("\n=== Audit Artifact ===")
    import image_processor

    original = image_processor.AUDIT_DIR
    image_processor.AUDIT_DIR = tempfile.mkdtemp()
    try:
        records = [
            {"ident": "xref:166", "page": 36, "status": "edit_ok", "scrim_blocks": 5},
            {"ident": "xref:43", "page": 1, "status": "aspect_kept", "untouched": True},
        ]
        path = _write_audit(records, "1 edit_ok, 1 aspect_kept")
        check("An audit file is written", bool(path) and os.path.exists(path or ""))
        if path:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            check("…with the run summary", payload["summary"] == "1 edit_ok, 1 aspect_kept")
            check("…and every record", len(payload["records"]) == 2)
            check("…including why an image was untouched",
                  payload["records"][1]["untouched"] is True
                  and payload["records"][1]["status"] == "aspect_kept")
    finally:
        image_processor.AUDIT_DIR = original


if __name__ == "__main__":
    print("Illustration Regions Detection Tests")
    print("=" * 60)

    test_imagemask_detection()
    test_imagemask_rasterization()
    test_backdrop_rejection()
    test_cover_illustration_detected()
    test_text_heavy_pages_rejected()
    test_dedup_content_key()
    test_text_free_render()
    test_in_place_swap_is_visible()
    test_punch_text_holes()
    test_palette_and_background_match()
    test_english_block_detection()
    test_text_surface_decisions()
    test_overlay_blocks_do_not_overlap()
    test_series_pages_are_protected()
    test_erase_box_snaps_to_the_ink()
    test_erased_blocks_are_identifiable()
    test_regeneration_mode()
    test_style_is_measured_not_asserted()
    test_live_text_pins_a_surface()
    test_a_raster_locks_every_page_it_is_printed_on()
    test_a_moved_placard_is_caught_in_pixels()
    test_a_pinned_picture_does_not_get_a_free_redraw()
    test_every_prompt_still_formats()
    test_the_prompts_name_the_props_that_must_go()
    test_a_redrawn_figure_leaves_no_ghost_of_the_old_one()
    test_cover_detection()
    test_apply_cover()
    test_background_match_leaves_a_picture_alone()
    test_background_is_measured_not_guessed()
    test_logos_are_never_changed()
    test_logo_text_is_never_translated()
    test_text_lands_on_the_moved_surface()
    test_text_on_a_photograph_gets_a_plate()
    test_a_badge_box_never_swallows_its_tile()
    test_no_english_survives_under_the_bangla()
    test_a_discarded_regeneration_is_never_reported_as_kept()
    test_numbers_survive_translation()
    test_smask_is_preserved()
    test_extreme_aspect_pictures_are_left_alone()
    test_overlay_skips_an_unreadably_small_box()
    test_a_cover_that_moves_its_panels_is_refused()
    test_a_picture_containing_a_mark_is_not_a_mark()
    test_nothing_the_overlay_draws_overlaps()
    test_divider_cartoons_are_found_in_a_translated_manual()
    test_a_panel_of_drawn_words_is_translated_but_not_redrawn()
    test_a_letter_is_not_a_checkbox()
    test_ocr_boxes_arrive_in_the_models_own_convention()
    test_unreadable_text_is_never_blanked()
    test_audit_is_written_to_disk()

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)
    elif skipped:
        print(f"OK: all tests passed ({len(skipped)} skipped)")
        sys.exit(0)
    else:
        print("OK: all tests passed")
        sys.exit(0)
