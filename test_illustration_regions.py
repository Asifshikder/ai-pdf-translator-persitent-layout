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
import os
import sys
from collections import defaultdict

import fitz
from PIL import Image, ImageDraw

from image_processor import (
    CONTEXT_MIN_WORDS,
    SIMPLE_MODE_MAX_PIXELS,
    _apply_cover,
    _block_key,
    _detect_cover_page,
    _downscale_for_model,
    _is_english_block,
    _match_background,
    _outside_logos,
    _overlay_text_blocks,
    _background_color,
    _page_ink_coverage,
    _restamp_logos,
    _palette_summary,
    _prepare_text_only_image,
    _punch_text_holes,
    _regeneration_mode,
    _series_xrefs,
    _snap_to_ink,
    _swap_image_in_place,
    _text_rects_in,
    _to_png,
)
from image_regions import (
    TextFreePages,
    _illustration_clusters,
    _is_backdrop,
    _is_image_mask,
    _rasterize_rect,
)
from pdf_processor import _panels, _rules, _vector_marks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PDF = os.path.join(BASE_DIR, "OriginalPDF", "Heart Manual_Post Myocardial Infarction (1).pdf")

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
        "background is flat #ffffff" in summary5,
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
    test_cover_detection()
    test_apply_cover()
    test_background_match_leaves_a_picture_alone()
    test_background_is_measured_not_guessed()
    test_logos_are_never_changed()
    test_logo_text_is_never_translated()

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
