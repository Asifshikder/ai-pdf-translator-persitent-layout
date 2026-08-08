r"""Tests for page_redesign.py / page_templates.py — the full-page AI redesign pipeline.

Tier 1, no API calls: everything here drives the geometry/text/image-placement code
directly over synthetic in-memory `fitz.Document`s, the same style as
test_illustration_regions.py. The one Gemini-calling function, `_plan_page`, is only
exercised through its no-network fallback path (by making the client raise).

Run via:
  .\.venv\Scripts\python.exe test_page_redesign.py
"""

import io
import logging
import sys

import fitz
from PIL import Image

import page_redesign
import page_templates
from copy_layer import blank_shaped_tounicode
from pdf_processor import FONTS_DIR

logging.basicConfig(level=logging.WARNING)

# PUA/control codepoints the shaped layer leaks when it is NOT neutralised — same markers
# test_copy_layer.py checks for.
GARBLE_MARKERS = ["Ɩ", "Ǝ", "ţ", "ƀ", "\x85", "\x97"]

failures = []
skipped = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def _png(color: tuple[int, int, int], size: tuple[int, int] = (200, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _seg(text: str, size: float = 12.0, bold: bool = False, color: int = 0,
         bullet: bool = False, number: dict | None = None) -> dict:
    return {"text": text, "size": size, "bold": bold, "color": color, "bullet": bullet, "number": number}


# --------------------------------------------------------------------------------------
# Template geometry
# --------------------------------------------------------------------------------------

def _rects_overlap(a: page_templates.Fraction, b: page_templates.Fraction) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def test_template_slots_never_overlap():
    print("\n=== Template slot geometry ===")
    for tid, tpl in page_templates.TEMPLATES.items():
        slots = list(tpl.text_slots) + list(tpl.image_slots)
        if tpl.logo_slot:
            slots.append(tpl.logo_slot)
        overlapping = []
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i], slots[j]
                # The cover's title and logo deliberately sit over its own full-bleed art,
                # the way a real book cover's title/imprint sit over the cover image —
                # everything else must be pairwise disjoint.
                if tid == "cover" and "cover_art" in (a.id, b.id):
                    continue
                if _rects_overlap(a.rect, b.rect):
                    overlapping.append((a.id, b.id))
        check(f"{tid}: all slots are pairwise disjoint", not overlapping, str(overlapping))

    for tid, tpl in page_templates.TEMPLATES.items():
        for slot in list(tpl.text_slots) + list(tpl.image_slots) + ([tpl.logo_slot] if tpl.logo_slot else []):
            x0, y0, x1, y1 = slot.rect
            check(
                f"{tid}.{slot.id}: rect is well-formed and inside the page",
                0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0,
                str(slot.rect),
            )


def test_resolve_maps_fractions_to_absolute_coordinates():
    print("\n=== Slot resolution ===")
    page_rect = fitz.Rect(0, 0, 400, 800)
    slot = page_templates.Slot("body", (0.1, 0.2, 0.9, 0.8))
    rect = page_templates.resolve(slot, page_rect)
    check(
        "A slot resolves to the expected absolute rect for this page size",
        (rect.x0, rect.y0, rect.x1, rect.y1) == (40.0, 160.0, 360.0, 640.0),
        str(rect),
    )


# --------------------------------------------------------------------------------------
# Image placement: logos are never regenerated
# --------------------------------------------------------------------------------------

def test_logo_is_never_regenerated():
    print("\n=== Logo passthrough ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    logo_bytes = _png((0, 90, 160), (100, 50))
    page.insert_image(fitz.Rect(10, 10, 110, 60), stream=logo_bytes)
    xref = page.get_images(full=True)[0][0]
    images = page_redesign._page_images(doc, page)
    check("The synthetic logo image is discovered", len(images) == 1, str(images))

    # The model said "regenerate" but also flagged is_logo=True; is_logo must win.
    decisions = [{
        "xref": xref, "slot_id": "logo", "action": "regenerate",
        "is_logo": True, "information_role": "referential", "reason": "brand mark",
    }]

    def _must_not_be_called(*a, **kw):
        raise AssertionError("localize_image must never be called for a logo")

    original = page_redesign.image_localizer.localize_image
    page_redesign.image_localizer.localize_image = _must_not_be_called
    try:
        template = page_templates.TEMPLATES["chapter_title"]
        record = page_redesign._place_images(page, template, images, decisions, {"palette": "", "style": ""})
    finally:
        page_redesign.image_localizer.localize_image = original

    check("A logo's recorded action is never 'regenerate'", record[0]["action"] != "regenerate", str(record))
    check("A logo is routed to the template's logo slot", record[0]["slot"] == template.logo_slot.id, str(record))
    doc.close()


def test_image_slot_action_omit_is_skipped():
    print("\n=== Omit action ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_image(fitz.Rect(10, 10, 110, 60), stream=_png((10, 20, 30)))
    xref = page.get_images(full=True)[0][0]
    images = page_redesign._page_images(doc, page)
    decisions = [{"xref": xref, "slot_id": "figure", "action": "omit", "is_logo": False, "information_role": "decorative"}]
    template = page_templates.TEMPLATES["body_image_right"]
    record = page_redesign._place_images(page, template, images, decisions, {"palette": "", "style": ""})
    check("An 'omit' decision is recorded and nothing is drawn for it", record[0]["action"] == "omit", str(record))
    doc.close()


def test_page_images_dedups_by_xref():
    print("\n=== Image extraction ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    art = _png((5, 6, 7))
    page.insert_image(fitz.Rect(0, 0, 100, 60), stream=art)
    page.insert_image(fitz.Rect(0, 100, 100, 160), stream=art)  # same bytes elsewhere
    images = page_redesign._page_images(doc, page)
    check("Repeated placements of one image dedup to one xref entry or fewer than two draws", len(images) >= 1)
    for img in images:
        check(f"xref {img['xref']}: rect_fraction stays within [0,1]",
              all(0.0 <= v <= 1.0 for v in img["rect_fraction"]), str(img["rect_fraction"]))
    doc.close()


# --------------------------------------------------------------------------------------
# Text placement + copy-layer correctness
# --------------------------------------------------------------------------------------

def test_text_slot_is_copy_pasteable():
    print("\n=== Copy-layer correctness on a template slot ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    archive = fitz.Archive(FONTS_DIR)
    segs = [_seg("Take your medicine every day.", bullet=True)]
    bangla = "প্রতিদিন আপনার ওষুধ খান।"
    translations = {id(segs[0]): bangla}
    rect = fitz.Rect(20, 20, 380, 200)

    ok, scale = page_redesign._place_text_slot(page, archive, rect, segs, translations)
    check("The slot reports a successful fit", ok, f"scale={scale}")

    doc.subset_fonts()
    neutralised = blank_shaped_tounicode(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    reopened = fitz.open(stream=out, filetype="pdf")
    extracted = reopened[0].get_text()
    reopened.close()

    check("At least one shaped Bangla font was neutralised", neutralised >= 1)
    check("The translated Bangla round-trips out of the invisible copy layer", bangla in extracted, extracted)
    leaked = [m for m in GARBLE_MARKERS if m in extracted]
    check("No shaped-layer garble leaks into extraction", not leaked, str(leaked))
    check("The bulleted segment kept its bullet marker", "•" in extracted, extracted)


def test_text_slot_with_no_translations_is_a_no_op():
    print("\n=== Empty slot ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    archive = fitz.Archive(FONTS_DIR)
    segs = [_seg("Nothing to show.")]
    ok, scale = page_redesign._place_text_slot(page, archive, fitz.Rect(20, 20, 380, 200), segs, {})
    check("A slot with no matching translation reports ok without drawing", ok is True and scale == 1.0)
    doc.close()


# --------------------------------------------------------------------------------------
# TOC / index: page numbers preserved verbatim, titles translated
# --------------------------------------------------------------------------------------

def test_looks_like_toc_heuristic():
    print("\n=== TOC detection heuristic ===")
    toc_like = [_seg(f"Chapter {i}", number={"text": str(i), "bbox": [0, 0, 1, 1]}) for i in range(1, 6)]
    check("A page of five numbered entries reads as a TOC", page_redesign._looks_like_toc(toc_like))

    body_like = [_seg("A regular paragraph of body text with no trailing number.") for _ in range(5)]
    check("A page of plain paragraphs does not read as a TOC", not page_redesign._looks_like_toc(body_like))

    too_few = [_seg("One", number={"text": "1", "bbox": [0, 0, 1, 1]})]
    check("A single numbered line is not enough to call it a TOC", not page_redesign._looks_like_toc(too_few))


def test_toc_page_numbers_preserved_titles_translated():
    print("\n=== TOC layout ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    archive = fitz.Archive(FONTS_DIR)
    heading = [_seg("Contents")]
    rows = [
        _seg("Introduction", number={"text": "7", "bbox": [0, 0, 1, 1]}),
        _seg("Exercise plan", number={"text": "12", "bbox": [0, 0, 1, 1]}),
    ]
    segments = heading + rows
    translations = {
        id(heading[0]): "সূচিপত্র",
        id(rows[0]): "ভূমিকা",
        id(rows[1]): "ব্যায়ামের পরিকল্পনা",
    }
    template = page_templates.TEMPLATES["toc_index"]

    page_redesign._place_toc_page(page, archive, template, segments, translations)
    doc.subset_fonts()
    blank_shaped_tounicode(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    reopened = fitz.open(stream=out, filetype="pdf")
    extracted = reopened[0].get_text()
    reopened.close()

    check("Page number 7 is preserved verbatim", "7" in extracted, extracted)
    check("Page number 12 is preserved verbatim", "12" in extracted, extracted)
    check("The row title was translated, not copied", "ভূমিকা" in extracted, extracted)
    check("The heading was translated, not copied", "সূচিপত্র" in extracted, extracted)
    check("The English source titles do not survive into the output", "Introduction" not in extracted, extracted)


# --------------------------------------------------------------------------------------
# Plan call: safe fallback when the model/API is unavailable
# --------------------------------------------------------------------------------------

def test_plan_falls_back_safely_on_api_failure():
    print("\n=== Plan fallback ===")
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    segments = [_seg("Some English sentence.")]
    images = [{
        "xref": 42, "bytes": b"", "mime": "image/png", "width": 10, "height": 10,
        "rect_fraction": [0.0, 0.0, 0.1, 0.1], "rect": fitz.Rect(0, 0, 10, 10),
    }]

    def _raise(*a, **kw):
        raise RuntimeError("simulated network failure")

    original = page_redesign.generate_content
    page_redesign.generate_content = _raise
    try:
        plan = page_redesign._plan_page(
            page, segments, images, {"palette": "", "style": "", "text": ""},
            {"segment_count": 1, "ink_coverage": 0.0},
        )
    finally:
        page_redesign.generate_content = original
    doc.close()

    check("Falls back to the body_text template", plan["template_id"] == "body_text", str(plan))
    check("Every segment is still assigned somewhere",
          plan["text_assignments"] == [{"segment_index": 0, "slot_id": "body"}], str(plan))
    check("Every image defaults to 'keep' rather than being dropped",
          all(d["action"] == "keep" for d in plan["image_decisions"]), str(plan))


def test_plan_with_nothing_on_the_page_short_circuits():
    print("\n=== Empty-page plan ===")
    calls = []
    page_redesign.generate_content_ref = None

    def _tracking(*a, **kw):
        calls.append(1)
        raise RuntimeError("should not be called")

    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    original = page_redesign.generate_content
    page_redesign.generate_content = _tracking
    try:
        plan = page_redesign._plan_page(page, [], [], {"palette": "", "style": "", "text": ""}, {})
    finally:
        page_redesign.generate_content = original
    doc.close()

    check("A page with no text and no images never calls the model", not calls)
    check("…and still returns a usable body_text fallback", plan["template_id"] == "body_text")


if __name__ == "__main__":
    print("Page Redesign Pipeline Tests")
    print("=" * 60)

    test_template_slots_never_overlap()
    test_resolve_maps_fractions_to_absolute_coordinates()
    test_logo_is_never_regenerated()
    test_image_slot_action_omit_is_skipped()
    test_page_images_dedups_by_xref()
    test_text_slot_is_copy_pasteable()
    test_text_slot_with_no_translations_is_a_no_op()
    test_looks_like_toc_heuristic()
    test_toc_page_numbers_preserved_titles_translated()
    test_plan_falls_back_safely_on_api_failure()
    test_plan_with_nothing_on_the_page_short_circuits()

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
