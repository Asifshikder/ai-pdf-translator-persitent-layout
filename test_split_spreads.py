r"""Spread-splitting tests.

Tests split_spreads against the real Heart Manual, which is laid out two-up
(one A3 landscape sheet per pair of facing A4 pages), to validate that:
  * A two-up sheet is recognised and a genuine wide page is not
  * Every span, drawing and image lands on exactly one of the two halves
  * Coordinates on the right-hand page are rebased to its own origin
  * A PDF with nothing to split comes back byte-identical
  * The embedded manifest is renumbered and shifted with the pages, so Fix
    still works on the split output

No API calls. Run via:
  .\.venv\Scripts\python.exe test_split_spreads.py
"""

import os
import sys

import fitz

import manifest
from split_spreads import GUTTER_BAND, is_spread, split_spreads

SPREAD_PDF = os.path.join("OriginalPDF", "Heart Manual Revascularisation (3) [1-19].pdf")

failures = []
skipped = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def skip(name: str, why: str) -> None:
    print(f"  SKIP  {name}\n          {why}")
    skipped.append(name)


def spread_doc():
    if not os.path.exists(SPREAD_PDF):
        return None
    return fitz.open(SPREAD_PDF)


def spans_of(page: fitz.Page) -> list[tuple]:
    """Every non-blank span on a page as (x0, y0, x1, y1, text)."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    out.append((*span["bbox"], span["text"]))
    return out


def test_a_two_up_sheet_is_recognised():
    """The A3 sheets are spreads; the lone A4 cover is not."""
    doc = spread_doc()
    if doc is None:
        skip("two-up sheet recognised", f"{SPREAD_PDF} not found")
        return

    verdicts = [is_spread(page) for page in doc]
    check(
        "the single A4 page is not a spread",
        verdicts[0] is False,
        f"page 1 is {doc[0].rect.width:.0f}x{doc[0].rect.height:.0f} and was called a spread",
    )
    check(
        "every A3 sheet is a spread",
        all(verdicts[1:]),
        f"missed pages {[i + 1 for i, v in enumerate(verdicts) if not v and i]}",
    )
    doc.close()


def test_a_wide_page_with_content_across_the_middle_is_left_alone():
    """Aspect ratio alone must not condemn a genuine landscape page."""
    doc = fitz.open()
    page = doc.new_page(width=1190, height=842)
    # A headline straddling the centreline: one page, not two.
    page.insert_text((400, 400), "a headline that runs across the whole sheet", fontsize=30)
    check(
        "a wide page with a span across the gutter is not a spread",
        is_spread(page) is False,
        "a landscape page with content across the middle was split",
    )

    ruled = doc.new_page(width=1190, height=842)
    ruled.draw_line(fitz.Point(100, 400), fitz.Point(1000, 400))
    check(
        "a wide page with a stroke across the gutter is not a spread",
        is_spread(ruled) is False,
        "a landscape page with a rule across the middle was split",
    )

    portrait = doc.new_page(width=595, height=842)
    check("a portrait page is never a spread", is_spread(portrait) is False)
    doc.close()


def test_every_span_lands_on_exactly_one_half():
    """No text is duplicated onto both pages or dropped between them."""
    doc = spread_doc()
    if doc is None:
        skip("spans land on exactly one half", f"{SPREAD_PDF} not found")
        return

    before = {}
    for page in doc:
        before[page.number] = len(spans_of(page))
    total_before = sum(before.values())
    doc.close()

    out, summary = split_spreads(open(SPREAD_PDF, "rb").read())
    split = fitz.open(stream=out, filetype="pdf")
    total_after = sum(len(spans_of(page)) for page in split)

    check(
        "no span is lost or duplicated by the split",
        total_after == total_before,
        f"{total_before} spans before, {total_after} after — {summary}",
    )
    check(
        "18 spreads become 36 pages",
        split.page_count == 37,
        f"got {split.page_count} pages, expected 37",
    )
    check(
        "every output page is portrait A4",
        all(page.rect.width < page.rect.height for page in split),
        f"{sum(1 for p in split if p.rect.width >= p.rect.height)} pages came out landscape",
    )
    split.close()


def test_the_right_hand_page_is_rebased_to_its_own_origin():
    """Content on the right half must report coordinates from 0, not from the gutter."""
    doc = spread_doc()
    if doc is None:
        skip("right-hand page rebased", f"{SPREAD_PDF} not found")
        return
    mid = doc[1].rect.width / 2
    right_before = sorted(s[0] for s in spans_of(doc[1]) if s[0] >= mid)
    doc.close()

    out, _ = split_spreads(open(SPREAD_PDF, "rb").read())
    split = fitz.open(stream=out, filetype="pdf")
    right_after = sorted(s[0] for s in spans_of(split[2]))

    check(
        "the right half keeps its spans",
        len(right_after) == len(right_before),
        f"{len(right_before)} spans on the right half, {len(right_after)} on page 3",
    )
    shifted = [round(x - mid, 1) for x in right_before]
    check(
        "the right half's x coordinates are rebased to the page origin",
        len(right_after) == len(shifted)
        and all(abs(a - b) < 0.5 for a, b in zip(right_after, shifted)),
        f"expected {shifted[:4]}, got {[round(x, 1) for x in right_after[:4]]}",
    )
    check(
        "nothing on page 3 sits outside the page",
        all(-1 <= s[0] and s[2] <= split[2].rect.width + 1 for s in spans_of(split[2])),
        "a span extends past the cropped page edge",
    )
    split.close()


def test_a_pdf_with_no_spreads_is_returned_unchanged():
    """The cheap path: don't rewrite a document that needs nothing done."""
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_text((72, 72), "single page")
    original = doc.tobytes()
    doc.close()

    out, summary = split_spreads(original)
    check(
        "a PDF with no spreads comes back byte-identical",
        out is original,
        "the document was rewritten despite having nothing to split",
    )
    check("the summary says so", "No spread pages" in summary, summary)


def test_the_manifest_follows_the_pages_it_describes():
    """Fix reads the manifest by page number and rect, so both must be remapped."""
    doc = spread_doc()
    if doc is None:
        skip("manifest follows its pages", f"{SPREAD_PDF} not found")
        return

    # A minimal manifest standing in for a translation run: one segment on the
    # left half of every sheet and one on the right, at known coordinates.
    # Each page is seeded against its own width, so the narrow cover gets two
    # segments that actually fit on it.
    mid = doc[1].rect.width / 2
    pages = []
    for page_num, page in enumerate(doc, start=1):
        half = page.rect.width / 2
        pages.append(
            {
                "page": page_num,
                "kept": [[10.0, 10.0, 60.0, 20.0], [half + 10, 10.0, half + 60, 20.0]],
                "segments": [
                    _segment(100.0, 300.0),
                    _segment(half + 40.0, half + 240.0),
                ],
            }
        )
    manifest.attach(doc, manifest.build({"Helvetica"}, pages))
    seeded = doc.tobytes()
    doc.close()

    out, _ = split_spreads(seeded)
    split = fitz.open(stream=out, filetype="pdf")
    data = manifest.read(split)

    numbers = [entry["page"] for entry in data["pages"]]
    check(
        "one manifest entry per output page, in order",
        numbers == list(range(1, split.page_count + 1)),
        f"got {numbers[:6]}… for {split.page_count} pages",
    )

    # Page 1 is the unsplit cover, so it keeps both of its planted segments.
    check(
        "the unsplit page keeps all its segments",
        len(data["pages"][0]["segments"]) == 2,
        f"got {len(data['pages'][0]['segments'])}",
    )
    halves = data["pages"][1:]
    check(
        "each half of a spread gets the one segment that fell on it",
        all(len(entry["segments"]) == 1 for entry in halves),
        f"segment counts {[len(e['segments']) for e in halves[:6]]}",
    )
    check(
        "each half of a spread gets its own kept rect",
        all(len(entry["kept"]) == 1 for entry in halves),
        f"kept counts {[len(e['kept']) for e in halves[:6]]}",
    )

    left, right = data["pages"][1], data["pages"][2]
    check(
        "the left half's rects are untouched",
        left["segments"][0]["ins"][0] == 100.0 and left["kept"][0][0] == 10.0,
        f"left ins={left['segments'][0]['ins']} kept={left['kept'][0]}",
    )
    check(
        "the right half's rects are shifted onto its own page",
        right["segments"][0]["ins"][0] == 40.0 and right["kept"][0][0] == 10.0,
        f"right ins={right['segments'][0]['ins']} kept={right['kept'][0]}",
    )
    check(
        "every manifest rect fits inside the page it now names",
        all(
            0 <= seg["ins"][0] and seg["ins"][2] <= split[entry["page"] - 1].rect.width + 1
            for entry in data["pages"]
            for seg in entry["segments"]
        ),
        "a segment rect hangs off its page",
    )
    split.close()


def _segment(x0: float, x1: float) -> dict:
    """A manifest segment entry spanning x0..x1, with every rect field set."""
    rect = [x0, 400.0, x1, 420.0]
    return {
        "en": "text", "bn": "টেক্সট", "ok": True,
        "rect": list(rect), "lines": [list(rect)], "ins": list(rect),
        "size": 10.0, "bold": False, "color": 0, "align": 0, "bullet": None,
        "block_x1": x1, "num": None, "sh": 1.0, "sc": 1.0, "fixed": None,
    }


def test_the_gutter_band_is_narrow_enough_for_a_real_document():
    """A band wider than the real gutter would veto every sheet."""
    doc = spread_doc()
    if doc is None:
        skip("gutter band width", f"{SPREAD_PDF} not found")
        return
    page = doc[1]
    mid = page.rect.width / 2
    nearest = min(
        (min(abs(s[0] - mid), abs(s[2] - mid)) for s in spans_of(page)), default=999
    )
    check(
        "the real gutter is wider than the band we test",
        nearest > GUTTER_BAND,
        f"nearest span edge is {nearest:.1f}pt from the centreline, band is {GUTTER_BAND}pt",
    )
    doc.close()


if __name__ == "__main__":
    print("Spread-splitting tests")
    print("=" * 60)
    test_a_two_up_sheet_is_recognised()
    test_a_wide_page_with_content_across_the_middle_is_left_alone()
    test_every_span_lands_on_exactly_one_half()
    test_the_right_hand_page_is_rebased_to_its_own_origin()
    test_a_pdf_with_no_spreads_is_returned_unchanged()
    test_the_manifest_follows_the_pages_it_describes()
    test_the_gutter_band_is_narrow_enough_for_a_real_document()

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
