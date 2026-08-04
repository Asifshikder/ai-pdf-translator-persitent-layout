"""Tests for the Page Fix pipeline. Makes NO API calls — the planner is stubbed.

What is worth testing here is the half that must be deterministic: the coordinate
convention, the redaction-then-draw ordering, style inheritance, and the refusal to act on
a rectangle that makes no sense. The planner itself is a model call and is exercised by
using the tool.

    .\\.venv\\Scripts\\python.exe test_page_fix.py
"""

import io
import sys

import fitz
from PIL import Image

import page_fix

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def sample_pdf(pages=2) -> bytes:
    """A page with a heading, a paragraph and a coloured panel, at known positions."""
    doc = fitz.open()
    for n in range(pages):
        page = doc.new_page(width=400, height=500)
        page.insert_text((50, 60), f"Chapter {n + 1}", fontname="helv", fontsize=20)
        page.insert_text((50, 100), "The quick brown fox jumps over the lazy dog.",
                         fontname="helv", fontsize=11)
        page.draw_rect(fitz.Rect(50, 300, 350, 400), color=None, fill=(0.85, 0.9, 1.0))
    out = doc.tobytes()
    doc.close()
    return out


def page_text(pdf: bytes, index=0) -> str:
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = doc[index].get_text()
    doc.close()
    return text


def pixel_at(pdf: bytes, x, y, index=0):
    """The colour of one page point, read from a render."""
    doc = fitz.open(stream=pdf, filetype="pdf")
    pix = doc[index].get_pixmap(dpi=72)
    doc.close()
    with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
        return img.convert("RGB").getpixel((int(x), int(y)))


def attachment(color=(255, 0, 0), size=(120, 80)) -> page_fix.Attachment:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return page_fix.Attachment("swatch.png", buf.getvalue(), *size)


# ---------------------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------------------


def test_box_roundtrip():
    page_rect = fitz.Rect(0, 0, 400, 500)
    rect = fitz.Rect(100, 250, 300, 375)
    box = page_fix._to_box2d(rect, page_rect)
    check("box_2d is [y0,x0,y1,x1] on a 0-1000 grid", box == [500, 250, 750, 750], str(box))
    back = page_fix._to_rect(box, page_rect)
    check("box_2d → rect round-trips", abs(back.x0 - 100) < 1 and abs(back.y1 - 375) < 1,
          str(back))


def test_box_is_clamped_and_ordered():
    page_rect = fitz.Rect(0, 0, 400, 500)
    # Reversed and out of range: both must still produce a sane rect inside the page.
    rect = page_fix._to_rect([900, 800, 100, 200], page_rect)
    check("reversed corners are normalized", rect.y0 < rect.y1 and rect.x0 < rect.x1, str(rect))
    rect = page_fix._to_rect([-200, -200, 5000, 5000], page_rect)
    check("out-of-range boxes are clipped to the page", rect in page_rect, str(rect))


def test_block_targeting_beats_boxes():
    pdf = sample_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    page = doc[0]
    blocks = page_fix._text_blocks(page)
    heading = next(b for b in blocks if "Chapter" in b["text"])
    rect = page_fix._resolve_rect({"block_id": heading["id"]}, page.rect, blocks)
    doc.close()
    check("block_id resolves to the measured block rect",
          rect is not None and rect.contains(fitz.Point(60, 55)), str(rect))


# ---------------------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------------------


def test_erase_removes_text():
    pdf = sample_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    blocks = page_fix._text_blocks(doc[0])
    doc.close()
    heading = next(b for b in blocks if "Chapter" in b["text"])
    out, report = page_fix.apply_plan(
        pdf, 0, [{"action": "erase", "block_id": heading["id"], "note": "drop heading"}], []
    )
    check("erase applies", report[0]["status"] == "applied", report[0]["detail"])
    check("erase removes the text objects", "Chapter 1" not in page_text(out),
          repr(page_text(out)[:60]))
    check("erase leaves the rest of the page", "quick brown fox" in page_text(out))


def test_replace_text_swaps_wording():
    pdf = sample_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    blocks = page_fix._text_blocks(doc[0])
    doc.close()
    heading = next(b for b in blocks if "Chapter" in b["text"])
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "replace_text", "block_id": heading["id"], "text": "Section One",
          "note": "rename"}],
        [],
    )
    text = page_text(out)
    check("replace_text applies", report[0]["status"] == "applied", report[0]["detail"])
    check("the old wording is gone", "Chapter 1" not in text)
    check("the new wording is on the page", "Section One" in text, repr(text[:80]))


def test_replace_text_inherits_size():
    """The heading is 20pt; a replacement that did not inherit would come out at 11pt."""
    pdf = sample_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    blocks = page_fix._text_blocks(doc[0])
    doc.close()
    heading = next(b for b in blocks if "Chapter" in b["text"])
    rect = heading["rect"]
    style = page_fix._inherited_style(rect, blocks)
    check("style is inherited from the text under the box", style["size"] > 15,
          f"size={style['size']}")


def test_insert_text_keeps_neighbours():
    pdf = sample_pdf()
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "insert_text", "box_2d": [840, 100, 900, 800], "text": "Added footnote",
          "font_size": 10, "note": "footnote"}],
        [],
    )
    text = page_text(out)
    check("insert_text applies", report[0]["status"] == "applied", report[0]["detail"])
    check("inserted text is present", "Added footnote" in text)
    check("insert_text erases nothing", "Chapter 1" in text and "quick brown fox" in text)


def test_bangla_text_renders():
    """Bangla cannot be read back out of a PDF once shaped, so the check is that ink
    appeared where there was none — the same limitation the translate pipeline lives with."""
    pdf = sample_pdf()
    box = [840, 100, 920, 800]
    before = pixel_at(pdf, 100, 440)
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "insert_text", "box_2d": box, "text": "অধ্যায় চার", "font_size": 18,
          "note": "bangla"}],
        [],
    )
    doc = fitz.open(stream=out, filetype="pdf")
    pix = doc[0].get_pixmap(clip=page_fix._to_rect(box, doc[0].rect), dpi=72)
    doc.close()
    with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
        dark = sum(1 for p in img.convert("L").getdata() if p < 128)
    check("Bangla insert reports applied", report[0]["status"] == "applied",
          report[0]["detail"])
    check("Bangla actually draws ink", dark > 20, f"{dark} dark pixels")


def test_insert_image_places_attachment():
    pdf = sample_pdf()
    swatch = attachment(color=(255, 0, 0), size=(300, 100))
    # A box with the swatch's own 3:1 proportions, over the blue panel.
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "insert_image", "box_2d": [600, 125, 800, 875], "attachment": 0,
          "erase_first": True, "note": "place photo"}],
        [swatch],
    )
    r, g, b = pixel_at(out, 200, 350)
    check("insert_image applies", report[0]["status"] == "applied", report[0]["detail"])
    check("the attachment is visible on the page", r > 200 and g < 80 and b < 80, f"{(r,g,b)}")


def test_insert_image_without_attachment_is_skipped():
    pdf = sample_pdf()
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "insert_image", "box_2d": [600, 125, 800, 875], "attachment": 0,
          "note": "place photo"}],
        [],
    )
    check("insert_image with nothing to place is skipped, not crashed",
          report[0]["status"] == "skipped", report[0]["detail"])
    check("the page is unchanged", "Chapter 1" in page_text(out))


def test_bad_operations_are_rejected():
    pdf = sample_pdf()
    out, report = page_fix.apply_plan(
        pdf, 0,
        [
            {"action": "teleport", "box_2d": [0, 0, 100, 100], "note": "nonsense"},
            {"action": "erase", "note": "no rectangle at all"},
            {"action": "erase", "box_2d": [500, 500, 500, 501], "note": "degenerate"},
        ],
        [],
    )
    check("an unknown action is skipped", report[0]["status"] == "skipped", report[0]["detail"])
    check("an operation with no rectangle is skipped", report[1]["status"] == "skipped",
          report[1]["detail"])
    check("a zero-size rectangle is skipped", report[2]["status"] == "skipped",
          report[2]["detail"])
    check("nothing was drawn", page_text(out).strip() == page_text(pdf).strip())


def test_erase_then_draw_ordering():
    """apply_redactions rewrites the content stream, so text inserted before it would be
    wiped. Both operations in one plan must therefore survive together."""
    pdf = sample_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    blocks = page_fix._text_blocks(doc[0])
    doc.close()
    heading = next(b for b in blocks if "Chapter" in b["text"])
    out, report = page_fix.apply_plan(
        pdf, 0,
        [
            {"action": "insert_text", "box_2d": [840, 100, 900, 800], "text": "Survivor",
             "note": "insert first in the list"},
            {"action": "erase", "block_id": heading["id"], "note": "erase second"},
        ],
        [],
    )
    text = page_text(out)
    check("both operations applied",
          all(r["status"] == "applied" for r in report), str([r["detail"] for r in report]))
    check("the insert survives a later erase in the same plan", "Survivor" in text)
    check("the erase still happened", "Chapter 1" not in text)


def test_replace_text_does_not_clip_the_rule_beneath_it():
    """A heading's bbox lands within a hair of its underline, and a redaction that touches
    a stroke clips it — the rule came back with a step in it exactly as wide as the box.
    Measured on the Heart Manual: title bbox y1=123.6, a 4pt rule centred at y=126.6."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    page.insert_text((50, 60), "Heading", fontname="hebo", fontsize=20)
    page.draw_line(fitz.Point(50, 68), fitz.Point(350, 68), color=(0.3, 0.5, 0.55), width=4)
    pdf = doc.tobytes()
    doc.close()

    doc = fitz.open(stream=pdf, filetype="pdf")
    blocks = page_fix._text_blocks(doc[0])
    doc.close()
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "replace_text", "block_id": blocks[0]["id"], "text": "Retitled",
          "note": "rename"}],
        [],
    )
    check("the replacement applied", report[0]["status"] == "applied", report[0]["detail"])

    band = fitz.Rect(0, 66.5, 400, 71)  # the rule and nothing else
    crops = []
    for data in (pdf, out):
        d = fitz.open(stream=data, filetype="pdf")
        crops.append(d[0].get_pixmap(clip=band, dpi=200).tobytes("png"))
        d.close()
    check("the rule under the heading is left byte-identical", crops[0] == crops[1])


def test_other_pages_untouched():
    pdf = sample_pdf(pages=3)
    out, _ = page_fix.apply_plan(
        pdf, 1, [{"action": "erase", "box_2d": [0, 0, 1000, 1000], "note": "wipe page 2"}], []
    )
    check("page 1 is untouched", "Chapter 1" in page_text(out, 0))
    check("page 2 was wiped", "Chapter 2" not in page_text(out, 1))
    check("page 3 is untouched", "Chapter 3" in page_text(out, 2))


def test_erase_matches_the_surrounding_colour():
    """An erase inside the blue panel must leave blue behind, not white."""
    pdf = sample_pdf()
    out, report = page_fix.apply_plan(
        pdf, 0,
        [{"action": "erase", "box_2d": [640, 200, 760, 800], "note": "clear inside panel"}],
        [],
    )
    r, g, b = pixel_at(out, 200, 350)
    check("erase inside a coloured panel keeps the panel colour",
          b > r > 100 and abs(b - 255) < 30, f"{(r, g, b)}")


# ---------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------


def test_session_lifecycle():
    pdf = sample_pdf()
    session = page_fix.create_session("doc.pdf", pdf)
    check("a session reports its page count", session.page_count == 2)
    check("the session can be fetched by id",
          page_fix.get_session(session.sid).filename == "doc.pdf")

    png = page_fix.render_page(session.pdf, 0)
    check("pages render to PNG", png.startswith(b"\x89PNG"), f"{len(png)} bytes")

    try:
        page_fix.render_page(session.pdf, 9)
        check("a page past the end is refused", False)
    except page_fix.PageFixError as exc:
        check("a page past the end is refused", True, str(exc))

    page_fix.drop_session(session.sid)
    try:
        page_fix.get_session(session.sid)
        check("a dropped session is gone", False)
    except page_fix.PageFixError:
        check("a dropped session is gone", True)


class stub_ai:
    """Stand in for both model calls, so the whole flow runs with no API traffic."""

    def __init__(self, pages, operations, reasoning="stubbed"):
        self.pages, self.operations, self.reasoning = pages, operations, reasoning
        self.planned = []

    def __enter__(self):
        self._locate, self._plan = page_fix._locate_pages, page_fix._plan
        page_fix._locate_pages = lambda *a, **k: (self.pages, self.reasoning)

        def plan(page_png, description, instruction, attachments):
            self.planned.append(instruction)
            return {"analysis": "stubbed", "operations": self.operations}

        page_fix._plan = plan
        return self

    def __exit__(self, *exc):
        page_fix._locate_pages, page_fix._plan = self._locate, self._plan


def test_the_user_never_names_a_page():
    """The whole point of the flow: a PDF and a sentence, no page number anywhere."""
    session = page_fix.create_session("doc.pdf", sample_pdf(pages=3))
    ops = [{"action": "replace_text", "box_2d": [90, 100, 150, 700], "text": "Renamed",
            "note": "rename the heading"}]
    try:
        with stub_ai(pages=[2], operations=ops) as ai:
            report = page_fix.fix_document(session.sid, "the heading on the second page is wrong")
        check("the located page is the one that gets fixed",
              [p["page"] for p in report["pages"]] == [2], str(report["summary"]))
        check("the instruction reaches the planner verbatim",
              ai.planned == ["the heading on the second page is wrong"])
        check("that page changed", "Renamed" in page_text(session.pdf, 1))
        check("the pages it did not name are untouched",
              "Chapter 1" in page_text(session.pdf, 0)
              and "Chapter 3" in page_text(session.pdf, 2))
    finally:
        page_fix.drop_session(session.sid)


def test_one_instruction_many_pages_is_one_undo():
    session = page_fix.create_session("doc.pdf", sample_pdf(pages=3))
    original = session.pdf
    ops = [{"action": "erase", "box_2d": [80, 100, 160, 700], "note": "drop the heading"}]
    try:
        with stub_ai(pages=[1, 2, 3], operations=ops):
            report = page_fix.fix_document(session.sid, "remove the heading from every page")
        check("every located page was fixed",
              [p["page"] for p in report["pages"]] == [1, 2, 3], report["summary"])
        check("all three pages changed",
              all("Chapter" not in page_text(session.pdf, i) for i in range(3)))
        check("a three-page instruction is a single undo step",
              page_fix.undo(session.sid) and session.pdf == original)
        check("...and there is nothing left to undo",
              page_fix.undo(session.sid) is False)
    finally:
        page_fix.drop_session(session.sid)


def test_summary_names_where_it_worked():
    session = page_fix.create_session("doc.pdf", sample_pdf(pages=3))
    ops = [{"action": "erase", "box_2d": [80, 100, 160, 700], "note": "drop"}]
    try:
        with stub_ai(pages=[2], operations=ops):
            one = page_fix.fix_document(session.sid, "x")
        check("a single-page fix names the page", "page 2" in one["summary"], one["summary"])
        with stub_ai(pages=[1, 3], operations=ops):
            many = page_fix.fix_document(session.sid, "y")
        check("a multi-page fix lists the pages",
              "1, 3" in many["summary"], many["summary"])
    finally:
        page_fix.drop_session(session.sid)


def test_nothing_found_changes_nothing():
    session = page_fix.create_session("doc.pdf", sample_pdf())
    original = session.pdf
    try:
        with stub_ai(pages=[], operations=[], reasoning="No page mentions a German paragraph."):
            report = page_fix.fix_document(session.sid, "fix the German paragraph")
        check("an unlocatable instruction reports no change", report["changed"] is False,
              report["summary"])
        check("...and says why", "German" in report["analysis"], report["analysis"])
        check("...and leaves the bytes alone", session.pdf == original)
    finally:
        page_fix.drop_session(session.sid)


def test_empty_plan_changes_nothing():
    session = page_fix.create_session("doc.pdf", sample_pdf())
    original = session.pdf
    try:
        # The page was found, but the planner declined to act on it.
        with stub_ai(pages=[1], operations=[]):
            report = page_fix.fix_document(session.sid, "make it better somehow")
        check("an empty plan reports no change", report["changed"] is False, report["summary"])
        check("an empty plan leaves the bytes alone", session.pdf == original)
    finally:
        page_fix.drop_session(session.sid)


def test_one_bad_page_does_not_lose_the_others():
    session = page_fix.create_session("doc.pdf", sample_pdf(pages=3))
    real = page_fix._fix_one_page

    def flaky(pdf_bytes, page_index, instruction, attachments):
        if page_index == 1:
            raise RuntimeError("this page exploded")
        return real(pdf_bytes, page_index, instruction, attachments)

    ops = [{"action": "erase", "box_2d": [80, 100, 160, 700], "note": "drop"}]
    try:
        page_fix._fix_one_page = flaky
        with stub_ai(pages=[1, 2, 3], operations=ops):
            report = page_fix.fix_document(session.sid, "remove every heading")
        check("the failing page is reported, not raised",
              report["pages"][1]["changed"] is False
              and "exploded" in report["pages"][1]["analysis"])
        check("the pages either side of it still got fixed",
              "Chapter 1" not in page_text(session.pdf, 0)
              and "Chapter 3" not in page_text(session.pdf, 2))
    finally:
        page_fix._fix_one_page = real
        page_fix.drop_session(session.sid)


def test_too_many_pages_is_capped():
    session = page_fix.create_session("doc.pdf", sample_pdf(pages=3))
    ops = [{"action": "erase", "box_2d": [80, 100, 160, 700], "note": "drop"}]
    try:
        page_fix.MAX_PAGES_PER_FIX, cap = 2, page_fix.MAX_PAGES_PER_FIX
        with stub_ai(pages=[1, 2, 3], operations=ops):
            report = page_fix.fix_document(session.sid, "every page")
        check("a runaway page list is capped", len(report["pages"]) == 2,
              report["summary"])
        check("...and the cap is reported", "stopped at the first" in report["summary"],
              report["summary"])
    finally:
        page_fix.MAX_PAGES_PER_FIX = cap
        page_fix.drop_session(session.sid)


def test_input_validation():
    session = page_fix.create_session("doc.pdf", sample_pdf())
    try:
        try:
            page_fix.fix_document(session.sid, "   ")
            check("an empty instruction is refused", False)
        except page_fix.PageFixError as exc:
            check("an empty instruction is refused", True, str(exc))
        try:
            page_fix.fix_document("no-such-session", "fix it")
            check("an unknown session is refused", False)
        except page_fix.PageFixError as exc:
            check("an unknown session is refused", True, str(exc))
    finally:
        page_fix.drop_session(session.sid)


def test_page_digests_cover_every_page():
    pdf = sample_pdf(pages=3)
    digests, thumbs = page_fix._page_digests(pdf)
    check("one digest per page", len(digests) == 3, str(len(digests)))
    check("digests carry the page's text",
          all(f"Page {i + 1}" in digests[i] and f"Chapter {i + 1}" in digests[i]
              for i in range(3)))
    check("thumbnails accompany a short document",
          len(thumbs) == 3 and all(t.startswith(b"\x89PNG") for t in thumbs))


def test_colour_parsing():
    cases = [
        ("#ff0000", (1.0, 0.0, 0.0)),
        ("#f00", (1.0, 0.0, 0.0)),
        ("0000ff", (0.0, 0.0, 1.0)),
        ("not a colour", (0.0, 0.0, 0.0)),
        ([255, 128, 0], (1.0, 128 / 255, 0.0)),
    ]
    ok = all(
        all(abs(a - b) < 0.01 for a, b in zip(page_fix._parse_color(value), expected))
        for value, expected in cases
    )
    check("colours parse from hex, short hex and triples", ok)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — a crashing test is a failing test
            import traceback
            traceback.print_exc()
            check(test.__name__, False, f"crashed: {exc}")

    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{'=' * 70}\n{len(results) - len(failures)}/{len(results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAIL  {name} — {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
