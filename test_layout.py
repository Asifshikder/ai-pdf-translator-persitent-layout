r"""Layout regression tests: segmentation and box planning, with no API calls.

Everything here runs on `_extract_segments` / `_plan_insert_rects` directly, so it
exercises the whole geometry half of the translation pipeline without spending a
Gemini call. Each case is a defect seen in real output, kept as a test because the
same three keep coming back:

  * separate list items merging into one run-on paragraph;
  * text boxes planned outside the speech bubble they belong to;
  * Bangla lines packed so tightly their ink collides.

Run:  .\.venv\Scripts\python.exe test_layout.py
"""

import os
import sys

import fitz  # PyMuPDF

from pdf_processor import (
    BANGLA_SIZE,
    CSS_TEMPLATE,
    FONTS_DIR,
    _extract_segments,
    _panel_home,
    _panels,
    _plan_insert_rects,
    _rules,
    _vector_marks,
    css_for,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUIZ_PDF = os.path.join(
    BASE_DIR, "OriginalPDF", "Heart Failure Manual v4 2021 (1) [81-120].pdf"
)
SAMPLE_PDF = os.path.join(
    BASE_DIR, "Heart Manual_Post Myocardial Infarction (1)-1-20.pdf"
)

failures = []
skipped = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def plan_page(doc: fitz.Document, page_num: int):
    """Segment and plan one page exactly as translate_pdf does."""
    page = doc[page_num - 1]
    segments, kept = _extract_segments(page, _vector_marks(page))
    if segments:
        _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))
    return page, segments, kept


def texts(segments):
    return [seg["text"] for seg in segments]


def test_quiz_options_stay_separate(doc):
    """Vector tick-boxes must keep their options apart.

    The boxes are stroked squares, not glyphs, so nothing marked the options as
    bullets and all three merged into one sentence that was then translated as
    prose and reflowed across three stationary boxes.
    """
    _, segments, kept = plan_page(doc, 7)

    fused = [
        t for t in texts(segments) if "when I cough a lot" in t and "regularly" in t
    ]
    check(
        "page 7: Q3's three options are separate segments",
        not fused,
        f"still fused into one segment: {fused[:1]}",
    )

    fused4 = [t for t in texts(segments) if "call the doctor" in t and "less medication" in t]
    check(
        "page 7: Q4's three options are separate segments",
        not fused4,
        f"still fused into one segment: {fused4[:1]}",
    )

    options = [t for t in texts(segments) if t.startswith("when I cough a lot")]
    check(
        "page 7: the first option keeps its continuation line",
        any("my heart failure medication" in t for t in options),
        f"option text: {options}",
    )

    boxed = [seg for seg in segments if seg["bullet"] and seg["rect"].x0 < 320]
    check(
        "page 7: options are flagged as bullets by their vector box",
        len(boxed) >= 6,
        f"only {len(boxed)} pieces got bullet=True",
    )

    marks_kept = [k for k in kept if 12 <= k.width <= 15 and 12 <= k.height <= 15]
    check(
        "page 7: the vector boxes are preserved as obstacles",
        len(marks_kept) >= 6,
        f"only {len(marks_kept)} checkbox marks reached `kept`",
    )


def test_weekly_plan_rows_stay_separate(doc):
    """A 2-col x 7-row table must not collapse into one paragraph.

    The 20.1pt gap between `MON` and `Go to library` is 1.2x the 17pt font, but it
    fell under the old flat 30pt column threshold, so the columns fused; that made
    the row a single piece, which re-enabled row merging.
    """
    _, segments, _ = plan_page(doc, 12)

    fused = [t for t in texts(segments) if "Go to library" in t and "Walk round harbour" in t]
    check(
        "page 12: the weekly plan is not one merged segment",
        not fused,
        f"all seven rows still fused: {fused[:1]}",
    )

    days = [
        t
        for t in texts(segments)
        if t.strip() in {"MON", "TUE", "WED", "THUR", "FRI", "SAT", "SUN"}
    ]
    check(
        "page 12: each day is its own column piece",
        len(days) >= 7,
        f"found {len(days)} day cells: {days}",
    )

    check(
        "page 12: 'Go to library' is its own cell",
        any(t.strip() == "Go to library" for t in texts(segments)),
        f"activity cells: {[t for t in texts(segments) if 'library' in t]}",
    )


def test_wrapped_cell_stays_one_segment(doc):
    """A cell whose text wraps must stay one segment.

    A line's box runs ascender-to-descender, and this table's leading is tighter
    than that, so line two starts 5.5pt *above* line one's bottom. Read as a
    column break, the cell split into two segments that were each grown a line and
    ended up drawing on top of each other.
    """
    _, segments, _ = plan_page(doc, 19)
    found = [t for t in texts(segments) if t.startswith("Once a week - Thursday")]
    check(
        "page 19: a wrapped table cell is one segment",
        found == ["Once a week - Thursday evening"],
        f"cell segmented as {found}",
    )
    check(
        "page 19: 'evening' is not stranded as its own segment",
        "evening" not in texts(segments),
        "'evening' became a separate segment and will overdraw its own cell",
    )


def test_segment_boxes_do_not_overlap(doc):
    """Two segments' boxes must not be planned on top of each other.

    Not a general guarantee — headings and their neighbours still graze — but the
    tall overlaps are the ones that put one line of text on top of another.
    """
    bad = []
    for page_num in range(1, doc.page_count + 1):
        _, segments, _ = plan_page(doc, page_num)
        for i, a in enumerate(segments):
            for b in segments[i + 1 :]:
                shared = a["insert_rect"] & b["insert_rect"]
                if shared.is_empty or shared.width <= 1:
                    continue
                # A graze of a few points is the metric boxes touching; a whole
                # line's worth of shared height means real text over real text.
                if shared.height >= 0.9 * min(a["size"], b["size"]):
                    bad.append(
                        f"p{page_num}: {shared.height:.0f}pt of overlap between "
                        f"{a['text'][:26]!r} and {b['text'][:26]!r}"
                    )
    check(
        "no two segment boxes overlap by a full line",
        len(bad) <= 8,
        f"{len(bad)} bad overlaps (was 18 before the BACKTRACK fix), e.g.\n          "
        + "\n          ".join(bad[:3]),
    )


def test_no_box_escapes_its_bubble(doc):
    """No text box may be planned outside the bubble its text sits in.

    Bubbles are filled curves, invisible to `_rules`, so quote boxes were grown
    11-42pt past the bubble's bottom edge and the text landed on the page.
    """
    escapes = []
    for page_num in range(1, doc.page_count + 1):
        page, segments, _ = plan_page(doc, page_num)
        panels = _panels(page)
        if not panels:
            continue
        for seg in segments:
            home = _panel_home(panels, seg["rect"])
            if home is None:
                continue
            ins = seg["insert_rect"]
            # The original box is the floor: a segment never shrinks below it, so
            # only growth beyond it and beyond the bubble counts as an escape.
            over_y = ins.y1 - max(home.y1, seg["rect"].y1)
            over_x = ins.x1 - max(home.x1, seg["rect"].x1)
            if max(over_y, over_x) > 0.5:
                escapes.append(
                    f"p{page_num}: box {tuple(round(v) for v in ins)} leaves bubble "
                    f"(over_y={over_y:.1f} over_x={over_x:.1f}) {seg['text'][:40]!r}"
                )
    check(
        "no text box is planned outside its bubble",
        not escapes,
        f"{len(escapes)} escapes, e.g.\n          " + "\n          ".join(escapes[:3]),
    )


def test_options_render_at_one_size(doc):
    """Tick-box options of one question must all render at the same size.

    A one-line option gets no room to grow: the next option starts ~1.8pt below,
    so its box stays the 14.2pt the English line left. Bangla needs 1.45x its
    font size against Latin's ~1.18x, so at the source's own size it cannot fit
    that box — and insert_htmlbox reports no failure, it just shrinks the segment
    on its own. That made the rendered size an accident of what sat below: every
    question's last option kept 100% (nothing under it but the next heading)
    while its siblings landed at 77%, three sizes inside one question.

    Text short enough to fit the width on one line, so height alone decides —
    which is exactly what BANGLA_SIZE has to buy back.
    """
    _, segments, _ = plan_page(doc, 7)
    options = [seg for seg in segments if seg["bullet"] and seg["rect"].x0 < 320]

    archive = fitz.Archive(FONTS_DIR)
    scratch = fitz.open()
    shrunk = []
    for seg in options:
        page = scratch.new_page(width=620, height=880)
        spare, scale = page.insert_htmlbox(
            seg["insert_rect"], "প্রতিদিন", css=css_for(seg),
            scale_low=0.0, archive=archive,
        )
        if scale < 0.999:
            shrunk.append(f"{scale:.2f} in {seg['insert_rect'].height:.1f}pt box")
    scratch.close()

    check(
        "page 7: no tick-box option has to shrink to fit its box",
        not shrunk,
        f"{len(shrunk)} of {len(options)} shrank below their intended size "
        f"(BANGLA_SIZE={BANGLA_SIZE}): {shrunk[:3]}",
    )


def test_list_items_are_left_aligned(doc):
    """A list item must not be guessed centred because its lines wrapped that way.

    Two lines of a hanging-indent list item look exactly like two centred ones.
    Q9's heading wrapped with its midpoints 1.9pt apart and was drawn centred;
    an option below it wrapped with its right edges 1.4pt apart and was drawn
    right-aligned — both on a page where every other option was flush left.
    """
    _, segments, _ = plan_page(doc, 7)

    strays = [
        f"{seg['align']} {seg['text'][:40]!r}"
        for seg in segments
        if seg["align"] != "left"
    ]
    check(
        "page 7: every quiz heading and option is left-aligned",
        not strays,
        f"{len(strays)} drifted: {strays}",
    )


def test_centred_quotes_stay_centred(doc):
    """Forcing list items left must not flatten genuinely centred text.

    The pull-quotes inside the speech bubbles carry no list marker and are really
    centred; a blanket "everything is left" would have been the easy wrong fix.
    """
    _, segments, _ = plan_page(doc, 4)
    quotes = [
        seg for seg in segments
        if seg["align"] == "center" and len(seg["line_rects"]) >= 3
    ]
    check(
        "page 4: the centred pull-quote is still centred",
        quotes,
        "no centred multi-line text left on the page — list-marker rule overreached",
    )


def test_fix_pipeline_sizes_match_translation(doc):
    """Fix must re-render text at exactly the size translation drew it.

    The two pipelines built their CSS separately, so a size factor applied in one
    and not the other would have Fix silently redraw any segment it touched at a
    size none of its neighbours share — the same ragged page, one segment at a
    time.
    """
    _, segments, _ = plan_page(doc, 7)
    seg = segments[0]
    # A manifest segment carries the source size, as `css_for` expects; the factor
    # is applied at render time and never stored, so this must stay idempotent.
    from_manifest = {
        "size": seg["size"], "color": seg["color"],
        "align": seg["align"], "bold": seg["bold"],
    }
    check(
        "fix and translate build identical CSS from the same segment",
        css_for(from_manifest) == css_for(seg),
        "the two pipelines would render the same text at different sizes",
    )
    css = css_for(seg)
    check(
        "css_for scales the source size rather than emitting it raw",
        f"font-size: {seg['size'] * BANGLA_SIZE:.1f}px" in css
        and f"font-size: {seg['size']:.1f}px" not in css,
        f"a {seg['size']}pt segment did not render at {BANGLA_SIZE}x: {css.strip()!r}",
    )


def test_bangla_lines_do_not_collide():
    """Two lines of Bangla in a real table cell must not have colliding ink.

    Geometry is the real cell from a translated page: 'Once a week - Thursday
    evening' at size 17 in a 126.21x48.25 box. insert_htmlbox called this a
    comfortable fit at line-height 1.15 while the lines sat 1.92pt apart.
    """
    text = "সপ্তাহে একবার - বৃহস্পতিবার সন্ধ্যায়"
    width, height = 126.21, 48.25
    css = CSS_TEMPLATE.format(size=17.0, color="#000000", align="left", bold="")

    doc = fitz.open()
    page = doc.new_page(width=width + 40, height=height + 40)
    page.insert_htmlbox(
        fitz.Rect(20, 20, 20 + width, 20 + height),
        text,
        css=css,
        scale_low=0.0,
        archive=fitz.Archive(FONTS_DIR),
    )
    pixmap = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
    w, h, samples = pixmap.width, pixmap.height, pixmap.samples
    rows = [y for y in range(h) if min(samples[y * w : (y + 1) * w]) < 240]
    bands = []
    for y in rows:
        if bands and y - bands[-1][1] <= 1:
            bands[-1][1] = y
        else:
            bands.append([y, y])
    doc.close()

    check(
        "two Bangla lines render as two separate ink bands",
        len(bands) == 2,
        f"got {len(bands)} bands — the lines merged into one",
    )
    if len(bands) == 2:
        gap = (bands[1][0] - bands[0][1]) * 72 / 300
        check(
            "the gap between Bangla lines is readable (>5pt)",
            gap > 5.0,
            f"gap is only {gap:.2f}pt",
        )


def test_sample_pdf_regressions(doc):
    """The previously-working sample must not regress.

    Its checkboxes are ZapfDingbats glyphs, and every page carries ~60 registration
    marks that are exactly checkbox-shaped — the reason a mark only counts as a
    bullet when it has a label beside it.
    """
    page, segments, kept = plan_page(doc, 5)

    glyph_boxes = [k for k in kept if abs(k.x0 - 92.87) < 1 and 480 < k.y0 < 660]
    check(
        "sample p5: ZapfDingbats checkboxes are still preserved",
        len(glyph_boxes) >= 7,
        f"only {len(glyph_boxes)} dingbat checkboxes reached `kept`",
    )

    # `kept` legitimately holds page numbers and letterless rows anywhere on the
    # page, so look only for what `_vector_marks` could have contributed: a
    # checkbox-shaped square out in the margins, where no label can sit beside it.
    marks = _vector_marks(page)
    check(
        "sample p5: the page really does carry checkbox-shaped furniture",
        len(marks) >= 20,
        f"only {len(marks)} square marks found — this page no longer tests anything",
    )
    strays = [
        k
        for k in kept
        if 5 <= k.width <= 24
        and 5 <= k.height <= 24
        and max(k.width, k.height) <= 1.35 * min(k.width, k.height)
        and (k.x0 < 40 or k.x1 > 600 or k.y0 < 30 or k.y1 > 870)
    ]
    check(
        "sample p5: registration marks are not mistaken for bullets",
        not strays,
        f"{len(strays)} page-furniture marks leaked into `kept`: {strays[:3]}",
    )

    check(
        "sample p5: text is still segmented",
        len(segments) > 5,
        f"only {len(segments)} segments found",
    )


def main() -> int:
    if os.path.exists(QUIZ_PDF):
        doc = fitz.open(QUIZ_PDF)
        print("Heart Failure Manual [81-120] — quiz, weekly plan, bubbles")
        test_quiz_options_stay_separate(doc)
        test_options_render_at_one_size(doc)
        test_list_items_are_left_aligned(doc)
        test_centred_quotes_stay_centred(doc)
        test_fix_pipeline_sizes_match_translation(doc)
        test_weekly_plan_rows_stay_separate(doc)
        test_wrapped_cell_stays_one_segment(doc)
        test_segment_boxes_do_not_overlap(doc)
        test_no_box_escapes_its_bubble(doc)
        doc.close()
    else:
        skipped.append(f"{QUIZ_PDF} not found")

    if os.path.exists(SAMPLE_PDF):
        doc = fitz.open(SAMPLE_PDF)
        print("\nHeart Manual 1-20 — regressions")
        test_sample_pdf_regressions(doc)
        doc.close()
    else:
        skipped.append(f"{SAMPLE_PDF} not found")

    print("\nBangla line spacing")
    test_bangla_lines_do_not_collide()

    for note in skipped:
        print(f"\n  SKIP  {note}")
    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
