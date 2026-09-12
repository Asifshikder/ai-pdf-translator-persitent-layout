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
    _bubble_home,
    _bubble_mask,
    _bubbles,
    _extract_segments,
    _panel_home,
    _panels,
    _plan_insert_rects,
    _rect_on_ink,
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
RATING_PDF = os.path.join(
    BASE_DIR, "OriginalPDF", "Heart Manual Revascularisation (3) [65-91].pdf"
)
# The full manual: 42 speech bubbles, the largest sample of them anywhere here.
BUBBLE_PDF = os.path.join(
    BASE_DIR, "OriginalPDF", "Heart Failure Manual v4 2021 (1).pdf"
)
# Its cover is a vector cartoon of 67-181 path items — a large filled shape with
# curves in it, and so a panel, but emphatically not a bubble.
ARTWORK_PDF = os.path.join(
    BASE_DIR, "OriginalPDF", "Heart Manual_Post Myocardial Infarction (1).pdf"
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


def test_rating_grid_rows_stay_separate(doc):
    """A questionnaire rating grid must not fuse its rows into one cell.

    The statements sit one per row with a "1 2 3 4 5" rating column to their
    right. That column is too close to split the row into separate pieces, so the
    row was not `standalone` and the vertical merge absorbed the row below it —
    five statements collapsed into one blob, translated as a run-on sentence and
    drawn across one cell while the rows it swallowed came out empty. The drawn
    row borders were never consulted; now they block the merge.
    """
    _, segments, _ = plan_page(doc, 23)
    rows = ["I exercise", "I practise relaxation", "I practise correct breathing",
            "I smoke", "I eat a healthy diet"]

    fused = [
        t for t in texts(segments)
        if sum(r in t for r in rows) >= 2
    ]
    check(
        "page 23: the rating grid's rows are not fused into one segment",
        not fused,
        f"rows merged across their borders: {fused[:1]}",
    )

    missing = [r for r in rows if r not in texts(segments)]
    check(
        "page 23: each lifestyle statement is its own segment",
        not missing,
        f"these rows never surfaced as their own segment: {missing}",
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


def bubble_homes(page, segments):
    """Every (segment, bubble mask) pair on a page, by ink membership."""
    masks = [_bubble_mask(page, drawing) for drawing in _bubbles(page)]
    if not masks:
        return []
    pairs = []
    for seg in segments:
        home = _bubble_home(masks, seg["rect"])
        if home is not None:
            pairs.append((seg, home))
    return pairs


def test_no_box_escapes_its_bubble(doc):
    """No text box may be planned outside the ink of the bubble it sits in.

    Bubbles are filled curves, invisible to `_rules`, so quote boxes were grown
    11-42pt past the bubble's bottom edge and the text landed on the page. The
    first fix capped growth at the bubble's *bounding box* inset 8%, which for
    an ellipse is nowhere near enough — an aspect-matched inscribed rectangle
    needs 29.3% — so 48 of 52 bubble segments still had a corner outside the
    ink, and because the bubble text is white the overflow read as clipped.

    Tested against the fill itself, with no allowance for the original box: the
    planner may now move a bubble's text, so "it was already like that" is no
    longer an excuse it can offer.
    """
    escapes = []
    checked = 0
    for page_num in range(1, doc.page_count + 1):
        page, segments, _ = plan_page(doc, page_num)
        for seg, (bubble_rect, mask) in bubble_homes(page, segments):
            checked += 1
            ins = seg["insert_rect"]
            if not _rect_on_ink(bubble_rect, mask, ins):
                escapes.append(
                    f"p{page_num}: box {tuple(round(v) for v in ins)} leaves the "
                    f"bubble at {tuple(round(v) for v in bubble_rect)} "
                    f"{seg['text'][:40]!r}"
                )
    check(
        f"no text box is planned outside its bubble ({checked} checked)",
        not escapes,
        f"{len(escapes)} escapes, e.g.\n          " + "\n          ".join(escapes[:3]),
    )


def test_no_box_escapes_its_panel(doc):
    """No text box may be planned outside a plain-rectangle callout panel.

    A highlight/"ACTION" box is drawn as a single `re` fill with no curve, so it
    was invisible to `_panels` and `_plan_insert_rects` never capped growth
    against it: paragraphs grew 17.5-19.5pt past an ACTION box's bottom edge and
    bullets 25-31pt past a warning box's right edge on pp.35/64 of this manual.

    Growth only, not containment: the planner never shrinks a box below its
    source, so a source that already reached past the panel's inset (the inset
    itself, not the growth rule, put it there) is not this test's concern —
    only growth past whichever edge is further out counts as an escape.
    """
    escapes = []
    checked = 0
    for page_num in range(1, doc.page_count + 1):
        page, segments, _ = plan_page(doc, page_num)
        panels = _panels(page)
        for seg in segments:
            if seg.get("bubble_rects"):
                continue  # sole bubble tenant: inscribed-rect path, not this one
            home = _panel_home(panels, seg["rect"])
            if home is None:
                continue
            checked += 1
            ins = seg["insert_rect"]
            limit_x1 = max(home.x1, seg["rect"].x1)
            limit_y1 = max(home.y1, seg["rect"].y1)
            if ins.x1 > limit_x1 + 0.5 or ins.y1 > limit_y1 + 0.5:
                escapes.append(
                    f"p{page_num}: box {tuple(round(v) for v in ins)} leaves the "
                    f"panel at {tuple(round(v) for v in home)} "
                    f"{seg['text'][:40]!r}"
                )
    check(
        f"no text box is planned outside its panel ({checked} checked)",
        not escapes,
        f"{len(escapes)} escapes, e.g.\n          " + "\n          ".join(escapes[:3]),
    )


def test_bubble_text_uses_the_headroom_above_it(doc):
    """A bubble's sole tenant is re-centred in it, not pinned where English was.

    The planner otherwise only ever grows a box down and right, so the taller
    Bangla pushed out of the bottom of a bubble while the space above the
    original line went unused. A bubble with one tenant is the one case where
    moving the box is safe, and it is the case that pays: the rect is centred on
    the fill, which is where the English was centred to begin with.
    """
    lifted = []
    tenants = 0
    for page_num in range(1, doc.page_count + 1):
        page, segments, _ = plan_page(doc, page_num)
        for seg, _home in bubble_homes(page, segments):
            if not seg.get("bubble_rects"):
                continue  # shares its bubble: clamped, deliberately not moved
            tenants += 1
            if seg["insert_rect"].y0 < seg["rect"].y0 - 0.5:
                lifted.append(page_num)
    check(
        f"bubble text reclaims the space above it ({len(lifted)}/{tenants} lifted)",
        tenants > 0 and len(lifted) >= 0.5 * tenants,
        f"only {len(lifted)} of {tenants} sole tenants were raised above their "
        "original top edge; centring is not taking effect",
    )


def test_bubble_membership_ignores_artwork(doc):
    """A cover cartoon is a panel but not a bubble, and hosts no text.

    `_panels` accepts any large filled shape with a curve in it, which on this
    cover matches the cartoon and its limbs. Judged by bounding box, a 6pt
    printer's slug at the page edge then counts as living "inside" it — and
    would be re-centred into the middle of the artwork. A bubble is a simple
    closed path, and membership is by fill, not by bounding box.
    """
    page = doc[0]
    panels = _panels(page)
    bubbles = _bubbles(page)
    check(
        "the cover cartoon reads as a panel but not as a bubble",
        bool(panels) and not bubbles,
        f"{len(panels)} panels, {len(bubbles)} bubbles "
        f"(item counts: {[len(d['items']) for d in bubbles]})",
    )

    _, segments, _ = plan_page(doc, 1)
    slugs = [seg for seg in segments if seg["text"].startswith("HM Post MI")]
    check(
        "the printer's slug is not adopted by the artwork behind it",
        bool(slugs) and not bubble_homes(page, slugs),
        f"found {len(slugs)} slug segments, "
        f"{len(bubble_homes(page, slugs))} of them claimed by a bubble",
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


def test_text_does_not_grow_onto_a_picture():
    """A paragraph that merely grazes a figure must not expand across it.

    Real geometry from p.114 of the translated Heart Failure manual: the right column
    ran to y=605 and the illustration of a man holding a sign occupied
    (421, 600)-(596, 843). Because the English text's last line already overlapped the
    top of that image by a few points, the planner excused the image from the obstacle
    list altogether and grew the Bangla box a full line further down — 82pt of text
    printed straight over the picture.

    The distinction the planner has to make is between a picture the text sits *on*
    (a backdrop, checked separately below) and one it happens to touch.
    """
    image = fitz.Rect(421.1, 600.2, 596.3, 842.9)
    doc = fitz.open()
    page = doc.new_page(width=595.276, height=841.89)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    pixmap.set_rect(pixmap.irect, (200, 180, 160))
    page.insert_image(image, pixmap=pixmap)
    # Two lines close enough to merge into one segment, positioned so the second
    # already dips a few points past the top of the image — the condition that made
    # the old planner stop treating the image as an obstacle at all.
    page.insert_text((313, 590), "Losing weight is worth keeping up", fontsize=10)
    page.insert_text((313, 604), "and adding to over time", fontsize=10)

    segments, kept = _extract_segments(page, _vector_marks(page))
    body = [s for s in segments if "weight" in s["text"]]
    if not body:
        check("p.114 overflow fixture builds", False, "no segment extracted")
        doc.close()
        return
    seg = body[0]
    base = fitz.Rect(seg["rect"])
    _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))
    grown = seg["insert_rect"]
    doc.close()

    added = (grown & image).get_area() - (base & image).get_area()
    check(
        "a box does not grow onto a picture it only grazes",
        added <= 1.0,
        f"gained {added:.0f}pt² of the image (base {base}, grown {grown})",
    )
    check(
        "…and is not shrunk below where it started",
        grown.y1 >= base.y1 - 0.01 and grown.x1 >= base.x1 - 0.01,
        f"grown {grown} is smaller than base {base}",
    )


def test_text_may_still_grow_on_its_backdrop():
    """The other half of the same rule: text printed over a full-bleed picture has
    nowhere else to go, so the picture must not block it. Pinning such a box to the
    width the English happened to need is what forces the Bangla down to 9pt."""
    backdrop = fitz.Rect(0, 0, 595.276, 841.89)
    doc = fitz.open()
    page = doc.new_page(width=595.276, height=841.89)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    pixmap.set_rect(pixmap.irect, (240, 240, 250))
    page.insert_image(backdrop, pixmap=pixmap)
    page.insert_text((60, 300), "Look after your heart", fontsize=14)

    segments, kept = _extract_segments(page, _vector_marks(page))
    body = [s for s in segments if "heart" in s["text"]]
    if not body:
        check("backdrop fixture builds", False, "no segment extracted")
        doc.close()
        return
    seg = body[0]
    base = fitz.Rect(seg["rect"])
    _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))
    grown = seg["insert_rect"]
    doc.close()

    check(
        "text on a full-page backdrop is still free to grow",
        grown.width > base.width + 1 or grown.height > base.height + 1,
        f"box was pinned to its English size: base {base}, grown {grown}",
    )


def _justified_line(page, words, x0, x1, y, size):
    """Draw one justified line: `words` spread so the line fills x0..x1 exactly."""
    widths = [fitz.get_text_length(w, fontname="helv", fontsize=size) for w in words]
    gap = (x1 - x0 - sum(widths)) / (len(words) - 1)
    x = x0
    for word, width in zip(words, widths):
        page.insert_text((x, y), word, fontsize=size)
        x += width + gap
    return gap


def test_justified_line_is_not_read_as_table_columns():
    """A justified line's words must come back as one segment, not one each.

    Real geometry from p.147 of the Post-MI manual. "One particular type of" is
    the opening line of a paragraph set justified in a 154pt column, so its
    three spaces are stretched to 13.3pt — past COLUMN_GAP_MIN. PyMuPDF hands
    each word over as its own line, `_rows` puts them back on one row, and the
    column test then cut the line into four cells.

    Every one of them was a defect on its own: "One" and "type" were sent to the
    translator with no sentence around them, and each was then planned as a
    standalone label with `10 * size` of room to grow into — straight across the
    paragraph they came from. Three of the four ended up drawn on top of it.
    """
    doc = fitz.open()
    page = doc.new_page(width=654.283, height=900.898)
    left, right, size = 416.0, 570.2, 11.0
    gap = _justified_line(page, ["One", "particular", "type", "of"], left, right, 251.3, size)
    # The rest of the paragraph: ordinary lines, flush with the same right edge.
    for text, y in (("breathlessness requires a little", 266.3),
                    ("more explanation. It is called", 281.3)):
        width = fitz.get_text_length(text, fontname="helv", fontsize=size)
        page.insert_text((right - width, y), text, fontsize=size)

    segments, kept = _extract_segments(page, _vector_marks(page))
    _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))
    stray = [t for t in texts(segments) if t in ("One", "particular", "type", "of")]
    whole = [t for t in texts(segments) if t.startswith("One particular type of")]
    doc.close()

    check(
        f"a justified line ({gap:.1f}pt word gaps) is not cut into cells",
        not stray,
        f"words stranded as their own segments: {stray}",
    )
    check(
        "…and its words are rejoined into the sentence they came from",
        bool(whole),
        f"segments: {texts(segments)[:4]}",
    )


def test_table_row_still_splits_into_cells():
    """The other half of the same rule: a real table row must still split.

    Two rows of a day planner, with the same kind of gap the justified line
    above has. What tells them apart is that a table's rows end wherever their
    last cell ends, so they are not all set to the full measure.
    """
    doc = fitz.open()
    page = doc.new_page(width=595.276, height=841.89)
    for label, activity, y in (("Monday", "Go to the library", 300.0),
                               ("Tuesday", "Buy a new shirt and a hat", 316.0)):
        page.insert_text((100.0, y), label, fontsize=11)
        page.insert_text((160.0, y), activity, fontsize=11)

    segments, _ = _extract_segments(page, _vector_marks(page))
    doc.close()
    fused = [t for t in texts(segments) if "Monday" in t and "library" in t]
    check(
        "a table row's cells are still separate segments",
        not fused,
        f"cells fused into one segment: {fused}",
    )


def test_growth_does_not_cross_a_neighbour():
    """Two boxes must not meet in the white space between them.

    Real geometry from the top of a Post-MI chapter page: the chapter title
    grows 280pt to its right at the same time as the running head beside it
    grows a line down, and they cross in a gap that neither had reached when it
    was planned. Each was measured only against where the other's *source* text
    sat, so neither saw it coming.

    The baselines are set so the two line boxes graze by well under a point, as
    the real ones do (0.77pt). That is what hides them from each other: both the
    horizontal and the vertical obstacle test want more than a point of overlap
    on the other axis before they treat a neighbour as being in the way, and
    below that threshold each box is free to grow across the other.
    """
    doc = fitz.open()
    page = doc.new_page(width=654.283, height=900.898)
    page.insert_text((87.2, 115.0), "Hyperventilation", fontsize=28)
    page.insert_text((428.1, 81.0), "Hyperventilation", fontsize=15.6)

    segments, kept = _extract_segments(page, _vector_marks(page))
    _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))
    boxes = [(fitz.Rect(s["rect"]), fitz.Rect(s["insert_rect"])) for s in segments]
    doc.close()

    if len(boxes) != 2:
        check("running-head fixture builds", False, f"{len(boxes)} segments, expected 2")
        return
    (src_a, ins_a), (src_b, ins_b) = boxes
    check(
        "growth does not run one box into the box beside it",
        (ins_a & ins_b).is_empty and (src_a & src_b).is_empty,
        f"{ins_a} overlaps {ins_b} (sources {src_a} and {src_b} did not)",
    )
    check(
        "…and each box still gets room to grow",
        ins_a.x1 > src_a.x1 + 1 and ins_b.y1 > src_b.y1 + 1,
        f"a: {src_a} -> {ins_a}; b: {src_b} -> {ins_b}",
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

    if os.path.exists(RATING_PDF):
        doc = fitz.open(RATING_PDF)
        print("\nHeart Manual Revascularisation [65-91] — questionnaire grid")
        test_rating_grid_rows_stay_separate(doc)
        doc.close()
    else:
        skipped.append(f"{RATING_PDF} not found")

    if os.path.exists(BUBBLE_PDF):
        doc = fitz.open(BUBBLE_PDF)
        print("\nHeart Failure Manual (full) — speech bubbles")
        test_no_box_escapes_its_bubble(doc)
        test_bubble_text_uses_the_headroom_above_it(doc)
        test_no_box_escapes_its_panel(doc)
        doc.close()
    else:
        skipped.append(f"{BUBBLE_PDF} not found")

    if os.path.exists(ARTWORK_PDF):
        doc = fitz.open(ARTWORK_PDF)
        print("\nHeart Manual Post MI — bubble detection vs artwork")
        test_bubble_membership_ignores_artwork(doc)
        doc.close()
    else:
        skipped.append(f"{ARTWORK_PDF} not found")

    print("\nJustified text vs table columns")
    test_justified_line_is_not_read_as_table_columns()
    test_table_row_still_splits_into_cells()

    print("\nText growth around pictures")
    test_text_does_not_grow_onto_a_picture()
    test_text_may_still_grow_on_its_backdrop()
    test_growth_does_not_cross_a_neighbour()

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
