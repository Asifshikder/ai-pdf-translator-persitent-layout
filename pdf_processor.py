"""PDF translation pipeline: extract text segments, translate, redact, re-insert Bangla."""

import html
import logging
import os
import re

import fitz  # PyMuPDF

import manifest
from translator import translate_batch_status

logger = logging.getLogger(__name__)

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Noto Sans Bengali draws ink over ~1.46x its font size — matras reach above the
# ascender and conjuncts below the baseline — so leading that suits Latin text
# packs Bangla lines into each other. Measured at 1.15: two lines of a real table
# cell sit 1.92pt apart, and at 12pt with conjunct-heavy text their ink merges
# into one unreadable band. 1.45 clears them by ~6.7pt.
#
# insert_htmlbox measures fit from *this* number rather than from the ink, so a
# value too low does not report a failure: it reports a comfortable fit and draws
# overlapping text. Nothing downstream notices.
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
    {bold}
}}
"""

# Bangla is rendered at a fraction of the size of the English it replaces.
#
# Not a matter of taste: a Latin line box is about 1.18x its font size, a Bangla
# one 1.45x (see CSS_TEMPLATE), so a line of Bangla set at the source's own size
# cannot fit the line box the source left behind. Nothing reported that as a
# failure — insert_htmlbox just shrank each segment on its own until it fit,
# which made the rendered size an accident of how much blank space happened to
# sit below it. On the quiz page that put every question's last option at 100%
# (nothing beneath it but the next heading) and its siblings at 77%, so the
# options of one question were visibly three different sizes. Choosing the size
# here is what makes it uniform: shrink-to-fit then never has to fire.
#
# The ceiling is what a segment's box can hold, so it moves with BELOW_GAP: at
# the old 2pt margin only 0.76 was safe, and reclaiming the gap bought 0.80.
# Measured across the 40-page manual at 0.80: 15 of 672 segments still shrink,
# none of them a tick-box option, and no page has options at mismatched sizes.
# Above this the boxes run out — 0.85 shrinks 70 — and buying more would mean
# re-stacking each option list and moving its checkbox down to follow the text.
BANGLA_SIZE = 0.80

# Bit 4 of span flags marks a bold font.
BOLD_FLAG = 1 << 4

# A segment is worth translating only if it contains at least one Latin letter.
HAS_LETTERS = re.compile(r"[A-Za-z]")

# A page number or page range as found in tables of contents: "7", "21-35", "101–118".
PAGE_NUMBER = re.compile(r"^\d{1,4}\s*(?:[–\-]\s*\d{1,4})?$")

# Fonts whose glyphs are decorative bullets, not readable text.
SYMBOL_FONT = re.compile(r"dingbat|wingding|webding|symbol", re.I)
# Bullets, and empty/checked tick-boxes: none of these exist in the Bangla font,
# so they must be preserved on the page rather than re-rendered (which would
# show a wrong letter, e.g. a checkbox becoming "T").
BULLET_CHARS = "•·◦▪▫■□‣∙❑❏❐❒☐☑☒✓✔✗✘➤➢▢○●◯"

# A list item: a numbered heading ("9.") or a line the translator's bullet glyph
# still leads. Its continuation lines hang indented under the text rather than
# under the marker, and two lines of that are geometrically indistinguishable
# from two centred ones — the shorter second line is inset on both sides by
# roughly the same amount, which is precisely what centring looks like. Guessing
# from the rects alone read "9. Which statement about exercise..." as centred
# (its two midpoints fell 1.9pt apart) and a tick-box option as right-aligned
# (its two right edges fell 1.4pt apart), both coincidences of where the words
# happened to wrap. A list item is always left-aligned, so it need not be judged
# on geometry at all — while a genuinely centred pull-quote carries no marker
# and is still measured.
LIST_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]\s|[" + re.escape(BULLET_CHARS) + r"])")

# Two lines belong to the same paragraph only if the vertical gap between them
# is below this fraction of the font size (real paragraphs run ~0.4–0.5; separate
# items such as form labels sit at ~0.7+).
MERGE_GAP = 0.6

# Lines of a paragraph run downwards; PDF block order does not. A line starting
# above where the previous one ended belongs to another column, cell or figure —
# never to the same paragraph. Some backtrack has to be allowed, though: a line's
# box runs ascender-to-descender (~1.2x the font size), and leading is routinely
# tighter than that, so consecutive lines of one paragraph genuinely overlap. At
# 17pt on 15pt leading they overlap by 5.5pt — an absolute allowance in points
# rejects that as a column break, splits the cell into two segments, and leaves
# them drawing on top of each other. Measured against the em, not the page.
BACKTRACK_EM = 0.5

# A horizontal hole this wide inside one row separates table columns. The test is
# relative to the font: a word space is about a quarter of the font size, so a gap
# approaching a full em is deliberate. A flat threshold cannot tell the two apart
# across sizes — at 17pt the 20pt gap between a day column and its activity reads
# as a wide space, and fusing them is what turns a seven-row table into one
# paragraph. The floor keeps a tab stop in small print from splitting a sentence.
COLUMN_GAP_EM = 0.9
COLUMN_GAP_MIN = 12.0

# Minimum gap between TOC entry text and its trailing page number.
NUMBER_GAP = 12.0

# A divider line — a table border or a heading underline — is a drawing thin
# enough not to be a box and long enough not to be a glyph artifact. Page
# backgrounds are filled rects hundreds of points thick, so they fail `THICK`
# and are correctly ignored.
RULE_THICK = 3.0
RULE_LONG = 20.0

# A checkbox drawn as line art rather than typed as a glyph: a small, roughly
# square box beside its label. Nothing else here can see one — `_is_bullet` reads
# spans, `_rules` takes only thin-and-long lines, and redaction preserves line art
# — so an undetected box stays put while the options it separates merge into a
# single paragraph and reflow straight across it.
MARK_MIN = 5.0
MARK_MAX = 24.0
MARK_ASPECT = 1.35
# A mark this close to the left of a piece is that piece's bullet.
MARK_GAP = 14.0

# A speech bubble or callout: a filled shape with curved edges, big enough to hold
# text. Unlike a background image — which text is merely laid over, and which must
# not pin the text to its original width — a bubble's edge is a hard boundary:
# grow past it and the text lands on the page background instead of the bubble.
PANEL_MIN_AREA = 8000.0
# A bubble's edge curves away from its bounding box, so text pushed into a corner
# falls outside the ink. Insetting approximates the curve without flattening the
# path; segments never shrink below their original box, so this only caps growth.
PANEL_INSET = 0.08

# How close a growing box may come to whatever sits below it.
#
# The source already leaves a gap between one line and the next — 1.8pt between
# the quiz options — and stopping a full 2pt short of the neighbour threw all of
# it away, leaving each option a box no taller than the single English line it
# replaced. That is what forced the Bangla down to 9pt. Nothing is gained by the
# clearance: insert_htmlbox keeps its ink well inside the box it is handed, so
# two boxes may nearly touch before any glyph does — measured on a 14.2pt option
# box, the ink left 3.7pt clear above it and 2.6pt below.
#
# Reclaiming the gap is what raised the Bangla from 9.0pt to 9.6pt (see
# BANGLA_SIZE): at 0.80 it takes the segments that must shrink from 59 to 15.
BELOW_GAP = 0.3

# Tried in order until the Bangla actually renders. insert_htmlbox draws nothing
# at all when it cannot meet its floor, so a floor it can always meet must come
# last: given no floor it is free to find whatever scale fits.
SCALE_LADDER = (0.6, 0.4, 0.0)


def _span_color_to_css(color_int: int) -> str:
    return f"#{color_int:06x}"


def css_for(seg: dict) -> str:
    """The CSS a segment renders with, in both the translate and fix pipelines.

    Shared rather than duplicated because the two must agree: fix re-renders from
    the manifest, and a fix pipeline that sized text even slightly differently
    would redraw every segment it touched at a size its neighbours don't share.

    `seg` holds the *source* size; BANGLA_SIZE is applied here, at render time,
    and never stored — so the manifest keeps recording the English size it
    measured and re-rendering an already-translated page stays idempotent.
    """
    return CSS_TEMPLATE.format(
        size=seg["size"] * BANGLA_SIZE,
        color=_span_color_to_css(seg["color"]),
        align=seg["align"],
        bold="font-weight: bold;" if seg["bold"] else "",
    )


def _rules(page: fitz.Page) -> list[fitz.Rect]:
    """Divider lines on the page: table borders, cell rules, heading underlines.

    Redaction preserves line art, so a table's borders survive translation — but
    they constrain nothing unless the geometry is read back, and text grown for
    the longer Bangla runs straight through them. Treating them as obstacles is
    what keeps a cell's text inside its cell.

    Deliberately not `page.find_tables()`: on a page whose only drawings are a
    background fill and printer's crop marks, that reports a table covering the
    whole page, which would put every segment on the page in one "cell".
    """
    rules = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        horizontal = rect.height <= RULE_THICK and rect.width >= RULE_LONG
        vertical = rect.width <= RULE_THICK and rect.height >= RULE_LONG
        if not (horizontal or vertical):
            continue
        # A hairline stroke has zero width or height, and a zero-area Rect is
        # "empty": it never intersects anything and would drop out of every
        # obstacle test. Give each rule a minimum thickness so it behaves.
        if horizontal:
            rect = fitz.Rect(rect.x0, rect.y0 - 0.5, rect.x1, rect.y1 + 0.5)
        else:
            rect = fitz.Rect(rect.x0 - 0.5, rect.y0, rect.x1 + 0.5, rect.y1)
        rules.append(rect)
    return rules


def _vector_marks(page: fitz.Page) -> list[fitz.Rect]:
    """Checkbox-shaped line art: candidates for a bullet that was never typed.

    Returned as candidates, not as bullets: a page's corner registration marks are
    the same size and shape as a tick-box. What separates them is having a label
    beside them, which only the row knows — see `_mark_bullets`.
    """
    marks = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if not (
            MARK_MIN <= rect.width <= MARK_MAX and MARK_MIN <= rect.height <= MARK_MAX
        ):
            continue
        if max(rect.width, rect.height) > MARK_ASPECT * min(rect.width, rect.height):
            continue
        marks.append(rect)
    return marks


def _panels(page: fitz.Page) -> list[fitz.Rect]:
    """Speech bubbles and callout panels: filled shapes with curved edges.

    A bubble's outline is a closed curve, so `_rules` — which looks for lines —
    cannot see it, and redaction preserves it. Without this the planner grows a
    quote's box straight through the bubble it lives in.
    """
    panels = []
    for drawing in page.get_drawings():
        if drawing["type"] not in ("f", "fs"):
            continue
        if not any(item[0] == "c" for item in drawing["items"]):
            continue
        rect = fitz.Rect(drawing["rect"])
        if rect.get_area() >= PANEL_MIN_AREA:
            panels.append(rect)
    return panels


def _panel_home(panels: list[fitz.Rect], rect: fitz.Rect) -> fitz.Rect | None:
    """The inner safe area of the smallest panel `rect` sits in, if any.

    Membership is judged by the centre: a bubble is sized to its English text, so
    that text often grazes the outline it is meant to sit inside.
    """
    middle = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    home = None
    for panel in panels:
        if middle not in panel:
            continue
        if home is None or panel.get_area() < home.get_area():
            home = panel
    if home is None:
        return None
    return home + (
        PANEL_INSET * home.width,
        PANEL_INSET * home.height,
        -PANEL_INSET * home.width,
        -PANEL_INSET * home.height,
    )


def _mark_bullets(
    row_rect: fitz.Rect, pieces: list[dict], marks: list[fitz.Rect]
) -> list[fitz.Rect]:
    """Flag pieces whose bullet is a vector box, and return the marks that matched.

    Being flagged is what makes such a piece `standalone`, and therefore what stops
    a column of tick-box options merging into one run-on paragraph.
    """
    used = set()
    for piece in pieces:
        for idx, mark in enumerate(marks):
            if idx in used:
                continue
            overlap = min(mark.y1, row_rect.y1) - max(mark.y0, row_rect.y0)
            if overlap <= 0.5 * min(mark.height, row_rect.height):
                continue
            if -1.0 <= piece["rect"].x0 - mark.x1 <= MARK_GAP:
                piece["bullet"] = True
                used.add(idx)
                break
    return [marks[idx] for idx in sorted(used)]


def _rows(block: dict) -> list[dict]:
    """Group a block's lines into visual rows (lines sharing a baseline).

    TOC entries store the title and the page number as two separate "lines"
    at the same height; printers' marks are sometimes duplicated verbatim.
    """
    lines = []
    seen = set()
    for line in block["lines"]:
        text = "".join(s["text"] for s in line["spans"])
        if not text.strip():
            continue
        bbox = line["bbox"]
        key = (
            re.sub(r"\s+", " ", text),
            round(bbox[0]),
            round(bbox[1]),
            round(bbox[2]),
            round(bbox[3]),
        )
        if key in seen:  # duplicated line (print artifacts)
            continue
        seen.add(key)
        lines.append(line)
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))

    rows = []
    for line in lines:
        rect = fitz.Rect(line["bbox"])
        if rows:
            last = rows[-1]["rect"]
            overlap = min(last.y1, rect.y1) - max(last.y0, rect.y0)
            if overlap > 0.5 * min(last.height, rect.height):
                rows[-1]["spans"].extend(line["spans"])
                rows[-1]["rect"] |= rect
                continue
        rows.append({"spans": list(line["spans"]), "rect": rect})
    for row in rows:
        row["spans"].sort(key=lambda s: s["bbox"][0])
    return rows


def _is_bullet(span: dict) -> bool:
    """True if a span is a decorative bullet/checkbox glyph, not readable text."""
    text = span["text"].strip()
    return bool(SYMBOL_FONT.search(span["font"])) or (
        len(text) == 1 and text in BULLET_CHARS
    )


def _parse_row(row: dict) -> tuple[list[dict], list[dict], dict | None]:
    """Split a row into (bullet spans, text column pieces, page-number span).

    Every column piece may carry its own leading bullet/checkbox glyph — grids
    of tick-boxes repeat one per column — so a bullet is stripped from the start
    of each piece, not just from the first span of the row. Stripped glyphs are
    returned so they can be preserved on the page instead of re-rendered in the
    Bangla font (which has no box/tick glyphs and would show a wrong letter).
    """
    spans = [s for s in row["spans"] if s["text"].strip()]

    number = None
    if spans and PAGE_NUMBER.match(spans[-1]["text"].strip()):
        prev_x1 = max((s["bbox"][2] for s in spans[:-1]), default=None)
        if prev_x1 is None or spans[-1]["bbox"][0] - prev_x1 > NUMBER_GAP:
            number = spans[-1]
            spans = spans[:-1]

    # Group spans into column pieces separated by wide horizontal gaps.
    groups = []
    for span in spans:
        if groups:
            prev = groups[-1][-1]
            limit = max(COLUMN_GAP_MIN, COLUMN_GAP_EM * max(prev["size"], span["size"]))
            if span["bbox"][0] - prev["bbox"][2] <= limit:
                groups[-1].append(span)
                continue
        groups.append([span])

    bullets = []
    pieces = []
    for group in groups:
        had_bullet = False
        while len(group) > 1 and _is_bullet(group[0]):
            bullets.append(group.pop(0))
            had_bullet = True
        if len(group) == 1 and _is_bullet(group[0]):
            bullets.append(group[0])  # the whole piece is just a bullet glyph
            continue
        piece = {"spans": group, "bullet": had_bullet}
        piece["rect"] = fitz.Rect(group[0]["bbox"])
        for span in group[1:]:
            piece["rect"] |= fitz.Rect(span["bbox"])
        piece["text"] = re.sub(r"\s+", " ", "".join(s["text"] for s in group)).strip()
        pieces.append(piece)
    return bullets, pieces, number


def _detect_align(line_rects: list[fitz.Rect], text: str, bullet: bool) -> str:
    """Infer text alignment from how a segment's lines line up.

    A list item is settled by what it is, not by where its lines fell: see
    LIST_MARKER for why its geometry cannot be told from centred text.
    """
    if bullet or LIST_MARKER.match(text):
        return "left"
    if len(line_rects) < 2:
        return "left"
    x0s = [r.x0 for r in line_rects]
    x1s = [r.x1 for r in line_rects]
    mids = [(r.x0 + r.x1) / 2 for r in line_rects]
    if max(x0s) - min(x0s) < 3:
        return "left"
    if max(x1s) - min(x1s) < 3:
        return "right"
    if max(mids) - min(mids) < 4:
        return "center"
    return "left"


def _extract_segments(
    page: fitz.Page, marks: list[fitz.Rect]
) -> tuple[list[dict], list[fitz.Rect]]:
    """Return (translatable segments, rects of text left untouched on the page).

    A segment is one translation/insertion unit: a paragraph, a heading, a
    bullet item, or a single TOC entry. Bullet glyphs and TOC page numbers are
    never part of a segment — they stay on the page unchanged.

    `marks` are checkbox-shaped drawings from `_vector_marks`. Only those that
    turn out to label a piece are kept: the rest are page furniture, and adding
    them would drag the page's text margins out to wherever they sit.
    """
    segments = []
    kept = []  # bullets, page numbers, digit-only rows: untouched, but obstacles

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        block_x1 = block["bbox"][2]
        prev = None  # last segment in this block that may accept continuations

        for row in _rows(block):
            bullets, pieces, number = _parse_row(row)
            for bullet in bullets:
                kept.append(fitz.Rect(bullet["bbox"]))
            if number is not None:
                kept.append(fitz.Rect(number["bbox"]))
            kept.extend(_mark_bullets(row["rect"], pieces, marks))

            for idx, piece in enumerate(pieces):
                if not HAS_LETTERS.search(piece["text"]):
                    kept.append(piece["rect"])
                    prev = None
                    continue

                main = max(piece["spans"], key=lambda s: len(s["text"]))
                size = main["size"]
                bold = bool(main["flags"] & BOLD_FLAG)
                color = main["color"]
                num = number if idx == len(pieces) - 1 else None
                standalone = piece["bullet"] or num is not None or len(pieces) > 1

                if (
                    not standalone
                    and prev is not None
                    and prev["number"] is None
                    and abs(size - prev["size"]) <= 0.12 * prev["size"]
                    and -BACKTRACK_EM * size
                    <= piece["rect"].y0 - prev["rect"].y1
                    < MERGE_GAP * size
                    and bold == prev["bold"]
                    and color == prev["color"]
                    and min(piece["rect"].x1, prev["rect"].x1)
                    - max(piece["rect"].x0, prev["rect"].x0)
                    > 0.3 * min(piece["rect"].width, prev["rect"].width)
                ):
                    prev["text"] += " " + piece["text"]
                    prev["rect"] |= piece["rect"]
                    prev["line_rects"].append(piece["rect"])
                    continue

                segment = {
                    "text": piece["text"],
                    "rect": fitz.Rect(piece["rect"]),
                    "line_rects": [fitz.Rect(piece["rect"])],
                    "size": size,
                    "bold": bold,
                    "color": color,
                    "number": num,
                    "bullet": piece["bullet"],
                    "block_x1": block_x1,
                }
                segments.append(segment)
                # Bullet items accept plain continuation lines; numbered ones
                # don't. Neither does a row that split into columns: `prev` would
                # be its rightmost cell, and the next full-width line overlaps a
                # narrow cell easily enough to merge into it and span the row.
                prev = segment if num is None and len(pieces) == 1 else None

    # Merge across blocks: consecutive same-style segments separated by less
    # than a line gap are one flowing paragraph (e.g. the cover tagline, which
    # PyMuPDF splits into one block per line). Blocks arrive in content order,
    # which is not reading order, so the gap must be checked in both directions:
    # an unbounded test reads a segment hundreds of points *above* the previous
    # one as a continuation and unions the two into one tall, narrow box.
    merged = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev["number"] is None
            and seg["number"] is None
            and not seg["bullet"]  # a bullet always starts a new item
            and abs(seg["size"] - prev["size"]) <= 0.12 * prev["size"]
            and -BACKTRACK_EM * seg["size"]
            <= seg["rect"].y0 - prev["rect"].y1
            < MERGE_GAP * seg["size"]
            and seg["bold"] == prev["bold"]
            and seg["color"] == prev["color"]
            and min(seg["rect"].x1, prev["rect"].x1) - max(seg["rect"].x0, prev["rect"].x0)
            > 0.3 * min(seg["rect"].width, prev["rect"].width)
        ):
            prev["text"] += " " + seg["text"]
            prev["rect"] |= seg["rect"]
            prev["line_rects"].extend(seg["line_rects"])
            continue
        merged.append(seg)

    for seg in merged:
        seg["align"] = _detect_align(seg["line_rects"], seg["text"], seg["bullet"])
    return merged, kept


def _plan_insert_rects(
    page: fitz.Page,
    segments: list[dict],
    kept: list[fitz.Rect],
    rules: list[fitz.Rect],
    panels: list[fitz.Rect],
) -> None:
    """Give each segment breathing room for the (usually longer) Bangla text.

    Rects only grow into space that is genuinely free: never across a table
    border or other divider line, into another segment, into a preserved
    bullet/number, or into an image the original text did not already sit on.
    """
    images = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
    # Only an image may be sat upon: text is routinely laid over a background
    # picture, and treating that picture as an obstacle would pin the text to
    # its original width. Everything else blocks even when it already touches —
    # PyMuPDF line boxes run ascender-to-descender and graze their neighbour by
    # a fraction of a point at tight leading, and excusing that overlap is what
    # lets a box grow a full line down over the text below it.
    solid = kept + rules + [seg["rect"] for seg in segments]
    # Never grow past the rightmost text on the page (stay inside its margins).
    text_x1 = max(r.x1 for r in kept + [seg["rect"] for seg in segments])

    for seg in segments:
        base = seg["rect"]
        others = [o for o in solid if o is not base and not o.is_empty]
        others += [i for i in images if not i.intersects(base) and not i.is_empty]
        # The one shape a segment may sit on but not leave. Text already inside a
        # bubble was fitted to it by the original layout, so this mostly forbids
        # growth outright — which is the point: there is nowhere for it to go.
        home = _panel_home(panels, base)

        # Horizontal: single lines may grow right (labels, captions, TOC titles);
        # paragraphs may reclaim their block's full width. Bangla is wider than
        # English and not proportional to a short label's width, so also grant an
        # absolute minimum of extra room (bounded below by obstacles/margins).
        if seg["number"] is not None:
            target_x1 = seg["number"]["bbox"][0] - 4
        elif len(seg["line_rects"]) == 1:
            target_x1 = base.x1 + max(0.6 * base.width, 10 * seg["size"])
        else:
            target_x1 = max(base.x1, seg["block_x1"])
        limit_x1 = min(page.rect.x1 - 20, text_x1 + 2)
        if home is not None:
            limit_x1 = min(limit_x1, home.x1)
        for o in others:
            if o.x0 >= base.x1 - 1 and min(o.y1, base.y1) - max(o.y0, base.y0) > 1:
                limit_x1 = min(limit_x1, o.x0 - 4)
        new_x1 = max(base.x1, min(target_x1, limit_x1))

        # Vertical: about one extra line of height, stopping above whatever
        # follows, so the taller Bangla text rarely has to shrink to fit.
        target_y1 = base.y1 + max(1.0 * seg["size"], 0.5 * base.height)
        limit_y1 = page.rect.y1 - 16
        if home is not None:
            limit_y1 = min(limit_y1, home.y1)
        for o in others:
            if o.y0 >= base.y1 - 1 and min(o.x1, new_x1) - max(o.x0, base.x0) > 1:
                limit_y1 = min(limit_y1, o.y0 - BELOW_GAP)
        new_y1 = max(base.y1, min(target_y1, limit_y1))

        # A small gap after a preserved bullet glyph (the original leading
        # space was stripped from the text).
        x0 = base.x0 + 2 if seg["bullet"] else base.x0
        seg["insert_rect"] = fitz.Rect(x0, base.y0, new_x1, new_y1)


def translate_pdf(pdf_bytes: bytes) -> bytes:
    """Translate all text in a PDF from English to Bangla, preserving layout."""
    import time
    start_time = time.time()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    archive = fitz.Archive(FONTS_DIR)
    pages_meta = []
    source_fonts = set()

    for page_num, page in enumerate(doc, start=1):
        page_start = time.time()
        # Record the page's fonts while the source text is still on it: the fix
        # pipeline tells the text it inserted from the text it must preserve by
        # asking which fonts were already here.
        source_fonts |= manifest.page_span_fonts(page)

        seg_start = time.time()
        segments, kept = _extract_segments(page, _vector_marks(page))
        seg_time = time.time() - seg_start

        if not segments:
            logger.debug("Page %d: no translatable segments (%.2fs)", page_num, seg_time)
            continue

        logger.debug(
            "Page %d: extracted %d segments in %.2fs",
            page_num, len(segments), seg_time
        )

        english = [seg["text"] for seg in segments]
        trans_start = time.time()
        translations, status = translate_batch_status(english)
        trans_time = time.time() - trans_start
        logger.debug("Page %d: translated in %.2fs", page_num, trans_time)
        # A page whose text the API refused is written back in English, which
        # looks exactly like a page the translator chose to skip. Say so, and
        # name the page: the manifest records the same flag, so Fix can retry it.
        failed = status.count(False)
        if failed:
            logger.warning(
                "Page %d: %d of %d segments left in English — the translation "
                "request failed. Run Fix on the output to retry them.",
                page_num,
                failed,
                len(segments),
            )
        _plan_insert_rects(page, segments, kept, _rules(page), _panels(page))

        # Remove only the translatable text: bullets, page numbers and images
        # are left untouched. Rects are shrunk a hair so redaction never bites
        # into an adjacent preserved glyph.
        for seg in segments:
            for line_rect in seg["line_rects"]:
                page.add_redact_annot(line_rect + (0.3, 0.3, -0.3, -0.3), fill=False)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )

        metrics = []
        for seg, translated in zip(segments, translations):
            css = css_for(seg)
            body = html.escape(translated)
            # The first floor keeps the font readable at 60%, and _plan_insert_rects
            # has already grown the rect so most text meets it without shrinking.
            # But a rect boxed in by table borders cannot grow, and insert_htmlbox
            # draws *nothing at all* when it cannot meet its floor — so drop the
            # floor rather than lose the text, and let the fix pipeline shorten
            # whatever ends up below READABLE_FLOOR. A failed call leaves no ink,
            # so retrying on the same page renders exactly as a clean pass would.
            for low in SCALE_LADDER:
                spare_height, scale = page.insert_htmlbox(
                    seg["insert_rect"], body, css=css, scale_low=low, archive=archive
                )
                if spare_height >= 0:
                    break
            metrics.append((spare_height, scale))
            if 0 < scale <= SCALE_LADDER[0]:
                logger.warning(
                    "Page %d: text hit the %.0f%% shrink floor and may overflow: %r at %s",
                    page_num,
                    scale * 100,
                    translated[:40],
                    seg["insert_rect"],
                )

        pages_meta.append(
            manifest.page_entry(
                page_num, segments, kept, english, translations, metrics, status
            )
        )

    # The manifest lets the fix pipeline re-render any segment later; the Bangla
    # on the page itself is unreadable once shaped.
    data = manifest.build(source_fonts, pages_meta)
    if data["fallback_collision"]:
        logger.warning(
            "Source PDF uses %s, which MuPDF also picks as a fallback font: the fix "
            "pipeline cannot tell inserted text from original text and will skip this "
            "document.",
            ", ".join(data["fallback_collision"]),
        )
    manifest.attach(doc, data)

    doc.subset_fonts()
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    total_time = time.time() - start_time
    logger.info("PDF translation complete: %.1f seconds", total_time)

    return out
