"""PDF translation pipeline: extract text segments, translate, redact, re-insert Bangla."""

import html
import logging
import math
import os
import re

import fitz  # PyMuPDF

import copy_layer
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
#
# Raised to 0.90 for print legibility at the user's request: the on-screen size
# was too small once printed. This trades uniformity for size — well past the 0.80
# ceiling, so more segments (roughly 100+, extrapolating from the 15→70 jump)
# shrink-to-fit and dense pages such as the quiz option lists may show text at
# mismatched sizes. Lower this back toward 0.80 if that crowding is unacceptable.
BANGLA_SIZE = 0.90

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

# A *justified* line defeats the column test above. Justification widens every
# space in the line until it reaches both margins, and a line holding few long
# words gets widened a lot: "One particular type of" is set across a 154pt
# column with 13.3pt between the words. PyMuPDF reports each word as its own
# line at that point, `_rows` puts them back on one row, and the column test
# then reads three ordinary word spaces as three column breaks.
#
# What follows is the damage, not a cosmetic split: each word becomes its own
# segment, is sent to the translator alone — "One", "particular", "type" carry
# no sentence to translate — and is then planned as a standalone label, which
# earns it `10 * size` of growth to the right, straight across the paragraph it
# was cut out of. Four segments end up drawn on top of each other. This is what
# put the overlapping Bangla on p.147 of the Post-MI manual.
#
# Three things together say "justified line" rather than "table row", measured
# over 923 multi-piece rows in four manuals — they fuse all 41 justified splits
# and leave every genuine table row (day-planner, food table, Likert header,
# TOC, nutrition panel) split:
#
#   * The gaps are *uniform*. One line's spaces are all stretched by the same
#     amount; column positions are set independently and rarely match.
#   * The gaps are *small*. Stretch is bounded by how much slack one line has;
#     2 ems covers every case seen (the widest was 20.5pt at 11pt) while the
#     narrowest genuine column gap sat far above it.
#   * The row is set to the full measure, and it is not the only row in its
#     block that is. That is what justification means, and it is what a table
#     row cannot fake: a table's rows end wherever their last cell ends, so
#     only the widest of them reaches the right margin.
JUSTIFY_STRETCH_EM = 2.0
JUSTIFY_UNIFORM = 1.3
JUSTIFY_FLUSH = 1.5

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

# A bubble is a *simple* closed path. Every speech bubble in the source manuals
# is 3, 4 or 8 path items — an ellipse is exactly 4 cubic Beziers — while the
# cover cartoons and illustrations that also read as "large filled shape with a
# curve" run 22 to 3007 items. The cut cleanly separates 42 bubbles from 121
# pieces of artwork across the three manuals.
#
# Only a bubble earns the treatment below: its outline is a hard boundary the
# text must stay inside, and it is small and regular enough that a rectangle
# centred in it is a fair approximation of the space available. Artwork keeps
# the looser PANEL_INSET cap, which is all it ever needed.
BUBBLE_MAX_ITEMS = 12
# Resolution of the shape mask, in pixels per point. The mask only has to
# resolve an edge to within a quarter point; 4 reproduces the analytic ellipse
# result to 0.1pt.
BUBBLE_MASK_DPI = 4
# Ink kept clear inside the outline, so a glyph never grazes the border. Mostly
# it covers the mask's own quarter-point quantisation — insert_htmlbox already
# keeps its ink 2.6-3.7pt inside the box it is handed (see BELOW_GAP). Measured
# across five bubbles, going from 2.0 to 1.0 buys back 0.2-0.4pt of type.
BUBBLE_MARGIN = 1.0
# The candidate rectangles, as an angle sweep: half-extents run
# (w/2 * cos t, h/2 * sin t), so a low angle is wide and short and a high one is
# narrow and tall. A one-line caption wants the first, a seven-line quote the
# last, and which of them a segment wants is not known until it is translated.
BUBBLE_ANGLES = tuple(range(20, 76, 5))
# Points sampled along each edge when testing a candidate against the mask.
BUBBLE_EDGE_SAMPLES = 20

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

# Sweeps of `_uncross`. Retracting one box can only ever free space for the
# rest, so the pass converges, and it stops early as soon as a sweep changes
# nothing. Across all 407 pages of the four manuals one sweep always settled
# the page (67 pages needed it at all); the spare passes are for the page where
# retracting one box first has to reveal the next pair.
UNCROSS_PASSES = 3

# Tried in order until the Bangla actually renders. insert_htmlbox draws nothing
# at all when it cannot meet its floor, so a floor it can always meet must come
# last: given no floor it is free to find whatever scale fits.
SCALE_LADDER = (0.6, 0.4, 0.0)

# How much of a segment an image must cover before the segment counts as printed
# *on* it. Text laid over a background picture has to stay free to grow, or it
# would be pinned to the width the English happened to need — but that licence
# belongs to backdrops only. See `_sits_on`.
BACKDROP_COVER = 0.9


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


def _divider_between(upper: fitz.Rect, lower: fitz.Rect, rules: list[fitz.Rect]) -> bool:
    """True if a horizontal rule separates two vertically-stacked boxes.

    The merge tests in `_extract_segments` fuse two pieces on font/gap/overlap
    geometry alone. In a table whose columns are too close to split a row into
    separate pieces, that let one cell absorb the row stacked below it — the
    drawn cell border between them was never consulted, because `_rules` reached
    only the box planner. This is that consultation: a row border sitting in the
    gap between two boxes, and spanning the width they share, is a hard divider
    and blocks the merge.
    """
    left = max(upper.x0, lower.x0)
    right = min(upper.x1, lower.x1)
    if right <= left:  # no horizontal overlap — not stacked in the same column
        return False
    top = min(upper.y1, lower.y1)
    bottom = max(upper.y0, lower.y0)
    for rule in rules:
        if rule.width < rule.height:  # a column border, not a row border
            continue
        mid_y = (rule.y0 + rule.y1) / 2
        if not (top - 1 <= mid_y <= bottom + 1):
            continue
        # The rule must actually run across the shared column, not merely touch
        # its edge — a stray tick beside the boxes is not a divider.
        if min(rule.x1, right) - max(rule.x0, left) > 0.5 * (right - left):
            return True
    return False


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
    """Speech bubbles and callout panels: filled shapes that contain text.

    A bubble's outline is a closed curve, so `_rules` — which looks for lines —
    cannot see it, and redaction preserves it. Without this the planner grows a
    quote's box straight through the bubble it lives in.

    A plain rectangle fill counts too — a highlight/"ACTION" box is drawn as a
    single `re` path with no curve at all, and without this a box's own text
    grew past its right and bottom edges onto the page background: 17.5-19.5pt
    below the ACTION panels and 25-31pt past the warning box's right edge on
    pp.35/64 of the Heart Failure manual. Restricted to a *single* `re` item so
    multi-path vector artwork (logos, illustrations) isn't misread as a box.
    """
    panels = []
    for drawing in page.get_drawings():
        if drawing["type"] not in ("f", "fs"):
            continue
        items = drawing["items"]
        has_curve = any(item[0] == "c" for item in items)
        plain_rect = len(items) == 1 and items[0][0] == "re"
        if not (has_curve or plain_rect):
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


def _bubbles(page: fitz.Page) -> list[dict]:
    """Speech bubbles: the simple closed shapes among `_panels`' candidates.

    Returned as the drawing dicts rather than their rects — the path items are
    what the mask below is built from, and a bubble's bounding box is exactly
    the thing that misleads about how much room it has.
    """
    found = []
    for drawing in page.get_drawings():
        if drawing["type"] not in ("f", "fs"):
            continue
        items = drawing["items"]
        if len(items) > BUBBLE_MAX_ITEMS:
            continue
        if not any(item[0] == "c" for item in items):
            continue
        if fitz.Rect(drawing["rect"]).get_area() >= PANEL_MIN_AREA:
            found.append(drawing)
    return found


def _bubble_mask(page: fitz.Page, drawing: dict) -> tuple[fitz.Rect, fitz.Pixmap]:
    """Redraw one bubble's path, filled, and read it back as a coverage bitmap.

    Rasterising rather than reasoning about the path keeps this honest for any
    outline the source cares to use — ellipse, rounded rectangle, cloud — and a
    bubble whose tail is part of the same path needs no special case: the tail
    is simply too narrow for any candidate rectangle to reach into.
    """
    rect = fitz.Rect(drawing["rect"])
    scratch = fitz.open()
    canvas = scratch.new_page(width=page.rect.x1, height=page.rect.y1)
    shape = canvas.new_shape()
    for item in drawing["items"]:
        op = item[0]
        if op == "l":
            shape.draw_line(item[1], item[2])
        elif op == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        elif op == "re":
            shape.draw_rect(item[1])
        elif op == "qu":
            shape.draw_quad(item[1])
    shape.finish(
        fill=(0, 0, 0),
        color=None,
        closePath=True,
        even_odd=drawing.get("even_odd", False),
    )
    shape.commit()
    pixmap = canvas.get_pixmap(
        clip=rect,
        matrix=fitz.Matrix(BUBBLE_MASK_DPI, BUBBLE_MASK_DPI),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    scratch.close()
    return rect, pixmap


def _on_ink(rect: fitz.Rect, pixmap: fitz.Pixmap, x: float, y: float) -> bool:
    """Whether a page-space point lands on the bubble's fill."""
    px = int((x - rect.x0) * BUBBLE_MASK_DPI)
    py = int((y - rect.y0) * BUBBLE_MASK_DPI)
    if not (0 <= px < pixmap.width and 0 <= py < pixmap.height):
        return False
    return pixmap.pixel(px, py)[0] < 128


def _rect_on_ink(rect: fitz.Rect, pixmap: fitz.Pixmap, candidate: fitz.Rect) -> bool:
    """Whether every edge of `candidate` lies on the bubble's fill.

    Sampling the perimeter is enough: a rectangle whose whole border sits on a
    bubble-shaped region has its interior there too.
    """
    for i in range(BUBBLE_EDGE_SAMPLES + 1):
        along = i / BUBBLE_EDGE_SAMPLES
        x = candidate.x0 + along * candidate.width
        y = candidate.y0 + along * candidate.height
        if not (
            _on_ink(rect, pixmap, x, candidate.y0)
            and _on_ink(rect, pixmap, x, candidate.y1)
            and _on_ink(rect, pixmap, candidate.x0, y)
            and _on_ink(rect, pixmap, candidate.x1, y)
        ):
            return False
    return True


def _inscribed_rects(rect: fitz.Rect, pixmap: fitz.Pixmap) -> list[fitz.Rect]:
    """Every shape of rectangle that fits inside the bubble, centred in it.

    A single rectangle cannot describe an ellipse's room: trading width for
    height buys back most of what the corners waste, and only the translated
    text knows which trade it wants. So the whole family is returned and the
    render step measures them.

    Centring is the point of the exercise. The English was laid out to fit the
    outline; the taller Bangla can only be laid out to fit it by using the
    headroom above the original box as well as below it.
    """
    mid_x = (rect.x0 + rect.x1) / 2
    mid_y = (rect.y0 + rect.y1) / 2
    family = []
    for degrees in BUBBLE_ANGLES:
        angle = math.radians(degrees)
        # Largest centred rectangle of this shape that the mask still accepts.
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = (low + high) / 2
            half_w = middle * rect.width / 2 * math.cos(angle)
            half_h = middle * rect.height / 2 * math.sin(angle)
            if _rect_on_ink(
                rect,
                pixmap,
                fitz.Rect(mid_x - half_w, mid_y - half_h, mid_x + half_w, mid_y + half_h),
            ):
                low = middle
            else:
                high = middle
        half_w = low * rect.width / 2 * math.cos(angle) - BUBBLE_MARGIN
        half_h = low * rect.height / 2 * math.sin(angle) - BUBBLE_MARGIN
        if half_w > 0 and half_h > 0:
            family.append(
                fitz.Rect(mid_x - half_w, mid_y - half_h, mid_x + half_w, mid_y + half_h)
            )
    return family


def _clip_to_ink(
    rect: fitz.Rect, pixmap: fitz.Pixmap, base: fitz.Rect, planned: fitz.Rect
) -> fitz.Rect | None:
    """The most of `planned` that stays on the bubble's fill, without moving it.

    For a segment that shares its bubble with another, position is not ours to
    change — moving it would stack it on its neighbour. Two ways to give ground,
    tried in that order because only the second costs type size:

    1. Hand back growth. `planned` is `base` grown down and right, and growth is
       free to surrender; trimming it is what a bubble-shaped boundary should
       have been doing all along.
    2. If `base` itself escapes the outline, shrink about its centre. Shrinking
       the *grown* box instead would throw away the original as well as the
       growth — measured on a nutrition pill, 11.9pt of label became 7.5pt.
    """
    inset = (BUBBLE_MARGIN, BUBBLE_MARGIN, -BUBBLE_MARGIN, -BUBBLE_MARGIN)

    def grown(fraction: float) -> fitz.Rect:
        return fitz.Rect(
            planned.x0,
            planned.y0,
            base.x1 + fraction * (planned.x1 - base.x1),
            base.y1 + fraction * (planned.y1 - base.y1),
        )

    if _rect_on_ink(rect, pixmap, grown(0.0) + inset):
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = (low + high) / 2
            if _rect_on_ink(rect, pixmap, grown(middle) + inset):
                low = middle
            else:
                high = middle
        return grown(low) + inset

    mid_x = (base.x0 + base.x1) / 2
    mid_y = (base.y0 + base.y1) / 2
    low, high = 0.0, 1.0
    for _ in range(16):
        middle = (low + high) / 2
        half_w = middle * base.width / 2
        half_h = middle * base.height / 2
        if _rect_on_ink(
            rect,
            pixmap,
            fitz.Rect(mid_x - half_w, mid_y - half_h, mid_x + half_w, mid_y + half_h),
        ):
            low = middle
        else:
            high = middle
    half_w = low * base.width / 2 - BUBBLE_MARGIN
    half_h = low * base.height / 2 - BUBBLE_MARGIN
    if half_w <= 0 or half_h <= 0:
        return None
    return fitz.Rect(mid_x - half_w, mid_y - half_h, mid_x + half_w, mid_y + half_h)


def _bubble_home(
    masks: list[tuple[fitz.Rect, fitz.Pixmap]], rect: fitz.Rect
) -> tuple[fitz.Rect, fitz.Pixmap] | None:
    """The smallest bubble whose *fill* the segment's centre lands on.

    Membership by ink rather than by bounding box. A bounding box reaches well
    past a curved outline, which is how a printer's slug at the page edge ends
    up "inside" a cover cartoon and how a nutrition label gets claimed by the
    neighbouring pill instead of its own.
    """
    mid_x = (rect.x0 + rect.x1) / 2
    mid_y = (rect.y0 + rect.y1) / 2
    home = None
    for bubble_rect, pixmap in masks:
        if not _on_ink(bubble_rect, pixmap, mid_x, mid_y):
            continue
        if home is None or bubble_rect.get_area() < home[0].get_area():
            home = (bubble_rect, pixmap)
    return home


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


def _piece(spans: list[dict], bullet: bool) -> dict:
    """One column piece: its spans, its bounding box and its text."""
    rect = fitz.Rect(spans[0]["bbox"])
    for span in spans[1:]:
        rect |= fitz.Rect(span["bbox"])
    return {
        "spans": spans,
        "bullet": bullet,
        "rect": rect,
        "text": re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip(),
    }


def _full_measure(rect: fitz.Rect, block_rect: fitz.Rect) -> bool:
    """True if a row is set to the full measure — flush with its block's right edge.

    Only the right margin is asked about. That is the one justification actually
    controls; the left is free to move under a bullet or a hanging indent. The
    bulleted item on p.78 of the Revascularisation manual sets its first line
    from the bullet and indents the continuations past it, so no two of its rows
    share a left edge and the paragraph did not read as justified at all — the
    one case in four manuals that this pass still left broken. Dropping the
    left-margin test fuses it and changes no other row of the 923 measured.
    """
    return block_rect.x1 - rect.x1 <= JUSTIFY_FLUSH


def _is_justified(
    row: dict, pieces: list[dict], block_rect: fitz.Rect, full_lines: int
) -> bool:
    """True if a row's pieces are the words of one justified line, not columns.

    See JUSTIFY_STRETCH_EM for what each test is doing and what it was measured
    against. All of them have to agree: any one of them alone also fires on a
    genuine table row somewhere in the manuals.
    """
    if len(pieces) < 2 or full_lines < 2:
        return False
    if not _full_measure(row["rect"], block_rect):
        return False
    gaps = [b["rect"].x0 - a["rect"].x1 for a, b in zip(pieces, pieces[1:])]
    if min(gaps) <= 0:
        return False  # pieces that touch or overlap were never stretched apart
    size = max(s["size"] for piece in pieces for s in piece["spans"])
    if max(gaps) > JUSTIFY_STRETCH_EM * size:
        return False
    if max(gaps) > JUSTIFY_UNIFORM * min(gaps):
        return False
    styles = {
        (round(s["size"], 1), s["font"], s["color"])
        for piece in pieces
        for s in piece["spans"]
    }
    return len(styles) == 1


def _parse_row(
    row: dict, block_rect: fitz.Rect, full_lines: int
) -> tuple[list[dict], list[dict], dict | None]:
    """Split a row into (bullet spans, text column pieces, page-number span).

    Every column piece may carry its own leading bullet/checkbox glyph — grids
    of tick-boxes repeat one per column — so a bullet is stripped from the start
    of each piece, not just from the first span of the row. Stripped glyphs are
    returned so they can be preserved on the page instead of re-rendered in the
    Bangla font (which has no box/tick glyphs and would show a wrong letter).

    `block_rect` and `full_lines` (how many of the block's rows run its full
    width) describe the block the row came from, and are only used to recognise
    a justified line that the column test has cut into its separate words.
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
        pieces.append(_piece(group, had_bullet))

    if _is_justified(row, pieces, block_rect, full_lines):
        # One line of one paragraph, not a row of cells: put its words back
        # together so the translator sees a sentence and the planner sees a
        # single box. Justification stretches every space alike, so the whole
        # row fuses or none of it does. The words are rejoined with a space of
        # our own rather than by concatenating the spans: the space that was
        # stretched is not reliably part of either fragment's text.
        fused = _piece(
            [span for piece in pieces for span in piece["spans"]],
            pieces[0]["bullet"],
        )
        fused["text"] = " ".join(piece["text"] for piece in pieces)
        pieces = [fused]
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
    # Table borders block a merge (see `_divider_between`): two cells stacked in
    # one column must not fuse just because their columns sat too close to split
    # the row. Computed here as well as in `_plan_insert_rects`; a second pass over
    # the drawings is negligible beside the translation request that follows.
    rules = _rules(page)

    segments = []
    kept = []  # bullets, page numbers, digit-only rows: untouched, but obstacles

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        block_x1 = block["bbox"][2]
        prev = None  # last segment in this block that may accept continuations

        rows = _rows(block)
        block_rect = fitz.Rect(block["bbox"])
        full_lines = sum(1 for r in rows if _full_measure(r["rect"], block_rect))

        for row in rows:
            bullets, pieces, number = _parse_row(row, block_rect, full_lines)
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
                    and not _divider_between(prev["rect"], piece["rect"], rules)
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
            and not _divider_between(prev["rect"], seg["rect"], rules)
        ):
            prev["text"] += " " + seg["text"]
            prev["rect"] |= seg["rect"]
            prev["line_rects"].extend(seg["line_rects"])
            continue
        merged.append(seg)

    for seg in merged:
        seg["align"] = _detect_align(seg["line_rects"], seg["text"], seg["bullet"])
    return merged, kept


def _sits_on(rect: fitz.Rect, image: fitz.Rect) -> bool:
    """True if `rect` is printed over `image` — i.e. the image is this segment's
    backdrop and must not block its growth.

    The distinction matters because the two cases look alike from a bounding box.
    A caption inside a speech bubble, or a paragraph over a full-bleed panel, is
    genuinely laid on the picture and has nowhere else to go. A body paragraph
    whose last line grazes the top edge of a figure is *not* on it, and treating
    it as though it were let the box grow a full line down across the artwork —
    82pt of Bangla over the photo on p.114 of the Heart Failure manual.
    """
    if rect.is_empty or image.is_empty:
        return False
    return (rect & image).get_area() >= BACKDROP_COVER * rect.get_area()


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

    # A bubble is the one place the planner may move a box rather than only grow
    # it, so work out who lives in which before planning anything. A bubble with
    # a single tenant is re-centred in it; one holding several (a two-line title,
    # a label above its value) is only clamped, since re-centring would stack
    # them on top of each other.
    masks = [_bubble_mask(page, drawing) for drawing in _bubbles(page)]
    homes = {}
    tenants = {}
    for seg in segments:
        home = _bubble_home(masks, seg["rect"])
        if home is not None:
            homes[id(seg)] = home
            tenants[id(home[0])] = tenants.get(id(home[0]), 0) + 1

    for seg in segments:
        base = seg["rect"]
        bubble = homes.get(id(seg))
        if bubble is not None and tenants[id(bubble[0])] == 1:
            # The whole bubble is this segment's to use. Hand the render step the
            # family and let it pick once the Bangla is known; the roomiest is
            # the right default for anything that never asks.
            inscribed = _inscribed_rects(*bubble)
            if inscribed:
                seg["bubble_rects"] = inscribed
                seg["insert_rect"] = max(inscribed, key=lambda r: r.get_area())
                continue
        others = [o for o in solid if o is not base and not o.is_empty]
        # Foreground pictures: every image except the one this segment is printed
        # on. Handled separately from `others` below, because an image that
        # already overlaps `base` still has to block — the old test excused any
        # image the text touched at all, which is how a paragraph came to grow
        # down across a figure it had merely grazed.
        pictures = [i for i in images if not i.is_empty and not _sits_on(base, i)]
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
        # Anything reaching further right than this box, on the same lines as it,
        # stops it — including a neighbour that *starts* to the left of the box's
        # own right edge. The old test only saw obstacles beginning at or beyond
        # `base.x1`, which let a box grow clean through a paragraph it was
        # interleaved with: the running head at the top of a Post-MI page grew
        # 146pt right across the chapter title beside it. Clamping to `base.x1`
        # rather than to the obstacle's edge keeps an already-overlapping
        # neighbour from *shrinking* the box — growth is refused, never reversed,
        # which is the same rule the pictures below have always followed.
        for o in others:
            if o.x1 > base.x1 and min(o.y1, base.y1) - max(o.y0, base.y0) > 1:
                limit_x1 = min(limit_x1, max(base.x1, o.x0 - 4))
        for i in pictures:
            if i.x1 > base.x1 and min(i.y1, base.y1) - max(i.y0, base.y0) > 1:
                limit_x1 = min(limit_x1, max(base.x1, i.x0 - 4))
        new_x1 = max(base.x1, min(target_x1, limit_x1))

        # Vertical: about one extra line of height, stopping above whatever
        # follows, so the taller Bangla text rarely has to shrink to fit.
        target_y1 = base.y1 + max(1.0 * seg["size"], 0.5 * base.height)
        limit_y1 = page.rect.y1 - 16
        if home is not None:
            limit_y1 = min(limit_y1, home.y1)
        # Same rule downwards: anything reaching below this box, in the columns
        # it occupies, stops it — a caption sitting inside a paragraph's own
        # bounding box blocks growth just as a paragraph below it does.
        for o in others:
            if o.y1 > base.y1 and min(o.x1, new_x1) - max(o.x0, base.x0) > 1:
                limit_y1 = min(limit_y1, max(base.y1, o.y0 - BELOW_GAP))
        for i in pictures:
            if i.y1 > base.y1 and min(i.x1, new_x1) - max(i.x0, base.x0) > 1:
                limit_y1 = min(limit_y1, max(base.y1, i.y0 - BELOW_GAP))
        new_y1 = max(base.y1, min(target_y1, limit_y1))

        # A small gap after a preserved bullet glyph (the original leading
        # space was stripped from the text).
        x0 = base.x0 + 2 if seg["bullet"] else base.x0
        seg["insert_rect"] = fitz.Rect(x0, base.y0, new_x1, new_y1)

        # Sharing a bubble: keep the planned position, but pull the box back
        # inside the outline. Refuse a clamp that would cost more than half the
        # box — overflowing a bubble is bad, vanishing into it is worse.
        if bubble is not None:
            clipped = _clip_to_ink(*bubble, base, seg["insert_rect"])
            if clipped is not None and clipped.get_area() >= 0.5 * base.get_area():
                seg["insert_rect"] = clipped

    _uncross(segments)


def _uncross(segments: list[dict]) -> None:
    """Take back any growth that ran one segment's box into another's.

    Every box above is planned on its own, against where the *source* text sat.
    That is enough to stop a box growing into an occupied space, but not enough
    to stop two boxes meeting in an empty one: the chapter title on a Post-MI
    page grows 280pt right while the running head beside it grows a line down,
    and they cross in the white space between them. Neither could see it coming
    — each was still clear of the other's source rect when it was planned.

    So the pair is settled afterwards, and only ever by giving growth back.
    Boxes grow right and down only, so a pair that the source kept apart was
    crossed by whichever of them sat on the near side of the gap; it is pulled
    back to the edge of that gap, never past its own source rect. A pair the
    source *already* overlapped is left alone — nothing here caused it, and the
    fix pipeline's `_deconflict` is where that case is answered.
    """
    for _ in range(UNCROSS_PASSES):
        settled = True
        for i, a in enumerate(segments):
            for b in segments[i + 1 :]:
                shared = a["insert_rect"] & b["insert_rect"]
                if shared.is_empty or shared.width <= 0 or shared.height <= 0:
                    continue

                # Each option is (area given back, segment, axis, new edge).
                options = []
                left = None
                if a["rect"].x1 <= b["rect"].x0:
                    left, right = a, b
                elif b["rect"].x1 <= a["rect"].x0:
                    left, right = b, a
                if left is not None:
                    edge = max(left["rect"].x1, right["rect"].x0 - 4)
                    if edge < left["insert_rect"].x1:
                        cost = (left["insert_rect"].x1 - edge) * left["insert_rect"].height
                        options.append((cost, left, "x", edge))

                upper = None
                if a["rect"].y1 <= b["rect"].y0:
                    upper, lower = a, b
                elif b["rect"].y1 <= a["rect"].y0:
                    upper, lower = b, a
                if upper is not None:
                    edge = max(upper["rect"].y1, lower["rect"].y0 - BELOW_GAP)
                    if edge < upper["insert_rect"].y1:
                        cost = (upper["insert_rect"].y1 - edge) * upper["insert_rect"].width
                        options.append((cost, upper, "y", edge))

                if not options:
                    continue
                _, seg, axis, edge = min(options, key=lambda option: option[0])
                box = seg["insert_rect"]
                seg["insert_rect"] = (
                    fitz.Rect(box.x0, box.y0, edge, box.y1)
                    if axis == "x"
                    else fitz.Rect(box.x0, box.y0, box.x1, edge)
                )
                settled = False
        if settled:
            return


def _probe(
    rect: fitz.Rect, body: str, css: str, archive: fitz.Archive, ladder: tuple
) -> tuple[float, float]:
    """Render `body` into `rect` off-page and report what it cost to fit."""
    scratch = fitz.open()
    canvas = scratch.new_page(width=rect.x1 + 2, height=rect.y1 + 2)
    spare_height, scale = -1.0, 0.0
    for low in ladder:
        spare_height, scale = canvas.insert_htmlbox(
            rect, body, css=css, scale_low=low, archive=archive
        )
        if spare_height >= 0:
            break
    scratch.close()
    return spare_height, scale


def render_segment(
    page: fitz.Page,
    seg: dict,
    translated: str,
    archive: fitz.Archive,
    ladder: tuple = SCALE_LADDER,
) -> tuple[float, float]:
    """Draw one translated segment; report insert_htmlbox's (spare, scale).

    Shared by both translate pipelines so the two cannot drift, and the only
    place a segment's `insert_rect` is settled: a bubble's tenant arrives with a
    family of candidate rectangles instead of one, because which shape of box
    suits it depends on how much Bangla the translation produced. Measuring
    off-page is the only way to ask — insert_htmlbox reports a fit, it cannot be
    asked for one.
    """
    css = css_for(seg)
    body = html.escape(translated)

    candidates = seg.get("bubble_rects")
    if candidates and len(candidates) > 1:
        best, best_scale = seg["insert_rect"], -1.0
        for candidate in candidates:
            spare_height, scale = _probe(candidate, body, css, archive, ladder)
            if spare_height >= 0 and scale > best_scale:
                best, best_scale = candidate, scale
        seg["insert_rect"] = best

    for low in ladder:
        spare_height, scale = page.insert_htmlbox(
            seg["insert_rect"], body, css=css, scale_low=low, archive=archive
        )
        if spare_height >= 0:
            break
    return spare_height, scale


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

        for seg in segments:
            for line_rect in seg["line_rects"]:
                page.add_redact_annot(line_rect + (0.3, 0.3, -0.3, -0.3), fill=False)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )

        metrics = []
        for seg, translated in zip(segments, translations):
            spare_height, scale = render_segment(page, seg, translated, archive)
            metrics.append((spare_height, scale))
            # Invisible logical-text layer so the shaped Bangla is copy-pasteable.
            # Uses the same rendered size (BANGLA_SIZE * winning scale) so the
            # selection box roughly tracks the visible text.
            copy_layer.add_invisible_text(
                page, seg["insert_rect"], translated, seg["size"] * BANGLA_SIZE * scale
            )
            if 0 < scale <= SCALE_LADDER[0]:
                logger.warning(
                    "Page %d: text hit the %.0f%% shrink floor%s: %r at %s",
                    page_num,
                    scale * 100,
                    " with the whole bubble already given to it"
                    if seg.get("bubble_rects")
                    else " and may overflow",
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
    # Neutralise the shaped Bangla layer so copy returns only the clean invisible
    # layer added above. Must run after subset_fonts(), which rewrites ToUnicode.
    copy_layer.blank_shaped_tounicode(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    total_time = time.time() - start_time
    logger.info("PDF translation complete: %.1f seconds", total_time)

    return out
