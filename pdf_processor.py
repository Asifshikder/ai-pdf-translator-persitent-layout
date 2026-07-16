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

CSS_TEMPLATE = """
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Regular.ttf); }}
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Bold.ttf); font-weight: bold; }}
* {{
    font-family: bengali, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.15;
    font-size: {size:.1f}px;
    color: {color};
    text-align: {align};
    {bold}
}}
"""

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

# Two lines belong to the same paragraph only if the vertical gap between them
# is below this fraction of the font size (real paragraphs run ~0.4–0.5; separate
# items such as form labels sit at ~0.7+).
MERGE_GAP = 0.6

# A horizontal hole this wide inside one row separates table columns.
COLUMN_GAP = 30.0

# Minimum gap between TOC entry text and its trailing page number.
NUMBER_GAP = 12.0


def _span_color_to_css(color_int: int) -> str:
    return f"#{color_int:06x}"


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
        if groups and span["bbox"][0] - groups[-1][-1]["bbox"][2] <= COLUMN_GAP:
            groups[-1].append(span)
        else:
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


def _detect_align(line_rects: list[fitz.Rect]) -> str:
    """Infer text alignment from how a segment's lines line up."""
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


def _extract_segments(page: fitz.Page) -> tuple[list[dict], list[fitz.Rect]]:
    """Return (translatable segments, rects of text left untouched on the page).

    A segment is one translation/insertion unit: a paragraph, a heading, a
    bullet item, or a single TOC entry. Bullet glyphs and TOC page numbers are
    never part of a segment — they stay on the page unchanged.
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
                    and bold == prev["bold"]
                    and color == prev["color"]
                    and piece["rect"].y0 - prev["rect"].y1 < MERGE_GAP * size
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
                # Bullet items accept plain continuation lines; numbered ones don't.
                prev = segment if num is None else None

    # Merge across blocks: consecutive same-style segments separated by less
    # than a line gap are one flowing paragraph (e.g. the cover tagline, which
    # PyMuPDF splits into one block per line).
    merged = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev["number"] is None
            and seg["number"] is None
            and not seg["bullet"]  # a bullet always starts a new item
            and abs(seg["size"] - prev["size"]) <= 0.12 * prev["size"]
            and seg["bold"] == prev["bold"]
            and seg["color"] == prev["color"]
            and seg["rect"].y0 - prev["rect"].y1 < MERGE_GAP * seg["size"]
            and min(seg["rect"].x1, prev["rect"].x1) - max(seg["rect"].x0, prev["rect"].x0)
            > 0.3 * min(seg["rect"].width, prev["rect"].width)
        ):
            prev["text"] += " " + seg["text"]
            prev["rect"] |= seg["rect"]
            prev["line_rects"].extend(seg["line_rects"])
            continue
        merged.append(seg)

    for seg in merged:
        seg["align"] = _detect_align(seg["line_rects"])
    return merged, kept


def _plan_insert_rects(page: fitz.Page, segments: list[dict], kept: list[fitz.Rect]) -> None:
    """Give each segment breathing room for the (usually longer) Bangla text.

    Rects only grow into space that is genuinely free: never into another
    segment, a preserved bullet/number, or an image the original text did not
    already sit on.
    """
    images = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
    obstacles = images + kept + [seg["rect"] for seg in segments]
    # Never grow past the rightmost text on the page (stay inside its margins).
    text_x1 = max(r.x1 for r in kept + [seg["rect"] for seg in segments])

    for seg in segments:
        base = seg["rect"]
        others = [
            o
            for o in obstacles
            if o is not base and not o.intersects(base) and not o.is_empty
        ]

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
        for o in others:
            if o.x0 >= base.x1 - 1 and min(o.y1, base.y1) - max(o.y0, base.y0) > 1:
                limit_x1 = min(limit_x1, o.x0 - 4)
        new_x1 = max(base.x1, min(target_x1, limit_x1))

        # Vertical: about one extra line of height, stopping above whatever
        # follows, so the taller Bangla text rarely has to shrink to fit.
        target_y1 = base.y1 + max(1.0 * seg["size"], 0.5 * base.height)
        limit_y1 = page.rect.y1 - 16
        for o in others:
            if o.y0 >= base.y1 - 1 and min(o.x1, new_x1) - max(o.x0, base.x0) > 1:
                limit_y1 = min(limit_y1, o.y0 - 2)
        new_y1 = max(base.y1, min(target_y1, limit_y1))

        # A small gap after a preserved bullet glyph (the original leading
        # space was stripped from the text).
        x0 = base.x0 + 2 if seg["bullet"] else base.x0
        seg["insert_rect"] = fitz.Rect(x0, base.y0, new_x1, new_y1)


def translate_pdf(pdf_bytes: bytes) -> bytes:
    """Translate all text in a PDF from English to Bangla, preserving layout."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    archive = fitz.Archive(FONTS_DIR)
    pages_meta = []
    source_fonts = set()

    for page_num, page in enumerate(doc, start=1):
        # Record the page's fonts while the source text is still on it: the fix
        # pipeline tells the text it inserted from the text it must preserve by
        # asking which fonts were already here.
        source_fonts |= manifest.page_span_fonts(page)

        segments, kept = _extract_segments(page)
        if not segments:
            continue

        english = [seg["text"] for seg in segments]
        translations, status = translate_batch_status(english)
        _plan_insert_rects(page, segments, kept)

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
            css = CSS_TEMPLATE.format(
                size=seg["size"],
                color=_span_color_to_css(seg["color"]),
                align=seg["align"],
                bold="font-weight: bold;" if seg["bold"] else "",
            )
            body = html.escape(translated)
            # scale_low=0.6 floors the font at 60% so Bangla text never shrinks to
            # an unreadable size; _plan_insert_rects already grows the rect so most
            # text fits without shrinking. Text that still overruns the rect at the
            # floor bleeds down into the free space reserved below it.
            spare_height, scale = page.insert_htmlbox(
                seg["insert_rect"], body, css=css, scale_low=0.6, archive=archive
            )
            metrics.append((spare_height, scale))
            if 0 < scale <= 0.6:
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
    return out
