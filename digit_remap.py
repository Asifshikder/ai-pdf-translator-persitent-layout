"""Digit-remap pipeline: Latin digits → Bengali digits, no AI.

Works on any PDF. Strategy:

  1. Prefer character-level replacement from `rawdict`: each Latin digit glyph is
     redacted and redrawn as its Bengali counterpart in the same box. Surrounding
     letters stay untouched — this is what catches numbers *inside* sentences.
  2. When a translation manifest is present, also remap digits inside `bn` and
     redraw those segments. HarfBuzz-shaped Bangla does not expose Latin digits
     to `get_text`, so step 1 alone cannot see them.

Examples: 2025 → ২০২৫, "between 2 to 3 months" → "between ২ to ৩ months".
"""

from __future__ import annotations

import html
import logging
import os
import re

import fitz

import copy_layer
import manifest
from fix_processor import SCALE_LADDER, _erase_inserted
from pdf_processor import BANGLA_SIZE, FONTS_DIR, css_for

logger = logging.getLogger(__name__)

DIGIT_SUFFIX = "_digits.pdf"  # keep in sync with MODES.digits.suffix in static/index.html

LATIN_TO_BENGALI = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
HAS_LATIN_DIGIT = re.compile(r"[0-9]")

SYMBOL_FONT = re.compile(r"dingbat|wingding|webding|symbol", re.I)
COPY_LAYER_TAG = "CopyLayer"

FONT_REGULAR = os.path.join(FONTS_DIR, "NotoSansBengali-Regular.ttf")
FONT_BOLD = os.path.join(FONTS_DIR, "NotoSansBengali-Bold.ttf")


class DigitReport:
    """Counters for one digit-remap run."""

    def __init__(self) -> None:
        self.pages_scanned = 0
        self.pages_changed = 0
        self.segments_remapped = 0
        self.chars_remapped = 0
        self.via_manifest = False

    def summary(self) -> str:
        mode = "manifest+chars" if self.via_manifest else "chars"
        return (
            f"{self.pages_scanned} pages scanned, {self.pages_changed} updated "
            f"via {mode} | {self.segments_remapped} segments, "
            f"{self.chars_remapped} digit glyphs"
        )


def remap_digits(text: str) -> str:
    """Replace Latin digits with Bengali digits; leave everything else alone."""
    return text.translate(LATIN_TO_BENGALI)


def _color_rgb(color_int: int) -> tuple[float, float, float]:
    """PyMuPDF span color int → RGB floats in 0..1."""
    return (
        ((color_int >> 16) & 255) / 255.0,
        ((color_int >> 8) & 255) / 255.0,
        (color_int & 255) / 255.0,
    )


def _collect_digit_chars(page: fitz.Page) -> list[dict]:
    """Every Latin digit glyph on the page, with its own bbox and style."""
    found = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                font = span["font"]
                if SYMBOL_FONT.search(font) or COPY_LAYER_TAG in font:
                    continue
                bold = bool(span["flags"] & (1 << 4))
                for ch in span.get("chars") or []:
                    if ch["c"] not in "0123456789":
                        continue
                    found.append(
                        {
                            "c": ch["c"],
                            "bbox": fitz.Rect(ch["bbox"]),
                            "origin": fitz.Point(ch["origin"]),
                            "size": span["size"],
                            "color": span["color"],
                            "bold": bold,
                        }
                    )
    return found


def _draw_digit(page: fitz.Page, item: dict, font_reg: fitz.Font, font_bold: fitz.Font) -> None:
    """Paint one Bengali digit at the Latin digit's origin/size/color."""
    bn = remap_digits(item["c"])
    font = font_bold if item["bold"] else font_reg
    # TextWriter places on the baseline (origin), matching the source glyph.
    writer = fitz.TextWriter(page.rect)
    writer.append(item["origin"], bn, font=font, fontsize=item["size"])
    writer.write_text(page, color=_color_rgb(item["color"]))


def _remap_chars_on_page(
    page: fitz.Page,
    font_reg: fitz.Font,
    font_bold: fitz.Font,
    report: DigitReport,
) -> bool:
    """Replace each Latin digit glyph in place. Returns True if anything changed."""
    chars = _collect_digit_chars(page)
    if not chars:
        return False

    # Shrink a hair so neighbouring letters are not clipped by the redact.
    for item in chars:
        page.add_redact_annot(item["bbox"] + (0.15, 0.15, -0.15, -0.15), fill=False)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
    )

    for item in chars:
        _draw_digit(page, item, font_reg, font_bold)
        report.chars_remapped += 1
    return True


def _segments_need_remap(entry: dict) -> bool:
    return any(HAS_LATIN_DIGIT.search(seg["bn"]) for seg in entry["segments"])


def _redraw_segments(
    page: fitz.Page,
    entry: dict,
    archive: fitz.Archive,
    report: DigitReport,
) -> None:
    """Redraw every segment after digits in `bn` have been remapped."""
    for seg in entry["segments"]:
        original = seg["bn"]
        remapped = remap_digits(original)
        if remapped != original:
            report.segments_remapped += 1
        seg["bn"] = remapped

        body = html.escape(remapped)
        css = css_for(seg)
        box = manifest.rect_of(seg["ins"])
        spare, scale = -1.0, 0.0
        for low in SCALE_LADDER:
            spare, scale = page.insert_htmlbox(
                box, body, css=css, scale_low=low, archive=archive
            )
            if spare >= 0:
                break
        seg["sh"] = round(spare, 2)
        seg["sc"] = round(scale, 3)
        copy_layer.add_invisible_text(
            page, box, remapped, seg["size"] * BANGLA_SIZE * max(scale, 0.0)
        )


def _remap_via_manifest(
    doc: fitz.Document,
    data: dict,
    archive: fitz.Archive,
    font_reg: fitz.Font,
    font_bold: fitz.Font,
    report: DigitReport,
) -> None:
    """Manifest path: reshape Bangla that hides digits, then char-fix leftovers."""
    report.via_manifest = True
    source_fonts = set(data["source_fonts"])
    entries = {entry["page"]: entry for entry in data["pages"]}

    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        entry = entries.get(page_num)
        segments_dirty = bool(entry and entry["segments"] and _segments_need_remap(entry))
        chars_dirty = bool(_collect_digit_chars(page))
        if not segments_dirty and not chars_dirty:
            continue

        if segments_dirty:
            logger.info("Page %d: remapping digits via manifest + chars", page_num)
            _erase_inserted(page, source_fonts)
            # Source-font page numbers / kept numerals survive erase.
            _remap_chars_on_page(page, font_reg, font_bold, report)
            _redraw_segments(page, entry, archive, report)
        else:
            logger.info("Page %d: remapping digit glyphs", page_num)
            _remap_chars_on_page(page, font_reg, font_bold, report)
        report.pages_changed += 1

    manifest.attach(doc, data)
    copy_layer.blank_shaped_tounicode(doc)


def _remap_via_chars(
    doc: fitz.Document,
    font_reg: fitz.Font,
    font_bold: fitz.Font,
    report: DigitReport,
) -> None:
    """Manifest-free path: replace every extractable Latin digit glyph."""
    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        n = len(_collect_digit_chars(page))
        if not n:
            continue
        logger.info("Page %d: remapping %d digit glyphs", page_num, n)
        if _remap_chars_on_page(page, font_reg, font_bold, report):
            report.pages_changed += 1


def remap_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Convert Latin digits to Bengali digits in a PDF.

    Uses the translation manifest when present (for shaped Bangla); always also
    remaps extractable digit glyphs so numbers inside sentences are caught.
    Never requires a prior Translate run.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    archive = fitz.Archive(FONTS_DIR)
    font_reg = fitz.Font(fontfile=FONT_REGULAR)
    font_bold = fitz.Font(fontfile=FONT_BOLD)
    report = DigitReport()

    try:
        data = manifest.read(doc)
    except (manifest.ManifestMissing, manifest.ManifestUnsupported):
        data = None

    if data is not None and data.get("fallback_collision"):
        logger.warning(
            "Manifest has fallback font collision (%s); remapping via chars only",
            ", ".join(data["fallback_collision"]),
        )
        data = None

    if data is not None:
        _remap_via_manifest(doc, data, archive, font_reg, font_bold, report)
    else:
        _remap_via_chars(doc, font_reg, font_bold, report)

    logger.info("Digit remap done: %s", report.summary())
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, report.summary()
