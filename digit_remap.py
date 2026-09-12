"""Digit-remap pipeline: Latin digits → Bengali digits, no AI.

Runs on an already-translated PDF (one with an embedded manifest). The translation
pipeline deliberately keeps Latin digits in the Bangla text; this pass converts
them afterwards with a plain character map — 2025 becomes ২০২৫, 2.5mg becomes
২.৫mg — then redraws only the pages that actually changed.

Also remaps digit-only source text the translator left untouched (TOC page
numbers, bare numerals): those still sit in the source fonts and extract cleanly.
"""

from __future__ import annotations

import html
import logging
import re

import fitz

import copy_layer
import manifest
from fix_processor import SCALE_LADDER, _erase_inserted
from pdf_processor import BANGLA_SIZE, CSS_TEMPLATE, FONTS_DIR, _span_color_to_css, css_for

logger = logging.getLogger(__name__)

DIGIT_SUFFIX = "_digits.pdf"  # keep in sync with MODES.digits.suffix in static/index.html

LATIN_TO_BENGALI = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
HAS_LATIN_DIGIT = re.compile(r"[0-9]")

# Source spans that are only digits / punctuation / whitespace — safe to rewrite
# without touching bullets, checkboxes, or mixed Latin prose.
DIGITISH = re.compile(r"^[\d\s.,:;/\-–—%+()]+$")


class DigitReport:
    """Counters for one digit-remap run."""

    def __init__(self) -> None:
        self.pages_scanned = 0
        self.pages_changed = 0
        self.segments_remapped = 0
        self.kept_spans_remapped = 0

    def summary(self) -> str:
        return (
            f"{self.pages_scanned} pages scanned, {self.pages_changed} updated | "
            f"{self.segments_remapped} translated segments, "
            f"{self.kept_spans_remapped} kept number spans"
        )


def remap_digits(text: str) -> str:
    """Replace Latin digits with Bengali digits; leave everything else alone."""
    return text.translate(LATIN_TO_BENGALI)


def _span_css(size: float, color_int: int, bold: bool) -> str:
    """CSS for a kept source span redrawn with the Bangla font."""
    return CSS_TEMPLATE.format(
        size=size,
        color=_span_color_to_css(color_int),
        align="left",
        bold="font-weight: bold;" if bold else "",
    )


def _kept_digit_spans(page: fitz.Page, source_fonts: set[str]) -> list[dict]:
    """Source-font spans whose text is digit-ish and still uses Latin digits."""
    found = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                if span["font"] not in source_fonts:
                    continue
                if not HAS_LATIN_DIGIT.search(text):
                    continue
                if not DIGITISH.match(text.strip()):
                    continue
                found.append(
                    {
                        "text": text,
                        "bbox": fitz.Rect(span["bbox"]),
                        "size": span["size"],
                        "color": span["color"],
                        "bold": bool(span["flags"] & (1 << 4)),
                    }
                )
    return found


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


def _remap_kept_spans(
    page: fitz.Page,
    source_fonts: set[str],
    archive: fitz.Archive,
    report: DigitReport,
) -> None:
    """Replace kept Latin digit spans (TOC numbers, bare numerals) in place."""
    spans = _kept_digit_spans(page, source_fonts)
    if not spans:
        return

    for span in spans:
        page.add_redact_annot(span["bbox"] + (0.3, 0.3, -0.3, -0.3), fill=False)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
    )

    for span in spans:
        remapped = remap_digits(span["text"])
        report.kept_spans_remapped += 1
        page.insert_htmlbox(
            span["bbox"],
            html.escape(remapped),
            css=_span_css(span["size"], span["color"], span["bold"]),
            scale_low=0.0,
            archive=archive,
        )


def remap_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Convert Latin digits to Bengali digits in a translated PDF.

    Returns the remapped PDF and a one-line summary. Pages with no Latin digits
    are left byte-identical; the run makes no API calls.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    data = manifest.read(doc)

    if data.get("fallback_collision"):
        doc.close()
        raise manifest.ManifestUnsupported(
            "The source PDF uses "
            + ", ".join(data["fallback_collision"])
            + ", which this tool also uses for its own text. Its translated text "
            "cannot be told apart from the original, so digits cannot be remapped."
        )

    archive = fitz.Archive(FONTS_DIR)
    source_fonts = set(data["source_fonts"])
    entries = {entry["page"]: entry for entry in data["pages"]}
    report = DigitReport()

    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        entry = entries.get(page_num)
        segments_dirty = bool(entry and entry["segments"] and _segments_need_remap(entry))
        kept_dirty = bool(_kept_digit_spans(page, source_fonts))
        if not segments_dirty and not kept_dirty:
            continue

        logger.info(
            "Page %d: remapping Latin digits (%s%s%s)",
            page_num,
            "segments" if segments_dirty else "",
            " + " if segments_dirty and kept_dirty else "",
            "kept numbers" if kept_dirty else "",
        )

        if segments_dirty:
            # Erasing takes out every inserted glyph; kept source numbers survive
            # and are remapped next, then the whole translated layer is redrawn.
            _erase_inserted(page, source_fonts)
            if kept_dirty:
                _remap_kept_spans(page, source_fonts, archive, report)
            _redraw_segments(page, entry, archive, report)
        else:
            _remap_kept_spans(page, source_fonts, archive, report)

        report.pages_changed += 1

    manifest.attach(doc, data)
    logger.info("Digit remap done: %s", report.summary())

    # No subset_fonts(): subsetting renumbers HarfBuzz glyph IDs and corrupts
    # complex-script Bangla the same way the old translate pipeline did.
    copy_layer.blank_shaped_tounicode(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, report.summary()
