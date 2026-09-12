"""Digit-remap pipeline: Latin digits → Bengali digits, no AI.

Works on any PDF. When a translation manifest is present (a `_bn.pdf` from this
tool), Bangla segments are redrawn from the manifest so shaped text stays
correct. Otherwise every extractable text span that still has Latin digits is
rewritten in place — enough for English PDFs, page numbers, and any document
whose text extracts cleanly.

Examples: 2025 → ২০২৫, 2.5mg → ২.৫mg. No API calls.
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

# Decorative fonts carry no readable digits worth remapping.
SYMBOL_FONT = re.compile(r"dingbat|wingding|webding|symbol", re.I)
# Invisible copy-layer overlays must not be rewritten as visible text.
COPY_LAYER_TAG = "CopyLayer"


class DigitReport:
    """Counters for one digit-remap run."""

    def __init__(self) -> None:
        self.pages_scanned = 0
        self.pages_changed = 0
        self.segments_remapped = 0
        self.spans_remapped = 0
        self.via_manifest = False

    def summary(self) -> str:
        mode = "manifest" if self.via_manifest else "text spans"
        return (
            f"{self.pages_scanned} pages scanned, {self.pages_changed} updated "
            f"via {mode} | {self.segments_remapped} segments, "
            f"{self.spans_remapped} spans"
        )


def remap_digits(text: str) -> str:
    """Replace Latin digits with Bengali digits; leave everything else alone."""
    return text.translate(LATIN_TO_BENGALI)


def _span_css(size: float, color_int: int, bold: bool) -> str:
    """CSS for a span redrawn with the Bangla font (needed for Bengali digits)."""
    return CSS_TEMPLATE.format(
        size=size,
        color=_span_color_to_css(color_int),
        align="left",
        bold="font-weight: bold;" if bold else "",
    )


def _collect_digit_spans(page: fitz.Page) -> list[dict]:
    """Every readable span on the page that still contains Latin digits."""
    found = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                if not text or not HAS_LATIN_DIGIT.search(text):
                    continue
                font = span["font"]
                if SYMBOL_FONT.search(font) or COPY_LAYER_TAG in font:
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


def _remap_spans_on_page(
    page: fitz.Page, archive: fitz.Archive, report: DigitReport
) -> bool:
    """Redact and redraw every span with Latin digits. Returns True if anything changed."""
    spans = _collect_digit_spans(page)
    if not spans:
        return False

    for span in spans:
        page.add_redact_annot(span["bbox"] + (0.3, 0.3, -0.3, -0.3), fill=False)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
    )

    for span in spans:
        remapped = remap_digits(span["text"])
        report.spans_remapped += 1
        page.insert_htmlbox(
            span["bbox"],
            html.escape(remapped),
            css=_span_css(span["size"], span["color"], span["bold"]),
            scale_low=0.0,
            archive=archive,
        )
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
    doc: fitz.Document, data: dict, archive: fitz.Archive, report: DigitReport
) -> None:
    """Prefer the manifest when present: shaped Bangla cannot be read back."""
    report.via_manifest = True
    source_fonts = set(data["source_fonts"])
    entries = {entry["page"]: entry for entry in data["pages"]}

    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        entry = entries.get(page_num)
        segments_dirty = bool(entry and entry["segments"] and _segments_need_remap(entry))
        spans_dirty = bool(_collect_digit_spans(page))
        if not segments_dirty and not spans_dirty:
            continue

        if segments_dirty:
            logger.info("Page %d: remapping Latin digits via manifest", page_num)
            _erase_inserted(page, source_fonts)
            if _collect_digit_spans(page):
                _remap_spans_on_page(page, archive, report)
            _redraw_segments(page, entry, archive, report)
        else:
            logger.info("Page %d: remapping digit spans (no segment changes)", page_num)
            _remap_spans_on_page(page, archive, report)
        report.pages_changed += 1

    manifest.attach(doc, data)
    copy_layer.blank_shaped_tounicode(doc)


def _remap_via_spans(
    doc: fitz.Document, archive: fitz.Archive, report: DigitReport
) -> None:
    """Manifest-free path: rewrite every extractable span that has Latin digits."""
    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        if not _collect_digit_spans(page):
            continue
        logger.info("Page %d: remapping Latin digits via text spans", page_num)
        if _remap_spans_on_page(page, archive, report):
            report.pages_changed += 1


def remap_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Convert Latin digits to Bengali digits in a PDF.

    Uses the translation manifest when the PDF has one; otherwise remaps from
    extractable page text. Never requires a prior Translate run.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    archive = fitz.Archive(FONTS_DIR)
    report = DigitReport()

    try:
        data = manifest.read(doc)
    except (manifest.ManifestMissing, manifest.ManifestUnsupported):
        data = None

    if data is not None and data.get("fallback_collision"):
        # Manifest is unusable for erase/redraw — fall back to span remapping.
        logger.warning(
            "Manifest has fallback font collision (%s); remapping via text spans instead",
            ", ".join(data["fallback_collision"]),
        )
        data = None

    if data is not None:
        _remap_via_manifest(doc, data, archive, report)
    else:
        _remap_via_spans(doc, archive, report)

    logger.info("Digit remap done: %s", report.summary())
    # No subset_fonts(): subsetting renumbers HarfBuzz glyph IDs and corrupts Bangla.
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, report.summary()
