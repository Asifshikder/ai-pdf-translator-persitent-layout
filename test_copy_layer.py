"""Test that translated Bangla is copy-pasteable from the output PDF.

The visible Bangla is HarfBuzz-shaped by `insert_htmlbox` and extracts as
garbage (conjuncts map to PUA codepoints, matras come out reordered). The
`copy_layer` module adds an invisible logical-text layer and neutralises the
shaped layer so text extraction returns clean, correctly-ordered Bangla.

Runs zero API calls: it drives `copy_layer` directly over a shaped fixture,
exactly as `pdf_processor.translate_pdf` does.
"""

import logging
import os

import fitz

import copy_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conjunct-heavy phrases with pre-base matras — the cases that garble worst.
SEGMENTS = [
    "হার্ট ফেইলিওর ম্যানুয়াল",
    "মতামত ফর্ম কিছু বিষয়",
    "কখন ডাক্তারকে ডাকতে হবে জীবনযাত্রার ধরন",
]

# PUA/control codepoints the shaped layer leaks when it is NOT neutralised.
GARBLE_MARKERS = ["Ɩ", "Ǝ", "ţ", "ƀ", "\x85", "\x97"]

CSS = """
@font-face { font-family: bengali; src: url(NotoSansBengali-Regular.ttf); }
* { font-family: bengali, sans-serif; margin: 0; padding: 0;
    line-height: 1.45; font-size: 14px; color: #000; }
"""


def _build_output() -> bytes:
    """Shape each segment (as the pipeline does) then apply the copy layer."""
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    doc = fitz.open()
    page = doc.new_page(width=420, height=80 + 60 * len(SEGMENTS))
    archive = fitz.Archive(fonts_dir)
    for i, text in enumerate(SEGMENTS):
        rect = fitz.Rect(20, 20 + 60 * i, 400, 60 + 60 * i)
        page.insert_htmlbox(rect, text, css=CSS, archive=archive)
        copy_layer.add_invisible_text(page, rect, text, 14 * 0.9)
    doc.subset_fonts()
    neutralised = copy_layer.blank_shaped_tounicode(doc)
    assert neutralised >= 1, "expected at least one shaped Bangla font to neutralise"
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out


def test_copy_layer_roundtrip() -> None:
    doc = fitz.open(stream=_build_output(), filetype="pdf")
    extracted = doc[0].get_text()
    doc.close()

    for text in SEGMENTS:
        assert text in extracted, f"segment did not round-trip: {text!r}"
    leaked = [m for m in GARBLE_MARKERS if m in extracted]
    assert not leaked, f"shaped-layer garble leaked into extraction: {leaked}"
    logger.info("✓ PASS: all %d segments copy-paste cleanly, no garble", len(SEGMENTS))


if __name__ == "__main__":
    test_copy_layer_roundtrip()
    logger.info("Copy-layer test complete.")
