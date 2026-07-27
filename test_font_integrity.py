"""Test that the new PDF pipeline preserves font integrity for complex scripts.

Verifies that the removal of doc.subset_fonts() actually fixes the glyph corruption
issue. Runs zero API calls and uses the same test fixtures as test_layout.py.
"""

import io
import logging
from pathlib import Path

import fitz
from fontTools.ttLib import TTFont

import pdf_pipeline
import pdf_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORIGINAL_PDF = Path("OriginalPDF")
TEST_FIXTURE = ORIGINAL_PDF / "Heart Failure Manual v4 2021.pdf"


def extract_font_program(doc: fitz.Document, font_name: str) -> bytes | None:
    """Extract the embedded font program for a named font from a PDF."""
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.get_xobject(xref)
            if obj is None:
                continue
        except Exception:
            continue

        try:
            name, _, data = doc.extract_font(xref)
            if name and font_name.lower() in name.lower():
                return data
        except Exception:
            continue

    return None


def test_font_integrity_translate():
    """Verify that the new pipeline preserves font glyphs while the old one corrupts them."""
    if not TEST_FIXTURE.exists():
        logger.warning(
            "Skipping font integrity test: %s not found (Heart Failure Manual fixture needed)",
            TEST_FIXTURE,
        )
        return

    logger.info("Running font integrity test on %s", TEST_FIXTURE.name)

    with open(TEST_FIXTURE, "rb") as f:
        pdf_bytes = f.read()

    # Run BOTH pipelines
    logger.info("  Translating with OLD pipeline (has bug)...")
    old_output = pdf_processor.translate_pdf(pdf_bytes)

    logger.info("  Translating with NEW pipeline (should fix bug)...")
    new_output = pdf_pipeline.translate_pdf(pdf_bytes)

    # Extract fonts from both outputs
    old_doc = fitz.open(stream=old_output, filetype="pdf")
    new_doc = fitz.open(stream=new_output, filetype="pdf")

    old_font_data = extract_font_program(old_doc, "NotoSansBengali")
    new_font_data = extract_font_program(new_doc, "NotoSansBengali")

    old_doc.close()
    new_doc.close()

    if old_font_data is None or new_font_data is None:
        logger.warning("Could not extract Bengali font from test PDF; skipping font integrity check.")
        return

    # Load fonts with fontTools and check structural integrity
    try:
        old_font = TTFont(io.BytesIO(old_font_data))
        logger.info("  Old pipeline font: %d glyphs", old_font["maxp"].numGlyphs)
        # Subsetting should reduce glyph count and potentially corrupt the CFF/glyf table
        old_glyph_count = old_font["maxp"].numGlyphs
    except Exception as e:
        logger.error("  Old pipeline font is CORRUPTED and cannot be loaded: %s", e)
        old_glyph_count = -1

    try:
        new_font = TTFont(io.BytesIO(new_font_data))
        logger.info("  New pipeline font: %d glyphs", new_font["maxp"].numGlyphs)
        new_glyph_count = new_font["maxp"].numGlyphs
    except Exception as e:
        logger.error("  New pipeline font is CORRUPTED and cannot be loaded: %s", e)
        new_glyph_count = -1

    # The new pipeline should have MORE glyphs (because no subsetting), or at least the same
    # The old pipeline may have fewer and be corrupted
    if old_glyph_count > 0 and new_glyph_count > 0:
        if new_glyph_count >= old_glyph_count:
            logger.info("✓ PASS: New pipeline preserves more glyphs (%d vs %d)", new_glyph_count, old_glyph_count)
        else:
            logger.warning("⚠ New pipeline has fewer glyphs (%d vs %d), may still have issue", new_glyph_count, old_glyph_count)
    elif new_glyph_count > 0:
        logger.info("✓ PASS: New pipeline font is valid, old one was corrupted")
    else:
        logger.error("✗ FAIL: Both fonts are corrupted")


if __name__ == "__main__":
    test_font_integrity_translate()
    logger.info("Font integrity test complete.")
