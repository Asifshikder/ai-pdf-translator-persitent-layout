"""Tests for the digit-remap pipeline (digit_remap.py).

No API calls: pure string mapping plus a tiny PDF fixture that carries a
manifest so remap_pdf can run end-to-end.
"""

from __future__ import annotations

import fitz

import manifest
from digit_remap import remap_digits, remap_pdf


def test_remap_digits_basic():
    assert remap_digits("2025") == "২০২৫"
    assert remap_digits("2.5mg") == "২.৫mg"
    assert remap_digits("30%") == "৩০%"
    assert remap_digits("21-35") == "২১-৩৫"
    assert remap_digits("no digits") == "no digits"
    assert remap_digits("ইতিমধ্যে ৯৯৯") == "ইতিমধ্যে ৯৯৯"  # already Bengali


def test_remap_digits_idempotent():
    once = remap_digits("Take 2 tablets for 6 weeks")
    assert once == "Take ২ tablets for ৬ weeks"
    assert remap_digits(once) == once


def _tiny_translated_pdf(bn: str = "প্রতিদিন 2 বার, 2025 সালে") -> bytes:
    """Minimal PDF with one segment and an embedded translation manifest."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    rect = fitz.Rect(40, 40, 260, 80)
    page.insert_htmlbox(rect, bn)

    source_fonts = sorted(manifest.page_span_fonts(page))
    # Pretend the drawn font is "inserted" Bangla by recording empty source fonts
    # so erase will strip it; the fixture only needs the remap path to run.
    data = {
        "v": manifest.SCHEMA_VERSION,
        "tool": "pdftranslator",
        "created": "2026-01-01T00:00:00+00:00",
        "fix_round": 0,
        "source_fonts": [],
        "fallback_collision": [],
        "pages": [
            {
                "page": 1,
                "kept": [],
                "segments": [
                    {
                        "en": "twice a day, in 2025",
                        "bn": bn,
                        "ok": True,
                        "rect": [40, 40, 260, 80],
                        "lines": [[40, 40, 260, 80]],
                        "ins": [40, 40, 260, 80],
                        "size": 12.0,
                        "bold": False,
                        "color": 0,
                        "align": "left",
                        "bullet": None,
                        "block_x1": 260.0,
                        "num": None,
                        "sh": 10.0,
                        "sc": 1.0,
                        "fixed": None,
                    }
                ],
            }
        ],
    }
    # Keep real source_fonts empty intentionally so _erase_inserted removes the
    # htmlbox glyphs we just drew (their font is not in source_fonts).
    assert source_fonts  # sanity: something was drawn
    del source_fonts
    manifest.attach(doc, data)
    out = doc.tobytes()
    doc.close()
    return out


def test_remap_pdf_updates_manifest_and_summary():
    pdf_bytes = _tiny_translated_pdf()
    out, summary = remap_pdf(pdf_bytes)
    assert "1 pages scanned, 1 updated" in summary
    assert "1 translated segments" in summary

    doc = fitz.open(stream=out, filetype="pdf")
    data = manifest.read(doc)
    bn = data["pages"][0]["segments"][0]["bn"]
    assert bn == "প্রতিদিন ২ বার, ২০২৫ সালে"
    assert "2" not in bn
    assert "2025" not in bn
    doc.close()


def test_remap_pdf_skips_when_no_latin_digits():
    pdf_bytes = _tiny_translated_pdf("শুধু বাংলা ১২৩")
    out, summary = remap_pdf(pdf_bytes)
    assert "0 updated" in summary
    assert "0 translated segments" in summary
    # Manifest bn unchanged
    doc = fitz.open(stream=out, filetype="pdf")
    data = manifest.read(doc)
    assert data["pages"][0]["segments"][0]["bn"] == "শুধু বাংলা ১২৩"
    doc.close()


def test_remap_pdf_rejects_missing_manifest():
    doc = fitz.open()
    doc.new_page()
    raw = doc.tobytes()
    doc.close()
    try:
        remap_pdf(raw)
        assert False, "expected ManifestMissing"
    except manifest.ManifestMissing:
        pass


if __name__ == "__main__":
    test_remap_digits_basic()
    test_remap_digits_idempotent()
    test_remap_pdf_updates_manifest_and_summary()
    test_remap_pdf_skips_when_no_latin_digits()
    test_remap_pdf_rejects_missing_manifest()
    print("OK")
