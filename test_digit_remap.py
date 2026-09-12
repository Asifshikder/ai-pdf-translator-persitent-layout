"""Tests for the digit-remap pipeline (digit_remap.py).

No API calls: pure string mapping plus tiny PDF fixtures — with and without
a translation manifest.
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
    assert remap_digits("ইতিমধ্যে ৯৯৯") == "ইতিমধ্যে ৯৯৯"


def test_remap_digits_idempotent():
    once = remap_digits("Take 2 tablets for 6 weeks")
    assert once == "Take ২ tablets for ৬ weeks"
    assert remap_digits(once) == once


def _plain_pdf_with_digits(text: str = "Year 2025 — dose 2.5mg") -> bytes:
    """Ordinary PDF with extractable Latin text (no translation manifest)."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), text, fontsize=14)
    out = doc.tobytes()
    doc.close()
    return out


def _tiny_translated_pdf(bn: str = "প্রতিদিন 2 বার, 2025 সালে") -> bytes:
    """Minimal PDF with one segment and an embedded translation manifest."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    rect = fitz.Rect(40, 40, 260, 80)
    page.insert_htmlbox(rect, bn)

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
    manifest.attach(doc, data)
    out = doc.tobytes()
    doc.close()
    return out


def test_remap_pdf_without_manifest():
    """Any PDF works — no Translate step required."""
    out, summary = remap_pdf(_plain_pdf_with_digits())
    assert "via text spans" in summary
    assert "1 updated" in summary

    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text()
    doc.close()
    assert "২০২৫" in text
    assert "২.৫" in text
    assert "2025" not in text


def test_remap_pdf_with_manifest_updates_bn():
    out, summary = remap_pdf(_tiny_translated_pdf())
    assert "via manifest" in summary
    assert "1 updated" in summary

    doc = fitz.open(stream=out, filetype="pdf")
    data = manifest.read(doc)
    bn = data["pages"][0]["segments"][0]["bn"]
    assert bn == "প্রতিদিন ২ বার, ২০২৫ সালে"
    assert "2" not in bn
    doc.close()


def test_remap_pdf_skips_when_no_latin_digits():
    # No Latin digits at all — pipeline should be a no-op.
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 80), "Hello world — no numbers here", fontsize=14)
    raw = doc.tobytes()
    doc.close()

    out, summary = remap_pdf(raw)
    assert "0 updated" in summary
    assert "0 spans" in summary
    doc = fitz.open(stream=out, filetype="pdf")
    assert "Hello world" in doc[0].get_text()
    doc.close()


if __name__ == "__main__":
    test_remap_digits_basic()
    test_remap_digits_idempotent()
    test_remap_pdf_without_manifest()
    test_remap_pdf_with_manifest_updates_bn()
    test_remap_pdf_skips_when_no_latin_digits()
    print("OK")
