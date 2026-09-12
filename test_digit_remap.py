"""Tests for the digit-remap pipeline (digit_remap.py)."""

from __future__ import annotations

import fitz

import manifest
from digit_remap import remap_digits, remap_pdf


def test_remap_digits_basic():
    assert remap_digits("2025") == "২০২৫"
    assert remap_digits("2.5mg") == "২.৫mg"
    assert remap_digits("30%") == "৩০%"
    assert remap_digits("no digits") == "no digits"
    assert remap_digits("ইতিমধ্যে ৯৯৯") == "ইতিমধ্যে ৯৯৯"


def test_remap_digits_idempotent():
    once = remap_digits("Take 2 tablets for 6 weeks")
    assert once == "Take ২ tablets for ৬ weeks"
    assert remap_digits(once) == once


def _plain_pdf(text: str = "Year 2025 — dose 2.5mg") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), text, fontsize=14)
    out = doc.tobytes()
    doc.close()
    return out


def test_remap_pdf_without_manifest_in_sentence():
    """Numbers inside a sentence must change; surrounding words must survive."""
    raw = _plain_pdf("The programme lasts between 2 to 3 months in 2025")
    out, summary = remap_pdf(raw)
    assert "via chars" in summary
    assert "digit glyphs" in summary

    text = fitz.open(stream=out, filetype="pdf")[0].get_text()
    assert "programme lasts between" in text
    assert "months" in text
    assert "২" in text and "৩" in text and "২০২৫" in text
    assert "2" not in text and "3" not in text and "2025" not in text


def test_remap_pdf_preserves_non_digit_spans():
    """Only digit glyphs are touched — neighbouring lines without digits stay."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=300)
    page.insert_text((40, 60), "1. First item mentions 2 things", fontsize=12)
    page.insert_text((40, 100), "No numbers on this line at all", fontsize=12)
    page.insert_text((40, 140), "Page 42", fontsize=12)
    raw = doc.tobytes()
    doc.close()

    out, _ = remap_pdf(raw)
    text = fitz.open(stream=out, filetype="pdf")[0].get_text()
    assert "First item mentions" in text
    assert "No numbers on this line at all" in text
    assert "Page" in text
    assert "১" in text and "২" in text and "৪২" in text
    assert not any(c in text for c in "0123456789")


def test_remap_pdf_with_manifest_updates_bn():
    bn = "প্রতিদিন 2 বার, 2025 সালে"
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
    raw = doc.tobytes()
    doc.close()

    out, summary = remap_pdf(raw)
    assert "manifest" in summary
    doc = fitz.open(stream=out, filetype="pdf")
    data = manifest.read(doc)
    assert data["pages"][0]["segments"][0]["bn"] == "প্রতিদিন ২ বার, ২০২৫ সালে"
    doc.close()


def test_remap_pdf_skips_when_no_latin_digits():
    out, summary = remap_pdf(_plain_pdf("Hello world — no numbers here"))
    assert "0 updated" in summary
    assert "0 digit glyphs" in summary


if __name__ == "__main__":
    test_remap_digits_basic()
    test_remap_digits_idempotent()
    test_remap_pdf_without_manifest_in_sentence()
    test_remap_pdf_preserves_non_digit_spans()
    test_remap_pdf_with_manifest_updates_bn()
    test_remap_pdf_skips_when_no_latin_digits()
    print("OK")
