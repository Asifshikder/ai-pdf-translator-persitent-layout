"""Fixed PDF translation pipeline: preserves fonts (no subsetting) to fix rendering corruption.

This pipeline is identical to pdf_processor.py and fix_processor.py except for one line:
it does NOT call doc.subset_fonts() before saving. Subsetting fonts corrupts complex-script
glyphs (Bangla conjuncts/matras) that were shaped by HarfBuzz: the subsetter can drop or
renumber glyph IDs that the content stream still points at. MuPDF's renderer is lenient
about this, but other engines (Word, Adobe Reader, other PDF viewers) are not, leading to
the garbled rendering visible when opening the PDF in any external tool.

Fonts are embedded in full instead of subsetted — this increases file size but is necessary
to preserve correct rendering in other tools and when converting to DOCX.
"""

import html
import logging
import re
import time

import fitz

import manifest
from fix_processor import (
    FIX_SUFFIX,
    SCALE_LADDER,
    FixReport,
    _collisions,
    _deconflict,
    _page_problems,
    _repair_page,
)
from pdf_processor import (
    FONTS_DIR,
    SCALE_LADDER as TRANSLATE_SCALE_LADDER,
    _extract_segments,
    _panels,
    _plan_insert_rects,
    _rules,
    _vector_marks,
    css_for,
)
from translator import translate_batch_status

logger = logging.getLogger(__name__)


def translate_pdf(pdf_bytes: bytes) -> bytes:
    """Translate all text in a PDF from English to Bangla, preserving layout.

    Same as pdf_processor.translate_pdf but omits doc.subset_fonts() to preserve
    correct rendering of HarfBuzz-shaped Bangla glyphs.
    """
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
            css = css_for(seg)
            body = html.escape(translated)
            for low in TRANSLATE_SCALE_LADDER:
                spare_height, scale = page.insert_htmlbox(
                    seg["insert_rect"], body, css=css, scale_low=low, archive=archive
                )
                if spare_height >= 0:
                    break
            metrics.append((spare_height, scale))
            if 0 < scale <= TRANSLATE_SCALE_LADDER[0]:
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

    data = manifest.build(source_fonts, pages_meta)
    if data["fallback_collision"]:
        logger.warning(
            "Source PDF uses %s, which MuPDF also picks as a fallback font: the fix "
            "pipeline cannot tell inserted text from original text and will skip this "
            "document.",
            ", ".join(data["fallback_collision"]),
        )
    manifest.attach(doc, data)

    # FIXED: omit doc.subset_fonts() to preserve glyph IDs for complex scripts
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    total_time = time.time() - start_time
    logger.info("PDF translation complete: %.1f seconds", total_time)

    return out


def fix_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Repair untranslated, dropped and overlapping text in a translated PDF.

    Returns the fixed PDF and a one-line summary of what was found and changed.
    Same as fix_processor.fix_pdf but omits doc.subset_fonts() to preserve correct
    rendering of HarfBuzz-shaped Bangla glyphs.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    data = manifest.read(doc)

    if data.get("fallback_collision"):
        doc.close()
        raise manifest.ManifestUnsupported(
            "The source PDF uses "
            + ", ".join(data["fallback_collision"])
            + ", which this tool also uses for its own text. Its translated text "
            "cannot be told apart from the original, so this PDF cannot be fixed."
        )

    archive = fitz.Archive(FONTS_DIR)
    source_fonts = set(data["source_fonts"])
    entries = {entry["page"]: entry for entry in data["pages"]}
    report = FixReport()

    for page_num, page in enumerate(doc, start=1):
        report.pages_scanned += 1
        entry = entries.get(page_num)
        if entry is None or not entry["segments"]:
            continue

        hits = _collisions(entry, archive)
        report.overlaps_before += len(hits)
        problems = _page_problems(entry, hits)
        if not problems:
            continue

        logger.info("Page %d: %s — repairing", page_num, ", ".join(problems))
        if hits:
            _deconflict(entry, hits)
        _repair_page(page, entry, source_fonts, archive, report)
        report.pages_repaired += 1
        report.overlaps_after += len(_collisions(entry, archive))

    data["fix_round"] += 1
    manifest.attach(doc, data)
    logger.info("Fix done: %s", report.summary())

    # FIXED: omit doc.subset_fonts() to preserve glyph IDs for complex scripts
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, report.summary()
