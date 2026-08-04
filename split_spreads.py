"""Split two-up spread pages into single pages.

Some source manuals are laid out for print as spreads: one A3 landscape sheet
carries two facing A4 pages side by side. The translation pipeline reproduces
whatever page geometry it is given, so a spread in is a spread out — the reader
gets one wide page showing two.

This is a terminal post-processing step: run it on the finished (translated,
fixed, localized) PDF, last. It halves each spread by narrowing the page's
CropBox rather than rebuilding the content, so glyphs, vectors and the invisible
copy layer are carried over byte-identically — important because the Bangla text
is HarfBuzz-shaped and does not survive being re-encoded.

The embedded translation manifest is remapped alongside, so Fix still works on
the split output.
"""

import logging

import fitz

import manifest

logger = logging.getLogger(__name__)

SPLIT_SUFFIX = "_split.pdf"

# A page is only a candidate if it is this much wider than it is tall. Two A4
# portraits side by side give 1.41. A genuine landscape page can clear this bar
# too, which is why the gutter test below has the final say.
SPREAD_ASPECT = 1.2

# Half-width of the band around the centreline that must be free of content for
# the sheet to be two pages rather than one wide one.
GUTTER_BAND = 1.0


def _crosses_gutter(page: fitz.Page) -> bool:
    """True if any span, stroke or image straddles the page's centreline."""
    mid = page.rect.width / 2
    lo, hi = mid - GUTTER_BAND, mid + GUTTER_BAND

    # Spans, not blocks: get_text("blocks") merges the two facing page numbers
    # ('6' at the far left, '7' at the far right) into a single block spanning
    # the whole sheet, which is not content crossing the gutter.
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0, _, x1, _ = span["bbox"]
                if x0 < lo and x1 > hi and span["text"].strip():
                    return True

    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.x0 < lo and rect.x1 > hi and rect.width > 2 * GUTTER_BAND:
            return True

    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            if rect.x0 < lo and rect.x1 > hi:
                return True

    return False


def is_spread(page: fitz.Page) -> bool:
    """True if this page is two portrait pages printed side by side."""
    rect = page.rect
    if page.rotation:
        # On a rotated page the centreline of page.rect is not the visual
        # gutter, and cropping in unrotated space would cut the wrong axis.
        return False
    if rect.height <= 0 or rect.width / rect.height < SPREAD_ASPECT:
        return False
    return not _crosses_gutter(page)


def _shift(values: list[float], dx: float) -> list[float]:
    """Move a serialised [x0, y0, x1, y1] left by dx."""
    return [round(values[0] - dx, 2), values[1], round(values[2] - dx, 2), values[3]]


def _on_this_page(values: list[float], mid: float | None, right: bool) -> bool:
    """True if a rect belongs to this half, judged by its centre.

    `mid` is None for a page that was not split, which keeps everything —
    without that case a wide-but-unsplit page would silently lose its
    right-hand segments.
    """
    if mid is None:
        return True
    return ((values[0] + values[2]) / 2 >= mid) == right


def _remap_manifest(data: dict, halves: dict[int, list[tuple[int, float, float | None]]]) -> None:
    """Rewrite manifest page numbers and rects to match the split pages.

    `halves` maps a 1-based source page number to the (new page number, x offset,
    gutter) of each page it became, with a gutter of None for a page that came
    through whole. Segments are assigned to a half by the centre of their rect,
    so a segment is never torn between two pages.
    """
    rewritten = []
    for entry in data["pages"]:
        for new_num, dx, mid in halves.get(entry["page"], []):
            right = dx > 0
            segments = [
                dict(
                    seg,
                    rect=_shift(seg["rect"], dx),
                    lines=[_shift(r, dx) for r in seg["lines"]],
                    ins=_shift(seg["ins"], dx),
                    block_x1=round(seg["block_x1"] - dx, 2),
                    num=_shift(seg["num"], dx) if seg["num"] else None,
                )
                for seg in entry["segments"]
                if _on_this_page(seg["rect"], mid, right)
            ]
            kept = [
                _shift(r, dx) for r in entry["kept"] if _on_this_page(r, mid, right)
            ]
            rewritten.append({"page": new_num, "kept": kept, "segments": segments})
    data["pages"] = rewritten


def split_spreads(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Split every two-up spread in the PDF into two single pages.

    Returns the new PDF and a one-line summary. A PDF with no spreads is
    returned unchanged.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Decide before touching anything: is_spread reads text and drawings, and
    # cropping a page changes what those calls report.
    spreads = [is_spread(page) for page in doc]
    if not any(spreads):
        doc.close()
        return pdf_bytes, "No spread pages found — the PDF is unchanged."

    try:
        data = manifest.read(doc)
    except (manifest.ManifestMissing, manifest.ManifestUnsupported):
        data = None  # not a translated PDF, or a manifest we cannot rewrite

    # Where each source page lands once the earlier spreads have doubled.
    halves: dict[int, list[tuple[int, float, float | None]]] = {}
    new_num = 0
    for pno, spread in enumerate(spreads):
        new_num += 1
        if not spread:
            halves[pno + 1] = [(new_num, 0.0, None)]
            continue
        mid = doc[pno].rect.width / 2
        halves[pno + 1] = [(new_num, 0.0, mid), (new_num + 1, mid, mid)]
        new_num += 1

    # Back to front, so copying a page never shifts the index of one not yet done.
    for pno in reversed(range(len(spreads))):
        if not spreads[pno]:
            continue
        page = doc[pno]
        mid, height = page.rect.width / 2, page.rect.height
        origin = page.cropbox.x0, doc[pno].mediabox.y1 - page.cropbox.y1
        # fullcopy_page inserts *in front of* `to`; -1 means append at the end,
        # which is where the copy belongs when the spread is the last page.
        doc.fullcopy_page(pno, -1 if pno + 1 >= doc.page_count else pno + 1)
        # set_cropbox works in unrotated mediabox space, so re-add the offset
        # that page.rect had already subtracted.
        left, top = origin
        doc[pno].set_cropbox(fitz.Rect(left, top, left + mid, top + height))
        doc[pno + 1].set_cropbox(fitz.Rect(left + mid, top, left + 2 * mid, top + height))

    if data is not None:
        _remap_manifest(data, halves)
        manifest.attach(doc, data)

    # No subset_fonts(): subsetting renumbers glyph IDs and corrupts the
    # HarfBuzz-shaped Bangla the content streams already point at.
    out = doc.tobytes(garbage=3, deflate=True)
    count = sum(spreads)
    doc.close()

    summary = (
        f"Split {count} spread page{'s' if count != 1 else ''} into "
        f"{count * 2} pages ({len(spreads)} pages in, {new_num} out)."
    )
    logger.info(summary)
    return out, summary
