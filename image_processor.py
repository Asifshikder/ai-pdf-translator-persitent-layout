"""Image localization pipeline: find images in a PDF, and for the ones that depict
culturally-specific content, replace them with a Bangladeshi-adapted version in place.

The existing text-translation pipeline (pdf_processor.py) is untouched; this is a
fully separate flow. Layout is preserved for free: PyMuPDF's page.replace_image swaps
the pixel content of an existing image object, so every placement of that image keeps
its exact position and size."""

import io
import logging

import fitz  # PyMuPDF
from PIL import Image

from image_localizer import classify_image, localize_image

logger = logging.getLogger(__name__)

# Images smaller than this (pixels, either dimension) are icons, bullets, rules or
# logos — never worth a model call. Skipped before any AI classification.
MIN_DIMENSION = 64


def _to_png(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """Normalize any extracted image to PNG bytes. Returns (png_bytes, w, h) or None
    if the bytes can't be decoded (in which case the original image is kept)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), width, height
    except Exception:
        logger.exception("Could not decode extracted image; keeping original")
        return None


def _resize_to(image_bytes: bytes, width: int, height: int) -> bytes:
    """Stretch an edited image to the original pixel dimensions so the replacement
    keeps the original aspect ratio inside its fixed placement rectangle."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.exception("Could not resize edited image; using it as-is")
        return image_bytes


def localize_pdf(pdf_bytes: bytes) -> bytes:
    """Return the PDF with culturally-specific images adapted to Bangladeshi culture.

    Each unique image (by xref) is decided once and cached, so an image reused
    across pages costs a single classify/edit. The localized image is then placed
    on every page that uses it. Any failure keeps the original image.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # xref -> localized image bytes, or None if the image is kept as-is (or failed).
    results: dict[int, bytes | None] = {}

    for page_num, page in enumerate(doc, start=1):
        # Unique xrefs used on this page, in first-seen order.
        xrefs: list[int] = []
        for img in page.get_images(full=True):
            if img[0] not in xrefs:
                xrefs.append(img[0])

        placements: list[tuple[fitz.Rect, bytes]] = []
        for xref in xrefs:
            if xref not in results:
                results[xref] = _decide(doc, xref, page_num)
            new_image = results[xref]
            if new_image is None:
                continue
            for rect in page.get_image_rects(xref):
                placements.append((rect, new_image))

        if not placements:
            continue

        # Remove the old images, then paint the localized ones back at the exact
        # same rectangles. redact+insert produces a single clean image object that
        # every PDF viewer renders (unlike replace_image, which can leave a
        # duplicate reference that some viewers resolve to the original). Text and
        # vector graphics under the rect are preserved.
        for rect, _ in placements:
            page.add_redact_annot(rect)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_NONE,
        )
        for rect, new_image in placements:
            page.insert_image(rect, stream=new_image, keep_proportion=False)

    replaced = sum(1 for v in results.values() if v is not None)
    kept = sum(1 for v in results.values() if v is None)
    logger.info("Image localization done: %d localized, %d kept", replaced, kept)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out


def _decide(doc: fitz.Document, xref: int, page_num: int) -> bytes | None:
    """Classify one image and, if it needs localization, return the edited bytes
    (resized to the original pixel dimensions). Returns None to keep the original."""
    try:
        extracted = doc.extract_image(xref)
    except Exception:
        logger.exception("Page %d: could not extract image xref %d", page_num, xref)
        return None
    if not extracted or not extracted.get("image"):
        return None
    if extracted["width"] < MIN_DIMENSION or extracted["height"] < MIN_DIMENSION:
        return None

    normalized = _to_png(extracted["image"])
    if normalized is None:
        return None
    png_bytes, width, height = normalized

    decision = classify_image(png_bytes, "image/png")
    if not decision.get("needs_localization"):
        logger.info(
            "Page %d: xref %d kept — %s",
            page_num,
            xref,
            decision.get("reason", "no localization needed"),
        )
        return None

    new_image = localize_image(png_bytes, "image/png", decision.get("categories", []))
    if not new_image:
        logger.warning("Page %d: xref %d edit failed; keeping original", page_num, xref)
        return None

    logger.info(
        "Page %d: xref %d localized (%s)",
        page_num,
        xref,
        ", ".join(decision.get("categories", [])) or "cultural content",
    )
    return _resize_to(new_image, width, height)
