"""Image localization pipeline: find images in a PDF, and for the ones that depict
culturally-specific content, replace them with a Bangladeshi-adapted version in place.

The existing text-translation pipeline (pdf_processor.py) is untouched; this is a
fully separate flow. Layout is preserved for free: PyMuPDF's page.replace_image swaps
the pixel content of an existing image object, so every placement of that image keeps
its exact position and size."""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image

from image_localizer import classify_image, localize_image

logger = logging.getLogger(__name__)

# Images smaller than this (pixels, either dimension) are icons, bullets, rules or
# logos — never worth a model call. Skipped before any AI classification.
MIN_DIMENSION = 40

# Number of concurrent image processing tasks (classify + edit)
CONCURRENT_IMAGES = 3


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


def _is_axis_aligned(matrix: fitz.Matrix) -> bool:
    """Check if a transformation matrix is axis-aligned (no rotation/shear).

    An axis-aligned matrix has only scale/translate, no rotation or shear.
    PyMuPDF matrices are [a,b,c,d,e,f] where [a,b] describes where X-axis
    maps and [c,d] describes where Y-axis maps. Axis-aligned means b≈0 and c≈0.
    """
    tolerance = 1e-6
    return abs(matrix.b) < tolerance and abs(matrix.c) < tolerance


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

    # Collect all unique xrefs across all pages, and their first-seen page number.
    unique_xrefs: dict[int, int] = {}
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref not in unique_xrefs:
                unique_xrefs[xref] = page_num

    # Extract page context strings in the main thread (PyMuPDF is not thread-safe).
    # Build a map of xref -> page_context for use in worker threads.
    xref_contexts: dict[int, str] = {}
    for xref, page_num in unique_xrefs.items():
        page = doc[page_num - 1]
        context = _get_page_context(page, fitz.Rect(0, 0, page.rect.width, page.rect.height))
        xref_contexts[xref] = context

    # Process all unique images concurrently.
    results: dict[int, bytes | None] = {}
    statuses: dict[int, str] = {}
    logger.info("Found %d unique images to process", len(unique_xrefs))
    logger.debug("Unique xrefs: %s", list(unique_xrefs.keys()))
    with ThreadPoolExecutor(max_workers=CONCURRENT_IMAGES) as executor:
        futures = {
            executor.submit(_decide, doc, xref, page_num, xref_contexts[xref]): xref
            for xref, page_num in unique_xrefs.items()
        }
        for future in as_completed(futures):
            xref = futures[future]
            try:
                image_bytes, status = future.result()
                results[xref] = image_bytes
                statuses[xref] = status
            except Exception as exc:
                logger.exception("Image xref %d processing failed", xref)
                results[xref] = None
                statuses[xref] = "edit_failed"

    # Apply the processed images to all pages.
    for page_num, page in enumerate(doc, start=1):
        # Unique xrefs used on this page, in first-seen order.
        xrefs: list[int] = []
        for img in page.get_images(full=True):
            if img[0] not in xrefs:
                xrefs.append(img[0])

        placements: list[tuple[fitz.Rect, bytes]] = []
        for xref in xrefs:
            new_image = results[xref]
            if new_image is None:
                continue
            rects_with_transforms = page.get_image_rects(xref, transform=True)
            for item in rects_with_transforms:
                if isinstance(item, tuple):
                    rect, matrix = item
                    if not _is_axis_aligned(matrix):
                        logger.debug(
                            "Page %d: xref %d skipped — rotated/sheared placement, "
                            "keeping original to avoid repainting bounding box",
                            page_num,
                            xref,
                        )
                        continue
                else:
                    rect = item
                placements.append((rect, new_image))

        if not placements:
            continue

        logger.debug(
            "Page %d: replacing %d images at rects: %s",
            page_num,
            len(placements),
            [f"({r.x0:.0f},{r.y0:.0f},{r.x1:.0f},{r.y1:.0f})" for r, _ in placements],
        )

        # For each placement: redact old image, reinsert new one. Do this one at a time
        # to keep page state consistent and avoid rendering artifacts from batch operations.
        for i, (rect, new_image) in enumerate(placements):
            try:
                # Remove the old image by marking and applying redaction
                page.add_redact_annot(rect)
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_REMOVE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_NONE,
                )
                # Insert the new image. Use overlay=True (put image on top) rather than
                # overlay=False (behind text), because after redacting the old image,
                # the rendering order can become ambiguous. Putting it on top ensures
                # the new image is always visible and not hidden behind text fragments.
                page.insert_image(rect, stream=new_image, keep_proportion=False, overlay=True)
                logger.debug(
                    "Page %d: placement %d/%d inserted at (%.0f,%.0f,%.0f,%.0f)",
                    page_num, i + 1, len(placements), rect.x0, rect.y0, rect.x1, rect.y1,
                )
            except Exception as exc:
                logger.exception(
                    "Page %d: failed to insert image at placement %d/%d", page_num, i + 1, len(placements)
                )

    status_counts = {}
    for status in statuses.values():
        status_counts[status] = status_counts.get(status, 0) + 1

    summary_parts = []
    for status in ["edit_ok", "classify_kept", "too_small", "extract_failed", "decode_failed", "classify_failed", "edit_failed"]:
        count = status_counts.get(status, 0)
        if count > 0:
            summary_parts.append(f"{count} {status}")

    summary = ", ".join(summary_parts) if summary_parts else "no images"
    logger.info("Image localization complete: %s", summary)

    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out


def _get_page_context(page: fitz.Page, rect: fitz.Rect, max_chars: int = 300) -> str:
    """Extract nearby text from the page as context for image editing.

    Expands the image rect by 50pt in each direction, then extracts text from
    that region, capped to max_chars.
    """
    if page is None:
        return ""
    try:
        margin = 50
        expanded = rect + (-margin, -margin, margin, margin)
        text = page.get_textbox(expanded)
        if text:
            return text[:max_chars].strip()
    except Exception:
        logger.debug("Could not extract page context for rect", exc_info=True)
    return ""


def _decide(
    doc: fitz.Document, xref: int, page_num: int, page_context: str = ""
) -> tuple[bytes | None, str]:
    """Classify one image and, if it needs localization, return the edited bytes
    (resized to the original pixel dimensions) and status. Returns (None, status) to keep original.

    Status is one of: too_small, extract_failed, decode_failed, classify_kept,
    classify_failed, edit_failed, edit_ok.
    """
    try:
        extracted = doc.extract_image(xref)
    except Exception:
        logger.debug("Page %d: xref %d extract_image failed", page_num, xref)
        return None, "extract_failed"

    if not extracted or not extracted.get("image"):
        return None, "extract_failed"

    if extracted["width"] < MIN_DIMENSION or extracted["height"] < MIN_DIMENSION:
        logger.debug(
            "Page %d: xref %d skipped — too small (%dx%d, min %d)",
            page_num, xref, extracted["width"], extracted["height"], MIN_DIMENSION
        )
        return None, "too_small"

    normalized = _to_png(extracted["image"])
    if normalized is None:
        logger.debug("Page %d: xref %d could not decode image", page_num, xref)
        return None, "decode_failed"

    png_bytes, width, height = normalized

    start = time.time()
    decision = classify_image(png_bytes, "image/png")
    classify_time = time.time() - start

    if not decision.get("needs_localization"):
        logger.debug(
            "Page %d: xref %d kept — %s (%.1fs)",
            page_num,
            xref,
            decision.get("reason", "no localization needed"),
            classify_time,
        )
        return None, "classify_kept"

    if decision.get("reason", "").lower().startswith("classify failed"):
        return None, "classify_failed"

    start = time.time()
    new_image = localize_image(
        png_bytes, "image/png", decision.get("categories", []), page_context
    )
    edit_time = time.time() - start

    if not new_image:
        logger.debug(
            "Page %d: xref %d edit failed (%.1fs)",
            page_num, xref, edit_time
        )
        return None, "edit_failed"

    logger.info(
        "Page %d: xref %d localized (%s) — classify: %.1fs, edit: %.1fs",
        page_num,
        xref,
        ", ".join(decision.get("categories", [])) or "cultural content",
        classify_time,
        edit_time,
    )
    return _resize_to(new_image, width, height), "edit_ok"
