"""Extract all images from a PDF and return them as a ZIP file."""

import io
import logging
import zipfile
from typing import BinaryIO

import fitz

logger = logging.getLogger(__name__)


def extract_images(pdf_bytes: bytes) -> bytes:
    """
    Extract all images from a PDF and return them as a ZIP file.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        ZIP file bytes containing all extracted images.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zip_buffer = io.BytesIO()

    image_count = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)

            for img_index, img_ref in enumerate(image_list, start=1):
                try:
                    xref = img_ref[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Convert to RGB if needed (handles CMYK, grayscale, etc.)
                    if pix.n - pix.alpha != 3:
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix = pix_rgb
                        ext = "jpg"
                        img_data = pix.tobytes("jpeg")
                    else:
                        ext = "png"
                        img_data = pix.tobytes("png")

                    # Create filename
                    filename = f"page_{page_num:03d}_image_{img_index:02d}.{ext}"
                    zf.writestr(filename, img_data)
                    image_count += 1
                    pix = None

                except Exception as e:
                    logger.warning(
                        f"Failed to extract image from page {page_num}, index {img_index}: {e}"
                    )

    doc.close()

    if image_count == 0:
        logger.warning("No images found in PDF")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
