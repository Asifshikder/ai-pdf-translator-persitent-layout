"""Standalone image localization pipeline: takes raw image bytes and returns localized image bytes.

Uses the same classify and localize logic as the PDF image localizer, but operates on
individual images instead of PDF-embedded images.
"""

import logging

from image_localizer import classify_image, localize_image

logger = logging.getLogger(__name__)


def localize_single_image(image_bytes: bytes, mime_type: str) -> bytes:
    """Localize a single image for Bangladeshi culture.

    Args:
        image_bytes: Raw image file bytes
        mime_type: Image MIME type (e.g., 'image/jpeg', 'image/png')

    Returns:
        Localized image bytes, or original image if localization fails or is not needed.
    """
    # Normalize MIME type
    if not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    # Classify the image
    classification = classify_image(image_bytes, mime_type)
    needs_localization = classification.get("needs_localization", False)

    if not needs_localization:
        logger.info(
            "Image does not need localization: %s",
            classification.get("reason", "unknown"),
        )
        return image_bytes

    categories = classification.get("categories", [])
    logger.info("Localizing image with categories: %s", categories)

    # Attempt localization
    localized_bytes = localize_image(image_bytes, mime_type, categories)

    # Return localized image if successful, otherwise original
    if localized_bytes:
        logger.info("Image localization succeeded")
        return localized_bytes

    logger.warning("Image localization failed; returning original")
    return image_bytes
