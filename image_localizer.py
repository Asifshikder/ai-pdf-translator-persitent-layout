"""Vertex AI (Gemini) image localizer: decide whether a PDF image needs cultural
adaptation for a Bangladeshi audience, and if so, edit it in place.

Two model calls per image:
  1. classify_image  — cheap multimodal Gemini call, structured JSON decision.
  2. localize_image  — image-editing model that returns an edited image.

Both fail safe: on any error the caller keeps the original image, so the PDF is
never corrupted (mirrors translator._translate_chunk)."""

import json
import logging

from google.genai import types

from vertex_client import generate_content

logger = logging.getLogger(__name__)

# Cheap multimodal model for the yes/no decision; dedicated image-editing model
# ("nano-banana") for the actual regeneration.
CLASSIFY_MODEL = "gemini-2.5-flash"
EDIT_MODEL = "gemini-2.5-flash-image"

MAX_ATTEMPTS = 3

# The four culturally-specific categories the user asked to localize. Anything
# that fits none of these (logos, charts, diagrams, icons, decorative graphics,
# UI screenshots, or already-Bangladeshi content) is left untouched.
CATEGORIES = ["people_attire", "scenes_settings", "food_objects", "signage_text"]

CLASSIFY_SYSTEM_PROMPT = """You decide whether an image inside a document should be \
adapted for a Bangladeshi audience, so the document feels local to Bangladeshi readers.

Return needs_localization=true if the image depicts culturally-adaptable content in one \
or more of these categories:
- people_attire: ANY people or faces (photo, illustration, cartoon, or icon of a person). \
Their appearance and clothing can always be made Bangladeshi (saree, salwar kameez, \
panjabi, hijab), so an image containing a person should almost always be localized.
- scenes_settings: streets, buildings, landscapes, rooms, or environments
- food_objects: meals, produce, or everyday cultural objects
- signage_text: readable signs, labels, or text baked into the image (render in Bangla)

Be decisive, not cautious: if any person is visible, return true even for a simple or \
stylized illustration. A generic "Western/international" look is exactly what should be \
localized — do not keep an image just because it looks neutral.

Return needs_localization=false ONLY for images with NO people and NO cultural content: \
logos, charts, graphs, diagrams, flowcharts, icons of objects, buttons, QR codes, \
UI screenshots, decorative shapes/rules, and abstract or medical/anatomical/technical \
illustrations (e.g. an organ or a cross-section). Also false if the image already clearly \
looks Bangladeshi.

List only the categories that apply."""

EDIT_INSTRUCTION = """Edit this image to reflect Bangladeshi culture. Keep the EXACT \
composition, framing, camera angle, poses, and overall color palette, and keep the same \
aspect ratio and dimensions. Change ONLY culturally-specific elements: make any people \
Bangladeshi with appropriate attire (saree, salwar kameez, panjabi, hijab as fitting), \
adapt settings and architecture to Bangladesh, swap food and objects for local \
equivalents, and render any signage or text visible in the image in Bangla. Do not add \
or remove objects, and do not change the layout. Focus especially on: {focus}."""

_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "needs_localization": {"type": "BOOLEAN"},
        "categories": {"type": "ARRAY", "items": {"type": "STRING", "enum": CATEGORIES}},
        "reason": {"type": "STRING"},
    },
    "required": ["needs_localization", "categories", "reason"],
}


def classify_image(image_bytes: bytes, mime: str) -> dict:
    """Return {"needs_localization": bool, "categories": [...], "reason": str}.

    On any failure returns needs_localization=False so the image is kept as-is.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[part],
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_CLASSIFY_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "needs_localization" in data:
                data.setdefault("categories", [])
                data.setdefault("reason", "")
                return data
            logger.warning(
                "Classify: unexpected response shape (attempt %d/%d): %r",
                attempt,
                MAX_ATTEMPTS,
                response.text[:200],
            )
        except Exception:
            logger.exception(
                "Classify request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Classify: giving up — treating image as no-localization")
    return {"needs_localization": False, "categories": [], "reason": "classify failed"}


def localize_image(image_bytes: bytes, mime: str, categories: list[str]) -> bytes | None:
    """Return edited image bytes adapted to Bangladeshi culture, or None on failure.

    None means the caller keeps the original image untouched.
    """
    focus = ", ".join(categories) if categories else "any culturally-specific content"
    instruction = EDIT_INSTRUCTION.format(focus=focus)
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=EDIT_MODEL,
                contents=[instruction, part],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    temperature=0.4,
                ),
            )
            out = _first_image_bytes(response)
            if out:
                return out
            logger.warning(
                "Localize: no image in response (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
        except Exception:
            logger.exception(
                "Localize request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Localize: giving up — keeping original image")
    return None


def _first_image_bytes(response) -> bytes | None:
    """Pull the first inline image payload out of a generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None
