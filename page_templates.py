"""Fixed-slot page templates for the full-page AI redesign pipeline (page_redesign.py).

Every template's slots are stored as (x0, y0, x1, y1) fractions of the page rect, so the
same small library reproduces consistently regardless of a page's actual trim size — what
keeps a whole book's redesign reading as one product instead of forty separate layouts.
Slot rects within a template are pairwise disjoint by construction (verified once in
test_page_redesign.py, not at runtime) except for `cover`, whose title deliberately sits
over its own full-bleed image slot the way a real book cover does.

The AI planning call in page_redesign.py picks a `template_id` from TEMPLATES and assigns
each page's text segments and images to the chosen template's slot ids; it never invents
geometry of its own. That is what turns "consistent layout across the book" and "nothing
overlaps" from a prompt-only hope into a structural guarantee.
"""

from dataclasses import dataclass

import fitz  # PyMuPDF

Fraction = tuple[float, float, float, float]


@dataclass(frozen=True)
class Slot:
    id: str
    rect: Fraction  # (x0, y0, x1, y1) as fractions of the page


@dataclass(frozen=True)
class Template:
    name: str
    text_slots: tuple[Slot, ...] = ()
    image_slots: tuple[Slot, ...] = ()
    logo_slot: Slot | None = None
    # Slot id (within text_slots) that holds a preserved-verbatim page-number column,
    # e.g. a table of contents. None for every template except toc_index.
    numeral_slot_id: str | None = None


def resolve(slot: Slot, page_rect: fitz.Rect) -> fitz.Rect:
    """A slot's fraction rect, in this page's absolute coordinates."""
    x0, y0, x1, y1 = slot.rect
    w, h = page_rect.width, page_rect.height
    return fitz.Rect(
        page_rect.x0 + x0 * w,
        page_rect.y0 + y0 * h,
        page_rect.x0 + x1 * w,
        page_rect.y0 + y1 * h,
    )


TEMPLATES: dict[str, Template] = {
    "cover": Template(
        name="cover",
        image_slots=(Slot("cover_art", (0.00, 0.00, 1.00, 1.00)),),
        # Deliberately overlaps cover_art: a title band drawn over the full-bleed art,
        # like a real book cover. page_redesign draws a scrim behind it for legibility.
        text_slots=(Slot("title", (0.08, 0.72, 0.92, 0.94)),),
        logo_slot=Slot("logo", (0.04, 0.04, 0.34, 0.14)),
    ),
    "chapter_title": Template(
        name="chapter_title",
        text_slots=(Slot("heading", (0.10, 0.14, 0.90, 0.42)),),
        image_slots=(Slot("motif", (0.20, 0.48, 0.80, 0.86)),),
        logo_slot=Slot("logo", (0.06, 0.03, 0.30, 0.10)),
    ),
    "body_text": Template(
        name="body_text",
        text_slots=(Slot("body", (0.08, 0.10, 0.92, 0.90)),),
        logo_slot=Slot("logo", (0.80, 0.02, 0.96, 0.08)),
    ),
    "body_image_right": Template(
        name="body_image_right",
        text_slots=(Slot("body", (0.06, 0.10, 0.58, 0.90)),),
        image_slots=(Slot("figure", (0.62, 0.10, 0.94, 0.60)),),
        logo_slot=Slot("logo", (0.62, 0.64, 0.94, 0.70)),
    ),
    "body_image_top": Template(
        name="body_image_top",
        image_slots=(Slot("figure", (0.10, 0.06, 0.90, 0.44)),),
        text_slots=(Slot("body", (0.08, 0.50, 0.92, 0.92)),),
        logo_slot=Slot("logo", (0.06, 0.94, 0.30, 0.99)),
    ),
    "two_column_body": Template(
        name="two_column_body",
        text_slots=(
            Slot("col_left", (0.06, 0.10, 0.47, 0.90)),
            Slot("col_right", (0.53, 0.10, 0.94, 0.90)),
        ),
        logo_slot=Slot("logo", (0.80, 0.02, 0.96, 0.08)),
    ),
    "exercise_grid": Template(
        name="exercise_grid",
        text_slots=(
            Slot("header", (0.08, 0.06, 0.92, 0.16)),
            Slot("item_1", (0.08, 0.20, 0.47, 0.30)),
            Slot("item_2", (0.53, 0.20, 0.92, 0.30)),
            Slot("item_3", (0.08, 0.33, 0.47, 0.43)),
            Slot("item_4", (0.53, 0.33, 0.92, 0.43)),
            Slot("item_5", (0.08, 0.46, 0.47, 0.56)),
            Slot("item_6", (0.53, 0.46, 0.92, 0.56)),
            Slot("item_7", (0.08, 0.59, 0.47, 0.69)),
            Slot("item_8", (0.53, 0.59, 0.92, 0.69)),
        ),
        logo_slot=Slot("logo", (0.80, 0.02, 0.96, 0.055)),
    ),
    "toc_index": Template(
        name="toc_index",
        text_slots=(
            Slot("title", (0.08, 0.06, 0.92, 0.16)),
            Slot("titles", (0.08, 0.20, 0.72, 0.92)),
            Slot("numbers", (0.76, 0.20, 0.92, 0.92)),
        ),
        logo_slot=Slot("logo", (0.06, 0.02, 0.30, 0.055)),
        numeral_slot_id="numbers",
    ),
}


def all_slot_ids(template: Template) -> set[str]:
    ids = {s.id for s in template.text_slots} | {s.id for s in template.image_slots}
    if template.logo_slot:
        ids.add(template.logo_slot.id)
    return ids
