"""Full-page AI redesign pipeline: translate, culturally localize images, and relay out
every page onto a small fixed library of page templates (see page_templates.py).

Unlike pdf_processor.translate_pdf (which re-inserts Bangla into the source layout) or
image_processor.localize_pdf (which swaps individual images in place), this pipeline
redraws each page from scratch: one AI call per page chooses a template and decides,
per image, whether it needs cultural adaptation or can pass through untouched (logos
always pass through untouched). Because a template's slots are fixed, non-overlapping
rectangles, visual consistency and overlap-freedom are structural guarantees rather than
something the prompt has to achieve on its own — the AI never invents page geometry.

Pagination stays 1:1 with the source (one output page per source page), which is what
keeps a table-of-contents page's page-number column correct without any renumbering: see
_place_toc_page.
"""

import html
import json
import logging
import time

import fitz  # PyMuPDF
from google.genai import types

import copy_layer
import image_localizer
import manifest
import page_templates
from image_processor import _page_ink_coverage, _palette_summary, _style_summary
from pdf_processor import FONTS_DIR, BANGLA_SIZE, SCALE_LADDER, _extract_segments, _span_color_to_css, _vector_marks
from shortener import shorten_batch
from translator import translate_batch_status
from vertex_client import generate_content

logger = logging.getLogger(__name__)

PLAN_MODEL = "gemini-3-flash-preview"
PLAN_DPI = 150
STYLE_DPI = 150

# Page templates the model may choose freely. "toc_index" is deliberately excluded: a
# table of contents needs its page-number column preserved verbatim (see _place_toc_page),
# which is a correctness requirement rather than a style choice, so it is forced
# server-side by _looks_like_toc rather than left to the model's judgement.
_FREE_TEMPLATE_IDS = [t for t in page_templates.TEMPLATES if t != "toc_index"]

BASE_CSS = """
@font-face { font-family: bengali; src: url(NotoSansBengali-Regular.ttf); }
@font-face { font-family: bengali; src: url(NotoSansBengali-Bold.ttf); font-weight: bold; }
* { font-family: bengali, sans-serif; margin: 0; padding: 0; line-height: 1.45; }
p { margin: 0 0 6px 0; }
"""


def _template_catalog_text() -> str:
    lines = []
    for tid in _FREE_TEMPLATE_IDS:
        tpl = page_templates.TEMPLATES[tid]
        text_ids = ", ".join(s.id for s in tpl.text_slots) or "none"
        image_ids = ", ".join(s.id for s in tpl.image_slots) or "none"
        logo = tpl.logo_slot.id if tpl.logo_slot else "none"
        lines.append(
            f"- {tid}: text slots [{text_ids}]; image slots [{image_ids}]; logo slot [{logo}]"
        )
    return "\n".join(lines)


PLAN_SYSTEM_PROMPT = """You are laying out one page of a redesigned Bangladeshi edition of a \
health manual. The page will be rebuilt from scratch onto one of a small set of fixed page \
templates, so your job is to choose the right template for this page's content and assign \
each text segment and each image to one of that template's named slots.

TEMPLATES AVAILABLE (choose exactly one template_id):
{catalog}

BOOK STYLE (for context only; you do not need to repeat it):
{style}

INPUT
- A rendered image of the current page.
- A JSON list of "segments": each has an "index" and its English "text", in reading order.
- A JSON list of "images": each has an "xref", its "rect_fraction" (approximate position on \
the page as [x0,y0,x1,y1] fractions), and pixel "width"/"height".
- "hints": cheap signals already computed (segment_count, ink_coverage). They are hints, not \
answers — use your own judgement from the rendered page.

WHAT TO DECIDE
1. template_id: the one template whose slot layout best fits this page's content. A page \
that is mostly one flow of paragraphs is body_text; a page with one clear picture and \
supporting text is body_image_right or body_image_top depending on whether the picture reads \
better beside or above the text; a title/section-opener page is chapter_title; a checklist or \
exercise page with several short discrete items is exercise_grid; a front cover is cover; \
anything else with two balanced blocks of text is two_column_body.
2. page_role: one short word describing the page's function — cover, chapter_title, body, \
exercise, or other.
3. text_assignments: for EVERY segment index given, which slot_id of the chosen template it \
belongs in. Segments that clearly belong together (a paragraph and the next paragraph of the \
same flow) should go to the same slot, in their given order. Do not invent slot ids that are \
not listed for your chosen template.
4. image_decisions: for EVERY image xref given, decide:
   - slot_id: which of the chosen template's image slots (or its logo slot) it belongs in. \
Skip an image (set action to "omit") if the template you chose has nowhere sensible for it.
   - is_logo: true only if this is an organisation's actual logo/brand mark. A logo is NEVER \
regenerated and never receives action "regenerate" — always "keep" (or "omit" if it does not \
fit anywhere).
   - information_role: "referential" if the picture itself IS a stated fact (a diagram, a \
chart, a labelled illustration whose content is the information — e.g. "your heart" anatomy \
diagram, a dosage chart), "decorative" otherwise (a generic photo/scene/illustration that only \
sets a mood or shows people/places).
   - action: "keep" (pixel-identical, no AI edit) for logos and referential images, or \
anything already fine for a Bangladeshi reader; "regenerate" (culturally adapt: people, \
clothing, settings, food, objects, signage redrawn as Bangladeshi) for decorative images whose \
content reads as a specific other culture/country and would look foreign if left as-is; \
"omit" if it has nowhere to go in the chosen template.
   - reason: one short phrase.

Be decisive about is_logo and about referential-vs-decorative: get these two judgements right \
above all, because a logo that gets redrawn or a diagram whose data gets altered is a much \
worse mistake than a merely suboptimal template choice."""


def _p(text: str, size: float, color: str, align: str, bold: bool) -> str:
    weight = "font-weight:bold;" if bold else ""
    return (
        f'<p style="font-size:{size:.1f}px;color:{color};text-align:{align};{weight}">'
        f"{html.escape(text)}</p>"
    )


def _fit_html(
    page: fitz.Page, rect: fitz.Rect, paragraphs: list[str], archive: fitz.Archive
) -> tuple[float, float]:
    """insert_htmlbox the given <p> paragraphs into rect, shrinking per SCALE_LADDER.

    Mirrors pdf_processor.render_segment's shrink-to-fit contract: tries each floor in
    SCALE_LADDER until insert_htmlbox reports a fit (spare_height >= 0). SCALE_LADDER's
    last rung is 0.0, so this always draws something once given non-empty paragraphs.
    """
    body = "".join(paragraphs)
    if not body:
        return 0.0, 1.0
    for low in SCALE_LADDER:
        spare_height, scale = page.insert_htmlbox(
            rect, body, css=BASE_CSS, scale_low=low, archive=archive
        )
        if spare_height >= 0:
            return spare_height, scale
    return spare_height, scale


def _place_text_slot(
    page: fitz.Page,
    archive: fitz.Archive,
    rect: fitz.Rect,
    segs: list[dict],
    translations: dict[int, str],
    heading: bool = False,
) -> tuple[bool, float]:
    """Draw every segment assigned to one slot as a flowing sequence of paragraphs.

    Each segment keeps its own size/color/weight (scaled by BANGLA_SIZE, as in
    pdf_processor.css_for) as an inline style, so a heading and a body paragraph
    routed to the same slot still render at their own sizes.
    """
    paragraphs = []
    plain_parts = []
    max_size = 0.0
    for seg in segs:
        text = translations.get(id(seg), "").strip()
        if not text:
            continue
        prefix = "• " if seg.get("bullet") else ""
        text = prefix + text
        size = seg["size"] * BANGLA_SIZE
        max_size = max(max_size, size)
        align = "center" if heading else "left"
        paragraphs.append(_p(text, size, _span_color_to_css(seg["color"]), align, seg["bold"]))
        plain_parts.append(text)
    if not paragraphs:
        return True, 1.0

    spare_height, scale = _fit_html(page, rect, paragraphs, archive)
    ok = spare_height >= 0
    if not ok:
        # Last resort: shorten every assigned segment's Bangla and try once more at the
        # shrink floor — the same fallback fix_processor.py uses for an overflowing box.
        items = [{"bn": t, "en": ""} for t in plain_parts]
        shortened = shorten_batch(items)
        paragraphs = [
            _p(prefix_text, s["size"] * BANGLA_SIZE, _span_color_to_css(s["color"]),
               "center" if heading else "left", s["bold"])
            for prefix_text, s in zip(shortened, segs)
        ]
        spare_height, scale = _fit_html(page, rect, paragraphs, archive)
        ok = spare_height >= 0
        plain_parts = shortened

    joined = " ".join(plain_parts)
    copy_layer.add_invisible_text(page, rect, joined, max(max_size * max(scale, 0.0), 1.0))
    return ok, scale


def _place_toc_page(
    page: fitz.Page,
    archive: fitz.Archive,
    template: page_templates.Template,
    segments: list[dict],
    translations: dict[int, str],
) -> list[dict]:
    """Deterministic TOC/index layout: page numbers are copied verbatim, never re-planned.

    Segments carrying a trailing page-number (see pdf_processor._extract_segments) are TOC
    rows; everything else is heading text ("Contents", "Index"). Because pagination stays
    1:1 with the source, the numbers need no renumbering — only the titles are translated.
    """
    slot_by_id = {s.id: s for s in template.text_slots}
    heading_segs = [s for s in segments if s.get("number") is None]
    row_segs = [s for s in segments if s.get("number") is not None]

    text_assignments = []
    if heading_segs and "title" in slot_by_id:
        ok, _ = _place_text_slot(
            page, archive, page_templates.resolve(slot_by_id["title"], page.rect),
            heading_segs, translations, heading=True,
        )
        text_assignments.append({"slot": "title", "segments": len(heading_segs), "ok": ok})

    if row_segs and "titles" in slot_by_id:
        ok, _ = _place_text_slot(
            page, archive, page_templates.resolve(slot_by_id["titles"], page.rect),
            row_segs, translations,
        )
        text_assignments.append({"slot": "titles", "segments": len(row_segs), "ok": ok})

    if row_segs and template.numeral_slot_id in slot_by_id:
        rect = page_templates.resolve(slot_by_id[template.numeral_slot_id], page.rect)
        # Plain Latin digits need no HarfBuzz shaping trick to stay copy-pasteable: the
        # whole reshape/reorder problem copy_layer works around is specific to Bengali
        # conjuncts and matras. Drawn directly with a base-14 font via insert_text, a
        # digit string already extracts correctly with no invisible overlay needed — and
        # it must be a font other than NotoSansBengali, which embeds no Latin digit
        # glyphs at all (PyMuPDF silently substitutes its own fallback font for them,
        # which blank_shaped_tounicode then correctly blanks as one of manifest.py's
        # FALLBACK_FONTS, erasing the numbers). See test_page_redesign.py for the
        # regression this guards.
        n = len(row_segs)
        row_h = rect.height / n
        for i, seg in enumerate(row_segs):
            text = seg["number"]["text"].strip()
            size = min(seg["size"], row_h * 0.7)
            width = fitz.get_text_length(text, fontname="helv", fontsize=size)
            baseline = fitz.Point(rect.x1 - width, rect.y0 + row_h * i + row_h * 0.5 + size * 0.35)
            page.insert_text(baseline, text, fontsize=size, fontname="helv")
        text_assignments.append({"slot": template.numeral_slot_id, "segments": n, "ok": True})

    return text_assignments


def _looks_like_toc(segments: list[dict]) -> bool:
    if len(segments) < 4:
        return False
    numbered = sum(1 for s in segments if s.get("number") is not None)
    return numbered / len(segments) >= 0.5


def _role_hints(page: fitz.Page, segments: list[dict]) -> dict:
    return {
        "segment_count": len(segments),
        "ink_coverage": round(_page_ink_coverage(page), 3),
    }


def _page_images(doc: fitz.Document, page: fitz.Page) -> list[dict]:
    """Every distinct raster image placed on this page, with its bytes and placement rect.

    Vector-drawn illustrations (no XObject image, just paths) are out of scope for this
    first version — image_processor.py's image_regions._illustration_clusters would be the
    place to add them later; here only page.get_images (real XObjects) is read.
    """
    result = []
    seen = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        try:
            extracted = doc.extract_image(xref)
        except Exception:
            logger.debug("Could not extract image xref %d", xref, exc_info=True)
            continue
        ext = extracted.get("ext", "png")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        rect = rects[0]
        result.append({
            "xref": xref,
            "rect": rect,
            "rect_fraction": [
                round((rect.x0 - page.rect.x0) / max(page.rect.width, 1), 3),
                round((rect.y0 - page.rect.y0) / max(page.rect.height, 1), 3),
                round((rect.x1 - page.rect.x0) / max(page.rect.width, 1), 3),
                round((rect.y1 - page.rect.y0) / max(page.rect.height, 1), 3),
            ],
            "bytes": extracted["image"],
            "mime": mime,
            "width": extracted.get("width", 0),
            "height": extracted.get("height", 0),
        })
    return result


_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "template_id": {"type": "STRING", "enum": _FREE_TEMPLATE_IDS},
        "page_role": {
            "type": "STRING",
            "enum": ["cover", "chapter_title", "body", "exercise", "other"],
        },
        "text_assignments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "segment_index": {"type": "INTEGER"},
                    "slot_id": {"type": "STRING"},
                },
                "required": ["segment_index", "slot_id"],
            },
        },
        "image_decisions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "xref": {"type": "INTEGER"},
                    "slot_id": {"type": "STRING"},
                    "action": {"type": "STRING", "enum": ["keep", "regenerate", "omit"]},
                    "is_logo": {"type": "BOOLEAN"},
                    "information_role": {"type": "STRING", "enum": ["decorative", "referential"]},
                    "reason": {"type": "STRING"},
                },
                "required": ["xref", "slot_id", "action", "is_logo", "information_role"],
            },
        },
    },
    "required": ["template_id", "page_role", "text_assignments", "image_decisions"],
}

MAX_PLAN_ATTEMPTS = 3


def _plan_page(
    page: fitz.Page, segments: list[dict], images: list[dict], style: dict, hints: dict
) -> dict:
    """One Gemini call: choose a template and assign this page's text/images to its slots.

    On any failure, falls back to template_id="body_text" with every segment routed to its
    single "body" slot and every image kept as-is — a safe, always-renderable default.
    """
    fallback = {
        "template_id": "body_text",
        "page_role": "other",
        "text_assignments": [{"segment_index": i, "slot_id": "body"} for i in range(len(segments))],
        "image_decisions": [
            {"xref": img["xref"], "slot_id": "logo", "action": "keep",
             "is_logo": False, "information_role": "referential", "reason": "fallback"}
            for img in images
        ],
    }
    if not segments and not images:
        return fallback

    png_bytes = page.get_pixmap(dpi=PLAN_DPI).tobytes("png")
    payload = json.dumps({
        "segments": [{"index": i, "text": s["text"]} for i, s in enumerate(segments)],
        "images": [
            {"xref": img["xref"], "rect_fraction": img["rect_fraction"],
             "width": img["width"], "height": img["height"]}
            for img in images
        ],
        "hints": hints,
    }, ensure_ascii=False)
    system_instruction = PLAN_SYSTEM_PROMPT.format(
        catalog=_template_catalog_text(),
        style=style.get("text", "no measured style available"),
    )
    contents = [types.Part.from_bytes(data=png_bytes, mime_type="image/png"), payload]

    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=PLAN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=_PLAN_SCHEMA,
                    temperature=0.0,
                ),
            )
            plan = json.loads(response.text)
            if isinstance(plan, dict) and plan.get("template_id") in _FREE_TEMPLATE_IDS:
                return plan
            logger.warning("Plan call returned an unusable template_id: %r", plan.get("template_id") if isinstance(plan, dict) else plan)
        except Exception:
            logger.warning("Plan call failed (attempt %d/%d)", attempt, MAX_PLAN_ATTEMPTS, exc_info=True)
    logger.warning("Falling back to body_text layout for a page the planner could not handle")
    return fallback


def _book_style_profile(doc: fitz.Document) -> dict:
    """A description of the book's look, measured once and reused on every page/image call.

    Naming the actual palette and drawing technique (see image_processor._palette_summary /
    _style_summary) is what keeps forty AI-regenerated pages and images reading as one
    coherent redesign instead of forty independent improvisations.
    """
    try:
        png_bytes = doc[0].get_pixmap(dpi=STYLE_DPI).tobytes("png")
        palette = _palette_summary(png_bytes)
        style = _style_summary(png_bytes)
    except Exception:
        logger.warning("Could not measure a book style profile", exc_info=True)
        palette, style = "", ""
    text = " ".join(p for p in (palette, style) if p) or "no measured style available"
    return {"palette": palette, "style": style, "text": text}


def _place_images(
    page: fitz.Page,
    template: page_templates.Template,
    images: list[dict],
    decisions: list[dict],
    style: dict,
) -> list[dict]:
    image_slot_by_id = {s.id: s for s in template.image_slots}
    logo_slot = template.logo_slot
    images_by_xref = {img["xref"]: img for img in images}
    decision_by_xref = {d.get("xref"): d for d in decisions if isinstance(d, dict)}

    record = []
    for img in images:
        decision = decision_by_xref.get(img["xref"]) or {
            "slot_id": next(iter(image_slot_by_id), None), "action": "keep",
            "is_logo": False, "information_role": "referential", "reason": "no decision",
        }
        is_logo = bool(decision.get("is_logo"))
        action = "keep" if is_logo else decision.get("action", "keep")

        if is_logo:
            slot = logo_slot
        else:
            slot = image_slot_by_id.get(decision.get("slot_id"))
        if slot is None or action == "omit":
            record.append({"xref": img["xref"], "slot": None, "action": "omit"})
            continue

        rect = page_templates.resolve(slot, page.rect)
        image_bytes = img["bytes"]
        final_action = action
        if action == "regenerate":
            edited = image_localizer.localize_image(
                image_bytes, img["mime"], categories=[], page_context="",
                palette=style.get("palette", ""), style=style.get("style", ""), mode="simple",
            )
            if edited:
                image_bytes = edited
            else:
                final_action = "keep_fallback"

        try:
            page.insert_image(rect, stream=image_bytes)
        except Exception:
            logger.warning("Could not place image xref %d in slot %s", img["xref"], slot.id, exc_info=True)
            continue

        record.append({
            "xref": img["xref"], "slot": slot.id, "action": final_action,
            "is_logo": is_logo, "reason": decision.get("reason", ""),
        })
    return record


def redesign_pdf(pdf_bytes: bytes) -> bytes:
    """Translate, culturally localize images, and relay out every page of a PDF."""
    start_time = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    archive = fitz.Archive(FONTS_DIR)
    style_profile = _book_style_profile(doc)
    source_fonts = set()
    pages_meta = []

    for page_num, page in enumerate(doc, start=1):
        source_fonts |= manifest.page_span_fonts(page)
        segments, _kept = _extract_segments(page, _vector_marks(page))
        images = _page_images(doc, page)
        hints = _role_hints(page, segments)
        is_toc = _looks_like_toc(segments)

        plan = _plan_page(page, segments, images, style_profile, hints)
        template_id = "toc_index" if is_toc else plan.get("template_id", "body_text")
        page_role = "toc" if is_toc else plan.get("page_role", "other")
        template = page_templates.TEMPLATES.get(template_id, page_templates.TEMPLATES["body_text"])

        english = [seg["text"] for seg in segments]
        bangla, status = translate_batch_status(english)
        translations = {id(seg): text for seg, text in zip(segments, bangla)}

        # Whole-page redact before anything is drawn: with every image and drawn path
        # removed, nothing can end up sitting above the new raster (see
        # image_processor._apply_cover, which this mirrors).
        page.add_redact_annot(page.rect)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=fitz.PDF_REDACT_TEXT_NONE,
        )

        image_record = _place_images(page, template, images, plan.get("image_decisions", []), style_profile)

        if is_toc:
            text_record = _place_toc_page(page, archive, template, segments, translations)
        else:
            slot_by_id = {s.id: s for s in template.text_slots}
            assigned: dict[str, list[dict]] = {s.id: [] for s in template.text_slots}
            default_slot = next(iter(assigned), None)
            for entry in plan.get("text_assignments", []):
                idx = entry.get("segment_index")
                if not isinstance(idx, int) or not (0 <= idx < len(segments)):
                    continue
                slot_id = entry.get("slot_id")
                if slot_id not in assigned:
                    slot_id = default_slot
                if slot_id is not None:
                    assigned[slot_id].append(segments[idx])
            text_record = []
            for slot_id, segs in assigned.items():
                if not segs:
                    continue
                rect = page_templates.resolve(slot_by_id[slot_id], page.rect)
                heading = slot_id in ("title", "heading")
                ok, scale = _place_text_slot(page, archive, rect, segs, translations, heading=heading)
                text_record.append({"slot": slot_id, "segments": len(segs), "ok": ok, "scale": round(scale, 3)})

        pages_meta.append(
            manifest.redesign_page_entry(page_num, template.name, page_role, text_record, image_record)
        )
        failed = status.count(False)
        if failed:
            logger.warning("Page %d: %d of %d segments left in English — translation failed", page_num, failed, len(segments))

    data = manifest.redesign_build(style_profile, source_fonts, pages_meta)
    manifest.redesign_attach(doc, data)

    doc.subset_fonts()
    # Must run after subset_fonts(), which rewrites ToUnicode — see copy_layer.py.
    copy_layer.blank_shaped_tounicode(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    logger.info("PDF redesign complete: %.1f seconds", time.time() - start_time)
    return out
