"""Prompt-driven PDF repair: a document, a sentence about what is wrong, and images.

A third, self-contained pipeline. Nothing else in the project imports it, and it changes
nothing that the translate / localize / regenerate / fix pipelines do. It exists for the
case those cannot serve: a human sees a specific problem, says in words what should happen,
and optionally hands over an image to put in the document. They do not say where it is —
"the subtitle that says 'Not just a book' is wrong" is the whole input, and finding it is
this module's job.

Design, and why it is this shape:

  * **Locate, then plan.** Two model calls, deliberately. Locating needs only a thumbnail
    and a text digest per page and answers "which page"; planning needs a full-resolution
    render and the page's measured geometry, which would be ruinous to send for all forty
    pages of a manual. So `_locate_pages` narrows and `_plan` is run only on what it picked.

  * **The model plans, PyMuPDF executes.** The model never produces a page. It answers with
    a short list of typed operations (erase / replace_text / insert_text / insert_image /
    regenerate_image / draw_box), and everything that actually touches the document is
    ordinary PyMuPDF applied to a rectangle. A model that misunderstands the request can
    therefore produce a wrong edit, but never a corrupt page, and every edit is reported
    and undoable.

  * **Boxes come back on a 0-1000 grid** (`box_2d = [y0, x0, y1, x1]`, Gemini's own
    convention), which this project has measured to be far more reliable than asking for
    0-1 fractions. Better still, the page's real text blocks are handed to the model with
    ids, and an operation may target `block_id` instead — then the rectangle is the one
    PyMuPDF measured, not one the model eyeballed.

  * **One redaction pass, then everything else.** `apply_redactions` rewrites the whole
    page content stream, so anything inserted before it would be wiped. Every erase is
    collected first, applied once, and only then is new content drawn.

  * **The undo step is the instruction, not the page.** One sentence may turn out to touch
    three pages; undoing it takes back all three, because that is the unit the user asked
    for. Sessions exist for that, and so the user can keep correcting the same document.

State lives in memory in this process only: this is a local single-user tool, and a
restart losing an unfinished session is cheaper than a temp-file lifecycle to get wrong.
"""

import html
import io
import json
import logging
import statistics
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from google.genai import types
from PIL import Image

from image_regen import EDIT_MODELS, _first_image_bytes, _resize_to
from pdf_processor import FONTS_DIR, _rules
from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

PAGEFIX_SUFFIX = "_pagefix.pdf"  # keep in sync with static/pagefix.html

# The planner. Vision + reasoning over a page render; every other pipeline in this project
# uses the same model for its judgement calls, and this one is verified enabled on both
# service accounts. Swap here if a stronger planner is provisioned.
PLAN_MODEL = "gemini-3-flash-preview"
PLAN_ATTEMPTS = 3

PREVIEW_DPI = 130   # what the browser shows
PLAN_DPI = 150      # the render the planner reads
CROP_DPI = 200      # region crops sent to the image model
MODEL_MAX_DIM = 1600  # longest side of any image handed to the planner

# Text that will not fit is shrunk rather than dropped. insert_htmlbox draws *nothing at
# all* when it cannot fit at the scale_low it was given, so the ladder must end at 0.0 —
# the same reasoning (and the same value) as SCALE_LADDER in pdf_processor.
SCALE_LADDER = (0.7, 0.45, 0.0)

# Bengali first so Bangla shapes correctly (HarfBuzz conjuncts/matras); sans-serif behind it
# so a Latin-only replacement still renders if a glyph is missing from Noto Sans Bengali.
PAGEFIX_CSS = """
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Regular.ttf); }}
@font-face {{ font-family: bengali; src: url(NotoSansBengali-Bold.ttf); font-weight: bold; }}
* {{
    font-family: bengali, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.45;
    font-size: {size:.1f}px;
    color: {color};
    text-align: {align};
    {bold}
}}
"""

ACTIONS = (
    "erase",
    "replace_text",
    "insert_text",
    "insert_image",
    "regenerate_image",
    "draw_box",
)

MAX_SESSIONS = 8
MAX_UNDO = 12
SESSION_TTL = 6 * 3600  # seconds of inactivity before a session is dropped

# A rectangle smaller than this in either direction is a mis-planned box, not an edit.
MIN_EDGE_PT = 3.0

# How far inside its rectangle a redaction is applied. See the call site: a hair, so that a
# box measured flush against a heading rule cannot clip it.
REDACT_INSET = 0.3


class PageFixError(Exception):
    """Anything the user can fix by doing something different (bad page, dead session)."""


# --------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------


@dataclass
class Attachment:
    name: str
    png: bytes
    width: int
    height: int


@dataclass
class Session:
    sid: str
    filename: str
    pdf: bytes
    page_count: int
    attachments: list[Attachment] = field(default_factory=list)
    history: list[bytes] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    touched: float = field(default_factory=time.time)


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.Lock()


def _evict() -> None:
    """Drop expired sessions, then the oldest ones over the cap. Caller holds the lock."""
    cutoff = time.time() - SESSION_TTL
    for sid in [s for s, sess in _sessions.items() if sess.touched < cutoff]:
        _sessions.pop(sid, None)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def create_session(filename: str, pdf_bytes: bytes) -> Session:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    doc.close()
    if not page_count:
        raise PageFixError("That PDF has no pages.")

    session = Session(
        sid=uuid.uuid4().hex,
        filename=filename or "document.pdf",
        pdf=pdf_bytes,
        page_count=page_count,
    )
    with _lock:
        _sessions[session.sid] = session
        _evict()
    return session


def get_session(sid: str) -> Session:
    with _lock:
        session = _sessions.get(sid)
        if session is None:
            raise PageFixError("That editing session has expired. Upload the PDF again.")
        session.touched = time.time()
        _sessions.move_to_end(sid)
        return session


def drop_session(sid: str) -> None:
    with _lock:
        _sessions.pop(sid, None)


def undo(sid: str) -> bool:
    """Roll the document back one applied instruction. False if there is nothing to undo."""
    session = get_session(sid)
    with _lock:
        if not session.history:
            return False
        session.pdf = session.history.pop()
        if session.log:
            session.log.pop()
        return True


def add_attachments(sid: str, images: list[tuple[str, bytes]]) -> list[Attachment]:
    """Register uploaded reference images with the session and return the full list."""
    session = get_session(sid)
    for name, raw in images:
        normalized = _normalize_image(raw)
        if normalized is None:
            logger.warning("Attachment %r could not be decoded and was ignored", name)
            continue
        png, width, height = normalized
        session.attachments.append(
            Attachment(name=name or f"image-{len(session.attachments) + 1}", png=png,
                       width=width, height=height)
        )
    return session.attachments


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------


def _normalize_image(raw: bytes) -> tuple[bytes, int, int] | None:
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), img.width, img.height
    except Exception:
        logger.debug("Undecodable image upload", exc_info=True)
        return None


def _shrink(png: bytes, max_dim: int = MODEL_MAX_DIM) -> bytes:
    """A copy whose longest side is <= max_dim. Boxes come back normalized, so this does
    not affect how the planner's coordinates map onto the page."""
    try:
        with Image.open(io.BytesIO(png)) as img:
            if max(img.size) <= max_dim:
                return png
            scale = max_dim / max(img.size)
            resized = img.convert("RGB").resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        logger.debug("Could not downscale for the model call", exc_info=True)
        return png


def render_page(pdf_bytes: bytes, page_index: int, dpi: int = PREVIEW_DPI) -> bytes:
    """Render one page to PNG."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 0 <= page_index < doc.page_count:
            raise PageFixError(f"This PDF has {doc.page_count} pages.")
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


# --------------------------------------------------------------------------------------
# Page geometry handed to the planner
# --------------------------------------------------------------------------------------


def _to_box2d(rect: fitz.Rect, page_rect: fitz.Rect) -> list[int]:
    w = max(page_rect.width, 1e-6)
    h = max(page_rect.height, 1e-6)
    return [
        max(0, min(1000, round((rect.y0 - page_rect.y0) / h * 1000))),
        max(0, min(1000, round((rect.x0 - page_rect.x0) / w * 1000))),
        max(0, min(1000, round((rect.y1 - page_rect.y0) / h * 1000))),
        max(0, min(1000, round((rect.x1 - page_rect.x0) / w * 1000))),
    ]


def _to_rect(box_2d: list[float], page_rect: fitz.Rect) -> fitz.Rect:
    y0, x0, y1, x1 = (float(v) for v in box_2d[:4])
    rect = fitz.Rect(
        page_rect.x0 + min(x0, x1) / 1000 * page_rect.width,
        page_rect.y0 + min(y0, y1) / 1000 * page_rect.height,
        page_rect.x0 + max(x0, x1) / 1000 * page_rect.width,
        page_rect.y0 + max(y0, y1) / 1000 * page_rect.height,
    )
    return rect & page_rect


def _text_blocks(page: fitz.Page) -> list[dict]:
    """The page's text, block by block, with the geometry and style PyMuPDF measured.

    Given to the planner so it can target an id instead of guessing a rectangle, and so a
    `replace_text` inherits the size and colour of the text it replaces.
    """
    blocks = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        text_parts, sizes, colors, bolds = [], [], [], []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                text_parts.append(line_text)
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                sizes.append(span.get("size", 0))
                colors.append(span.get("color", 0))
                bolds.append(bool(span.get("flags", 0) & (1 << 4)))
        text = " ".join(text_parts).strip()
        if not text:
            continue
        blocks.append(
            {
                "id": len(blocks),
                "rect": fitz.Rect(block["bbox"]),
                "text": text,
                "size": round(statistics.median(sizes), 1) if sizes else 10.0,
                "color": statistics.mode(colors) if colors else 0,
                "bold": bool(bolds) and sum(bolds) > len(bolds) / 2,
            }
        )
    return blocks


def _image_rects(page: fitz.Page) -> list[dict]:
    out = []
    for info in page.get_image_info(xrefs=True):
        rect = fitz.Rect(info["bbox"])
        if rect.is_empty or rect.width < MIN_EDGE_PT or rect.height < MIN_EDGE_PT:
            continue
        out.append({"id": len(out), "rect": rect, "xref": info.get("xref", 0)})
    return out


def _int_color_to_hex(color_int: int) -> str:
    return "#{:06x}".format(int(color_int) & 0xFFFFFF)


def _parse_color(value, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    """'#rrggbb' / '#rgb' / [r,g,b] → PyMuPDF's 0..1 float triple."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        nums = [float(v) for v in value]
        if max(nums) > 1.0:
            nums = [v / 255 for v in nums]
        return tuple(max(0.0, min(1.0, v)) for v in nums)
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))
            except ValueError:
                pass
    return default


def _hex_color(value, default="#000000") -> str:
    rgb = _parse_color(value, default=_parse_color(default))
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgb))


# --------------------------------------------------------------------------------------
# Background sampling
# --------------------------------------------------------------------------------------


def _surrounding_color(page_png: Image.Image, rect: fitz.Rect, page_rect: fitz.Rect
                       ) -> tuple[float, float, float]:
    """The colour of the page just *outside* a rectangle — what an erase should leave behind.

    Sampled from a frame around the box rather than inside it: the inside is the thing being
    removed. Median, not mean, so a neighbouring letter clipping the frame cannot tint the
    fill grey.
    """
    try:
        sx = page_png.width / max(page_rect.width, 1e-6)
        sy = page_png.height / max(page_rect.height, 1e-6)
        pad = max(3, int(min(page_png.size) * 0.006))
        x0 = int((rect.x0 - page_rect.x0) * sx)
        y0 = int((rect.y0 - page_rect.y0) * sy)
        x1 = int((rect.x1 - page_rect.x0) * sx)
        y1 = int((rect.y1 - page_rect.y0) * sy)

        outer = (
            max(0, x0 - pad), max(0, y0 - pad),
            min(page_png.width, x1 + pad), min(page_png.height, y1 + pad),
        )
        if outer[2] - outer[0] < 2 or outer[3] - outer[1] < 2:
            return (1.0, 1.0, 1.0)

        crop = page_png.crop(outer).convert("RGB")
        inner = (x0 - outer[0], y0 - outer[1], x1 - outer[0], y1 - outer[1])
        pixels = []
        step = max(1, (crop.width + crop.height) // 200)
        for y in range(0, crop.height, step):
            inside_rows = inner[1] <= y < inner[3]
            for x in range(0, crop.width, step):
                if inside_rows and inner[0] <= x < inner[2]:
                    continue
                pixels.append(crop.getpixel((x, y)))
        if not pixels:
            return (1.0, 1.0, 1.0)
        return tuple(
            statistics.median(p[channel] for p in pixels) / 255 for channel in range(3)
        )
    except Exception:
        logger.debug("Background sampling failed; falling back to white", exc_info=True)
        return (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------


PLAN_PROMPT = """You repair a single page of a PDF on behalf of a user who is looking at
that page and has told you, in their own words, what is wrong with it.

You do not produce a page. You produce a short list of edit operations, which a
deterministic PDF engine then applies. Answer ONLY with the JSON the schema describes.

COORDINATES
Every rectangle is `box_2d` = [y0, x0, y1, x1] on a 0-1000 grid over the page image:
y from the top edge, x from the left edge. Verify each box against the page image before
you emit it — a box in the wrong place damages a part of the page nobody asked you to touch.

TARGETING
The page's own text blocks are listed below with their measured rectangles and ids.
When your edit concerns one of them, set `block_id` and OMIT `box_2d`: the engine then uses
the exact rectangle PyMuPDF measured, which is always better than one you estimated.
Use `box_2d` only for a place that is not an existing text block (empty space, an image,
part of a block).

OPERATIONS
- `erase` — cover a rectangle with the surrounding page colour. Removes the text objects
  under it. Use for "delete this", "remove that logo", "get rid of the duplicated line".
- `replace_text` — erase a rectangle and write `text` into it. This is the normal way to
  correct wrong, garbled, mistranslated, or untranslated wording. Carry over the original
  size, colour, bold and alignment unless the user asked for a change; omit `font_size`
  and it is inherited from the text being replaced.
- `insert_text` — write `text` into a rectangle without erasing first. For adding something
  where there is blank space.
- `insert_image` — place attachment number `attachment` (0-based, from the list below) into
  a rectangle. Set `erase_first` true when it must cover existing content. Keep the
  rectangle's proportions close to the attachment's, or it will be letterboxed inside it.
- `regenerate_image` — hand the pixels currently inside a rectangle to an image model with
  your `prompt`, and put the result back in the same rectangle. Use only when the fix is a
  change to the artwork itself. Never use it over body text.
- `draw_box` — draw a filled and/or outlined rectangle. For redaction bars, panels, rules.

RULES
1. Do exactly what the user asked, on this page only. No unrequested tidying, restyling or
   re-translating, however tempting — an edit they did not ask for is a defect.
2. Prefer the fewest operations that accomplish it. If one `replace_text` does the job, one
   is the whole answer.
3. Bangla text: write real Bangla in `text`. The engine has a Bengali font and shapes it
   correctly. Never transliterate.
4. If the request cannot be carried out with these operations, or you cannot locate what
   the user means on the page, return an empty `operations` list and say why in `analysis`.
   An honest refusal is much cheaper than a wrong edit.
5. `note` on each operation says, in one short phrase, what that operation is for. The user
   reads these to decide whether to keep the result.
"""


_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "analysis": {"type": "STRING"},
        "operations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING", "enum": list(ACTIONS)},
                    "box_2d": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                    "block_id": {"type": "INTEGER"},
                    "text": {"type": "STRING"},
                    "font_size": {"type": "NUMBER"},
                    "color": {"type": "STRING"},
                    "align": {"type": "STRING", "enum": ["left", "center", "right", "justify"]},
                    "bold": {"type": "BOOLEAN"},
                    "attachment": {"type": "INTEGER"},
                    "erase_first": {"type": "BOOLEAN"},
                    "fill": {"type": "STRING"},
                    "stroke": {"type": "STRING"},
                    "prompt": {"type": "STRING"},
                    "note": {"type": "STRING"},
                },
                "required": ["action", "note"],
            },
        },
    },
    "required": ["analysis", "operations"],
}


def _describe_page(page: fitz.Page, blocks: list[dict], images: list[dict]) -> str:
    page_rect = page.rect
    lines = [
        f"Page size: {page_rect.width:.0f} x {page_rect.height:.0f} pt.",
        "",
        "TEXT BLOCKS (id | box_2d [y0,x0,y1,x1] | size pt | colour | bold | text):",
    ]
    for block in blocks:
        text = block["text"]
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(
            f"{block['id']} | {_to_box2d(block['rect'], page_rect)} | {block['size']} | "
            f"{_int_color_to_hex(block['color'])} | {'bold' if block['bold'] else 'regular'} "
            f"| {text}"
        )
    if not blocks:
        lines.append("(none — this page carries no extractable text)")

    lines += ["", "IMAGES ALREADY ON THE PAGE (id | box_2d):"]
    for image in images:
        lines.append(f"{image['id']} | {_to_box2d(image['rect'], page_rect)}")
    if not images:
        lines.append("(none)")
    return "\n".join(lines)


def _plan(page_png: bytes, description: str, instruction: str,
          attachments: list[Attachment]) -> dict:
    """Ask the planner what to do. Raises PageFixError if it never answers usefully."""
    request = [
        "USER'S INSTRUCTION FOR THIS PAGE:\n" + instruction.strip(),
        "",
        description,
    ]
    if attachments:
        request.append(
            "\nATTACHMENTS the user supplied, in order, are appended as images after the "
            "page. Reference them by 0-based index in `attachment`:\n"
            + "\n".join(f"{i}: {a.name} ({a.width}x{a.height})"
                        for i, a in enumerate(attachments))
        )
    else:
        request.append("\nThe user supplied no attachments; `insert_image` is unavailable.")

    contents = [
        "\n".join(request),
        types.Part.from_bytes(data=_shrink(page_png), mime_type="image/png"),
    ]
    for attachment in attachments:
        contents.append(types.Part.from_bytes(data=_shrink(attachment.png),
                                              mime_type="image/png"))

    last_error = None
    for attempt in range(1, PLAN_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=PLAN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=PLAN_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_PLAN_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and isinstance(data.get("operations"), list):
                return data
            logger.warning("Plan: unexpected shape (%d/%d): %r",
                           attempt, PLAN_ATTEMPTS, response.text[:200])
        except Exception as exc:  # noqa: BLE001 — retried, then surfaced
            last_error = exc
            logger.exception("Plan request failed (%d/%d)", attempt, PLAN_ATTEMPTS)
    raise PageFixError(
        "The page could not be analysed — the AI request failed. "
        + (f"({str(last_error)[:120]})" if last_error else "")
    )


# --------------------------------------------------------------------------------------
# Executing a plan
# --------------------------------------------------------------------------------------


def _resolve_rect(op: dict, page_rect: fitz.Rect, blocks: list[dict]) -> fitz.Rect | None:
    """The rectangle an operation acts on: a measured block if it named one, else its box."""
    block_id = op.get("block_id")
    if isinstance(block_id, int) and 0 <= block_id < len(blocks):
        return fitz.Rect(blocks[block_id]["rect"]) & page_rect

    box = op.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            return _to_rect(list(box), page_rect)
        except (TypeError, ValueError):
            return None
    return None


def _inherited_style(rect: fitz.Rect, blocks: list[dict]) -> dict:
    """Size, colour, boldness of whatever text the rectangle sits over."""
    overlapping = [b for b in blocks if (b["rect"] & rect).get_area() > 0]
    if not overlapping:
        return {"size": 11.0, "color": "#000000", "bold": False}
    best = max(overlapping, key=lambda b: (b["rect"] & rect).get_area())
    return {
        "size": best["size"],
        "color": _int_color_to_hex(best["color"]),
        "bold": best["bold"],
    }


def _free_rect(rect: fitz.Rect, align: str, obstacles: list[fitz.Rect],
               page_rect: fitz.Rect) -> fitz.Rect:
    """The rectangle grown into the empty page around it, without disturbing its anchor.

    A text block's bbox hugs its glyphs, so a replacement even one word longer than the
    original does not fit it and gets shrunk — a corrected heading coming out visibly
    smaller than the one beside it. The space to set it in is usually right there: the
    margin the line stopped short of, the gap before the next paragraph. Growing into that
    space first means shrinking stays what it is meant to be, a last resort.

    Growth is only ever away from the anchor the text is aligned to, so the words do not
    move: right for left-aligned text, left for right-aligned, symmetrically for centred.
    """
    margin = 18.0  # keep off the trim edge
    gap = 4.0      # never touch a neighbour
    bounds = page_rect + (margin, margin, -margin, -margin)
    if bounds.is_empty:
        bounds = page_rect
    grown = fitz.Rect(rect)

    def limit(axis: str) -> float:
        """How far the rect can extend in one direction before hitting something."""
        if axis in ("left", "right"):
            band = [o for o in obstacles
                    if o.y1 > rect.y0 + 1 and o.y0 < rect.y1 - 1]
            if axis == "right":
                stops = [o.x0 for o in band if o.x0 >= rect.x1 - 1]
                return min([bounds.x1] + [s - gap for s in stops])
            stops = [o.x1 for o in band if o.x1 <= rect.x0 + 1]
            return max([bounds.x0] + [s + gap for s in stops])
        band = [o for o in obstacles if o.x1 > rect.x0 + 1 and o.x0 < rect.x1 - 1]
        stops = [o.y0 for o in band if o.y0 >= rect.y1 - 1]
        return min([bounds.y1] + [s - gap for s in stops])

    if align == "right":
        grown.x0 = min(rect.x0, limit("left"))
    elif align == "center":
        # Symmetric, so the centre line stays where it is.
        room = min(rect.x0 - limit("left"), limit("right") - rect.x1)
        if room > 0:
            grown.x0, grown.x1 = rect.x0 - room, rect.x1 + room
    else:
        grown.x1 = max(rect.x1, limit("right"))
    grown.y1 = max(rect.y1, limit("down"))
    return grown & page_rect


def _probe(rect: fitz.Rect, body: str, css: str, archive: fitz.Archive) -> float:
    """Measure what a rectangle would cost to fit, off-page. Returns the scale, 0 if it
    cannot be fitted at all — insert_htmlbox reports a fit, it cannot be asked for one."""
    scratch = fitz.open()
    canvas = scratch.new_page(width=rect.x1 + 2, height=rect.y1 + 2)
    scale = 0.0
    for low in SCALE_LADDER:
        spare, scale = canvas.insert_htmlbox(rect, body, css=css, scale_low=low,
                                             archive=archive)
        if spare >= 0:
            break
    scratch.close()
    return scale


def _write_text(page: fitz.Page, rect: fitz.Rect, text: str, style: dict,
                archive: fitz.Archive, room: fitz.Rect | None = None) -> tuple[float, bool]:
    """Draw text into a rectangle, shrinking until it fits.

    Returns (scale, grew). `room` is the same rectangle grown into the free page around it;
    it is used only when the text does not fit the original at full size, so an edit that
    fits stays exactly where the original text was.
    """
    css = PAGEFIX_CSS.format(
        size=max(4.0, float(style["size"])),
        color=style["color"],
        align=style["align"],
        bold="font-weight: bold;" if style["bold"] else "",
    )
    body = html.escape(text).replace("\n", "<br>")

    target, grew = rect, False
    if room is not None and room != rect:
        as_is = _probe(rect, body, css, archive)
        if as_is < 0.99 and _probe(room, body, css, archive) > as_is:
            target, grew = room, True

    scale = 0.0
    for low in SCALE_LADDER:
        spare, scale = page.insert_htmlbox(target, body, css=css, scale_low=low,
                                           archive=archive)
        if spare >= 0:
            break
    return scale, grew


def _regenerate_region(pdf_bytes: bytes, page_index: int, rect: fitz.Rect, prompt: str,
                       attachments: list[Attachment]) -> bytes | None:
    """Redraw the artwork inside a rectangle. None if every image model declines."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page_index].get_pixmap(clip=rect, dpi=CROP_DPI)
        crop = pix.tobytes("png")
        crop_w, crop_h = pix.width, pix.height
    finally:
        doc.close()

    instruction = (
        "Redraw this picture, applying exactly this change and nothing else:\n"
        f"{prompt}\n\n"
        "Keep the framing, proportions, palette and background of the original so the "
        "result drops back into the same place on the page. Return only the image."
    )
    parts = [instruction, types.Part.from_bytes(data=crop, mime_type="image/png")]
    for attachment in attachments:
        parts.append(types.Part.from_bytes(data=_shrink(attachment.png),
                                           mime_type="image/png"))

    for model in EDIT_MODELS:
        for attempt in range(1, 3):
            try:
                response = generate_content(
                    model=model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=0.2 + 0.3 * (attempt - 1),
                    ),
                    timeout_ms=IMAGE_TIMEOUT_MS,
                )
            except Exception:
                logger.warning("%s: region regeneration request failed (%d/2)", model, attempt)
                continue
            data = _first_image_bytes(response)
            if data:
                return _resize_to(data, crop_w, crop_h)
            logger.warning("%s: returned no image for the region (%d/2)", model, attempt)
    return None


def apply_plan(pdf_bytes: bytes, page_index: int, operations: list[dict],
               attachments: list[Attachment]) -> tuple[bytes, list[dict]]:
    """Execute a plan against one page. Returns the new PDF and a per-operation report.

    Redactions run as a single pass before anything is drawn: `apply_redactions` rewrites
    the page's whole content stream, so content inserted first would not survive it.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: list[dict] = []
    try:
        page = doc[page_index]
        page_rect = page.rect
        if page.rotation:
            logger.warning("Page %d is rotated %d°; boxes are read in the rotated view",
                           page_index + 1, page.rotation)
        blocks = _text_blocks(page)
        # Divider rules count as obstacles too: a box grown past a heading's underline
        # would set its second line straight through it (the reasoning behind _rules).
        obstacles = (
            [b["rect"] for b in blocks]
            + [i["rect"] for i in _image_rects(page)]
            + _rules(page)
        )

        with Image.open(io.BytesIO(page.get_pixmap(dpi=PREVIEW_DPI).tobytes("png"))) as raw:
            page_png = raw.convert("RGB")

        # Phase 1 — resolve every operation to a rectangle, rejecting the unusable.
        planned = []
        for index, op in enumerate(operations):
            action = op.get("action")
            note = (op.get("note") or "").strip()
            entry = {"index": index, "action": action, "note": note, "status": "", "detail": ""}
            if action not in ACTIONS:
                entry.update(status="skipped", detail=f"unknown action {action!r}")
                results.append(entry)
                continue
            rect = _resolve_rect(op, page_rect, blocks)
            if rect is None or rect.is_empty or rect.width < MIN_EDGE_PT or rect.height < MIN_EDGE_PT:
                entry.update(status="skipped", detail="no usable rectangle")
                results.append(entry)
                continue
            entry["box"] = _to_box2d(rect, page_rect)
            planned.append((op, rect, entry))
            results.append(entry)

        # Phase 2 — one redaction pass for everything that has to be cleared.
        erased = False
        for op, rect, entry in planned:
            action = op["action"]
            clears = action in ("erase", "replace_text") or (
                action == "insert_image" and op.get("erase_first")
            )
            if not clears:
                continue
            fill = (_parse_color(op["fill"], default=(1.0, 1.0, 1.0))
                    if op.get("fill") else _surrounding_color(page_png, rect, page_rect))
            # Inset by a third of a point, the same margin the translate pipeline redacts
            # with. A text block's bbox routinely lands within a hair of the heading rule
            # beneath it, and a redaction touching a 4pt stroke clips it: the rule comes
            # back with a step in it exactly as wide as the box. The inset is far too small
            # to leave any of the removed text behind.
            page.add_redact_annot(rect + (REDACT_INSET, REDACT_INSET,
                                          -REDACT_INSET, -REDACT_INSET), fill=fill)
            erased = True
        if erased:
            # Images and vector art are covered by the redaction's own fill rectangle but
            # left intact as objects — the same choice the translate pipeline makes, and
            # what keeps an erase over a photo from deleting the photo everywhere it is used.
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
            page = doc[page_index]

        # Phase 3 — draw.
        archive = fitz.Archive(FONTS_DIR)
        for op, rect, entry in planned:
            action = op["action"]
            try:
                if action == "erase":
                    entry.update(status="applied", detail="cleared")

                elif action in ("replace_text", "insert_text"):
                    text = (op.get("text") or "").strip()
                    if not text:
                        entry.update(status="skipped", detail="no text given")
                        continue
                    inherited = _inherited_style(rect, blocks)
                    style = {
                        "size": float(op["font_size"]) if op.get("font_size") else inherited["size"],
                        "color": _hex_color(op.get("color"), inherited["color"]),
                        "align": op.get("align") or "left",
                        "bold": bool(op["bold"]) if "bold" in op else inherited["bold"],
                    }
                    # Everything on the page except what this edit is replacing.
                    around = [o for o in obstacles
                              if (o & rect).get_area() <= 0.5 * max(o.get_area(), 1e-6)]
                    room = _free_rect(rect, style["align"], around, page_rect)
                    scale, grew = _write_text(page, rect, text, style, archive, room)
                    if scale <= 0:
                        entry.update(status="failed", detail="text did not fit the box")
                    else:
                        detail = f"wrote {len(text)} chars at {style['size']:.1f}pt"
                        if grew:
                            detail += " (grown into the space beside it)"
                        if scale < 1:
                            detail += f" (shrunk to {scale:.0%})"
                        entry.update(status="applied", detail=detail)

                elif action == "insert_image":
                    index = op.get("attachment", 0) or 0
                    if not attachments:
                        entry.update(status="skipped", detail="no attachment uploaded")
                        continue
                    if not 0 <= index < len(attachments):
                        entry.update(status="skipped", detail=f"no attachment {index}")
                        continue
                    page.insert_image(rect, stream=attachments[index].png,
                                      keep_proportion=True, overlay=True)
                    entry.update(status="applied",
                                 detail=f"placed {attachments[index].name}")

                elif action == "regenerate_image":
                    prompt = (op.get("prompt") or op.get("text") or "").strip()
                    if not prompt:
                        entry.update(status="skipped", detail="no prompt given")
                        continue
                    # Rendered from the in-progress document so earlier operations are visible.
                    new_image = _regenerate_region(
                        doc.tobytes(garbage=3, deflate=True), page_index, rect, prompt,
                        attachments,
                    )
                    if not new_image:
                        entry.update(status="failed", detail="the image model declined")
                        continue
                    page.insert_image(rect, stream=new_image, keep_proportion=False,
                                      overlay=True)
                    entry.update(status="applied", detail="artwork redrawn")

                elif action == "draw_box":
                    fill = _parse_color(op["fill"]) if op.get("fill") else None
                    stroke = _parse_color(op["stroke"]) if op.get("stroke") else None
                    if fill is None and stroke is None:
                        fill = (1.0, 1.0, 1.0)
                    page.draw_rect(rect, color=stroke, fill=fill,
                                   width=1.0 if stroke else 0)
                    entry.update(status="applied", detail="box drawn")
            except Exception as exc:  # noqa: BLE001 — one bad op must not lose the others
                logger.exception("Operation %d (%s) failed", entry["index"], action)
                entry.update(status="failed", detail=str(exc)[:150])

        # No subset_fonts(): subsetting renumbers the glyph ids HarfBuzz-shaped Bangla
        # points at, which garbles it in every viewer but MuPDF. See pdf_pipeline.py.
        out = doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()
    return out, results


# --------------------------------------------------------------------------------------
# Finding the pages an instruction is about
#
# The user does not say which page. They say "the subtitle on the contents page is wrong"
# or "remove the grey box under the exercise table", and finding it is the tool's job.
#
# Locating is a separate, cheap call from planning: a thumbnail and a text digest per page
# are enough to say *which* page, and planning an actual edit needs a full-resolution
# render and the page's measured geometry — which would be ruinous to send for all 40 pages
# of a manual. So this pass narrows, and the planner is then run only on what it picked.
# --------------------------------------------------------------------------------------


LOCATE_MODEL = PLAN_MODEL
LOCATE_DPI = 40            # thumbnails: enough to recognise a page, cheap to send
LOCATE_THUMB_DIM = 420
DIGEST_CHARS = 400         # of each page's text
# Beyond this, thumbnails for every page stop being worth their tokens and the text digests
# carry the search on their own.
THUMBNAIL_PAGE_LIMIT = 40
# An instruction that seems to touch more pages than this is a document-wide edit the user
# should confirm before it runs, not something to apply silently to a whole book.
MAX_PAGES_PER_FIX = 25

LOCATE_PROMPT = """You are given a user's instruction for repairing a PDF, and a list of
every page in it — its number, a digest of its text, and (usually) a thumbnail in the same
order. Say which pages the instruction is about. Answer ONLY with the JSON described.

- Page numbers are positions in the file, starting at 1. A number *printed* on a page is
  often different (front matter, or a chapter starting at 41); when the user names a page,
  prefer the printed folio if you can see one that matches, and say so in `analysis`.
- Most instructions concern exactly one page. Return one.
- Return several only when the instruction genuinely spans them ("every page", "all the
  chapter headings", "pages 4 to 6").
- Return an empty list if you cannot tell which page they mean, or if nothing in the
  document matches what they describe. Say why in `analysis`. Guessing wastes the user's
  time repairing the wrong page.
"""

_LOCATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "analysis": {"type": "STRING"},
        "pages": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["analysis", "pages"],
}


def _page_digests(pdf_bytes: bytes) -> tuple[list[str], list[bytes]]:
    """One text digest per page, and thumbnails while the document is small enough."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    digests, thumbs = [], []
    want_thumbs = doc.page_count <= THUMBNAIL_PAGE_LIMIT
    try:
        for number, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            digests.append(
                f"Page {number}: " + (text[:DIGEST_CHARS] + "…" if len(text) > DIGEST_CHARS
                                      else text or "(no text — images or blank)")
            )
            if want_thumbs:
                thumbs.append(_shrink(page.get_pixmap(dpi=LOCATE_DPI).tobytes("png"),
                                      LOCATE_THUMB_DIM))
    finally:
        doc.close()
    return digests, thumbs


def _locate_pages(pdf_bytes: bytes, instruction: str,
                  attachments: list[Attachment]) -> tuple[list[int], str]:
    """Which pages the instruction is about, and the reasoning. May be empty."""
    digests, thumbs = _page_digests(pdf_bytes)
    request = [
        "USER'S INSTRUCTION:\n" + instruction.strip(),
        "",
        f"The document has {len(digests)} pages.",
        "\n".join(digests),
    ]
    if thumbs:
        request.append(f"\nThumbnails of pages 1-{len(thumbs)} follow, in order.")
    if attachments:
        request.append(
            "\nAfter the thumbnails come the images the user attached, which they want "
            "placed into the document: " + ", ".join(a.name for a in attachments)
        )

    contents = ["\n".join(request)]
    for thumb in thumbs:
        contents.append(types.Part.from_bytes(data=thumb, mime_type="image/png"))
    for attachment in attachments:
        contents.append(types.Part.from_bytes(data=_shrink(attachment.png, LOCATE_THUMB_DIM),
                                              mime_type="image/png"))

    for attempt in range(1, PLAN_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=LOCATE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=LOCATE_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_LOCATE_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and isinstance(data.get("pages"), list):
                pages = sorted({int(p) for p in data["pages"]
                                if isinstance(p, (int, float)) and 1 <= int(p) <= len(digests)})
                return pages, (data.get("analysis") or "").strip()
            logger.warning("Locate: unexpected shape (%d/%d)", attempt, PLAN_ATTEMPTS)
        except Exception:
            logger.exception("Locate request failed (%d/%d)", attempt, PLAN_ATTEMPTS)
    raise PageFixError(
        "The document could not be searched — the AI request failed. Check the server logs."
    )


# --------------------------------------------------------------------------------------
# The public operation
# --------------------------------------------------------------------------------------


def _fix_one_page(pdf_bytes: bytes, page_index: int, instruction: str,
                  attachments: list[Attachment]) -> tuple[bytes, dict]:
    """Plan and apply one instruction against one page. Returns the new PDF and a report.

    Pure: it does not touch session state, so a multi-page instruction can be applied to
    each page in turn and still be undone as the single edit the user asked for.
    """
    page_png = render_page(pdf_bytes, page_index, dpi=PLAN_DPI)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        description = _describe_page(page, _text_blocks(page), _image_rects(page))
    finally:
        doc.close()

    plan = _plan(page_png, description, instruction, attachments)
    operations = plan.get("operations") or []
    analysis = (plan.get("analysis") or "").strip()
    report = {"page": page_index + 1, "analysis": analysis, "operations": [],
              "changed": False}

    if not operations:
        report["analysis"] = analysis or "Nothing to change on this page."
        return pdf_bytes, report

    after, results = apply_plan(pdf_bytes, page_index, operations, attachments)
    applied = sum(1 for r in results if r["status"] == "applied")
    report["operations"] = results
    report["changed"] = bool(applied)
    return (after if applied else pdf_bytes), report


def _summarize(pages: list[dict]) -> str:
    changed = [p for p in pages if p["changed"]]
    applied = sum(1 for p in pages for op in p["operations"] if op["status"] == "applied")
    failed = sum(1 for p in pages for op in p["operations"] if op["status"] == "failed")
    if not applied:
        return "No change — nothing was found that could be fixed safely."
    where = (f"page {changed[0]['page']}" if len(changed) == 1
             else f"{len(changed)} pages ({', '.join(str(p['page']) for p in changed)})")
    summary = f"{applied} edit{'s' if applied != 1 else ''} applied on {where}"
    return summary + (f"; {failed} could not be applied" if failed else "")


def fix_document(sid: str, instruction: str,
                 new_attachments: list[tuple[str, bytes]] | None = None) -> dict:
    """Find the pages an instruction is about, fix each, and record one undo step.

    This is the whole tool: a PDF, a sentence about what is wrong, and optionally images to
    put in it. The undo step covers the instruction as a whole, however many pages it
    turned out to touch — that is the unit the user asked for.
    """
    session = get_session(sid)
    if not (instruction or "").strip():
        raise PageFixError("Describe what needs fixing.")
    if new_attachments:
        add_attachments(sid, new_attachments)

    targets, reasoning = _locate_pages(session.pdf, instruction, session.attachments)
    if not targets:
        return {
            "instruction": instruction,
            "analysis": reasoning or "Nothing in the document matches that.",
            "pages": [],
            "changed": False,
            "summary": "No change — the pages that instruction refers to could not be found.",
        }

    truncated = len(targets) > MAX_PAGES_PER_FIX
    targets = targets[:MAX_PAGES_PER_FIX]

    before = session.pdf
    working = before
    reports = []
    for page_number in targets:
        try:
            working, report = _fix_one_page(working, page_number - 1, instruction,
                                            session.attachments)
        except Exception as exc:  # noqa: BLE001 — one bad page must not lose the others
            logger.exception("Page %d could not be fixed", page_number)
            reports.append({"page": page_number, "analysis": f"This page failed: {exc}",
                            "operations": [], "changed": False})
            continue
        reports.append(report)

    changed = any(r["changed"] for r in reports)
    if changed:
        with _lock:
            session.history.append(before)
            del session.history[:-MAX_UNDO]
            session.pdf = working
            session.log.append({"instruction": instruction,
                                "pages": [r["page"] for r in reports if r["changed"]]})

    summary = _summarize(reports)
    if truncated:
        summary += f" (stopped at the first {MAX_PAGES_PER_FIX} pages)"
    return {
        "instruction": instruction,
        "analysis": reasoning,
        "pages": reports,
        "changed": changed,
        "summary": summary,
    }
