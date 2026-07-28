"""PDF fix pipeline: find and repair defects in an already-translated PDF.

A fully separate flow from the translation and image pipelines, which it leaves
untouched. It repairs the three defects the translation pipeline can leave behind:

  * untranslated text — the translator returns the original English when its API
    call fails, so a failed chunk leaves a whole page in English;
  * dropped text — `insert_htmlbox` draws NOTHING when the text cannot fit even
    at its shrink floor, so a segment can vanish from the page entirely;
  * overlapping text — two segments whose boxes were planned over each other.

None of it is found by reading the page. The Bangla drawn by `insert_htmlbox`
cannot be read back (see manifest.py), so the text comes from the manifest
embedded during translation and the page is consulted only for geometry.
"""

import html
import logging
import re

import fitz  # PyMuPDF

import manifest
from pdf_processor import FONTS_DIR, css_for
from shortener import shorten_batch
from translator import translate_batch_status

logger = logging.getLogger(__name__)

FIX_SUFFIX = "_fix.pdf"  # keep in sync with MODES.fix.suffix in static/index.html

# The smallest a block may shrink and still sit comfortably on the page. Text
# that needs less than this is asked to get shorter instead.
READABLE_FLOOR = 0.4

# Tried in order until the text actually renders. Never leave a segment on a
# floor it cannot meet: insert_htmlbox draws nothing at all in that case. A
# floor of 0 is the guarantee — given no floor it always finds a scale that fits.
SCALE_LADDER = (READABLE_FLOOR, 0.0)

# Rendering resolution for measuring where the ink actually lands. The glyph
# boxes PyMuPDF reports run from the font's ascender to its descender — for Noto
# Sans Bengali that is ~1.46x the font size, far taller than the visible text —
# so boxes touch long before any glyph does. Only pixels settle it.
INK_DPI = 144
INK_MIN = 240  # a grey below this counts as ink

# Ink this far into another segment is a real collision, not a rounding artifact.
COLLIDE_MIN = 1.0

WHITESPACE = re.compile(r"\s+")


class FixReport:
    """Counters for one fix run, for the log line and the response header."""

    def __init__(self) -> None:
        self.pages_scanned = 0
        self.pages_repaired = 0
        self.retranslated = 0
        self.accepted = 0
        self.restored = 0
        self.shortened = 0
        self.shrunk = 0
        self.overlaps_before = 0
        self.overlaps_after = 0

    def summary(self) -> str:
        return (
            f"{self.pages_scanned} pages scanned, {self.pages_repaired} repaired | "
            f"{self.retranslated} re-translated, {self.accepted} left as-is, "
            f"{self.restored} dropped-text restored, {self.shortened} shortened, "
            f"{self.shrunk} shrunk to fit | "
            f"overlaps {self.overlaps_before} -> {self.overlaps_after}"
        )


def _norm(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().casefold()


def _is_miss(seg: dict) -> bool:
    """True if this segment's translation request failed.

    Not "the text still looks English": plenty of segments are meant to stay that
    way — an acronym, a URL, a printer's mark — and re-asking about them would
    redraw a page and spend a call to be told the same thing. The translation
    pipeline records whether the request itself succeeded, which is the only
    thing that distinguishes the two. Segments settled by an earlier fix run are
    left alone.
    """
    return seg["fixed"] is None and not seg.get("ok", True)


def _is_dropped(seg: dict) -> bool:
    """True if this segment's text is missing from the page.

    `insert_htmlbox` returns spare_height -1 when the text does not fit at the
    scale floor it was given — and in that case it draws nothing at all, rather
    than overflowing. So this is silent data loss, and the manifest holds the
    only surviving copy of the text.
    """
    return seg["sh"] < 0


def _ink_span(page: fitz.Page) -> tuple[float, float] | None:
    """Top and bottom of the inked pixels on a page, in points."""
    pixmap = page.get_pixmap(dpi=INK_DPI, colorspace=fitz.csGRAY)
    width, height, samples = pixmap.width, pixmap.height, pixmap.samples
    scale = 72 / INK_DPI
    rows = [y for y in range(height) if min(samples[y * width : (y + 1) * width]) < INK_MIN]
    if not rows:
        return None
    return rows[0] * scale, (rows[-1] + 1) * scale


def _probe(
    scratch: fitz.Document, box: fitz.Rect, text: str, css: str, archive: fitz.Archive, low: float
) -> tuple[float, float, fitz.Rect | None]:
    """Render `text` into a copy of `box` off-page and report how it landed.

    `insert_htmlbox` draws as it measures, so the only way to ask "would this
    fit?" is to render it somewhere disposable. Returns the spare height, the
    scale it chose, and where the ink actually fell in page coordinates.
    """
    page = scratch.new_page(width=box.width, height=box.height)
    spare, scale = page.insert_htmlbox(
        fitz.Rect(0, 0, box.width, box.height),
        html.escape(text),
        css=css,
        scale_low=low,
        archive=archive,
    )
    rows = _ink_span(page)
    if rows is None:
        return spare, scale, None

    xs = [
        span["bbox"]
        for block in page.get_text("dict")["blocks"]
        if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    x0 = min(b[0] for b in xs) if xs else 0
    x1 = max(b[2] for b in xs) if xs else box.width
    ink = fitz.Rect(box.x0 + x0, box.y0 + rows[0], box.x0 + x1, box.y0 + rows[1])
    return spare, scale, ink


def _fit(
    scratch: fitz.Document, seg: dict, text: str, archive: fitz.Archive
) -> tuple[float, float, fitz.Rect | None, float]:
    """Find the highest scale floor at which `text` actually renders."""
    box = manifest.rect_of(seg["ins"])
    css = css_for(seg)
    for low in SCALE_LADDER:
        spare, scale, ink = _probe(scratch, box, text, css, archive, low)
        if spare >= 0:
            return spare, scale, ink, low
    return spare, scale, ink, SCALE_LADDER[-1]


def _erase_inserted(page: fitz.Page, source_fonts: set[str]) -> int:
    """Remove every glyph the translation pipeline drew, and nothing else.

    A span belongs to the translation exactly when its font was not on the page
    before translation. Matching on Bangla font names instead would miss the
    Latin runs (acronyms, URLs) that MuPDF renders in a fallback font, and they
    would then be drawn a second time on top of themselves.

    Bullets, checkboxes and TOC page numbers were deliberately preserved by the
    translation pipeline — they are in the source fonts, so they survive here
    too. The Bangla font has no box or tick glyphs and would render them as a
    wrong letter.
    """
    erased = 0
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if not span["text"].strip() or span["font"] in source_fonts:
                    continue
                # Shrunk a hair so redaction never bites an adjacent glyph we keep.
                page.add_redact_annot(fitz.Rect(span["bbox"]) + (0.3, 0.3, -0.3, -0.3))
                erased += 1
    if erased:
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )
    return erased


def _clashes(rects: dict[int, fitz.Rect]) -> list[tuple[int, int]]:
    """Pairs sharing more than a rounding error's worth of space."""
    hits = []
    idxs = sorted(rects)
    for i, a in enumerate(idxs):
        for b in idxs[i + 1 :]:
            shared = rects[a] & rects[b]
            if (
                not shared.is_empty
                and shared.width > COLLIDE_MIN
                and shared.height > COLLIDE_MIN
            ):
                hits.append((a, b))
    return hits


def _collisions(entry: dict, archive: fitz.Archive) -> list[tuple[int, int]]:
    """Segment pairs whose text genuinely sits on top of other text.

    Text never escapes its box — insert_htmlbox shrinks it until it fits — so
    boxes that do not overlap cannot possibly collide. That check is free, and
    on a healthy page it answers the whole question without rendering anything.
    Only the few pairs it cannot rule out are worth measuring the ink for.
    """
    boxes = {
        idx: manifest.rect_of(seg["ins"])
        for idx, seg in enumerate(entry["segments"])
        if not _is_dropped(seg)  # nothing is drawn, so nothing can collide
    }
    candidates = _clashes(boxes)
    if not candidates:
        return []

    scratch = fitz.open()
    inks = {}
    for idx in {i for pair in candidates for i in pair}:
        seg = entry["segments"][idx]
        # No floor. Handing the recorded scale back as `scale_low` was meant to
        # reproduce the page's own render, but it cannot: the manifest rounds
        # the scale to three decimals and the box to two, so re-fitting the text
        # lands a shade under the floor it was given — and PyMuPDF asserts on
        # that rather than reporting a miss, which killed the whole Fix run on
        # p.158 of the Revascularisation manual (`scale_low=0.945 scale=0.944…`).
        # Nothing is lost by dropping it: insert_htmlbox returns the largest
        # scale that fits whatever floor it is under, so the ink measured here
        # is the same ink, and the segments that genuinely failed to render were
        # already excluded above by `_is_dropped`.
        _, _, ink = _probe(scratch, boxes[idx], seg["bn"], css_for(seg), archive, 0)
        if ink is not None:
            inks[idx] = ink
    scratch.close()

    confirmed = set(_clashes(inks))
    return [pair for pair in candidates if pair in confirmed]


def _deconflict(entry: dict, hits: list[tuple[int, int]]) -> int:
    """Pull colliding boxes apart so the text below is not written over.

    The higher segment yields: its box is cut short above the lower one, which
    makes its text shrink to fit the smaller box. Shrinking the block that runs
    long is what keeps the page readable — moving the lower text instead would
    push the rest of the page out of place.
    """
    changed = 0
    for a, b in hits:
        seg_a, seg_b = entry["segments"][a], entry["segments"][b]
        box_a, box_b = manifest.rect_of(seg_a["ins"]), manifest.rect_of(seg_b["ins"])
        upper, lower = (seg_a, box_b) if box_a.y0 <= box_b.y0 else (seg_b, box_a)
        limit = lower.y0 - 2
        box = manifest.rect_of(upper["ins"])
        if limit <= box.y0 + 4 or limit >= box.y1:
            continue  # no room to give; the shrink below has to carry it
        upper["ins"] = manifest.rect_list(fitz.Rect(box.x0, box.y0, box.x1, limit))
        changed += 1
    return changed


def _retranslate_misses(entry: dict, report: FixReport) -> None:
    """Translate the segments whose translation never happened.

    Retried once only: text that comes back unchanged is genuinely untranslatable
    (an acronym like ECG, a URL), and marking it accepted stops this run — and
    every later run — from asking again.
    """
    misses = [seg for seg in entry["segments"] if _is_miss(seg)]
    if not misses:
        return
    if len(misses) == len(entry["segments"]):
        logger.info("Page %d was not translated at all — re-translating", entry["page"])

    translations, status = translate_batch_status([seg["en"] for seg in misses])
    for seg, new, ok in zip(misses, translations, status):
        seg["ok"] = ok
        if not ok or _norm(new) == _norm(seg["en"]):
            # Failed again, or came back unchanged because it is untranslatable.
            # Either way, stop asking: this run and every later one.
            seg["fixed"] = "accepted"
            report.accepted += 1
        else:
            seg["bn"] = new
            seg["fixed"] = "retranslated"
            report.retranslated += 1


def _resolve(entry: dict, archive: fitz.Archive, report: FixReport) -> dict[int, tuple[str, float]]:
    """Settle the final text and scale floor for every segment, before drawing.

    Runs for every segment on the page, not only the broken ones: re-translated
    text is new text, and a de-conflicted box is a new box, so any of them may
    now land differently.
    """
    scratch = fitz.open()
    plan = {}
    tight = []

    for idx, seg in enumerate(entry["segments"]):
        _, _, _, low = _fit(scratch, seg, seg["bn"], archive)
        plan[idx] = (seg["bn"], low)
        if low < READABLE_FLOOR:
            tight.append(idx)

    # Only text that cannot fit at a readable size is worth an API call: shorter
    # words are the only thing left that makes it bigger.
    if tight:
        items = [
            {"bn": entry["segments"][i]["bn"], "en": entry["segments"][i]["en"]}
            for i in tight
        ]
        for idx, shorter in zip(tight, shorten_batch(items)):
            seg = entry["segments"][idx]
            if shorter == seg["bn"]:
                continue  # the shortener could not help; keep the unbounded fit
            _, _, _, low = _fit(scratch, seg, shorter, archive)
            plan[idx] = (shorter, low)

    for idx, seg in enumerate(entry["segments"]):
        text, low = plan[idx]
        # One segment can be several of these at once — text that was dropped is
        # often also the text that had to be shortened — so they count separately.
        dropped, shortened = _is_dropped(seg), text != seg["bn"]
        if dropped:
            report.restored += 1
        if shortened:
            report.shortened += 1
        if low < READABLE_FLOOR and not shortened:
            report.shrunk += 1
        if seg["fixed"] is None:
            seg["fixed"] = (
                "restored" if dropped
                else "shortened" if shortened
                else "shrunk" if low < READABLE_FLOOR
                else None
            )
        if low < READABLE_FLOOR:
            logger.warning(
                "Page %d: text only fits well below its original size: %r at %s",
                entry["page"],
                text[:40],
                seg["ins"],
            )

    scratch.close()
    return plan


def _repair_page(
    page: fitz.Page,
    entry: dict,
    source_fonts: set[str],
    archive: fitz.Archive,
    report: FixReport,
) -> None:
    """Rebuild every segment on a page from the manifest.

    The whole page is redrawn rather than just the broken segments: erasing text
    is done by font, which takes out all of the inserted text at once, and the
    manifest can redraw every piece of it. That is cheaper to reason about than
    erasing pieces of a page selectively, and it makes the result identical no
    matter which defect triggered the repair.
    """
    _retranslate_misses(entry, report)
    plan = _resolve(entry, archive, report)

    _erase_inserted(page, source_fonts)

    for idx, seg in enumerate(entry["segments"]):
        text, low = plan[idx]
        spare, scale = page.insert_htmlbox(
            manifest.rect_of(seg["ins"]),
            html.escape(text),
            css=css_for(seg),
            scale_low=low,
            archive=archive,
        )
        seg["bn"] = text
        seg["sh"] = round(spare, 2)
        seg["sc"] = round(scale, 3)


def _page_problems(entry: dict, hits: list[tuple[int, int]]) -> list[str]:
    """Why this page needs repairing, if it does."""
    problems = []
    if hits:
        problems.append(f"{len(hits)} overlapping")
    misses = sum(1 for seg in entry["segments"] if _is_miss(seg))
    if misses:
        problems.append(f"{misses} untranslated")
    dropped = sum(1 for seg in entry["segments"] if _is_dropped(seg))
    if dropped:
        problems.append(f"{dropped} with dropped text")
    return problems


def fix_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Repair untranslated, dropped and overlapping text in a translated PDF.

    Returns the fixed PDF and a one-line summary of what was found and changed.
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
            continue  # nothing was translated here, so nothing can be broken

        hits = _collisions(entry, archive)
        report.overlaps_before += len(hits)
        problems = _page_problems(entry, hits)
        if not problems:
            continue  # leave the page exactly as it is

        logger.info("Page %d: %s — repairing", page_num, ", ".join(problems))
        if hits:
            _deconflict(entry, hits)
        _repair_page(page, entry, source_fonts, archive, report)
        report.pages_repaired += 1
        report.overlaps_after += len(_collisions(entry, archive))

    data["fix_round"] += 1
    manifest.attach(doc, data)
    logger.info("Fix done: %s", report.summary())

    doc.subset_fonts()
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, report.summary()
