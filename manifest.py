"""Translation manifest embedded inside a translated PDF.

The Bangla text written by `insert_htmlbox` cannot be read back out: HarfBuzz
shapes conjuncts into glyphs that carry no reverse mapping to Unicode, so
'হার্ট' extracts as 'হাƖট'. Any pipeline that wants to re-render a segment
therefore needs the text from somewhere other than the page.

So the translation pipeline records what it did — every segment's English,
Bangla, geometry and fit result — and embeds it in the PDF it produces. The
fix pipeline reads it back as ground truth.
"""

import datetime
import json

import fitz  # PyMuPDF

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"

# MuPDF falls back to these when a glyph is missing from the Bangla fonts, so a
# source document already using one would make `source_fonts` ambiguous and
# break the fix pipeline's "font not in source_fonts means we drew it" rule.
FALLBACK_FONTS = ("noto sans bengali", "noto serif bengali", "noto serif regular")


class ManifestMissing(Exception):
    """The PDF carries no manifest (it was not produced by this translator)."""


class ManifestUnsupported(Exception):
    """The PDF carries a manifest written by an incompatible version."""


def rect_list(r: fitz.Rect) -> list[float]:
    return [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]


def rect_of(v: list[float]) -> fitz.Rect:
    return fitz.Rect(*v)


def page_span_fonts(page: fitz.Page) -> set[str]:
    """Font names of every span on a page, as `get_text` reports them.

    Must be read before redaction, while the source text is still present.
    These names are truncated to 24 characters, but the fix pipeline reads them
    the same way, so the two sets stay comparable.
    """
    fonts = set()
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    fonts.add(span["font"])
    return fonts


def segment_entry(
    seg: dict, english: str, bangla: str, metric: tuple[float, float], ok: bool
) -> dict:
    """One segment, with everything needed to re-render it without the page."""
    spare_height, scale = metric
    number = seg.get("number")
    return {
        "en": english,
        "bn": bangla,
        "ok": ok,
        "rect": rect_list(seg["rect"]),
        "lines": [rect_list(r) for r in seg["line_rects"]],
        "ins": rect_list(seg["insert_rect"]),
        "size": round(seg["size"], 2),
        "bold": seg["bold"],
        "color": seg["color"],
        "align": seg["align"],
        "bullet": seg["bullet"],
        "block_x1": round(seg["block_x1"], 2),
        "num": [round(v, 2) for v in number["bbox"]] if number else None,
        "sh": round(spare_height, 2),
        "sc": round(scale, 3),
        "fixed": None,
    }


def page_entry(
    page_num: int,
    segments: list[dict],
    kept: list[fitz.Rect],
    english: list[str],
    bangla: list[str],
    metrics: list[tuple[float, float]],
    status: list[bool],
) -> dict:
    return {
        "page": page_num,
        "kept": [rect_list(r) for r in kept],
        "segments": [
            segment_entry(seg, en, bn, metric, ok)
            for seg, en, bn, metric, ok in zip(segments, english, bangla, metrics, status)
        ],
    }


def build(source_fonts: set[str], pages: list[dict]) -> dict:
    collisions = sorted(
        f for f in source_fonts if any(f.lower().startswith(x) for x in FALLBACK_FONTS)
    )
    return {
        "v": SCHEMA_VERSION,
        "tool": "pdftranslator",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fix_round": 0,
        "source_fonts": sorted(source_fonts),
        "fallback_collision": collisions,
        "pages": pages,
    }


def attach(doc: fitz.Document, data: dict) -> None:
    """Embed the manifest, replacing any previous one."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        doc.embfile_del(MANIFEST_FILE)
    except Exception:
        pass  # no previous manifest
    doc.embfile_add(
        MANIFEST_FILE,
        payload,
        filename=MANIFEST_FILE,
        desc="pdftranslator segment manifest",
    )


def read(doc: fitz.Document) -> dict:
    try:
        payload = doc.embfile_get(MANIFEST_FILE)
    except Exception:
        raise ManifestMissing(
            "This PDF has no translation manifest. Re-run Translate to Bangla to "
            "produce a fixable PDF, then run Fix on the new file."
        )
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        raise ManifestUnsupported("The translation manifest in this PDF is unreadable.")
    if data.get("v") != SCHEMA_VERSION:
        raise ManifestUnsupported(
            f"This PDF's manifest is version {data.get('v')}, but this server writes "
            f"version {SCHEMA_VERSION}. Re-run Translate to Bangla on the original PDF."
        )
    return data
