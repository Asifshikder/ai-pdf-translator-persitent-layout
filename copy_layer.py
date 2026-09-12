"""Invisible logical-text layer that makes translated Bangla copy-pasteable.

The visible Bangla is drawn by `page.insert_htmlbox`, which HarfBuzz-shapes it
into glyphs. That output cannot be copied correctly: the font's ToUnicode maps
shaped conjunct/reph glyphs to garbage codepoints, and pre-base matras
(ি ে ৈ ো ৌ) are stored in *visual* order. A ToUnicode patch cannot fix the
reordering, so instead we overlay the correct *logical* string as invisible
text (the standard "searchable PDF" technique, honoured by every viewer) and
then blank the shaped layer's ToUnicode so it contributes nothing on copy.

See the module `manifest.py` for why the shaped text is unreadable, and the
plan in .claude/plans for the full rationale.
"""

import io
import logging
import os
import re

import fitz  # PyMuPDF
from fontTools.ttLib import TTFont

import manifest

logger = logging.getLogger(__name__)

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# A unique family name for the overlay font. It must NOT collide with any name
# the shaped-layer fonts carry ("Noto Sans/Serif Bengali"), so that
# `blank_shaped_tounicode` can tell the two apart even after subsetting renames
# the BaseFont with a random 6-letter prefix. The tag survives subsetting; only
# the prefix changes.
COPY_LAYER_FAMILY = "NotoCopyLayerBengali"
COPY_LAYER_TAG = "CopyLayer"  # substring searched for when protecting the font

_overlay_font: fitz.Font | None = None


def _renamed_font_bytes() -> bytes:
    """Load Noto Sans Bengali and rename it so it embeds as a distinct font.

    A byte-distinct program with a unique family name guarantees PyMuPDF will
    not dedup it against the htmlbox `@font-face` font, and lets the blanking
    pass identify (and spare) it by name.
    """
    src = os.path.join(FONTS_DIR, "NotoSansBengali-Regular.ttf")
    tt = TTFont(src)
    name = tt["name"]
    # 1=Family, 2=Subfamily, 4=Full, 6=PostScript, 16=Typo Family, 17=Typo Sub.
    for rec in name.names:
        if rec.nameID in (1, 16):
            rec.string = COPY_LAYER_FAMILY
        elif rec.nameID == 4:
            rec.string = COPY_LAYER_FAMILY + " Regular"
        elif rec.nameID == 6:
            rec.string = COPY_LAYER_FAMILY + "-Regular"
    buf = io.BytesIO()
    tt.save(buf)
    return buf.getvalue()


def overlay_font() -> fitz.Font:
    """The cached, uniquely-named Bengali font used for the invisible layer."""
    global _overlay_font
    if _overlay_font is None:
        _overlay_font = fitz.Font(fontbuffer=_renamed_font_bytes())
    return _overlay_font


# Matches CSS_TEMPLATE's line-height for Bangla, so the invisible lines stack at
# roughly the same pitch as the visible ones.
_LINE_HEIGHT = 1.45


def _wrap(font: fitz.Font, text: str, fontsize: float, width: float) -> list[str]:
    """Greedily wrap `text` into lines no wider than `width` (best-effort)."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.text_length(trial, fontsize=fontsize) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def add_invisible_text(
    page: fitz.Page, insert_rect: fitz.Rect, text: str, fontsize: float
) -> None:
    """Draw `text` invisibly (render mode 3) as a copy-pasteable logical layer.

    The glyphs are written unshaped, so they carry a clean per-codepoint
    ToUnicode and extract in logical order. They are never seen, so the wrong
    visual shaping of an unshaped run does not matter. The text is wrapped to the
    segment's box and stacked top-to-bottom so selection roughly tracks the
    visible lines; exact placement is not important for copy correctness.
    """
    text = " ".join(text.split())  # collapse the source's own line breaks
    if not text:
        return
    font = overlay_font()
    fontsize = max(fontsize, 1.0)
    width = max(insert_rect.width, fontsize)  # guard against zero-width boxes
    tw = fitz.TextWriter(page.rect)
    y = insert_rect.y0 + fontsize  # first baseline sits below the box top
    wrote = False
    for line in _wrap(font, text, fontsize, width):
        try:
            tw.append(fitz.Point(insert_rect.x0, y), line, font=font, fontsize=fontsize)
            wrote = True
        except Exception:
            logger.exception("copy-layer append failed for %r", line[:40])
        y += fontsize * _LINE_HEIGHT
    if wrote:
        tw.write_text(page, render_mode=3)


# Codepoint every shaped glyph is remapped to. U+200B (zero-width space) makes
# the shaped layer extract as *nothing visible*, so copy returns only the clean
# invisible layer. It must be a real, defined codepoint: mapping to U+0000 or an
# empty string makes MuPDF/Adobe treat the entry as "undefined" and fall back to
# the font's own (garbled) cmap. A single giant `bfrange` cannot be used either
# — a bfrange *increments* its destination per code, so every glyph must get its
# own `bfchar` entry.
_NEUTRAL_CP = 0x200B
_RE_REF = re.compile(rb"(\d+) 0 R")
_FALLBACK_GLYPH_COUNT = 2048  # if the subset's glyph count can't be read


def _blank_cmap(glyph_count: int) -> bytes:
    """A ToUnicode CMap mapping CIDs 0..glyph_count-1 to the neutral codepoint."""
    dst = b"%04x" % _NEUTRAL_CP
    chars, i = [], 0
    while i < glyph_count:
        chunk = min(100, glyph_count - i)  # PDF caps a bfchar block at 100
        chars.append(b"%d beginbfchar" % chunk)
        chars.extend(b"<%04x> <%s>" % (c, dst) for c in range(i, i + chunk))
        chars.append(b"endbfchar")
        i += chunk
    return (
        b"/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n"
        b"/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n"
        + b"\n".join(chars)
        + b"\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    )


def _first_ref(value) -> int | None:
    """The first `N 0 R` xref in a `xref_get_key` value tuple, or None."""
    if not value:
        return None
    m = _RE_REF.search(value[1].encode() if isinstance(value[1], str) else value[1])
    return int(m.group(1)) if m else None


def _subset_glyph_count(doc: fitz.Document, type0_xref: int) -> int:
    """Number of glyphs in a Type0 font's embedded (subset) program.

    Type0 -> DescendantFonts[0] -> FontDescriptor -> FontFile2/3. For a subset
    this is small (a few hundred), and CID == GID under Identity-H, so mapping
    0..count-1 covers every code the content stream can reference.
    """
    try:
        desc = _first_ref(doc.xref_get_key(type0_xref, "DescendantFonts"))
        fd = _first_ref(doc.xref_get_key(desc, "FontDescriptor"))
        for key in ("FontFile2", "FontFile3", "FontFile"):
            ff = _first_ref(doc.xref_get_key(fd, key))
            if ff is not None:
                from fontTools.ttLib import TTFont

                tt = TTFont(io.BytesIO(doc.xref_stream(ff)))
                return tt["maxp"].numGlyphs
    except Exception:
        logger.exception("could not read glyph count for font xref %d", type0_xref)
    return _FALLBACK_GLYPH_COUNT


def _is_shaped_bangla(ftype: str, basefont: str, keep_tag: str) -> bool:
    """True for a shaped Bangla font we must blank; False for the overlay/others.

    Matches the Bangla fonts the pipeline draws with (`Noto Sans/Serif Bengali`,
    plus the serif fallbacks MuPDF substitutes), but never the uniquely-named
    overlay font, which must stay copyable.
    """
    if ftype != "Type0":
        return False
    low = (basefont or "").lower()
    if keep_tag.lower() in low:
        return False  # our own invisible layer
    return "bengali" in low or any(f in low for f in manifest.FALLBACK_FONTS)


def blank_shaped_tounicode(doc: fitz.Document, keep_tag: str = COPY_LAYER_TAG) -> int:
    """Neutralise the ToUnicode of every shaped Bangla font, sparing the overlay.

    Remaps each shaped glyph to a zero-width space so the shaped layer yields no
    text on copy, leaving only the clean invisible layer. Must run *after*
    `subset_fonts()`, which rewrites ToUnicode. Returns the number of fonts
    neutralised.
    """
    # xref of the Type0 font dict -> its ToUnicode CMap xref. Dedup by CMap xref
    # so a font reused across pages is rewritten once.
    cmap_to_font: dict[int, int] = {}
    for pno in range(doc.page_count):
        for f in doc[pno].get_fonts(full=True):
            xref, ftype, basefont = f[0], f[2], f[3]
            if not _is_shaped_bangla(ftype, basefont, keep_tag):
                continue
            cmap_xref = _first_ref(doc.xref_get_key(xref, "ToUnicode"))
            if cmap_xref is not None:
                cmap_to_font.setdefault(cmap_xref, xref)

    for cmap_xref, font_xref in cmap_to_font.items():
        try:
            n = _subset_glyph_count(doc, font_xref)
            doc.update_stream(cmap_xref, _blank_cmap(n))
        except Exception:
            logger.exception("failed to neutralise ToUnicode stream %d", cmap_xref)

    logger.info("copy-layer: neutralised %d shaped Bangla ToUnicode CMaps", len(cmap_to_font))
    return len(cmap_to_font)
