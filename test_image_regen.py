"""Tests for the image regeneration pipeline (image_regen.py).

Runs zero API calls: the two model functions are stubbed, so what is exercised is the part
that can damage a document — extraction, placement, the vector guards, and the text baking.

    PYTHONIOENCODING=utf-8 .\\.venv\\Scripts\\python.exe test_image_regen.py
"""

import hashlib
import io
import os
import sys

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

import image_regen
from image_regen import (
    _accept_vector_regions,
    _apply_vectors,
    _bake_text,
    _denorm,
    _resize_to,
    _summarize,
    Job,
    regenerate_pdf,
)

# The one manual actually present in the repo. Every test using it is guarded, because the
# other fixtures the older test files reference are not committed.
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "OriginalPDF",
    "Heart Failure Manual v4 2021 (1) [41-80].pdf",
)

failures: list[str] = []
skipped: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        if detail:
            print(f"      {detail}")
        failures.append(name)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def png_of(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def build_pdf() -> bytes:
    """A two-page document with real text and three pictures of different shapes."""
    doc = fitz.open()
    for page_index in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Heart failure page {page_index + 1}", fontsize=14)
        page.insert_text((72, 130), "Keep this text exactly as it is.", fontsize=11)
        page.insert_image(fitz.Rect(100, 300, 300, 450), stream=png_of(200, 150, (200, 30, 30)))
        page.insert_image(fitz.Rect(350, 300, 450, 400), stream=png_of(120, 120, (30, 30, 200)))
    out = doc.tobytes()
    doc.close()
    return out


def page_texts(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = [page.get_text("text") for page in doc]
    doc.close()
    return texts


def image_placements(pdf_bytes: bytes) -> list[tuple]:
    """Every image placement in the document: (page, x0, y0, x1, y1), sorted.

    Keyed by geometry rather than by xref, because saving a PDF with garbage collection
    renumbers its objects — an xref that changes number has not moved on the page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    placements = [
        (page_num,) + tuple(round(v, 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1))
        for page_num, page in enumerate(doc, start=1)
        for entry in page.get_images(full=True)
        for rect in page.get_image_rects(entry[0])
    ]
    doc.close()
    return sorted(placements)


def image_digests(pdf_bytes: bytes) -> list[str]:
    """A fingerprint of every distinct image's decoded pixels, sorted.

    Identity again comes from content, not from the xref number: this answers "are these the
    same pictures?" across a save that renumbered everything.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    digests = []
    for xref in {entry[0] for page in doc for entry in page.get_images(full=True)}:
        try:
            with Image.open(io.BytesIO(doc.extract_image(xref)["image"])) as img:
                rgb = img.convert("RGB")
                digests.append(f"{rgb.size}:{hashlib.md5(rgb.tobytes()).hexdigest()}")
        except Exception:
            digests.append(f"unreadable:{xref}")
    doc.close()
    return sorted(digests)


class stub_models:
    """Replace the two model calls for the duration of a block.

    The generated image is deliberately a different size and aspect from every picture in the
    test document: nothing downstream is allowed to care, because the placement rect lives in
    the page's content stream and the image is painted into a unit square inside it.
    """

    def __init__(self, decision: dict, generated: bytes | None = None):
        self.decision = decision
        self.generated = generated if generated is not None else png_of(777, 333, (0, 180, 0))
        self.classified = 0

    def _classify(self, png_bytes):
        self.classified += 1
        return dict(self.decision)

    def __enter__(self):
        self._saved = (image_regen._classify, image_regen._regenerate)
        image_regen._classify = self._classify
        image_regen._regenerate = lambda png, categories, context: self.generated
        return self

    def __exit__(self, *exc):
        image_regen._classify, image_regen._regenerate = self._saved
        return False


REGENERATE = {
    "is_logo": False,
    "needs_regeneration": True,
    "has_text": False,
    "categories": ["people_attire"],
    "reason": "test",
}
KEEP_AS_LOGO = {**REGENERATE, "is_logo": True, "needs_regeneration": False}


class FakeRegion:
    """Stands in for image_regions.IllustrationRegion — the guards only read key and rect."""

    def __init__(self, key: str, rect: fitz.Rect):
        self.key = key
        self.rect = rect


# --------------------------------------------------------------------------------------
# The requirements, as checks
# --------------------------------------------------------------------------------------


def test_text_is_untouched():
    """Regenerating every picture must not change one character of the page text.

    This is the whole promise of the in-place swap: the content stream is never edited, so the
    text objects cannot move, be redacted, or be covered.
    """
    source = build_pdf()
    with stub_models(REGENERATE):
        out, _ = regenerate_pdf(source)

    before, after = page_texts(source), page_texts(out)
    check(
        "regenerating every image leaves the page text byte-identical",
        before == after,
        f"page texts changed: {before!r} -> {after!r}",
    )


def test_placement_is_unchanged():
    """Every picture stays exactly where it was, at exactly the size it was.

    The replacement is 777x333 against sources of 200x150 and 120x120 — a different pixel count
    and a different aspect ratio. If the placement survived that, resolution cannot move
    anything, which is the property the whole pipeline rests on.
    """
    source = build_pdf()
    with stub_models(REGENERATE):
        out, _ = regenerate_pdf(source)

    before, after = image_placements(source), image_placements(out)
    check(
        "every image keeps its exact placement rect after the swap",
        before == after,
        f"{before} -> {after}",
    )
    check("no image placement is lost or added", len(before) == len(after) > 0, f"{before}")
    check(
        "every image really was replaced",
        image_digests(source) != image_digests(out),
        "the pictures came back unchanged — the swap did nothing",
    )


def test_no_image_is_skipped():
    """Every image in the document is accounted for in the summary, with a real status."""
    source = build_pdf()
    with stub_models(REGENERATE) as stub:
        _, summary = regenerate_pdf(source)

    expected = len(image_digests(source))
    check(
        "the summary counts exactly the images the document contains",
        summary.startswith(f"{expected} images:"),
        f"expected {expected} images, summary was {summary!r}",
    )
    check(
        "every image was actually classified rather than passed over",
        stub.classified == expected,
        f"{stub.classified} classify calls for {expected} images",
    )
    check(
        "every image reports the regenerated status",
        f"{expected} regenerated" in summary,
        summary,
    )


def test_logo_is_left_byte_identical():
    """A logo is not merely 'not redrawn' — its pixels must come back untouched."""
    source = build_pdf()
    with stub_models(KEEP_AS_LOGO):
        out, summary = regenerate_pdf(source)

    check("a logo is reported as kept", "logo_kept" in summary, summary)
    check(
        "a logo's pixels come back byte-identical",
        image_digests(source) == image_digests(out),
        f"{image_digests(source)} -> {image_digests(out)}",
    )
    check(
        "a logo is not even re-encoded, so its placement is untouched too",
        image_placements(source) == image_placements(out),
        "a kept logo moved on the page",
    )


def test_decorative_strip_is_left_alone():
    """A sliver too thin to be a picture is kept, and is reported rather than silently dropped.

    Regenerating one is all risk and no gain: it holds nothing cultural, and a few percent of
    drift in a shape whose image the page mostly crops moves that shape out of view entirely.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Body text", fontsize=12)
    page.insert_image(fitz.Rect(0, 100, 20, 700), stream=png_of(114, 1381, (150, 220, 40)))
    source = doc.tobytes()
    doc.close()

    with stub_models(REGENERATE) as stub:
        out, summary = regenerate_pdf(source)

    check("a 12:1 sliver is reported as a decorative strip", "decorative_strip" in summary, summary)
    check(
        "a decorative strip is never sent to the model",
        stub.classified == 0,
        f"{stub.classified} classify calls for an image that should not have been sent",
    )
    check(
        "a decorative strip's pixels are untouched",
        image_digests(source) == image_digests(out),
        "the strip was redrawn anyway",
    )


def test_smask_is_preserved():
    """A cut-out picture keeps its soft mask, so it cannot come back as an opaque rectangle
    printed over the panel it was floating on."""
    if not os.path.exists(FIXTURE):
        skipped.append("SMask preservation (fixture missing)")
        return

    source = open(FIXTURE, "rb").read()
    doc = fitz.open(stream=source, filetype="pdf")
    masked = [
        entry[0]
        for page in doc
        for entry in page.get_images(full=True)
        if doc.xref_get_key(entry[0], "SMask")[0] != "null"
    ]
    doc.close()
    if not masked:
        skipped.append("SMask preservation (no masked image in the fixture)")
        return

    with stub_models(REGENERATE):
        out, _ = regenerate_pdf(source)

    doc = fitz.open(stream=out, filetype="pdf")
    still_masked = [x for x in masked if doc.xref_get_key(x, "SMask")[0] != "null"]
    doc.close()
    check(
        "an image with a soft mask still has one after being swapped",
        sorted(still_masked) == sorted(masked),
        f"{len(masked)} masked images went in, {len(still_masked)} came out masked",
    )


def test_real_document_survives_intact():
    """The same two guarantees, against a real 40-page manual rather than a built fixture."""
    if not os.path.exists(FIXTURE):
        skipped.append("real-document round trip (fixture missing)")
        return

    source = open(FIXTURE, "rb").read()
    with stub_models(REGENERATE):
        out, summary = regenerate_pdf(source)

    check(
        "40-page manual: every page's text is byte-identical",
        page_texts(source) == page_texts(out),
        "page text changed somewhere in the manual",
    )
    check(
        "40-page manual: every image placement is unchanged",
        image_placements(source) == image_placements(out),
        "an image moved or changed size on the page",
    )
    print(f"      summary: {summary}")


# --------------------------------------------------------------------------------------
# The vector path — the one place page content is edited
# --------------------------------------------------------------------------------------


def test_vector_guards_reject_unsafe_regions():
    """A vector region is only taken when a foreground insert over it can bury nothing."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((120, 220), "A caption inside the region", fontsize=11)

    over_text = FakeRegion("vec:1:0", fitz.Rect(100, 200, 300, 300))
    over_raster = FakeRegion("vec:1:1", fitz.Rect(350, 500, 450, 600))
    clean = FakeRegion("vec:1:2", fitz.Rect(60, 600, 260, 780))
    overlapping = FakeRegion("vec:1:3", fitz.Rect(100, 650, 300, 800))

    statuses: dict[str, str] = {}
    accepted = _accept_vector_regions(
        page,
        [over_text, over_raster, clean, overlapping],
        [fitz.Rect(340, 490, 460, 610)],
        statuses,
    )
    doc.close()

    check(
        "a region containing page text is rejected",
        statuses.get("vec:1:0") == "vector_has_text",
        f"got {statuses.get('vec:1:0')!r}",
    )
    check(
        "a region sitting over a raster image is rejected",
        statuses.get("vec:1:1") == "vector_over_raster",
        f"got {statuses.get('vec:1:1')!r}",
    )
    check(
        "a region overlapping an accepted region is rejected",
        statuses.get("vec:1:3") == "vector_overlap",
        f"got {statuses.get('vec:1:3')!r}",
    )
    check(
        "the clean region is the only one accepted",
        [r.key for r in accepted] == ["vec:1:2"],
        f"accepted {[r.key for r in accepted]}",
    )


def test_vector_replacement_keeps_the_page_text():
    """Redacting the artwork and inserting over it must not disturb text elsewhere on the page."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "A heading well above the artwork", fontsize=14)
    shape = page.new_shape()
    for offset in range(0, 120, 12):
        shape.draw_bezier(
            fitz.Point(120 + offset, 400), fitz.Point(160, 360),
            fitz.Point(200, 460), fitz.Point(240 - offset, 420),
        )
    shape.finish(color=(0.1, 0.2, 0.8), width=1.5)
    shape.commit()
    source = doc.tobytes()
    doc.close()

    doc = fitz.open(stream=source, filetype="pdf")
    rect = fitz.Rect(100, 340, 280, 480)
    job = Job(
        key="vec:1:0", kind="vector", page_num=1, png=b"", width=180, height=140,
        context="", rect=rect, result=png_of(360, 280, (0, 160, 0)),
    )
    _apply_vectors(doc, [job])
    out = doc.tobytes()
    doc.close()

    check(
        "vector replacement leaves the rest of the page's text intact",
        page_texts(source) == page_texts(out),
        f"{page_texts(source)!r} -> {page_texts(out)!r}",
    )

    doc = fitz.open(stream=out, filetype="pdf")
    placed = doc[0].get_image_rects(doc[0].get_images(full=True)[0][0])
    doc.close()
    check(
        "the replacement lands exactly on the region it replaced",
        placed and tuple(round(v) for v in placed[0]) == tuple(round(v) for v in rect),
        f"placed at {placed}",
    )


# --------------------------------------------------------------------------------------
# Baking translated text into a picture
# --------------------------------------------------------------------------------------


def test_bake_text_erases_english_and_draws_bangla():
    """The English is painted out and real Bangla glyphs are drawn in its place.

    Rendered through insert_htmlbox rather than PIL: Bengali needs conjunct shaping and
    pre-base matra reordering, which PIL only does when built against libraqm.
    """
    # Sized past TEXT_BAKE_MIN_DIM so no enlargement happens and the two images stay directly
    # comparable pixel for pixel; the enlargement itself is covered by its own test.
    scratch = fitz.open()
    page = scratch.new_page(width=900, height=450)
    page.insert_text((90, 225), "BLOOD PRESSURE", fontsize=54)
    source = page.get_pixmap(dpi=72).tobytes("png")
    scratch.close()

    block = {"text": "BLOOD PRESSURE", "lang": "en", "bbox": [0.08, 0.38, 0.85, 0.55],
             "bn": "রক্তচাপ"}
    baked = _bake_text(source, [block])

    with Image.open(io.BytesIO(source)) as a, Image.open(io.BytesIO(baked)) as b:
        before, after = a.convert("RGB"), b.convert("RGB")
        box = _denorm(block["bbox"], before.width, before.height)
        ink_before = sum(1 for px in before.crop(box).getdata() if sum(px) < 400)
        ink_after = sum(1 for px in after.crop(box).getdata() if sum(px) < 400)
        outside_same = before.crop((0, 0, before.width, box[1])).tobytes() == \
            after.crop((0, 0, after.width, box[1])).tobytes()

    check("baking text draws ink where the English was", ink_after > 0, f"{ink_after} dark pixels")
    check(
        "the baked box is genuinely redrawn, not left as the English",
        ink_before != ink_after,
        f"{ink_before} dark pixels before, {ink_after} after",
    )
    check(
        "baking text changes nothing outside the text box",
        outside_same,
        "pixels above the text box changed",
    )


def test_bake_text_keeps_the_bar_a_caption_sits_on():
    """White type on a coloured bar comes back as type on that same bar.

    The regression this pins: an OCR box is drawn around the *words*, so on a caption bar
    barely taller than its text the box's border ring catches as much of the white paper
    beside the bar as of the bar itself. Sampling the ring painted the bar out.
    """
    img = Image.new("RGB", (900, 450), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((90, 180, 810, 279), fill=(0, 90, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    # A box drawn around the words, so it overhangs the bar top and bottom — the shape that
    # made the ring sample come back white.
    block = {"text": "PINT OF LAGER", "bbox": [0.11, 0.38, 0.89, 0.64], "bn": "এক পাইন্ট"}
    baked = _bake_text(buf.getvalue(), [block])

    with Image.open(io.BytesIO(baked)) as out:
        bar_pixels = list(out.convert("RGB").crop((90, 185, 810, 275)).get_flattened_data())
    blue = sum(1 for px in bar_pixels if px[2] > px[0] + 50)
    check(
        "the coloured bar under a caption survives the erase",
        blue > len(bar_pixels) * 0.5,
        f"only {blue} of {len(bar_pixels)} pixels on the bar are still blue",
    )


def test_bake_text_enlarges_a_small_picture():
    """A small tile is enlarged before the Bangla is drawn, so the glyphs actually resolve.

    Free on the page: the placement rect is fixed, so a bigger image only prints sharper.
    """
    baked = _bake_text(
        png_of(180, 177, (0, 90, 200)),
        [{"text": "PINT", "bbox": [0.1, 0.4, 0.9, 0.7], "bn": "এক পাইন্ট"}],
    )
    with Image.open(io.BytesIO(baked)) as out:
        size = out.size
    check(
        "a 180px tile is enlarged before its text is baked in",
        max(size) >= 700 and abs(size[0] / size[1] - 180 / 177) < 0.01,
        f"came out {size}",
    )

    # A picture already big enough is left at its own size — nothing to gain, and re-encoding
    # a large photo for no reason costs quality.
    baked = _bake_text(
        png_of(1200, 900, (0, 90, 200)),
        [{"text": "PINT", "bbox": [0.1, 0.4, 0.9, 0.7], "bn": "এক পাইন্ট"}],
    )
    with Image.open(io.BytesIO(baked)) as out:
        check("a picture already large enough is not enlarged", out.size == (1200, 900), str(out.size))


def test_bake_text_survives_an_untranslatable_block():
    """A block with no resolved text is left alone rather than erased into a blank patch."""
    source = png_of(200, 80, (240, 240, 240))
    baked = _bake_text(source, [{"text": "x", "bbox": [0.1, 0.1, 0.9, 0.9], "bn": ""}])
    with Image.open(io.BytesIO(source)) as a, Image.open(io.BytesIO(baked)) as b:
        same = a.convert("RGB").tobytes() == b.convert("RGB").tobytes()
    check("an empty translation leaves the picture untouched", same, "the picture was altered")


# --------------------------------------------------------------------------------------
# Small pieces
# --------------------------------------------------------------------------------------


def test_denorm_clamps_to_the_image():
    check(
        "an out-of-range bbox is clamped inside the image",
        _denorm([-0.5, -0.5, 1.5, 1.5], 100, 50) == (0, 0, 100, 50),
        str(_denorm([-0.5, -0.5, 1.5, 1.5], 100, 50)),
    )
    box = _denorm([0.5, 0.5, 0.5, 0.5], 100, 50)
    check(
        "a degenerate bbox still yields a drawable box",
        box[2] > box[0] and box[3] > box[1],
        str(box),
    )


def test_resize_keeps_aspect_ratio():
    """Extra resolution is kept; a mismatched aspect ratio is resampled, because the placement
    rect is fixed and a wrong aspect would print stretched."""
    with Image.open(io.BytesIO(_resize_to(png_of(400, 300, (1, 2, 3)), 200, 150))) as img:
        kept = img.size
    with Image.open(io.BytesIO(_resize_to(png_of(400, 400, (1, 2, 3)), 200, 150))) as img:
        corrected = img.size
    check("a matching aspect ratio keeps the higher resolution", kept == (400, 300), str(kept))
    check(
        "a mismatched aspect ratio is resampled to the original dimensions",
        corrected == (200, 150),
        str(corrected),
    )


def test_summary_counts_every_image():
    summary = _summarize({"a": "regenerated", "b": "regenerated", "c": "logo_kept"})
    check(
        "the summary tallies each status",
        summary == "3 images: 2 regenerated, 1 logo_kept",
        summary,
    )


def main() -> int:
    for test in (
        test_text_is_untouched,
        test_placement_is_unchanged,
        test_no_image_is_skipped,
        test_logo_is_left_byte_identical,
        test_decorative_strip_is_left_alone,
        test_smask_is_preserved,
        test_real_document_survives_intact,
        test_vector_guards_reject_unsafe_regions,
        test_vector_replacement_keeps_the_page_text,
        test_bake_text_erases_english_and_draws_bangla,
        test_bake_text_keeps_the_bar_a_caption_sits_on,
        test_bake_text_enlarges_a_small_picture,
        test_bake_text_survives_an_untranslatable_block,
        test_denorm_clamps_to_the_image,
        test_resize_keeps_aspect_ratio,
        test_summary_counts_every_image,
    ):
        test()

    print("=" * 60)
    for name in skipped:
        print(f"SKIP  {name}")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
