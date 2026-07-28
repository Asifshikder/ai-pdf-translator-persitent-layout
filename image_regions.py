"""Detection and rasterization of vector-drawn illustrations and ImageMask images.

This module finds culturally-adaptable illustrations that are not discovered by the
standard raster-image extraction pipeline — specifically vector-drawn content (like
the manual's cover cartoon, drawn as 200+ vector paths) and PDF ImageMask stencils
(like monochrome handwritten task lists, drawn as 1-bit fax masks, which lose their
color context during extraction).

All detection logic is kept here separate from image_processor.py to be unit-testable
without vertex_client dependencies.
"""

import hashlib
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Thresholds for vector-cluster detection, empirically validated against the real document.
# Verified on 176-page manual: correctly catches the cover illustration, correctly rejects
# all corner crop-marks and page decorations.
CLUSTER_PAD = 12.0  # pt: merge distance for spatially-adjacent drawing fragments
MIN_CLUSTER_ITEMS = 15  # min drawing-item count per cluster (real: 69; false positives: 6-8)
MIN_CLUSTER_AREA = 10_000.0  # pt²: above PANEL_MIN_AREA=8000 (pdf_processor.py)
MIN_CLUSTER_DIM = 40.0  # pt: excludes thin decorative bars
MAX_CLUSTER_COVERAGE = 0.92  # fraction of page rect: excludes full-bleed watermarks
MAX_REGIONS_PER_PAGE = 6  # circuit breaker: keep largest clusters
MAX_ILLUSTRATION_REGIONS_PER_DOC = 20  # circuit breaker across the whole document
ENABLE_VECTOR_ILLUSTRATION_LOCALIZATION = True  # feature flag: can be toggled off

# A backdrop is a plain filled rectangle big enough to sit *behind* things: the cover's
# lavender panel, or the white matte a layout draws under a photo. It is not part of any
# illustration, and letting it into a cluster is what fuses the cover cartoon with the
# whole page — its bbox then covers everything and the region is useless.
BACKDROP_MIN_COVERAGE = 0.10  # fraction of the page rect
BACKDROP_MAX_ITEMS = 2  # a rectangle is one "re"; anything richer is real artwork
# A plain fill running nearly the whole height or width of the page is page decoration
# even when it is too narrow to reach BACKDROP_MIN_COVERAGE — the divider pages carry an
# 8.5pt lavender spine bar down their left edge, and clustering it with the artwork drags
# the region's bbox to the top of the paper.
BACKDROP_MIN_SPAN = 0.80  # fraction of a page dimension

# How much live text may sit INSIDE a cluster before it stops being a picture.
#
# What has to be kept out is the line art of tables and diagrams — the "Exercise can:" grid,
# a flow diagram, a checkbox list — because the image model regenerates those as pictures,
# i.e. destroys them. This used to be asked of the whole page ("is this page essentially a
# picture?", limit 12 lines), which is a proxy, and the proxy broke the moment the manual was
# translated: Bangla wraps into more lines than the English it was measured on, so a divider
# page went from 12 lines to 16 and its cartoon became invisible. On the Revascularisation
# manual that silently cost every divider illustration in the book — the audit for a 91-page
# run recorded zero vector regions.
#
# The cluster's own text is the real signal, and it separates the two cases cleanly. Measured
# on that manual: the divider cartoons carry 3-6 lines inside their bbox, while table and
# diagram clusters carry 38, 49, 81, 90, 97, 239 and 286. Anything in between is a judgement
# the classifier is better placed to make than a threshold.
MAX_TEXT_LINES_IN_REGION = 12

# Drawing fragments this close to the page edge are printer's furniture — crop marks,
# registration targets, colour bars. They cluster with real art and drag its bbox out to
# the paper's corner (seen on pages 19 and 37: a cover cluster starting at 21,21).
TRIM_MARGIN = 28.0  # pt


@dataclass(frozen=True)
class IllustrationRegion:
    """A vector-drawn or masked illustration region detected on a page."""
    key: str  # f"vec:{page_num}:{index}" — unique within one localize_pdf() run
    content_key: str  # hash of member drawings (translation-invariant) — used for dedup
    page_num: int
    rect: fitz.Rect
    item_count: int  # total number of drawing items in the cluster
    page_context: str  # nearby text context for the edit model


def _is_image_mask(doc: fitz.Document, xref: int) -> bool:
    """Check whether an image XObject is a PDF stencil mask (1-bit, no inherent color)."""
    try:
        _, value = doc.xref_get_key(xref, "ImageMask")
        return value == "true"
    except Exception:
        return False


def _is_backdrop(drawing: dict, page: fitz.Page) -> bool:
    """True if a drawing is a plain background panel rather than part of an illustration.

    The cover's cartoon is drawn on a full-bleed lavender rectangle, and every photo in
    the manual sits on a white matte of its own. Both are filled, both are large, and
    both touch everything near them — so clustering pulls the entire page into one
    region and its bbox becomes useless. What separates a backdrop from artwork is
    complexity: a backdrop is one or two straight-edged path items, while the cartoon's
    body is a single path of 181 items with curves.
    """
    if drawing["type"] not in ("f", "fs"):
        return False
    items = drawing.get("items", [])
    if len(items) > BACKDROP_MAX_ITEMS or any(item[0] == "c" for item in items):
        return False
    rect = fitz.Rect(drawing["rect"])
    if rect.get_area() > BACKDROP_MIN_COVERAGE * page.rect.get_area():
        return True
    return (
        rect.width >= BACKDROP_MIN_SPAN * page.rect.width
        or rect.height >= BACKDROP_MIN_SPAN * page.rect.height
    )


def _in_trim_margin(rect: fitz.Rect, page: fitz.Page) -> bool:
    """True if a drawing sits in the paper margin outside the trimmed page."""
    return (
        rect.x1 <= page.rect.x0 + TRIM_MARGIN
        or rect.x0 >= page.rect.x1 - TRIM_MARGIN
        or rect.y1 <= page.rect.y0 + TRIM_MARGIN
        or rect.y0 >= page.rect.y1 - TRIM_MARGIN
    )


def _text_line_count(page: fitz.Page) -> int:
    """Number of text lines on the page — how much of it is a document, not a picture."""
    try:
        return sum(
            len(block["lines"])
            for block in page.get_text("dict")["blocks"]
            if block["type"] == 0
        )
    except Exception:
        logger.debug("Could not count text lines", exc_info=True)
        return 0


def _text_lines_in(page: fitz.Page, rect: fitz.Rect) -> int:
    """Number of the page's text lines that lie inside `rect`.

    Majority containment, not intersection: a caption printed hard against an illustration
    clips its bbox by a hair, and counting that as text *in* the picture would veto exactly
    the pictures this is meant to admit. See MAX_TEXT_LINES_IN_REGION.
    """
    count = 0
    try:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_rect = fitz.Rect(line["bbox"])
                area = line_rect.get_area()
                if area > 0 and (line_rect & rect).get_area() > 0.5 * area:
                    count += 1
    except Exception:
        logger.debug("Could not count text lines in a region", exc_info=True)
    return count


class TextFreePages:
    """Serves copies of a document's pages with the live text layer removed.

    The region sent to the edit model must not carry the page's English text: the model
    treats baked-in words as part of the picture and redraws them, so the localized
    raster arrives with English already burnt into it — under the real text objects the
    redaction step deliberately preserved. Stripping the text first gives the model the
    artwork alone.

    Pages are stripped lazily and cached, because the strip costs a redaction pass and
    only a handful of pages in a document ever hold an illustration region. The shadow
    documents are held open: a `fitz.Page` does not outlive its document.
    """

    def __init__(self, doc: fitz.Document) -> None:
        self._doc = doc
        self._shadows: dict[int, fitz.Document] = {}

    def page(self, page_num: int) -> fitz.Page:
        """The 1-based page with its text removed, or the original page if stripping fails."""
        if page_num not in self._shadows:
            try:
                shadow = fitz.open()
                shadow.insert_pdf(self._doc, from_page=page_num - 1, to_page=page_num - 1)
                shadow_page = shadow[0]
                shadow_page.add_redact_annot(shadow_page.rect)
                shadow_page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )
                self._shadows[page_num] = shadow
            except Exception:
                logger.exception("Page %d: could not strip text; rasterizing as-is", page_num)
                return self._doc[page_num - 1]
        return self._shadows[page_num][0]

    def close(self) -> None:
        for shadow in self._shadows.values():
            try:
                shadow.close()
            except Exception:
                pass
        self._shadows.clear()


def _rasterize_rect(
    page: fitz.Page, rect: fitz.Rect, target_dpi: int = 200, max_dim_px: int = 2400
) -> tuple[bytes, int, int] | None:
    """Render the page as it visually appears inside a rect, composited and anti-aliased.

    This is the core fix for both vector art (which get_images doesn't find) and ImageMask
    stencils (which extract_image returns as uncomposited 1-bit masks with no color).
    get_pixmap(alpha=False) composites the stencil against the real page background
    (white paper) instead of returning the bare mask. It is also how an image stored in a
    non-RGB colour space keeps its colours, which extraction does not.

    Note for callers that measure the result: a rect rendered at an arbitrary dpi does not
    land on an exact pixel boundary, so the outermost row and column can carry a sliver of
    whatever lies beyond the rect (page white, usually). Sample inside the edge, not on it —
    see image_processor._border_ring.

    Returns (png_bytes, width, height) or None on failure.
    """
    longest_pt = max(rect.width, rect.height, 1.0)
    dpi = int(min(target_dpi, max_dim_px / (longest_pt / 72.0)))
    try:
        pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
        return pix.tobytes("png"), pix.width, pix.height
    except Exception:
        logger.exception("Failed to rasterize rect %s on page", rect)
        return None


def _enclosed_by_any(rect: fitz.Rect, members: list[fitz.Rect], tolerance: float = 1.0) -> bool:
    """True if `rect` sits inside one of `members` — i.e. it is drawn on top of the art."""
    for member in members:
        grown = member + (-tolerance, -tolerance, tolerance, tolerance)
        if grown.contains(rect):
            return True
    return False


def _union_find_merge(rects: list[fitz.Rect]) -> list[list[int]]:
    """Group overlapping rects (after padding) via union-find.

    Returns list of clusters, each cluster is a list of indices into the input rects.
    """
    if not rects:
        return []
    n = len(rects)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Pad each rect and check pairwise overlaps
    padded = [r + (-CLUSTER_PAD, -CLUSTER_PAD, CLUSTER_PAD, CLUSTER_PAD) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            if padded[i].intersects(padded[j]):
                union(i, j)

    # Group indices by their root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


def _illustration_clusters(
    page: fitz.Page, page_num: int, protected_rects: list[fitz.Rect], image_placement_rects: list[fitz.Rect]
) -> list[IllustrationRegion]:
    """Detect vector-drawn illustration clusters on a page.

    protected_rects: bounding boxes of checkboxes, rules, and panels (from pdf_processor
                     _rules/_vector_marks/_panels) that must be preserved. A cluster whose
                     bbox intersects one of these is dropped — unless the protected rect
                     sits inside the cluster's own ink, in which case it is not furniture
                     at all but a piece of the illustration (see below).
    image_placement_rects: placement rects of raster images on this page. Drawings >50%
                           contained in these are excluded from clustering (prevents
                           double-counting a decorative frame drawn over a photo).

    Returns a list of IllustrationRegion candidates, sorted by area (largest first).
    """
    if not ENABLE_VECTOR_ILLUSTRATION_LOCALIZATION:
        return []

    drawings = page.get_drawings()
    if not drawings:
        return []

    # Crop marks and registration targets match the checkbox test exactly, and one sitting
    # in the paper margin would veto any region that reached up to the trim line.
    protected_rects = [pr for pr in protected_rects if not _in_trim_margin(pr, page)]

    # Exclude drawings that match the existing preservation rules
    # (checkboxes, rules, panels — via their exact size/shape criteria).
    excluded_rects: set[tuple] = set()

    for drawing in drawings:
        rect = fitz.Rect(drawing["rect"])

        # Rule: thin and long (table borders, heading underlines)
        if rect.width <= 3.0 and rect.height >= 20.0:
            excluded_rects.add(tuple(rect))
            continue
        if rect.height <= 3.0 and rect.width >= 20.0:
            excluded_rects.add(tuple(rect))
            continue

        # Mark (checkbox): small and roughly square
        if 5.0 <= rect.width <= 24.0 and 5.0 <= rect.height <= 24.0:
            aspect = max(rect.width, rect.height) / max(min(rect.width, rect.height), 0.1)
            if aspect <= 1.35:
                excluded_rects.add(tuple(rect))
                continue

        # Background panel: the page's own backdrop, not part of any illustration.
        #
        # Deliberately not the `_panels` (speech bubble) test used elsewhere — that one
        # accepts any large filled shape with a curve in it, which on this cover matches
        # the cartoon's t-shirt and both of its arms. Excluding those is what left the
        # cover with nothing to localize.
        if _is_backdrop(drawing, page):
            excluded_rects.add(tuple(rect))
            continue

        # Printer's furniture in the paper margin (crop marks, registration targets).
        if _in_trim_margin(rect, page):
            excluded_rects.add(tuple(rect))
            continue

        # Exclude drawings >50% contained in a raster image placement
        drawing_center_x = (rect.x0 + rect.x1) / 2
        drawing_center_y = (rect.y0 + rect.y1) / 2
        for img_rect in image_placement_rects:
            if img_rect.contains((drawing_center_x, drawing_center_y)):
                excluded_rects.add(tuple(rect))
                break

    # Cluster the remaining drawings, keeping track of original drawing indices
    candidate_rects: list[fitz.Rect] = []
    candidate_drawing_indices: list[int] = []
    for drawing_idx, d in enumerate(drawings):
        if tuple(fitz.Rect(d["rect"])) not in excluded_rects:
            candidate_rects.append(fitz.Rect(d["rect"]))
            candidate_drawing_indices.append(drawing_idx)

    if not candidate_rects:
        return []

    clusters = _union_find_merge(candidate_rects)

    # Build candidates from clusters, with filtering
    candidates: list[IllustrationRegion] = []
    for idx, cluster_indices in enumerate(clusters):
        cluster_rects = [candidate_rects[i] for i in cluster_indices]
        cluster_drawings = [drawings[candidate_drawing_indices[i]] for i in cluster_indices]

        # Union the bounding boxes (unpadded), clipped to the page: this document's
        # cover art runs 25pt past the bottom edge, and an insert rect reaching outside
        # the page squashes the replacement raster.
        bbox = fitz.Rect()
        for rect in cluster_rects:
            bbox |= rect
        bbox &= page.rect
        if bbox.is_empty:
            continue

        item_count = sum(len(d.get("items", [])) for d in cluster_drawings)

        # Apply filtering thresholds
        if item_count < MIN_CLUSTER_ITEMS:
            continue
        if bbox.get_area() < MIN_CLUSTER_AREA:
            continue
        if bbox.width < MIN_CLUSTER_DIM or bbox.height < MIN_CLUSTER_DIM:
            continue
        if bbox.get_area() / page.rect.get_area() > MAX_CLUSTER_COVERAGE:
            continue

        # The line art of a table, a checkbox list or a flow diagram, which the image model
        # would regenerate as a picture and so destroy. Such a cluster has the page's words
        # *inside* it; an illustration does not. See MAX_TEXT_LINES_IN_REGION.
        region_lines = _text_lines_in(page, bbox)
        if region_lines > MAX_TEXT_LINES_IN_REGION:
            logger.debug(
                "Page %d: cluster at %s rejected — %d text lines sit inside it, so it is a "
                "table or diagram rather than a picture",
                page_num, bbox, region_lines,
            )
            continue

        # Veto: reject if a checkbox, rule or panel that is *not part of this drawing*
        # falls inside the region — flattening one into a raster loses it.
        #
        # Membership matters because `_vector_marks` and `_panels` read shapes, not
        # meaning: on the cover they match 73 "checkboxes" and 7 "speech bubbles" that
        # are really the cartoon's own fingers, arms and t-shirt. A blanket veto let the
        # illustration veto itself, which is why no page of the manual detected anything.
        # A protected shape enclosed by one of the cluster's own paths is a piece of the
        # illustration; one sitting beside it is furniture, and still vetoes.
        blocker = next(
            (
                pr
                for pr in protected_rects
                if bbox.intersects(pr) and not _enclosed_by_any(pr, cluster_rects)
            ),
            None,
        )
        if blocker is not None:
            logger.debug(
                "Page %d: cluster at %s rejected — protected region %s (checkbox/rule/panel) "
                "lies inside it",
                page_num,
                bbox,
                blocker,
            )
            continue

        # Compute content-hash for deduplication (translation-invariant).
        #
        # Rounded to whole points, not tenths: the same cover cartoon is placed on seven
        # pages of this manual and one of its 75 paths measures 4.2pt wide on one page and
        # 4.3pt on the others, which at tenth-point precision split it into two signatures
        # — two AI edits, and two different-looking cartoons in one book. A 75-path
        # signature carrying each path's type, size, item count and fill stays far more
        # than specific enough at whole-point precision.
        content_sig = tuple(
            sorted(
                (
                    d["type"],
                    round(fitz.Rect(d["rect"]).width),
                    round(fitz.Rect(d["rect"]).height),
                    len(d.get("items", [])),
                    str(d.get("fill")),
                )
                for d in cluster_drawings
            )
        )
        content_key = hashlib.md5(str(content_sig).encode()).hexdigest()

        key = f"vec:{page_num}:{idx}"
        candidates.append(
            IllustrationRegion(
                key=key,
                content_key=content_key,
                page_num=page_num,
                rect=bbox,
                item_count=item_count,
                page_context="",  # filled in by localize_pdf when context is available
            )
        )

    # Sort by area and cap at MAX_REGIONS_PER_PAGE
    candidates.sort(key=lambda c: c.rect.get_area(), reverse=True)
    if len(candidates) > MAX_REGIONS_PER_PAGE:
        logger.info(
            "Page %d: %d clusters exceed cap %d, keeping largest %d",
            page_num,
            len(candidates),
            MAX_REGIONS_PER_PAGE,
            MAX_REGIONS_PER_PAGE,
        )
        candidates = candidates[:MAX_REGIONS_PER_PAGE]

    return candidates
