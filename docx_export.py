"""Export translated segments from a manifest to DOCX format."""

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from pdf_processor import BANGLA_SIZE


def build_docx(manifest_data: dict) -> bytes:
    """Build a DOCX document from the translation manifest.

    The manifest records every segment's original and translated text, geometry,
    and styling. This function reconstructs a clean DOCX with the Bangla text,
    preserving size, color, bold, and alignment from the original PDF layout.

    Word will apply its own complex-script shaping (not HarfBuzz), so the visual
    result differs slightly from the PDF but is correct, readable Bangla — and
    avoids the glyph-corruption issue inherent to PDFs with shaped fonts.
    """
    doc = Document()

    for page_data in manifest_data.get("pages", []):
        for seg in page_data.get("segments", []):
            text = seg["bn"]
            if not text or not text.strip():
                continue

            para = doc.add_paragraph(text)
            para_format = para.paragraph_format

            # Alignment: PDF align field maps directly to Word enum
            align_map = {
                "left": WD_ALIGN_PARAGRAPH.LEFT,
                "center": WD_ALIGN_PARAGRAPH.CENTER,
                "right": WD_ALIGN_PARAGRAPH.RIGHT,
                "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
            }
            para_format.alignment = align_map.get(seg["align"], WD_ALIGN_PARAGRAPH.LEFT)

            # Run styling: size, bold, color, font
            for run in para.runs:
                run.font.name = "Nirmala UI"
                run.font.size = Pt(seg["size"] * BANGLA_SIZE)
                if seg["bold"]:
                    run.font.bold = True
                # Color is stored as 0xRRGGBB integer
                run.font.color.rgb = RGBColor.from_string(f"{seg['color']:06x}")

    # Write to bytes
    import io
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
