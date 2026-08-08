"""FastAPI app: upload an English PDF, download it translated to Bangla."""

import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

import fitz

import manifest
import page_fix
from docx_export import build_docx
from extract_images import extract_images
from fix_processor import FIX_SUFFIX, fix_pdf
from image_localizer_pipeline import localize_single_image
from image_processor import localize_pdf
from image_regen import REGEN_SUFFIX, regenerate_pdf
from manifest import ManifestMissing, ManifestUnsupported
from page_redesign import redesign_pdf
from pdf_processor import translate_pdf
from split_spreads import SPLIT_SUFFIX, split_spreads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="PDF Translator (English → Bangla)")


async def _read_pdf(file: UploadFile) -> bytes:
    """Validate the upload is a PDF and return its bytes."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    pdf_bytes = await file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    return pdf_bytes


def _pdf_response(
    file: UploadFile, content: bytes, suffix: str, extra: dict[str, str] | None = None
) -> Response:
    out_name = os.path.splitext(os.path.basename(file.filename))[0] + suffix
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    headers.update(extra or {})
    return Response(content=content, media_type="application/pdf", headers=headers)




@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        translated = translate_pdf(pdf_bytes)
    except Exception:
        logger.exception("Translation pipeline failed")
        raise HTTPException(status_code=500, detail="Translation failed. Check the server logs.")

    # Print-ready sources are often laid out two-up: one A3 sheet carrying two
    # facing A4 pages. Translation reproduces the geometry it is given, so the
    # spreads have to come apart afterwards.
    try:
        translated, _ = split_spreads(translated)
    except Exception:
        logger.exception("Spread splitting failed — returning the unsplit translation")

    return _pdf_response(file, translated, "_bn.pdf")


@app.post("/export_docx")
async def export_docx(file: UploadFile = File(...)):
    """Build a DOCX from a translated PDF's embedded manifest.

    Re-extracts nothing: the manifest already carries every segment's Bangla
    text and styling, so this is cheap and needs no AI calls.
    """
    pdf_bytes = await _read_pdf(file)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        data = manifest.read(doc)
    except (ManifestMissing, ManifestUnsupported) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        doc.close()

    try:
        docx_bytes = build_docx(data)
    except Exception:
        logger.exception("DOCX export failed")
        raise HTTPException(status_code=500, detail="DOCX export failed. Check the server logs.")

    out_name = os.path.splitext(os.path.basename(file.filename))[0] + ".docx"
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )


@app.post("/localize")
async def localize(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        localized = localize_pdf(pdf_bytes)
    except Exception:
        logger.exception("Image localization pipeline failed")
        raise HTTPException(
            status_code=500, detail="Image localization failed. Check the server logs."
        )

    return _pdf_response(file, localized, "_localized.pdf")


@app.post("/regenerate")
async def regenerate(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        regenerated, summary = regenerate_pdf(pdf_bytes)
    except Exception:
        logger.exception("Image regeneration pipeline failed")
        raise HTTPException(
            status_code=500, detail="Image regeneration failed. Check the server logs."
        )

    # Expose-Headers: fetch() cannot read a custom header unless it is listed.
    return _pdf_response(
        file,
        regenerated,
        REGEN_SUFFIX,
        {"X-Regen-Summary": summary, "Access-Control-Expose-Headers": "X-Regen-Summary"},
    )


@app.post("/fix")
async def fix(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        fixed, summary = fix_pdf(pdf_bytes)
    # Must precede the catch-all below, or a fixable-input problem reads as a crash.
    except (ManifestMissing, ManifestUnsupported) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Fix pipeline failed")
        raise HTTPException(status_code=500, detail="Fix failed. Check the server logs.")

    # Expose-Headers: fetch() cannot read a custom header unless it is listed.
    return _pdf_response(
        file,
        fixed,
        FIX_SUFFIX,
        {"X-Fix-Summary": summary, "Access-Control-Expose-Headers": "X-Fix-Summary"},
    )


@app.post("/redesign")
async def redesign(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        redesigned = redesign_pdf(pdf_bytes)
    except Exception:
        logger.exception("Page redesign pipeline failed")
        raise HTTPException(status_code=500, detail="Page redesign failed. Check the server logs.")

    return _pdf_response(file, redesigned, "_redesigned.pdf")


@app.post("/split")
async def split(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        split_bytes, summary = split_spreads(pdf_bytes)
    except Exception:
        logger.exception("Spread splitting failed")
        raise HTTPException(
            status_code=500, detail="Splitting failed. Check the server logs."
        )

    # Expose-Headers: fetch() cannot read a custom header unless it is listed.
    return _pdf_response(
        file,
        split_bytes,
        SPLIT_SUFFIX,
        {"X-Split-Summary": summary, "Access-Control-Expose-Headers": "X-Split-Summary"},
    )


@app.post("/extract_images")
async def extract_images_endpoint(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        zip_bytes = extract_images(pdf_bytes)
    except Exception:
        logger.exception("Image extraction pipeline failed")
        raise HTTPException(status_code=500, detail="Image extraction failed. Check the server logs.")

    out_name = os.path.splitext(os.path.basename(file.filename))[0] + "_images.zip"
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.post("/localize_images")
async def localize_images_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        localized_bytes = localize_single_image(image_bytes, file.content_type or "image/jpeg")
    except Exception:
        logger.exception("Image localization pipeline failed")
        raise HTTPException(
            status_code=500, detail="Image localization failed. Check the server logs."
        )

    out_name = os.path.splitext(os.path.basename(file.filename))[0] + "_localized"
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    out_name += ext
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return Response(content=localized_bytes, media_type=file.content_type or "image/jpeg", headers=headers)


# --------------------------------------------------------------------------------------
# Page Fix — its own window, its own session-based pipeline (page_fix.py).
#
# Unlike every endpoint above, this one is a conversation rather than a single
# request/response: the user uploads once, then issues instructions page by page and
# watches the result, so the document lives in a server-side session until they download
# it. Nothing here touches the pipelines above.
# --------------------------------------------------------------------------------------


@app.get("/pagefix")
def pagefix_window():
    return FileResponse(os.path.join(BASE_DIR, "static", "pagefix.html"))


@app.post("/pagefix/upload")
async def pagefix_upload(file: UploadFile = File(...)):
    pdf_bytes = await _read_pdf(file)
    try:
        session = page_fix.create_session(os.path.basename(file.filename or ""), pdf_bytes)
    except page_fix.PageFixError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Page fix: could not open the upload")
        raise HTTPException(status_code=500, detail="Could not open that PDF.")

    return JSONResponse(
        {"session": session.sid, "filename": session.filename, "pages": session.page_count}
    )


@app.get("/pagefix/{sid}/page/{page_number}")
def pagefix_page(sid: str, page_number: int, dpi: int = page_fix.PREVIEW_DPI):
    try:
        session = page_fix.get_session(sid)
        png = page_fix.render_page(
            session.pdf, page_number - 1, dpi=max(40, min(300, dpi))
        )
    except page_fix.PageFixError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("Page fix: rendering page %d failed", page_number)
        raise HTTPException(status_code=500, detail="Could not render that page.")

    # no-store: the same URL returns different pixels after every edit.
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/pagefix/{sid}/fix")
async def pagefix_fix(
    sid: str,
    instruction: str = Form(...),
    images: list[UploadFile] = File(default=[]),
):
    attachments = []
    for upload in images or []:
        raw = await upload.read()
        if raw:
            attachments.append((os.path.basename(upload.filename or "image"), raw))

    try:
        report = page_fix.fix_document(sid, instruction, attachments)
    except page_fix.PageFixError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Page fix pipeline failed")
        raise HTTPException(status_code=500, detail="The fix failed. Check the server logs.")

    return JSONResponse(report)


@app.post("/pagefix/{sid}/undo")
def pagefix_undo(sid: str):
    try:
        undone = page_fix.undo(sid)
    except page_fix.PageFixError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"undone": undone})


@app.get("/pagefix/{sid}/download")
def pagefix_download(sid: str):
    try:
        session = page_fix.get_session(sid)
    except page_fix.PageFixError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    out_name = os.path.splitext(session.filename)[0] + page_fix.PAGEFIX_SUFFIX
    return Response(
        content=session.pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
