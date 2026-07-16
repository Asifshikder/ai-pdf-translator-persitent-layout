"""FastAPI app: upload an English PDF, download it translated to Bangla."""

import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from fix_processor import FIX_SUFFIX, fix_pdf
from image_processor import localize_pdf
from manifest import ManifestMissing, ManifestUnsupported
from pdf_processor import translate_pdf

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

    return _pdf_response(file, translated, "_bn.pdf")


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
