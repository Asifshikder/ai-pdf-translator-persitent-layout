# PDF Translator (English → Bangla)

Upload an English PDF, get the same PDF back with the text translated to Bangla —
layout, images, and graphics preserved. Translation runs on Vertex AI (Gemini 2.5 Flash).

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

Then open http://127.0.0.1:8000 in your browser, drop in a PDF, and click **Translate to Bangla**.

## How it works

1. `pdf_processor.py` extracts every text block with its position and style (PyMuPDF).
2. `translator.py` sends all blocks of a page to Gemini 2.5 Flash on Vertex AI in one
   batched call with structured JSON output (guaranteed one translation per block).
3. Original text is removed with redactions that leave images and vector graphics untouched.
4. Bangla text is inserted into the exact original box with `insert_htmlbox`, which
   shapes Bangla properly (conjuncts/matras via HarfBuzz) and **shrinks the font to fit**,
   down to a floor of 60% (`scale_low=0.6`) — so longer Bangla text never overlaps
   neighboring content. Fonts: Noto Sans Bengali (in `fonts/`).
5. `manifest.py` embeds a hidden JSON manifest in the output recording every segment's
   English, Bangla, geometry and fit result. Shaped Bangla cannot be read back out of a
   PDF, so this is the only record of what was written — and it is what the Fix tab uses.

## Fix a translated PDF

Upload a translated `_bn.pdf` to the **Fix** tab to get a `_bn_fix.pdf`. It reads the
embedded manifest and repairs, page by page:

- **untranslated text** — re-translates segments whose original request failed
  (text that is merely untranslatable, like `ECG` or a URL, is left alone);
- **dropped text** — text that did not fit at the 60% floor was not drawn *at all*;
  it is re-rendered at whatever scale does fit, and shortened via Gemini if that
  would be too small to read;
- **overlapping text** — pulls colliding boxes apart.

Pages with no defects are left byte-identical, and the run costs no API calls if there
is nothing to fix. A PDF without a manifest is rejected with a 400 — re-run Translate
first. Re-running Fix on its own output is a no-op.

## Notes

- Credentials: `vertextaiproject2.json` (service account key) in the project root.
  It is git-ignored — never commit or share it.
- If translation fails for a page (API error, quota), the original English text is
  kept for that page instead of corrupting the document. The Fix tab repairs those.
- If a block had to shrink below 60% of its original font size, a warning is logged.
  **Text that still does not fit at that floor is silently dropped from the page** —
  `insert_htmlbox` draws nothing rather than overflowing. The Fix tab restores it from
  the manifest; lowering `scale_low` in `pdf_processor.py` would prevent it up front.

## Setup from scratch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fonts are already in `fonts/`. The GCP project needs the Vertex AI API enabled and
the service account needs the "Vertex AI User" role.
