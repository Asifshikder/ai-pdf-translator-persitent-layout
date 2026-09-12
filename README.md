# PDF Translator (English → Bangla)

Upload an English PDF, get the same PDF back with the text translated to Bangla —
layout, images, and graphics preserved. Translation runs on Vertex AI (Gemini 2.5 Flash).

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

Then open http://127.0.0.1:8000 in your browser, drop in a PDF, and click **Translate to Bangla**.

## How it works

1. `pdf_processor.py` extracts every text block with its position and style (PyMuPDF),
   and groups it into segments — a segment is one paragraph, heading, bullet item or
   table cell, and is both the unit of translation and the unit of insertion.
2. `translator.py` sends all segments of a page to Gemini 2.5 Flash on Vertex AI in one
   batched call with structured JSON output (guaranteed one translation per segment).
3. Original text is removed with redactions that leave images and vector graphics untouched.
4. Bangla text is inserted into the original box (grown into whatever free space is
   next to it) with `insert_htmlbox`, which shapes Bangla properly (conjuncts/matras
   via HarfBuzz) and shrinks the font until it fits. Fonts: Noto Sans Bengali (in `fonts/`).
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

## Fix a PDF by asking (Page Fix)

The Fix tab above repairs defects this tool can recognise on its own. **Fix a PDF with AI**
(`page_fix.py`, its own window at <http://127.0.0.1:8000/pagefix>) handles the rest: drop
in a PDF, type what is wrong with it, attach any images it needs, and click **Fix it**.

You do not say which page — describe what you can see and it finds it:

- *"The subtitle that says 'Not just a book' should say 'More than a book'"* — the wording
  is replaced in the original size, colour and weight, growing into the free space beside
  it rather than shrinking;
- *"Replace the cartoon of the woman at the computer with the attached photo"* — the
  attachment is placed in exactly that spot;
- *"Remove the empty grey box under the table on page 6"* — naming a page works too (and
  a page's *printed* folio is reconciled with its position in the file); the region is
  cleared to the colour of the page around it;
- *"Make the person in this illustration Bangladeshi"* — that region alone is handed to an
  image model and redrawn back into the same rectangle.

Instructions stack: each one applies on top of the last, **Undo** takes back the whole of
the previous instruction however many pages it touched, and **Download PDF** gives you
`<name>_pagefix.pdf`. Every changed page is shown back to you with a list of what was done
to it.

Two model calls per instruction, for a reason. Locating the page needs only a thumbnail
and a text digest of each page; planning the edit needs a full-resolution render and the
page's measured geometry, which would be ruinous to send for all forty pages of a manual.
The planner then answers with a short list of typed operations (erase / replace_text /
insert_text / insert_image / regenerate_image / draw_box) which are applied as ordinary
PyMuPDF calls on a rectangle — it never produces a page. So a misunderstood instruction
can produce a wrong edit, but never a corrupt page. If it cannot find what you mean, it
changes nothing and says so rather than guessing.

Sessions live in the server's memory (six hours, eight documents), so download the result
before restarting uvicorn.

## Notes

- Credentials: `vertextaiproject2.json` (service account key) in the project root.
  It is git-ignored — never commit or share it.
- If translation fails for a page (API error, quota), the original English text is
  kept for that page instead of corrupting the document. The Fix tab repairs those.
- `insert_htmlbox` draws **nothing at all** when text cannot fit at the `scale_low` it
  was given — not a warning, not an overflow, just a blank space and a `-1` return. So
  `SCALE_LADDER` in `pdf_processor.py` ends at `0.0`: with no floor it is free to find
  whatever scale fits, and text is never lost. A block that lands below 60% is logged.
- `insert_htmlbox` decides whether text fits from the CSS `line-height`, not from the
  ink. Bangla draws over ~1.46x its font size, so leading tuned for Latin text reports
  a comfortable fit while the lines physically overlap — nothing downstream notices.
  This is why `CSS_TEMPLATE` uses 1.45 and why `test_layout.py` measures rendered pixels.

## Tests

```powershell
.\.venv\Scripts\python.exe test_layout.py
```

```powershell
.\.venv\Scripts\python.exe test_page_fix.py
```

`test_page_fix.py` covers the Page Fix pipeline with the planner stubbed, so it makes no
API calls either: the coordinate convention, the redaction-then-draw ordering, style
inheritance, session/undo, and the rule under a heading surviving a replacement above it.

`test_layout.py` runs the segmentation and box-planning stages against real PDFs and
makes **no API calls**, so it is free and fast. It covers the defects that keep coming
back: list items merging into run-on prose, text boxes planned outside the speech bubble
they belong to, and Bangla lines packed until their ink collides. The Heart Failure
Manual cases need `OriginalPDF/` and are skipped if it is absent.

## Setup from scratch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fonts are already in `fonts/`. The GCP project needs the Vertex AI API enabled and
the service account needs the "Vertex AI User" role.
