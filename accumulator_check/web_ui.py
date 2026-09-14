"""Browser upload UI for the accumulator (battery pack) checker.

Upload a top-down photo of the 6x6 cell pack, get back PASS/FAIL plus an
annotated view of every cell's detected +/- and which ones disagree with
their own column - see pack_checker.py for the check itself and its
current calibration status (per-cell classification is not fully
reliable yet - treat a FAIL here as "worth a human look", not a
certain defect).

Run with:
    python3 -m uvicorn accumulator_check.web_ui:app --reload --port 8001

Then open http://127.0.0.1:8001
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from accumulator_check.pack_checker import check_pack

app = FastAPI(title="Accumulator Checker")

UPLOAD_DIR = Path(__file__).resolve().parent / "dataset" / "incoming"

UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
<title>Accumulator Checker</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; color: #222; }
  h1 { font-size: 1.4rem; }
  .dropzone {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    width: 100%; box-sizing: border-box; min-height: 160px;
    border: 2px dashed #999; border-radius: 12px; padding: 48px 20px;
    text-align: center; color: #666; cursor: pointer; transition: border-color .15s, background .15s;
  }
  .dropzone.drag { border-color: #2a7; background: #f3fbf6; }
  input[type=file] { display: none; }
  button {
    margin-top: 16px; padding: 10px 20px; font-size: 1rem; border-radius: 8px;
    border: none; background: #2a7; color: white; cursor: pointer;
  }
  button:disabled { background: #aaa; cursor: default; }
  #preview { max-width: 100%; margin-top: 16px; border-radius: 8px; display: none; }
</style>
</head>
<body>
<h1>Accumulator Checker</h1>
<p>Upload a top-down photo of the 6x6 battery pack. Checks that every cell's terminal
orientation (+/-) alternates consistently with its own column.</p>
<p><em>Per-cell classification is still being calibrated - see the result page's notes
before trusting a FAIL as a certain defect.</em></p>
<form id="form" action="/check" method="post" enctype="multipart/form-data">
  <label class="dropzone" id="dropzone">
    <input type="file" id="file" name="file" accept="image/*">
    <div id="dz-text">Click to choose a photo, or drag one here</div>
    <img id="preview">
  </label>
  <br>
  <button id="submit" type="submit" disabled>Check pack</button>
</form>
<script>
  const fileInput = document.getElementById('file');
  const dropzone = document.getElementById('dropzone');
  const preview = document.getElementById('preview');
  const dzText = document.getElementById('dz-text');
  const submitBtn = document.getElementById('submit');

  function showFile(file) {
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    dzText.textContent = file.name;
    submitBtn.disabled = false;
  }
  fileInput.addEventListener('change', () => showFile(fileInput.files[0]));
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    fileInput.files = e.dataTransfer.files;
    showFile(fileInput.files[0]);
  });
  document.getElementById('form').addEventListener('submit', () => {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Checking...';
  });
</script>
</body>
</html>
"""


def _encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _draw_annotations(image: np.ndarray, result: dict) -> np.ndarray:
    """Circle every detected cell: green + its own sign if it matches its
    column's majority, red + its sign if it doesn't. A cell in
    low_confidence gets a dashed-looking amber outline underneath its
    pass/fail color, since that's the set most likely to be a
    measurement error rather than a real defect (see pack_checker.py)."""
    out = image.copy()
    low_confidence = set(result["low_confidence"])
    for (row, col), info in result["cells"].items():
        x, y, r = result["grid"][(row, col)]
        x, y, r = int(x), int(y), int(r)
        color = (0, 200, 0) if info["pass"] else (0, 0, 220)
        if (row, col) in low_confidence:
            cv2.circle(out, (x, y), int(r * 1.05), (0, 165, 255), 3)
        cv2.circle(out, (x, y), r, color, 4)
        label = info["sign"]
        cv2.putText(out, label, (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.post("/check", response_class=HTMLResponse)
async def check(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    tmp_path = str(UPLOAD_DIR / f"{timestamp}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(contents)

    result = check_pack(tmp_path)

    if result["pass"] is None:
        body = f"""
        <p class="verdict fail">COULD NOT ANALYZE THIS PHOTO</p>
        <p>{result.get("error", "Unknown error.")}</p>
        <p>Make sure the photo shows the full 6x6 grid from directly above, with all 36
        cell tops visible and reasonably well lit.</p>
        """
        css_class = "fail"
    else:
        image = cv2.imread(tmp_path)
        annotated = _draw_annotations(image, result)
        img_b64 = _encode_image(annotated)

        verdict = "PASS" if result["pass"] else "CHECK PACK"
        css_class = "pass" if result["pass"] else "fail"

        mismatch_rows = "".join(
            f"<li>row {r}, col {c} - detected {result['cells'][(r,c)]['sign']}, "
            f"expected {result['cells'][(r,c)]['expected']} (this column's majority)</li>"
            for (r, c) in result["mismatches"]
        )
        low_conf_rows = "".join(f"<li>row {r}, col {c}</li>" for (r, c) in result["low_confidence"])

        body = f"""
        <p class="verdict {css_class}">{verdict}</p>
        <img src="data:image/jpeg;base64,{img_b64}">
        <table>
          <tr><td>Total detected "+"</td><td>{result['plus_count']}</td></tr>
          <tr><td>Total detected "-"</td><td>{result['minus_count']}</td></tr>
          <tr><td>Cells disagreeing with their column</td><td>{len(result['mismatches'])}</td></tr>
        </table>
        {"<p><strong>Mismatched cells:</strong></p><ul>" + mismatch_rows + "</ul>" if mismatch_rows else ""}
        {"<p><em>Low-confidence cells (amber outline above) - measurement is uncertain here, not necessarily wrong:</em></p><ul>" + low_conf_rows + "</ul>" if low_conf_rows else ""}
        <p><em>Note: per-cell +/- classification is still being calibrated (see pack_checker.py) -
        even known-good packs currently show a few false mismatches. Treat CHECK PACK as
        "worth a second look", not a certain defect.</em></p>
        """

    return f"""
    <!doctype html>
    <html>
    <head>
    <title>Accumulator Checker - Result</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #222; }}
      a {{ color: #2a7; }}
      .verdict {{ font-size: 1.5rem; font-weight: bold; padding: 14px 18px; border-radius: 10px; margin-bottom: 12px; }}
      .verdict.pass {{ color: #2a7; background: #f3fbf6; border: 2px solid #2a7; }}
      .verdict.fail {{ color: #d33; background: #fdf3f3; border: 2px solid #d33; }}
      img {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
      ul {{ line-height: 1.6; }}
    </style>
    </head>
    <body>
    <p><a href="/">&larr; upload another photo</a></p>
    {body}
    </body>
    </html>
    """
