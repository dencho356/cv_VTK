"""Browser upload UI for the wire checker (Step 6/7 groundwork).

Upload a photo, get back a verdict for every harness defined in
spec.json (currently the primary connector and the power connector),
plus an annotated view of exactly what was detected - same pipeline as
the live webcam tool and check_image.py, just reachable from a browser
instead of a terminal.

Run with:
    uvicorn app.web_ui:app --reload

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from app.solder_checker import count_solder_joints
from app.solder_quality import assess_joint_quality
from app.wire_checker import check_wires, load_spec

app = FastAPI(title="Wire Checker")

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "dataset" / "incoming"

UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
<title>Wire Checker</title>
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
<h1>Wire Checker</h1>
<p>Upload a photo of the plate. Checks every harness defined in spec.json.</p>
<p><a href="/grade-joint">&rarr; Grade a single joint's solder quality instead (close-up photo)</a></p>
<form id="form" action="/check" method="post" enctype="multipart/form-data">
  <label class="dropzone" id="dropzone">
    <input type="file" id="file" name="file" accept="image/*">
    <div id="dz-text">Click to choose a photo, or drag one here</div>
    <img id="preview">
  </label>
  <br>
  <button id="submit" type="submit" disabled>Check wires</button>
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

JOINT_UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
<title>Grade Solder Joint</title>
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
<p><a href="/">&larr; back to wire checker</a></p>
<h1>Grade a Solder Joint</h1>
<p>Upload a CLOSE-UP photo of a single joint - it should fill most of the frame, like a macro shot.
This does not work well on a whole-board photo (not enough pixels per joint to judge quality - use the
wire checker for finding/counting joints on those).</p>
<form id="form" action="/grade-joint" method="post" enctype="multipart/form-data">
  <label class="dropzone" id="dropzone">
    <input type="file" id="file" name="file" accept="image/*">
    <div id="dz-text">Click to choose a photo, or drag one here</div>
    <img id="preview">
  </label>
  <br>
  <button id="submit" type="submit" disabled>Grade joint</button>
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
    submitBtn.textContent = 'Grading...';
  });
</script>
</body>
</html>
"""


def _encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _draw_harness_annotations(debug: dict, wire_slots: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (original_with_outline, warped_with_wire_boxes)."""
    warped = debug["warped"].copy()
    offset_y = debug["strip_offset_y"]
    in_order = debug["in_order"]
    check_order = debug["check_order"]

    for r in debug["raw_results"]:
        # This wire's own box is green iff it was individually found/matched -
        # a missing or out-of-order sibling wire doesn't make THIS one wrong.
        color_bgr = (0, 200, 0) if r["found"] else (0, 0, 220)
        if r["found"]:
            x, y = r["centroid"]
            y += offset_y
            cv2.circle(warped, (x, y), 22, color_bgr, 4)
            order_note = " (order?)" if check_order and not in_order else ""
            label = f"P{r['slot']} {r['expected_color']}{order_note}"
            cv2.putText(warped, label, (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        else:
            label = f"P{r['slot']} {r['expected_color']} MISSING"
            cv2.putText(
                warped, label, (10, 30 + 25 * r["slot"]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2
            )
    return warped


def _draw_solder_blob_annotations(original_image: np.ndarray, joint_result: dict) -> np.ndarray | None:
    """Circle each detected solder joint on the original photo, then crop
    to a padded bounding box around just the found joints - count_solder_
    joints returns coordinates in the original image's own resolution
    (see its docstring for why: each joint is found in a small window
    anchored to its own wire, not one shared rectified/resized crop), so
    circles are drawn directly at those coordinates, no offset math
    needed. Returns None if no joints were found (nothing to usefully
    crop to)."""
    joints = joint_result["joints"]
    if not joints:
        return None

    out = original_image.copy()
    for i, j in enumerate(joints, 1):
        cv2.circle(out, (j["x"], j["y"]), 90, (0, 255, 0), 6)
        cv2.putText(out, str(i), (j["x"] - 15, j["y"] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 3)

    pad = 160
    xs = [j["x"] for j in joints]
    ys = [j["y"] for j in joints]
    h, w = out.shape[:2]
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    return out[y0:y1, x0:x1]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UPLOAD_PAGE


@app.post("/check", response_class=HTMLResponse)
async def check(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    # Saved persistently (not a temp file that gets deleted after the
    # request) into dataset/incoming - the folder the original plan
    # designated for "where a new photo would land". This doubles as the
    # traceability record and means a failure can actually be inspected
    # afterward instead of being lost the moment the response is sent.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    tmp_path = str(UPLOAD_DIR / f"{timestamp}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(contents)

    spec = load_spec()
    sections = []
    original_outline_img: np.ndarray | None = None

    for harness_type, harness in spec["harness_types"].items():
        if not harness.get("wire_slots"):
            continue

        result, debug = check_wires(tmp_path, harness_type=harness_type, return_debug=True)

        if debug.get("corners") is None:
            sections.append(
                f"""
                <div class="harness fail">
                  <h2>{harness_type}</h2>
                  <p class="verdict">Could not detect the plate.</p>
                </div>
                """
            )
            continue

        if original_outline_img is None:
            original_outline_img = cv2.imread(tmp_path)

        corners = debug["corners"]
        cv2.polylines(original_outline_img, [corners.astype(np.int32)], True, (255, 200, 0), 4)

        strip_img = _draw_harness_annotations(debug, harness["wire_slots"])
        strip_b64 = _encode_image(strip_img)

        wire_rows = "".join(
            f"""<tr class="{'pass' if w['pass'] else 'fail'}">
                  <td>{w['pad_label']}</td><td>{w['expected_color']}</td>
                  <td>{w['detected_color']}</td><td>{'PASS' if w['pass'] else 'FAIL'}</td>
                </tr>"""
            for w in result["wires"]
        )
        verdict = "GOOD" if result["pass"] else "CHECK WIRES"
        css_class = "pass" if result["pass"] else "fail"

        # Solder-joint circling: a separate capability from the wire-color
        # check above (detects joints by their specular highlight, not by
        # color - see app/solder_checker.py) - shown as long as the plate
        # was found, independent of whether the wire-color check passed.
        joint_section = ""
        try:
            joint_result = count_solder_joints(tmp_path, harness_type=harness_type)
            joint_img = _draw_solder_blob_annotations(original_outline_img, joint_result)
            if joint_img is not None:
                joint_b64 = _encode_image(joint_img)
                joint_section = f"""
                <p><strong>{joint_result['count']} solder joint(s) detected</strong> (of {len(result['wires'])} wires found)</p>
                <img src="data:image/jpeg;base64,{joint_b64}">
                """
            else:
                joint_section = "<p><em>No solder joints detected near the found wires.</em></p>"
        except Exception as exc:  # noqa: BLE001 - a joint-circling failure shouldn't hide the wire result
            joint_section = f"<p><em>Solder joint detection failed: {exc}</em></p>"

        sections.append(
            f"""
            <div class="harness {css_class}">
              <h2>{harness_type}</h2>
              <p class="verdict">{verdict}</p>
              <img src="data:image/jpeg;base64,{strip_b64}">
              <table>
                <tr><th>Pad</th><th>Expected</th><th>Detected</th><th></th></tr>
                {wire_rows}
              </table>
              {joint_section}
            </div>
            """
        )

    original_b64 = _encode_image(original_outline_img) if original_outline_img is not None else ""

    return f"""
    <!doctype html>
    <html>
    <head>
    <title>Wire Checker - Result</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #222; }}
      a {{ color: #2a7; }}
      .harness {{ border-radius: 10px; padding: 16px; margin: 16px 0; border: 2px solid #ddd; }}
      .harness.pass {{ border-color: #2a7; background: #f3fbf6; }}
      .harness.fail {{ border-color: #d33; background: #fdf3f3; }}
      .verdict {{ font-size: 1.3rem; font-weight: bold; }}
      .harness.pass .verdict {{ color: #2a7; }}
      .harness.fail .verdict {{ color: #d33; }}
      img {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      td, th {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }}
      tr.fail td {{ color: #d33; }}
    </style>
    </head>
    <body>
    <p><a href="/">&larr; upload another photo</a></p>
    {'<img src="data:image/jpeg;base64,' + original_b64 + '">' if original_b64 else ''}
    {''.join(sections)}
    </body>
    </html>
    """


@app.get("/grade-joint", response_class=HTMLResponse)
def grade_joint_form() -> str:
    return JOINT_UPLOAD_PAGE


@app.post("/grade-joint", response_class=HTMLResponse)
async def grade_joint(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    tmp_path = UPLOAD_DIR / f"joint_{timestamp}{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    image = cv2.imread(str(tmp_path))
    if image is None:
        body = "<p class='verdict'>Could not read that image.</p>"
        css_class = "fail"
        verdict_label = "ERROR"
    else:
        result = assess_joint_quality(image)
        verdict = result["verdict"]
        low_confidence = verdict.endswith("_low_confidence")
        base_verdict = verdict.removesuffix("_low_confidence")
        css_class = "warn" if low_confidence else {"good": "pass", "suspect": "fail"}.get(base_verdict, "warn")
        verdict_label = {
            "good": "GOOD SOLDERING",
            "suspect": "SUSPECT - CHECK THIS JOINT",
            "no_joint_found": "NO JOINT FOUND IN PHOTO",
        }.get(base_verdict, base_verdict.upper())
        if low_confidence:
            verdict_label += " (LOW CONFIDENCE - photo may be too far for a reliable verdict)"

        reasons_html = (
            "<ul>" + "".join(f"<li>{r}</li>" for r in result["reasons"]) + "</ul>"
            if result["reasons"]
            else "<p><em>No issues found.</em></p>"
        )
        metrics = result["metrics"]
        metrics_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in metrics.items()
            if k != "center"
        )
        img_b64 = _encode_image(image)
        body = f"""
        <img src="data:image/jpeg;base64,{img_b64}">
        <p class="verdict">{verdict_label}</p>
        {reasons_html}
        {'<table><tr><th>metric</th><th>value</th></tr>' + metrics_rows + '</table>' if metrics else ''}
        """

    return f"""
    <!doctype html>
    <html>
    <head>
    <title>Joint Quality Result</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; color: #222; }}
      a {{ color: #2a7; }}
      .verdict {{ font-size: 1.3rem; font-weight: bold; padding: 12px; border-radius: 8px; }}
      .pass .verdict, body.pass .verdict {{ color: #2a7; background: #f3fbf6; }}
      .fail .verdict, body.fail .verdict {{ color: #d33; background: #fdf3f3; }}
      .warn .verdict, body.warn .verdict {{ color: #b80; background: #fdf8f0; }}
      img {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      td, th {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }}
      ul {{ line-height: 1.6; }}
    </style>
    </head>
    <body class="{css_class}">
    <p><a href="/grade-joint">&larr; grade another joint photo</a></p>
    {body}
    </body>
    </html>
    """
