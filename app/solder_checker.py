"""Solder joint quality checker (Step 5 of the plan).

check_solder (per-joint good/bad quality) is a placeholder until enough
good-joint images exist to train an anomaly detector (recommended:
PatchCore via Anomalib, see spec.json thresholds). Kept structurally
identical to what the trained version will return, so main.py never needs
to change when the real model is dropped in.

count_solder_joints is a separate, already-working capability added
2026-08-20: not joint quality, just "how many joints are there and where"
- see detect_solder_blobs in app/plate_detector.py for the technique
(specular highlight, not solder color) and its real-photo validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.plate_detector import detect_solder_blobs
from app.wire_checker import TOP_MARGIN_FRACTION, check_wires

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "solder_patchcore.ckpt"

# Half-size (px, in the ORIGINAL photo's own resolution) of the search
# window placed around each found wire's own seam point - see
# count_solder_joints. Swept against a real photo (dataset/incoming
# 20260820T094903929566.jpeg, 2026-08-20): real joints sat up to ~590px
# inward of the detected seam there, well past an initial 260px guess.
JOINT_WINDOW_OUTWARD_PX = 90
JOINT_WINDOW_INWARD_PX = 650
JOINT_WINDOW_PERP_PX = 110


def check_solder(image_path: str) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "pass": None,
            "confidence": 0.0,
            "status": "untrained",
            "defects": [],
            "note": (
                "No trained solder model yet. Collect 100-300 images of good "
                "joints (Step 3) and train PatchCore via Anomalib before this "
                "check can return a real result."
            ),
        }

    raise NotImplementedError("Trained-model inference path not implemented yet.")


def count_solder_joints(
    image_path: str, harness_type: str = "default", return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Count solder joints and return their pixel locations in the
    ORIGINAL image (not a rectified/resized one).

    Three approaches tried and rejected before this one, each verified
    against real photos:
    1. Scan a crop of "the margin near the board edge" for specular
       highlights (see detect_solder_blobs), keep anything with any
       wire-colored pixel nearby. Worked on one close-up photo, found 40+
       false hits (screws, IC legs, QR code flecks) on a wider photo of
       the whole board.
    2. Anchor a small search window to each wire's OWN detected position
       (mapped from its color-blob centroid in the small rectified strip
       through the inverse perspective transform), require that wire's
       own color to touch the candidate blob. Fixed the false-positive
       problem, but the anchor itself turned out to be imprecise for
       wires in a tightly-twisted bundle - traced one failure all the way
       to its root cause: the wire's centroid in the small rectified
       strip was measurably off (color blobs fragment/overlap there for
       twisted wires), so the window it built never even contained the
       real dome. No amount of downstream color-matching logic can fix a
       search window that's looking in the wrong place.
    3. Tried requiring the DOMINANT color (not just present) to touch
       each blob - broke because this board's own navy PCB color falls
       inside the loose "blue" wire-color range, so "blue" trivially won
       almost everywhere regardless of what wire was actually there.

    This instead treats it as a COUNTING and ORDERING problem, not a
    per-wire anchoring problem: detect every real dome ONCE across the
    combined area spanning all the found wires (the union of what
    approach 2's per-wire windows would have covered, so still anchored
    to real wire detections and not a blind whole-board scan), then match
    dome order to wire order left-to-right - the same "position, not
    pixel-perfect anchoring" logic the wire-color check itself already
    relies on (find_wire_slots_by_color assigns same-colored blobs to
    slots by left-to-right order, not by an assumed exact x). A precise
    per-wire anchor was the fragile part throughout every failed
    attempt; how many wires were found and their relative order is not.

    Needs the same close-up-enough, well-lit, in-focus photo detect_
    solder_blobs does - the specular highlight a dome needs only survives
    at real resolution, which is why this re-reads the original image
    rather than reusing check_wires' small rectified strip.
    """
    wire_result, wire_debug = check_wires(image_path, harness_type=harness_type, return_debug=True)
    if wire_debug.get("corners") is None:
        result = {"count": 0, "joints": [], "error": wire_result.get("error", "Could not detect the plate.")}
        return (result, {"corners": None}) if return_debug else result

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h_img, w_img = image.shape[:2]

    inverse_transform = wire_debug["inverse_transform"]
    strip_offset_y = wire_debug["strip_offset_y"]
    strip_position = "top" if strip_offset_y == 0 else "bottom"
    board_px = 500
    top_margin_px = int(board_px * TOP_MARGIN_FRACTION)
    # The seam (board edge the wires cross) is the strip's edge nearest
    # the board, not the wire's own detected y - that's somewhere along
    # its free-hanging length, not at the joint.
    seam_y_warped = (top_margin_px - 1) if strip_position == "top" else strip_offset_y
    inward_sign = 1 if strip_position == "top" else -1

    found_wires = [r for r in wire_debug["raw_results"] if r["found"]]
    if not found_wires:
        result = {"count": 0, "joints": []}
        return (result, {"corners": wire_debug["corners"]}) if return_debug else result

    # Union of every found wire's own seam-anchored window - each wire's
    # individual anchor point is only used to bound the SHARED search
    # area (forgiving: even an imprecise anchor is still in the general
    # vicinity), not to pick out that wire's own blob directly, which is
    # what made the earlier per-wire approach fragile.
    union_x0, union_y0 = w_img, h_img
    union_x1, union_y1 = 0, 0
    wire_anchors = []
    for r in found_wires:
        wx = r["centroid"][0]
        pt_seam = np.array([[[wx, seam_y_warped]]], dtype=np.float32)
        pt_inward = np.array([[[wx, seam_y_warped + inward_sign * 120]]], dtype=np.float32)
        seam_orig = cv2.perspectiveTransform(pt_seam, inverse_transform)[0][0]
        inward_orig = cv2.perspectiveTransform(pt_inward, inverse_transform)[0][0]
        direction = inward_orig - seam_orig
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-3 else np.array([0.0, 1.0])
        perp = np.array([-direction[1], direction[0]])

        corners_win = np.array(
            [
                seam_orig - direction * JOINT_WINDOW_OUTWARD_PX + perp * JOINT_WINDOW_PERP_PX,
                seam_orig - direction * JOINT_WINDOW_OUTWARD_PX - perp * JOINT_WINDOW_PERP_PX,
                seam_orig + direction * JOINT_WINDOW_INWARD_PX + perp * JOINT_WINDOW_PERP_PX,
                seam_orig + direction * JOINT_WINDOW_INWARD_PX - perp * JOINT_WINDOW_PERP_PX,
            ]
        )
        wx0, wy0, ww, wh = cv2.boundingRect(corners_win.astype(np.int32))
        union_x0, union_y0 = min(union_x0, wx0), min(union_y0, wy0)
        union_x1, union_y1 = max(union_x1, wx0 + ww), max(union_y1, wy0 + wh)
        wire_anchors.append((r["slot"], seam_orig[0], seam_orig[1]))

    # Padded past the exact per-wire windows' extent - verified against a
    # real photo that a real dome sitting right at one window's edge got
    # cropped by the union boundary itself, truncating its width and
    # producing a falsely tall/thin bbox that the aspect-ratio filter in
    # detect_solder_blobs then wrongly rejected as a sliver artifact.
    union_pad_px = 80
    union_x0, union_y0 = max(0, union_x0 - union_pad_px), max(0, union_y0 - union_pad_px)
    union_x1, union_y1 = min(w_img, union_x1 + union_pad_px), min(h_img, union_y1 + union_pad_px)
    if union_x1 <= union_x0 or union_y1 <= union_y0:
        result = {"count": 0, "joints": []}
        return (result, {"corners": wire_debug["corners"]}) if return_debug else result

    region = image[union_y0:union_y1, union_x0:union_x1]
    blobs = detect_solder_blobs(region)
    dome_positions = [(b["centroid"][0] + union_x0, b["centroid"][1] + union_y0, b["area"]) for b in blobs]

    # Match wires to domes by GLOBAL nearest-pair-first greedy matching
    # (full 2D distance to each wire's own seam anchor, not just x) -
    # repeatedly pick whichever (wire, dome) pair is closest among
    # everything still unmatched, not a fixed per-wire processing order.
    # x-only distance was tried first and broke on a photo where one
    # wire's anchor (imprecise - see the module docstring) happened to
    # sit near-ish in x to an entirely different, unrelated row of pads
    # further up the board; nothing about x-only distance penalizes that
    # huge y mismatch. Nearest-pair-first (rather than "process wire A's
    # closest choice, then wire B's, in some fixed order") is also what
    # avoids the earlier bug where two wires independently rank the SAME
    # correct blob as their own top choice and whichever is processed
    # first steals it from the other.
    pairs = []
    for slot, wx, wy in wire_anchors:
        for dx, dy, area in dome_positions:
            pairs.append((((wx - dx) ** 2 + (wy - dy) ** 2), slot, (dx, dy, area)))
    pairs.sort(key=lambda p: p[0])

    used_slots: set[int] = set()
    used_domes: set[tuple[int, int]] = set()
    joints = []
    for _dist, slot, dome in pairs:
        if slot in used_slots or (dome[0], dome[1]) in used_domes:
            continue
        used_slots.add(slot)
        used_domes.add((dome[0], dome[1]))
        joints.append({"slot": slot, "x": dome[0], "y": dome[1], "area": dome[2]})

    joints.sort(key=lambda j: j["x"])
    result = {"count": len(joints), "joints": joints}
    if not return_debug:
        return result

    debug = {"corners": wire_debug["corners"]}
    return result, debug
