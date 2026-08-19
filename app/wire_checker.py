"""Wire color/order checker (Step 4 of the plan).

No machine learning: detects the plate via its 4 mounting grommets, warps
it to an upright view, and searches for each wire slot's own expected
color independently within the resulting strip - see
app/plate_detector.py for the detection pipeline itself, which is shared
unchanged with the live webcam tool (scripts/webcam_live.py). This module
is the phase-1 (static photo) entry point into that same pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.plate_detector import (
    evaluate_wire_results,
    find_plate_corners,
    find_plate_corners_by_color,
    find_plate_corners_by_grommets,
    find_wire_slots_by_color,
    is_monotonic,
    rectify_plate,
)

SPEC_PATH = Path(__file__).resolve().parent.parent / "spec.json"

BOTTOM_MARGIN_FRACTION = 0.18
TOP_MARGIN_FRACTION = 0.4


def load_spec() -> dict[str, Any]:
    with open(SPEC_PATH) as f:
        return json.load(f)


def classify_hsv_color(hsv_crop: np.ndarray, color_ranges: dict[str, Any]) -> tuple[str, float]:
    """Return (best_color_name, fraction_of_pixels_matched) for a cropped HSV region."""
    total_pixels = hsv_crop.shape[0] * hsv_crop.shape[1]
    best_color, best_fraction = "unknown", 0.0

    for color_name, ranges in color_ranges.items():
        if color_name.startswith("_"):
            continue
        mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
        fraction = float(np.count_nonzero(mask)) / total_pixels if total_pixels else 0.0
        if fraction > best_fraction:
            best_color, best_fraction = color_name, fraction

    return best_color, best_fraction


def _find_plate(image: np.ndarray, spec: dict[str, Any], harness_type: str) -> np.ndarray | None:
    grommet_range = spec.get("grommet_color_range")
    if grommet_range:
        corners = find_plate_corners_by_grommets(image, grommet_range["lower"], grommet_range["upper"])
        if corners is not None:
            return corners

    plate_color = spec.get("plate_color_ranges", {}).get(harness_type) or spec.get("plate_color_ranges", {}).get(
        "default"
    )
    if plate_color:
        corners = find_plate_corners_by_color(image, plate_color["lower"], plate_color["upper"])
        if corners is not None:
            return corners

    return find_plate_corners(image)


def check_wires(
    image_path: str, harness_type: str = "default", return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """return_debug=True additionally returns a dict of intermediate
    artifacts (corners, the rectified image, the strip's offset within it,
    the raw per-slot centroids) so a caller can draw exactly what was
    detected - e.g. for a web UI showing the result visually, not just as
    numbers."""
    spec = load_spec()
    harness = spec["harness_types"].get(harness_type)
    if harness is None:
        raise ValueError(f"Unknown harness_type: {harness_type!r}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    corners = _find_plate(image, spec, harness_type)
    if corners is None:
        result = {
            "pass": None,
            "confidence": 0.0,
            "wires": [],
            "error": "Could not detect the plate in this image.",
        }
        return (result, {"corners": None}) if return_debug else result

    strip_position = harness.get("strip_position", "bottom")
    warped, _, top_margin_px, board_px, _bottom_margin_px = rectify_plate(
        image,
        corners,
        board_px=500,
        top_margin_fraction=TOP_MARGIN_FRACTION,
        bottom_margin_fraction=BOTTOM_MARGIN_FRACTION,
    )
    strip = warped[:top_margin_px, :] if strip_position == "top" else warped[top_margin_px + board_px :, :]

    plate_color = spec.get("plate_color_ranges", {}).get(harness_type) or spec.get("plate_color_ranges", {}).get(
        "default"
    )
    exclude_range = (plate_color["lower"], plate_color["upper"]) if plate_color else None
    # The board seam is at the opposite edge of the strip from strip_position:
    # a "bottom" strip has the board right above it (seam at local y=0, "top"),
    # a "top" strip has the board right below it (seam at the far edge, "bottom").
    exclude_edge = "bottom" if strip_position == "top" else "top"

    wire_slots = harness["wire_slots"]
    check_order = harness.get("check_order", True)
    require_solder_reach = harness.get("require_solder_reach", False)
    color_ranges = spec["hsv_color_ranges"]

    results = find_wire_slots_by_color(
        strip,
        wire_slots,
        color_ranges,
        exclude_hsv_range=exclude_range,
        require_solder_reach=require_solder_reach,
        exclude_near_edge=exclude_edge,
    )
    overall_pass = evaluate_wire_results(results, check_order)

    found_centroids_x = [r["centroid"][0] for r in results if r["found"]]
    in_order = is_monotonic(found_centroids_x)

    per_wire_results = []
    for r in results:
        # A wire's own pass/fail reflects whether it was found matching its
        # own expected (or alt) color - find_wire_slots_by_color never sets
        # found=True any other way, so this alone is the right condition.
        # Order is a harness-level property (see overall_pass above), not
        # something one specific wire can individually be blamed for - a
        # single missing/misordered wire was previously failing every
        # other correctly-detected wire's row too, which was confusing.
        wire_pass = r["found"]
        per_wire_results.append(
            {
                "slot": r["slot"],
                "pad_label": r["pad_label"],
                "expected_color": r["expected_color"],
                "detected_color": r["matched_color"] if r["found"] else "none",
                "confidence": round(min(1.0, r["pixel_count"] / 5000), 3) if r["found"] else 0.0,
                "pass": wire_pass,
            }
        )

    overall_confidence = (
        sum(w["confidence"] for w in per_wire_results) / len(per_wire_results) if per_wire_results else 0.0
    )

    result = {
        "pass": overall_pass,
        "confidence": round(overall_confidence, 3),
        "wires": per_wire_results,
    }
    if not return_debug:
        return result

    strip_offset_y = 0 if strip_position == "top" else top_margin_px + board_px
    debug = {
        "corners": corners,
        "warped": warped,
        "strip_offset_y": strip_offset_y,
        "raw_results": results,
        "in_order": in_order,
        "check_order": check_order,
    }
    return result, debug
