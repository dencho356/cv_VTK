"""Pipeline for a dedicated close-up photo of ONE connector's wires and
solder joints - the two-photo capture setup (an "upper" photo of the
harness whose wires exit upward, a "bottom" photo of the one whose wires
hang down), replacing the single whole-plate photo app/wire_checker.py +
app/solder_checker.py were built around.

Each photo already frames just that harness's wires and solder joints
(plus a sliver of board at the seam, per the user) - so unlike the
single-photo pipeline, there's no plate/grommet detection or perspective
warp here: the photo already IS the "strip" that pipeline had to
construct by rectifying a wider shot (see rectify_plate in
app/plate_detector.py). This also sidesteps app/solder_checker.py's
central complication - needing the ORIGINAL full-board photo's resolution
for detect_solder_blobs' specular-highlight signal while wire-color
matching ran on a separately rectified small strip - since one photo now
serves both checks directly, at whatever resolution it was actually shot
at.

UNVALIDATED against a real two-photo capture yet (no sample photos existed
at time of writing) - see the project's own feedback memory on live-
camera/photo verification: a design that's internally consistent and
reuses already-validated pieces (find_wire_slots_by_color,
count_wire_strands_at_row, detect_solder_blobs, the pitch-scaled joint
windows from app/solder_checker.py) is not the same as a design confirmed
against a real photo from this exact setup. REFERENCE_STRIP_WIDTH_PX below
is a reasoned starting point for scaling the wire-color thresholds, not a
calibrated one - expect to retune once real sample photos exist.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.plate_detector import (
    count_wire_strands_at_row,
    detect_solder_blobs,
    evaluate_wire_results,
    find_wire_slots_by_color,
    is_monotonic,
)
from app.solder_checker import REFERENCE_PITCH_PX, _scaled_joint_params
from app.wire_checker import load_spec

# find_wire_slots_by_color/count_wire_strands_at_row's default thresholds
# (min_pixel_count=60, min_run_px=12, gap_merge_px=6) were tuned against
# the ~500px-wide rectified strip app/wire_checker.py builds via
# rectify_plate. A dedicated close-up photo is shot at whatever native
# resolution the camera captures - almost certainly much wider - so using
# those same absolute pixel counts unscaled would accept tiny noise as a
# real wire blob. Scaled by this photo's own width against that same
# 500px reference, the same reasoning app/solder_checker.py uses to scale
# its own joint-detection windows by measured wire pitch instead of a
# fixed px count.
REFERENCE_STRIP_WIDTH_PX = 500.0


def _seam_edge(strip_position: str) -> str:
    """Which edge of the PHOTO the board sliver sits at, for
    exclude_near_edge - same convention app/wire_checker.check_wires uses
    for its own rectified strips: strip_position "top" (wires exit upward
    from the board, e.g. power_connector) means the board sits at the
    strip's own BOTTOM edge; strip_position "bottom" (wires hang down,
    e.g. default) means the board sits at the strip's own TOP edge."""
    return "bottom" if strip_position == "top" else "top"


def check_harness_photo(
    image_path: str, harness_type: str, return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Wire color/order + physical wire count for ONE dedicated harness
    photo - the two-photo-setup counterpart to
    app/wire_checker.check_wires, minus the plate-detection/rectification
    step that function needs when working from a single whole-plate
    photo.
    """
    spec = load_spec()
    harness = spec["harness_types"].get(harness_type)
    if harness is None:
        raise ValueError(f"Unknown harness_type: {harness_type!r}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    strip_position = harness.get("strip_position", "bottom")
    exclude_edge = _seam_edge(strip_position)
    scale = image.shape[1] / REFERENCE_STRIP_WIDTH_PX

    plate_color = spec.get("plate_color_ranges", {}).get(harness_type) or spec.get(
        "plate_color_ranges", {}
    ).get("default")
    exclude_range = (plate_color["lower"], plate_color["upper"]) if plate_color else None

    wire_slots = harness["wire_slots"]
    check_order = harness.get("check_order", True)
    require_solder_reach = harness.get("require_solder_reach", False)
    color_ranges = spec["hsv_color_ranges"]

    results = find_wire_slots_by_color(
        image,
        wire_slots,
        color_ranges,
        exclude_hsv_range=exclude_range,
        require_solder_reach=require_solder_reach,
        exclude_near_edge=exclude_edge,
        min_pixel_count=max(10, int(round(60 * scale**2))),
    )
    overall_pass = evaluate_wire_results(results, check_order)

    found_centroids_x = [r["centroid"][0] for r in results if r["found"]]
    in_order = is_monotonic(found_centroids_x)

    background_color = spec.get("background_color_ranges", {}).get(harness_type) or spec.get(
        "background_color_ranges", {}
    ).get("default")
    background_range = (background_color["lower"], background_color["upper"]) if background_color else None
    strand_runs = count_wire_strands_at_row(
        image,
        background_hsv_range=background_range,
        exclude_hsv_range=exclude_range,
        exclude_near_edge=exclude_edge,
        min_run_px=max(3, int(round(12 * scale))),
        gap_merge_px=max(2, int(round(6 * scale))),
    )
    expected_wire_count = harness.get("expected_wire_strand_count", len(wire_slots))
    found_wire_count = len(strand_runs)
    wire_count_match = found_wire_count == expected_wire_count
    check_wire_count = harness.get("check_wire_count", False)
    if check_wire_count and not wire_count_match:
        overall_pass = False

    per_wire_results = []
    for r in results:
        per_wire_results.append(
            {
                "slot": r["slot"],
                "pad_label": r["pad_label"],
                "expected_color": r["expected_color"],
                "detected_color": r["matched_color"] if r["found"] else "none",
                # Same confidence formula as check_wires (pixel_count / 5000),
                # scaled by area (scale**2) since a higher-res close-up photo
                # naturally produces far more matched pixels per wire than the
                # 500px-wide reference strip that constant was set against.
                "confidence": round(min(1.0, r["pixel_count"] / (5000 * scale**2)), 3) if r["found"] else 0.0,
                "pass": r["found"],
            }
        )
    overall_confidence = (
        sum(w["confidence"] for w in per_wire_results) / len(per_wire_results) if per_wire_results else 0.0
    )

    result = {
        "pass": overall_pass,
        "confidence": round(overall_confidence, 3),
        "wires": per_wire_results,
        "wire_count": {
            "expected": expected_wire_count,
            "found": found_wire_count,
            "match": wire_count_match,
            "enforced": check_wire_count,
        },
    }
    if not return_debug:
        return result

    debug = {
        "raw_results": results,
        "in_order": in_order,
        "check_order": check_order,
        "strand_runs": strand_runs,
        "exclude_edge": exclude_edge,
        "image_shape": image.shape,
    }
    return result, debug


def count_harness_solder_joints(
    image_path: str, harness_type: str, return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Solder joint count/location for ONE dedicated harness photo - the
    two-photo-setup counterpart to app/solder_checker.count_solder_joints,
    minus the inverse-perspective-transform step that function needs to
    anchor windows back into a whole-plate photo from a separately
    rectified strip. Here each wire's centroid from check_harness_photo is
    already in this photo's own pixel coordinates, so its search window is
    anchored directly at (wire_x, seam_y) with a straight up/down search
    direction from the photo's own seam edge (see _seam_edge) - no
    perspective transform needed since this is a dedicated, roughly
    fronto-parallel close-up, not a warped region of a wider shot.

    Keeps count_solder_joints' two hard-won properties: windows are pitch-
    scaled per photo (_scaled_joint_params) rather than fixed-px, and
    blobs are detected ONCE over the union of every wire's own window then
    matched by global nearest-pair-first greedy matching - not detected
    independently per wire, which that function's own docstring (approach
    2) found non-deterministic for a blob two overlapping windows both
    cover, since morphological closing/opening see different surrounding
    context depending on exactly where a crop's edge falls.
    """
    wire_result, wire_debug = check_harness_photo(image_path, harness_type, return_debug=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h_img, w_img = image.shape[:2]

    found_wires = [r for r in wire_debug["raw_results"] if r["found"]]
    if not found_wires:
        result = {"count": 0, "joints": []}
        return (result, {}) if return_debug else result

    exclude_edge = wire_debug["exclude_edge"]
    seam_y = 0.0 if exclude_edge == "top" else float(h_img - 1)
    inward_dir = np.array([0.0, 1.0]) if exclude_edge == "top" else np.array([0.0, -1.0])

    anchors = [
        (r["slot"], np.array([r["centroid"][0], seam_y], dtype=np.float32), inward_dir) for r in found_wires
    ]

    # Median gap between adjacent wires' own seam x-positions, in this
    # photo's own pixels - the per-photo scale reference every
    # window/kernel/area size is derived from (see _scaled_joint_params).
    xs = sorted(a[1][0] for a in anchors)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    pitch_px = float(np.median(gaps)) if gaps else REFERENCE_PITCH_PX
    outward_px, inward_px, perp_px, close_kernel_px, min_area_px, max_area_px = _scaled_joint_params(pitch_px)

    union_x0, union_y0 = w_img, h_img
    union_x1, union_y1 = 0, 0
    for _slot, seam_orig, direction in anchors:
        perp_v = np.array([-direction[1], direction[0]])
        corners_win = np.array(
            [
                seam_orig - direction * outward_px + perp_v * perp_px,
                seam_orig - direction * outward_px - perp_v * perp_px,
                seam_orig + direction * inward_px + perp_v * perp_px,
                seam_orig + direction * inward_px - perp_v * perp_px,
            ]
        )
        wx0, wy0, ww, wh = cv2.boundingRect(corners_win.astype(np.int32))
        union_x0, union_y0 = min(union_x0, wx0), min(union_y0, wy0)
        union_x1, union_y1 = max(union_x1, wx0 + ww), max(union_y1, wy0 + wh)
    union_x0, union_y0 = max(0, union_x0), max(0, union_y0)
    union_x1, union_y1 = min(w_img, union_x1), min(h_img, union_y1)
    if union_x1 <= union_x0 or union_y1 <= union_y0:
        result = {"count": 0, "joints": []}
        return (result, {}) if return_debug else result

    region = image[union_y0:union_y1, union_x0:union_x1]
    blobs = detect_solder_blobs(
        region, close_kernel_px=close_kernel_px, min_area_px=min_area_px, max_area_px=max_area_px
    )
    dome_positions = [(b["centroid"][0] + union_x0, b["centroid"][1] + union_y0, b["area"]) for b in blobs]

    pairs = []
    for slot, seam_orig, direction in anchors:
        perp_v = np.array([-direction[1], direction[0]])
        for dx, dy, area in dome_positions:
            rel = np.array([dx, dy]) - seam_orig
            along = float(np.dot(rel, direction))
            across = float(np.dot(rel, perp_v))
            if -outward_px <= along <= inward_px and abs(across) <= perp_px:
                pairs.append(((seam_orig[0] - dx) ** 2 + (seam_orig[1] - dy) ** 2, slot, (dx, dy, area)))
    pairs.sort(key=lambda p: p[0])

    used_slots: set[int] = set()
    used_domes: set[tuple[float, float]] = set()
    joints = []
    for _sq_dist, slot, dome in pairs:
        if slot in used_slots or (dome[0], dome[1]) in used_domes:
            continue
        used_slots.add(slot)
        used_domes.add((dome[0], dome[1]))
        joints.append({"slot": slot, "x": int(round(dome[0])), "y": int(round(dome[1])), "area": dome[2]})

    if len(joints) >= 3:
        areas = sorted(j["area"] for j in joints)
        mid = len(areas) // 2
        median_area = areas[mid] if len(areas) % 2 else (areas[mid - 1] + areas[mid]) / 2
        joints = [j for j in joints if j["area"] >= 0.3 * median_area]

    joints.sort(key=lambda j: j["x"])
    result = {"count": len(joints), "joints": joints}
    return (result, {}) if return_debug else result
