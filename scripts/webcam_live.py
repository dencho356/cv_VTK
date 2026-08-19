"""Live webcam wire color/order checker.

Hold the plate up to the webcam and get a live PASS/FAIL overlay per wire,
based on the HSV color spec in spec.json. This is a standalone calibration +
demo tool for the wire check only (Step 4) - it does not touch the solder
check or the folder-based phase 1 pipeline.

Default mode is 'auto': it looks for the plate in your hands each frame
(app/plate_detector.py) and samples wire colors near its bottom edge
automatically - no manual clicking needed. First run: hold the plate
centered in the on-screen box and press 'p' to calibrate on its color (a
solid-colored real part is detected far more reliably by color-thresholding
than by tracing its outline, which breaks easily on silkscreen text, solder
highlights, and components). Until calibrated, it falls back to edge-based
detection, which is slower to lock on. If it still struggles, switch to
'manual' mode - the older click-to-calibrate flow with fixed crop boxes
(better once you have a fixed camera rig, per the original plan).

Once the plate is found, it's warped into an upright, un-tilted view (see
app/plate_detector.rectify_plate) and the wire colors it already knows
about (from spec.json) are searched for within that straightened strip,
rather than assumed to be evenly spread across the full board width -
wires rarely span the whole edge, and a hand-held plate is rarely
perfectly level, so both those assumptions were putting sample boxes over
background/skin instead of wire insulation. A second "Rectified" window
shows exactly what's being sampled, top and bottom. The top strip is shown
for visibility only (no pass/fail yet - see spec.json's power_connector
harness, which has no wire_slots defined).

Controls (auto mode):    'p' calibrate plate color, 'm' manual mode, 's' log, 'q' quit
Controls (manual mode):
  during calibration: click each wire slot in order, 'r' to reset clicks,
                       's' to save once all slots are clicked, 'q' to quit
  during live check:  'c' to recalibrate, 'a' switch to auto mode, 'q' quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.plate_detector import (
    evaluate_wire_results,
    find_plate_corners,
    find_plate_corners_by_color,
    find_plate_corners_by_grommets,
    find_wire_slots_by_color,
    is_monotonic,
    locate_color_cluster_x_range,
    rectify_plate,
    sample_plate_color,
)
from app.wire_checker import SPEC_PATH, classify_hsv_color, load_spec

PRINT_INTERVAL_S = 0.5  # how often to print live status to the terminal
LOG_PATH = SPEC_PATH.parent / "logs" / "webcam_results.jsonl"


def ui_scale(frame_width: int) -> tuple[int, float]:
    """(box_half, font_scale) sized relative to the camera's actual
    resolution. A fixed 20px/0.45-scale pair (this tool's first version)
    is essentially invisible on a 3024px-wide phone camera frame - about
    1.3% of the frame width - which is what made manual mode look like it
    wasn't showing pass/fail feedback at all once the preview window was
    scaled down; the boxes and labels were being drawn, just too small to
    register. box_half is baked into the saved crop region at calibration
    time, so this must be applied there, not just at display time."""
    box_half = max(20, int(frame_width * 0.012))
    font_scale = max(0.45, frame_width / 2400)
    return box_half, font_scale


def print_status(prefix: str, all_pass: bool, details: list[dict]) -> None:
    verdict = "GOOD" if all_pass else "CHECK WIRES"
    parts = " | ".join(f"{d['pad_label']} exp:{d['expected']} det:{d['detected']}" for d in details)
    print(f"{prefix}[{verdict}] {parts}")


def log_snapshot(harness_type: str, all_pass: bool, details: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "harness_type": harness_type,
        "result": "pass" if all_pass else "fail",
        "wires": details,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged snapshot to {LOG_PATH}")


def save_crop_regions(harness_type: str, points: list[tuple[int, int]], box_half: int) -> None:
    spec = load_spec()
    regions = [[x - box_half, y - box_half, 2 * box_half, 2 * box_half] for x, y in points]
    spec["wire_crop_regions"][harness_type] = regions
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved {len(regions)} crop regions for harness '{harness_type}' to {SPEC_PATH}")


def run_calibration(cap: cv2.VideoCapture, harness_type: str, num_slots: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    window = "Wire Checker - Calibration"
    cv2.namedWindow(window)

    def on_click(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_slots:
            points.append((x, y))

    cv2.setMouseCallback(window, on_click)

    print(f"Calibration mode: click {num_slots} wire slots in left-to-right order.")
    print("Keys: 'r' reset, 's' save (once all clicked), 'q' quit without saving.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read from webcam.")
        box_half, font_scale = ui_scale(frame.shape[1])

        for i, (x, y) in enumerate(points):
            cv2.rectangle(frame, (x - box_half, y - box_half), (x + box_half, y + box_half), (0, 255, 255), 3)
            cv2.putText(frame, str(i + 1), (x - box_half, y - box_half - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 3)

        status = f"{len(points)}/{num_slots} slots clicked"
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("s") and len(points) == num_slots:
            save_crop_regions(harness_type, points, box_half)
            cv2.destroyWindow(window)
            return points
        elif key == ord("q"):
            cv2.destroyWindow(window)
            sys.exit(0)


def run_live_check(cap: cv2.VideoCapture, harness_type: str) -> None:
    window = "Wire Checker - Live"
    cv2.namedWindow(window)
    last_print = 0.0

    while True:
        spec = load_spec()
        harness = spec["harness_types"][harness_type]
        wire_slots = harness["wire_slots"]
        color_ranges = spec["hsv_color_ranges"]
        regions = spec["wire_crop_regions"][harness_type]

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read from webcam.")
        _, font_scale = ui_scale(frame.shape[1])

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        all_pass = True
        details = []

        for slot, (x, y, w, h) in zip(wire_slots, regions):
            x, y, w, h = max(x, 0), max(y, 0), w, h
            crop = hsv_frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            detected, confidence = classify_hsv_color(crop, color_ranges)
            expected = slot["expected_color"]
            alt = slot.get("alt_color")
            wire_pass = detected in (expected, alt)
            all_pass &= wire_pass
            details.append(
                {
                    "slot": slot["slot"],
                    "pad_label": slot.get("pad_label", slot["slot"]),
                    "expected": expected,
                    "detected": detected,
                    "confidence": round(confidence, 3),
                    "pass": wire_pass,
                }
            )

            color_bgr = (0, 200, 0) if wire_pass else (0, 0, 220)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
            label = f"P{slot['slot']} exp:{expected} det:{detected} ({confidence:.0%})"
            cv2.putText(frame, label, (x, max(y - 12, 16)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2)

        if time.time() - last_print > PRINT_INTERVAL_S:
            print_status("[manual] ", all_pass, details)
            last_print = time.time()

        banner_text = "GOOD" if all_pass else "CHECK WIRES"
        banner_color = (0, 200, 0) if all_pass else (0, 0, 220)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), banner_color, -1)
        cv2.putText(frame, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "'c' recal  'a' auto  's' log  'q' quit", (frame.shape[1] - 290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            return
        elif key == ord("c"):
            cv2.destroyWindow(window)
            points = run_calibration(cap, harness_type, len(wire_slots))
            if not points:
                return
            cv2.namedWindow(window)
        elif key == ord("a"):
            cv2.destroyWindow(window)
            run_auto_check(cap, harness_type)
            cv2.namedWindow(window)
        elif key == ord("s"):
            log_snapshot(harness_type, all_pass, details)


def save_plate_color(harness_type: str, lower: list[float], upper: list[float]) -> None:
    spec = load_spec()
    spec["plate_color_ranges"][harness_type] = {"lower": lower, "upper": upper}
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved plate color range for harness '{harness_type}' to {SPEC_PATH}")


def classify_strip_by_color_search(
    strip: np.ndarray,
    wire_slots: list[dict],
    color_ranges: dict,
    exclude_hsv_range: tuple[list[float], list[float]] | None = None,
    check_order: bool = True,
    require_solder_reach: bool = False,
    exclude_near_edge: str = "top",
) -> tuple[bool, list[dict]]:
    """Find each wire slot's own expected color independently (see
    find_wire_slots_by_color - even spacing between the outermost wires
    was verified wrong against a real photo, since bending wire doesn't
    sit at the positions that predicts), check the found colors' left-to-
    right order against the expected slot order (unless check_order is
    False - not physically meaningful for a twisted-pair connector, see
    evaluate_wire_results), and draw boxes/labels directly onto strip in
    place."""
    results = find_wire_slots_by_color(
        strip,
        wire_slots,
        color_ranges,
        exclude_hsv_range,
        require_solder_reach=require_solder_reach,
        exclude_near_edge=exclude_near_edge,
    )
    overall_pass = evaluate_wire_results(results, check_order)

    found_centroids_x = [r["centroid"][0] for r in results if r["found"]]
    in_order = is_monotonic(found_centroids_x)

    details = []
    box_half = 30
    for r in results:
        # This wire's own pass/fail reflects whether it was individually
        # found/matched - a missing or out-of-order sibling wire doesn't
        # make THIS one wrong. Order is a harness-level property, reflected
        # in overall_pass above and the " ORDER?" label suffix below, not
        # in this wire's own pass flag.
        wire_pass = r["found"]
        detected = r["matched_color"] if r["found"] else "none"
        details.append(
            {
                "slot": r["slot"],
                "pad_label": r["pad_label"],
                "expected": r["expected_color"],
                "detected": detected,
                "confidence": min(1.0, r["pixel_count"] / 5000) if r["found"] else 0.0,
                "pass": wire_pass,
            }
        )

        color_bgr = (0, 200, 0) if wire_pass else (0, 0, 220)
        if r["found"]:
            x, y = r["centroid"]
            x0, y0 = max(x - box_half, 0), max(y - box_half, 0)
            cv2.rectangle(strip, (x0, y0), (x0 + 2 * box_half, y0 + 2 * box_half), color_bgr, 2)
            reason = " ORDER?" if (check_order and not in_order) else ""
            label = f"P{r['slot']} exp:{r['expected_color']} det:{detected}{reason}"
            cv2.putText(strip, label, (x0, max(y0 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1)
        else:
            label = f"P{r['slot']} exp:{r['expected_color']} NOT FOUND"
            cv2.putText(strip, label, (10, 20 + 15 * r["slot"]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1)
    return overall_pass, details


def save_plate_color(harness_type: str, lower: list[float], upper: list[float]) -> None:
    spec = load_spec()
    spec["plate_color_ranges"][harness_type] = {"lower": lower, "upper": upper}
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved plate color range for harness '{harness_type}' to {SPEC_PATH}")


def run_auto_check(
    cap: cv2.VideoCapture,
    harness_type: str,
    wire_offset_fraction: float = 0.18,
    top_harness_type: str | None = "power_connector",
    top_wire_offset_fraction: float | None = None,
) -> None:
    window = "Wire Checker - Auto"
    rect_window = "Wire Checker - Rectified"
    cv2.namedWindow(window)

    spec = load_spec()
    harness = spec["harness_types"][harness_type]
    wire_slots = harness["wire_slots"]
    check_order = harness.get("check_order", True)
    require_solder_reach = harness.get("require_solder_reach", False)
    color_ranges = spec["hsv_color_ranges"]
    num_slots = len(wire_slots)
    plate_color = spec.get("plate_color_ranges", {}).get(harness_type)
    grommet_range = spec.get("grommet_color_range")
    all_known_colors = [c for c in color_ranges if not c.startswith("_")]

    top_harness = spec["harness_types"].get(top_harness_type) if top_harness_type else None
    top_wire_slots = top_harness["wire_slots"] if top_harness else []
    top_check_order = top_harness.get("check_order", True) if top_harness else True
    top_require_solder_reach = top_harness.get("require_solder_reach", False) if top_harness else False

    last_print = 0.0
    last_details: list[dict] = []
    last_all_pass = False
    last_top_details: list[dict] = []
    last_top_pass = False
    reticle_half = 20

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read from webcam.")

        if grommet_range is not None:
            corners = find_plate_corners_by_grommets(frame, grommet_range["lower"], grommet_range["upper"])
        elif plate_color is not None:
            corners = find_plate_corners_by_color(frame, plate_color["lower"], plate_color["upper"])
        else:
            corners = find_plate_corners(frame)

        if corners is None:
            if time.time() - last_print > PRINT_INTERVAL_S:
                print("[auto] no plate detected")
                last_print = time.time()
            last_details = []
            msg = "Show the plate to the camera..."
            if plate_color is None:
                msg += "  (press 'p' with plate centered to calibrate color - much more stable)"
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                cv2.rectangle(
                    frame,
                    (cx - reticle_half, cy - reticle_half),
                    (cx + reticle_half, cy + reticle_half),
                    (0, 165, 255),
                    2,
                )
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        else:
            cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 200, 0), 2)

            warped, _, top_margin_px, board_px, bottom_margin_px = rectify_plate(
                frame,
                corners,
                board_px=500,
                top_margin_fraction=top_wire_offset_fraction if top_wire_offset_fraction is not None else wire_offset_fraction,
                bottom_margin_fraction=wire_offset_fraction,
            )
            bottom_strip = warped[top_margin_px + board_px :, :]
            top_strip = warped[:top_margin_px, :]

            exclude_range = (plate_color["lower"], plate_color["upper"]) if plate_color is not None else None

            last_all_pass, last_details = classify_strip_by_color_search(
                bottom_strip,
                wire_slots,
                color_ranges,
                exclude_hsv_range=exclude_range,
                check_order=check_order,
                require_solder_reach=require_solder_reach,
                exclude_near_edge="top",
            )

            if top_wire_slots:
                last_top_pass, last_top_details = classify_strip_by_color_search(
                    top_strip,
                    top_wire_slots,
                    color_ranges,
                    exclude_hsv_range=exclude_range,
                    check_order=top_check_order,
                    require_solder_reach=top_require_solder_reach,
                    exclude_near_edge="bottom",
                )
                top_banner_color = (0, 200, 0) if last_top_pass else (0, 0, 220)
                cv2.rectangle(top_strip, (0, 0), (top_strip.shape[1], 24), top_banner_color, -1)
                cv2.putText(
                    top_strip, f"{top_harness_type}: {'GOOD' if last_top_pass else 'CHECK WIRES'}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
            else:
                last_top_pass, last_top_details = False, []
                top_range = locate_color_cluster_x_range(
                    top_strip, color_ranges, all_known_colors, exclude_hsv_range=exclude_range, exclude_near_edge="bottom"
                )
                if top_range is not None:
                    tx_min, tx_max, _tx_y = top_range
                    cv2.rectangle(top_strip, (tx_min, 5), (tx_max, top_strip.shape[0] - 5), (0, 200, 255), 2)
                    cv2.putText(
                        top_strip, "unspec'd connector - see spec.json", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1,
                    )
                else:
                    cv2.putText(
                        top_strip, "no wires detected at top", (10, top_strip.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                    )

            if time.time() - last_print > PRINT_INTERVAL_S:
                print_status("[auto-bottom] ", last_all_pass, last_details)
                if top_wire_slots:
                    print_status("[auto-top]    ", last_top_pass, last_top_details)
                last_print = time.time()

            banner_text = "GOOD" if last_all_pass else "CHECK WIRES"
            banner_color = (0, 200, 0) if last_all_pass else (0, 0, 220)
            cv2.rectangle(warped, (0, top_margin_px + board_px), (warped.shape[1], top_margin_px + board_px + 30), banner_color, -1)
            cv2.putText(warped, banner_text, (10, top_margin_px + board_px + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(rect_window, warped)

        cv2.putText(
            frame,
            "'p' calib color  'm' manual  's' log  'q' quit",
            (max(frame.shape[1] - 340, 0), frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            cv2.destroyWindow(rect_window)
            return
        elif key == ord("m"):
            cv2.destroyWindow(window)
            cv2.destroyWindow(rect_window)
            existing_regions = spec["wire_crop_regions"].get(harness_type, [])
            if len(existing_regions) != num_slots:
                run_calibration(cap, harness_type, num_slots)
            run_live_check(cap, harness_type)
            cv2.namedWindow(window)
        elif key == ord("s") and last_details:
            log_snapshot(harness_type, last_all_pass, last_details)
            if top_wire_slots and last_top_details:
                log_snapshot(top_harness_type, last_top_pass, last_top_details)
        elif key == ord("p"):
            lower, upper, median = sample_plate_color(frame)
            if median[1] < 60:
                print(
                    f"WARNING: sampled saturation is low ({median[1]:.0f}) - this often means the box "
                    "wasn't actually centered on the board (background/skin/wall instead). Saved anyway, "
                    "but if detection breaks, that's likely why - recenter the board and press 'p' again."
                )
            save_plate_color(harness_type, lower, upper)
            plate_color = {"lower": lower, "upper": upper}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam wire color/order checker.")
    parser.add_argument("--harness-type", default="default")
    parser.add_argument(
        "--top-harness-type",
        default="power_connector",
        help="Harness checked in the top margin strip. Pass '' to disable and just show an "
        "unclassified color outline there instead.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument(
        "--wire-offset",
        type=float,
        default=0.18,
        help="Auto mode only: size of the bottom margin strip searched for wire colors, "
        "as a fraction of the plate's own height (e.g. 0.18 = 18%% of board height). "
        "Increase if the wires aren't fully inside the strip in the Rectified window; decrease "
        "to search a tighter region.",
    )
    parser.add_argument(
        "--top-wire-offset",
        type=float,
        default=0.4,
        help="Auto mode only: same as --wire-offset but for the top margin strip. Defaults larger "
        "than --wire-offset since a connector exiting upward tends to need more room before it's "
        "occluded (hair, cap, hand) - if the top connector's wires are still mostly outside the "
        "strip in the Rectified window, increase this further.",
    )
    args = parser.parse_args()

    spec = load_spec()
    harness = spec["harness_types"].get(args.harness_type)
    if harness is None:
        raise ValueError(f"Unknown harness_type: {args.harness_type!r}")
    num_slots = len(harness["wire_slots"])

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}.")

    try:
        if args.mode == "auto":
            run_auto_check(cap, args.harness_type, args.wire_offset, args.top_harness_type or None, args.top_wire_offset)
        else:
            existing_regions = spec["wire_crop_regions"].get(args.harness_type, [])
            if len(existing_regions) != num_slots:
                run_calibration(cap, args.harness_type, num_slots)
            run_live_check(cap, args.harness_type)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
