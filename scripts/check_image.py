"""Check wire color/order against a single static photo instead of a live
camera - much faster to iterate with than a shaky handheld feed, and this
is what actually validated the core detection logic (see
app/plate_detector.find_wire_slots_by_color) against a real reference
photo before trusting it on live video.

Searches a region of the image (drag-select interactively, or pass
--region) for each wire slot's own expected color independently, rather
than assuming the wires sit at evenly-spaced positions - verified that
assumption is wrong for real, bendable hookup wire.

Usage:
  python scripts/check_image.py --image path/to/photo.jpg
      Drag a rectangle around the wires, release to confirm, 'r' to redo,
      's' to run the check, 'q' to quit.

  python scripts/check_image.py --image path/to/photo.jpg --region x,y,w,h
      Skip the interactive step (region in original-image pixel coords).

Either way, an annotated copy is saved next to the source image
(<name>_checked.jpg) so results can be reviewed without a live GUI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.plate_detector import evaluate_wire_results, find_wire_slots_by_color, is_monotonic
from app.wire_checker import load_spec


def select_region_interactively(image) -> tuple[int, int, int, int]:
    window = "Check Image - drag a box around the wires, 's' to confirm, 'r' to redo, 'q' to quit"
    cv2.namedWindow(window)
    state = {"start": None, "end": None, "dragging": False}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"], state["end"], state["dragging"] = (x, y), (x, y), True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"], state["dragging"] = (x, y), False

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = image.copy()
        if state["start"] and state["end"]:
            cv2.rectangle(display, state["start"], state["end"], (0, 255, 255), 3)
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            state["start"], state["end"] = None, None
        elif key == ord("s") and state["start"] and state["end"]:
            cv2.destroyWindow(window)
            (x0, y0), (x1, y1) = state["start"], state["end"]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            return x, y, w, h
        elif key == ord("q"):
            cv2.destroyWindow(window)
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check wire color/order against a static photo.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--harness-type", default="default")
    parser.add_argument("--region", help="x,y,w,h in original-image pixel coords; skips the interactive step.")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    spec = load_spec()
    harness = spec["harness_types"].get(args.harness_type)
    if harness is None:
        raise ValueError(f"Unknown harness_type: {args.harness_type!r}")
    wire_slots = harness["wire_slots"]
    check_order = harness.get("check_order", True)
    require_solder_reach = harness.get("require_solder_reach", False)
    color_ranges = spec["hsv_color_ranges"]
    plate_color = spec.get("plate_color_ranges", {}).get(args.harness_type)
    exclude_range = (plate_color["lower"], plate_color["upper"]) if plate_color else None

    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
    else:
        x, y, w, h = select_region_interactively(image)

    region = image[y : y + h, x : x + w]
    results = find_wire_slots_by_color(
        region, wire_slots, color_ranges, exclude_hsv_range=exclude_range, require_solder_reach=require_solder_reach
    )
    all_pass = evaluate_wire_results(results, check_order)

    centroids_x = [r["centroid"][0] for r in results if r["found"]]
    in_order = is_monotonic(centroids_x)

    print(f"Region: x={x} y={y} w={w} h={h}")
    print(f"Order check: {'on' if check_order else 'off (twisted-pair harness - color/count only)'}")
    print(f"Result: {'GOOD' if all_pass else 'CHECK WIRES'}")
    for r in results:
        if not r["found"]:
            status = "MISSING"
        elif check_order and not in_order:
            status = "ORDER?"
        else:
            status = "PASS"
        print(f"  [{status}] {r['pad_label']}: expected={r['expected_color']} "
              f"{'found at x=' + str(r['centroid'][0]) if r['found'] else 'not found'} "
              f"(pixels={r['pixel_count']})")

    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 3)
    for r in results:
        # This wire's own box is green iff it was individually found/matched -
        # a missing or out-of-order sibling wire doesn't make THIS one wrong.
        color_bgr = (0, 200, 0) if r["found"] else (0, 0, 220)
        if r["found"]:
            cx, cy = r["centroid"]
            cv2.circle(annotated, (x + cx, y + cy), 25, color_bgr, 4)
            label = f"P{r['slot']} {r['expected_color']}"
        else:
            label = f"P{r['slot']} {r['expected_color']} MISSING"
            cx, cy = 10, 40 * r["slot"]
        cv2.putText(annotated, label, (x + cx - 20, y + cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)

    out_path = image_path.with_name(f"{image_path.stem}_checked{image_path.suffix}")
    cv2.imwrite(str(out_path), annotated)
    print(f"Annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
