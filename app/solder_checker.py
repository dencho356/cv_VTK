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
from app.solder_quality import assess_joint_quality
from app.wire_checker import TOP_MARGIN_FRACTION, check_wires

# Half-size (px, in the ORIGINAL photo's own resolution) of the crop
# handed to assess_joint_quality for each located joint - see
# grade_solder_joints. Scaled from that JOINT's own detected blob area
# (not a fixed px count, same reasoning as _scaled_joint_params below)
# so a bigger, closer-up dome gets a bigger crop and a small, far-back
# one gets a small one, rather than either cutting off a big dome or
# mostly-background-padding a small one.
# Kept tight (not padded generously like the *_PX window constants
# above) specifically to avoid pulling in neighboring board material -
# verified 2026-08-24 against a real whole-plate photo that a looser
# 3.5x scale let a joint's crop reach a NEIGHBORING wire/pad, inflating
# assess_joint_quality's own measured size well past this joint's real
# extent and making the too-small-to-grade check underfire.
QUALITY_CROP_SCALE = 1.6
QUALITY_CROP_MIN_HALF_PX = 40

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "solder_patchcore.ckpt"

# Every JOINT_WINDOW_*/detect_solder_blobs size parameter below is scaled
# by (this photo's own wire pitch) / REFERENCE_PITCH_PX rather than used
# as a fixed px count - see count_solder_joints. REFERENCE_PITCH_PX is
# the median wire pitch measured on the original close-up calibration
# photos (dataset/incoming 20260820T113028353581.jpeg and similar,
# 2026-08-20/24: ~105-150px), which is what every *_PX constant below
# was originally tuned against before scaling was added.
REFERENCE_PITCH_PX = 150.0


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


def _scaled_joint_params(pitch_px: float) -> tuple[float, float, float, int, int, int]:
    """Derive this photo's own search-window and blob-detection sizes from
    its wire pitch, instead of the fixed px counts count_solder_joints
    used through 2026-08-24.

    That fixed-px approach was calibrated against close-up photos where
    the board fills most of the frame (pitch ~105-150px) and broke down
    completely on real whole-board phone photos at the same height/angle
    production will actually use (pitch ~18-63px, board ~330-780px wide
    vs ~2600px on the calibration set) - verified 2026-08-24 against 3 such
    photos (dataset/incoming 20260824T105103384502.jpg and siblings): the
    old fixed 41px closing kernel bridged FOUR adjacent real pad domes
    (only ~24-54px apart at that scale) into one 19724px^2 blob, which
    then exceeded max_area_px=9000 and got discarded entirely - starving
    those wire slots of any real candidate and leaving the matcher no
    choice but a stray reflection off the USB-C shell or an IC leg,
    which it then confidently reported as a real joint.

    Returns (outward_px, inward_px, perp_px, close_kernel_px, min_area_px,
    max_area_px) - see count_solder_joints for outward/inward/perp and
    detect_solder_blobs for the rest.
    """
    scale = pitch_px / REFERENCE_PITCH_PX
    outward_px = max(10.0, min(150.0, 90 * scale))
    # inward reach needs a much steeper ratio to pitch than the others -
    # a twisted-pair connector's two wire ends splay apart from their
    # shared joint at very different rates, and its far dome sat ~586-
    # 596px out on a photo whose pitch was only ~105px (~5.6x) - verified
    # against a real photo (20260820T094903929566.jpeg) that a shallower
    # ratio here misses that real, far dome entirely.
    inward_px = max(60.0, min(700.0, 6.0 * pitch_px))
    # also steeper than a flat 1:1 with outward - a twisted pair's real
    # dome can sit ~100px sideways of its own (imprecise, see this
    # module's pre-2026-08-24 history) anchor even at pitch~105px;
    # verified a shallower ratio here misses that same real dome above.
    perp_px = max(15.0, min(150.0, 0.85 * pitch_px))
    close_kernel_px = max(3, min(41, int(round(0.22 * pitch_px)) | 1))
    min_area_px = max(6, min(400, int(0.012 * pitch_px**2)))
    max_area_px = max(200, min(9000, int(0.4 * pitch_px**2)))
    return outward_px, inward_px, perp_px, close_kernel_px, min_area_px, max_area_px


def count_solder_joints(
    image_path: str, harness_type: str = "default", return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Count solder joints and return their pixel locations in the
    ORIGINAL image (not a rectified/resized one).

    Several approaches tried and rejected before this one, each verified
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
       real dome.
    3. Tried requiring the DOMINANT color (not just present) to touch
       each blob - broke because this board's own navy PCB color falls
       inside the loose "blue" wire-color range, so "blue" trivially won
       almost everywhere regardless of what wire was actually there.
    4. (2026-08-20 through 2026-08-24) Detect every dome ONCE across a
       shared UNION of all found wires' own windows, then greedy-match
       dome order to wire order by nearest 2D distance, with a distance-
       outlier check afterward to catch a wire matched to a far, unrelated
       blob when its own dome wasn't detected. This worked well on
       close-up photos, but real whole-board phone photos at production
       height/angle (2026-08-24, see _scaled_joint_params) exposed two
       compounding problems: the shared union window let a photo-scale-
       mismatched fixed detect_solder_blobs kernel bridge adjacent real
       domes into one oversized, discarded blob, AND even after fixing
       the scale, a shared search area still let one wire's window reach
       into a completely unrelated part of the board (a whole-board photo
       has proportionally much more "unrelated board" inside the same
       fixed-px reach than a close-up crop does).

    This instead keeps each wire's search LOCAL to its own window (no
    shared union) and scales every window/kernel/area size from this
    photo's own measured wire pitch (see _scaled_joint_params) rather
    than a fixed px count - so a whole-board phone photo gets
    proportionally small windows/kernels instead of the close-up-
    calibrated ones. A blob is still only ever found because it fell
    inside SOME wire's own window, never a blind board-wide scan, and a
    blob claimed by more than one wire's overlapping window is still
    resolved by the same global nearest-pair-first greedy matching as
    approach 4 (repeatedly pick whichever (wire, dome) pair is closest
    among everything still unmatched) so two wires can't both claim the
    same dome.

    Known remaining limitation, verified against a real, blurrier phone
    photo (20260824T105218491488.jpg): if a wire's OWN dome highlight
    doesn't register at all (too dim, motion blur, or just small at this
    photo's scale) AND an unused, factory-tinned header row happens to
    sit inside that wire's local window, the unused row's own tiny
    highlight can still be matched in the real dome's place. Local
    windowing eliminates false matches to anything far away (a USB-C
    shell, the main IC) but does not fully eliminate a false match to
    something this physically close - narrowing the window further to
    close that gap risks losing real, legitimately spread-out matches
    (see the twisted-pair case in _scaled_joint_params) without more
    real photos at this scale to validate against.

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

    anchors = []
    for r in found_wires:
        wx = r["centroid"][0]
        pt_seam = np.array([[[wx, seam_y_warped]]], dtype=np.float32)
        pt_inward = np.array([[[wx, seam_y_warped + inward_sign * 120]]], dtype=np.float32)
        seam_orig = cv2.perspectiveTransform(pt_seam, inverse_transform)[0][0]
        inward_orig = cv2.perspectiveTransform(pt_inward, inverse_transform)[0][0]
        direction = inward_orig - seam_orig
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-3 else np.array([0.0, 1.0])
        anchors.append((r["slot"], seam_orig, direction))

    # Median gap between adjacent wires' own seam x-positions, in this
    # photo's own original-resolution pixels - the per-photo scale
    # reference every window/kernel/area size is derived from. Median
    # (not mean) so one unusually large gap - e.g. a wire whose neighbor
    # wasn't found - doesn't skew the estimate for every other wire.
    xs = sorted(a[1][0] for a in anchors)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    pitch_px = float(np.median(gaps)) if gaps else REFERENCE_PITCH_PX
    outward_px, inward_px, perp_px, close_kernel_px, min_area_px, max_area_px = _scaled_joint_params(pitch_px)

    # Blobs are detected ONCE, over the union of every wire's own window
    # (not once per wire) - verified against a real photo
    # (20260820T113028353581.jpeg) that running detect_solder_blobs
    # separately per wire is NOT deterministic for a blob two overlapping
    # windows both cover: the same highlight pixels produced a blob in
    # one wire's crop but not a neighboring, differently-bounded crop of
    # the same pixels, because MORPH_CLOSE/OPEN see a different amount of
    # surrounding context depending on exactly where that crop's edge
    # falls. Each wire's own window is still used afterward (see
    # eligible_slots below) to decide which wires a given blob may match
    # - so this keeps the earlier fix's "never even see a connector shell
    # or IC" property, just with consistent detection.
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
        return (result, {"corners": wire_debug["corners"]}) if return_debug else result

    region = image[union_y0:union_y1, union_x0:union_x1]
    blobs = detect_solder_blobs(
        region,
        close_kernel_px=close_kernel_px,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    )
    dome_positions = [(b["centroid"][0] + union_x0, b["centroid"][1] + union_y0, b["area"]) for b in blobs]

    # A (wire, blob) pair only enters matching if the blob actually falls
    # inside THAT wire's own oriented window (projected onto its
    # outward/inward/perp axes) - not just anywhere in the shared union
    # crop above, which exists only to detect blobs consistently in one
    # pass, not to widen any individual wire's own reach.
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

    # Match wires to domes by GLOBAL nearest-pair-first greedy matching
    # (full 2D distance to each wire's own seam anchor, not just x) -
    # repeatedly pick whichever (wire, dome) pair is closest among
    # everything still unmatched, not a fixed per-wire processing order.
    # Nearest-pair-first (rather than "process wire A's closest choice,
    # then wire B's, in some fixed order") is what avoids two wires whose
    # windows both reach the SAME correct blob having whichever is
    # processed first steal it from the other.
    used_slots: set[int] = set()
    used_domes: set[tuple[float, float]] = set()
    joints = []
    for _sq_dist, slot, dome in pairs:
        if slot in used_slots or (dome[0], dome[1]) in used_domes:
            continue
        used_slots.add(slot)
        used_domes.add((dome[0], dome[1]))
        joints.append({"slot": slot, "x": int(round(dome[0])), "y": int(round(dome[1])), "area": dome[2]})

    # A wire whose OWN dome wasn't detected can still end up matched to
    # some OTHER wire's leftover, weaker candidate once the real domes
    # are claimed - verified against a real photo
    # (20260820T113028353581.jpeg) that two wires' windows both reached
    # the same one real dome, the closer wire correctly won it, and the
    # other wire then matched a genuinely empty pad's much weaker
    # highlight (area 416) rather than being reported as unfound, even
    # though that same photo's other 4 real matches all measured
    # 1352-1611. A real dome and an empty pad's highlight are not the
    # same kind of feature and don't produce comparably-sized blobs, so
    # an match whose area is a clear outlier against this photo's OTHER
    # matches is treated as no match at all, the same way the earlier
    # distance-outlier check (since replaced by local windowing, which
    # covers the far-away case this doesn't) treated an implausible
    # match - not enough matches to judge "outlier" from below 3.
    if len(joints) >= 3:
        areas = sorted(j["area"] for j in joints)
        mid = len(areas) // 2
        median_area = areas[mid] if len(areas) % 2 else (areas[mid - 1] + areas[mid]) / 2
        joints = [j for j in joints if j["area"] >= 0.3 * median_area]

    joints.sort(key=lambda j: j["x"])
    result = {"count": len(joints), "joints": joints}
    if not return_debug:
        return result

    debug = {"corners": wire_debug["corners"]}
    return result, debug


def grade_solder_joints(image_path: str, harness_type: str = "default") -> dict[str, Any]:
    """Locate every joint (count_solder_joints) and grade each one's
    quality (assess_joint_quality, app/solder_quality.py) from a crop of
    the same ORIGINAL photo centered on it - the two-stage plan a real
    inspection needs: first find every joint reliably, THEN judge each
    one's soldering quality, rather than one check trying to do both.

    2026-08-24: assess_joint_quality was calibrated against dedicated
    macro close-ups (a joint filling most of a ~340x340px crop) - it has
    not been validated on a crop pulled from a whole-board photo, where
    a joint is only ~10-50px across (see _scaled_joint_params' own
    real-photo pitch measurements). Every joint still gets graded here
    regardless, but "no_joint_found" or a low-confidence verdict on a
    whole-board photo may mean "not enough pixels to judge", not "bad
    joint" - worth checking the returned crop_bbox against a couple of
    real results before trusting this at that scale.
    """
    result = count_solder_joints(image_path, harness_type=harness_type)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h_img, w_img = image.shape[:2]

    graded_joints = []
    for j in result["joints"]:
        diameter_px = 2 * (j["area"] / np.pi) ** 0.5
        half = max(QUALITY_CROP_MIN_HALF_PX, QUALITY_CROP_SCALE * diameter_px)
        x0, y0 = int(max(0, j["x"] - half)), int(max(0, j["y"] - half))
        x1, y1 = int(min(w_img, j["x"] + half)), int(min(h_img, j["y"] + half))
        crop = image[y0:y1, x0:x1]
        quality = (
            assess_joint_quality(crop)
            if crop.size
            else {"verdict": "no_joint_found", "reasons": ["crop was empty"], "metrics": {}}
        )
        graded_joints.append({**j, "quality": quality, "crop_bbox": (x0, y0, x1, y1)})

    return {"count": result["count"], "joints": graded_joints}
