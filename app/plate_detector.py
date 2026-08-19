"""Finds a hand-held, roughly-square/rectangular plate in a webcam frame and
derives wire sample points along its bottom edge, so the live checker does
not need manual click-calibration every session.

This trades the plan's original fixed-camera assumption (Step 4: "camera
position will be fixed in production, so wire positions are roughly fixed
too") for a per-frame detection step, since a hand-held plate moves and
rotates. It is a best-effort contour detector, not a trained model - it
needs reasonable contrast between the plate edges and the background
(a plain table or mat behind your hand works much better than clutter).
"""
from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def get_edge_map(frame: np.ndarray) -> np.ndarray:
    """The edge map find_plate_corners hunts for contours in."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)
    return edges


def _largest_rectangle_ish_contour(
    contours: list[np.ndarray],
    frame_area: float,
    min_area_fraction: float,
    max_area_fraction: float,
    min_fill_ratio: float,
    max_aspect_ratio: float,
) -> np.ndarray | None:
    """Shared filtering: fits a rotated bounding rectangle (minAreaRect) to
    each contour rather than requiring approxPolyDP to collapse to exactly 4
    points - a real PCB has rounded corners, mounting holes, a connector
    notch, and pin headers, none of which trace to a clean quadrilateral.
    min_fill_ratio (contour area / rect area) and max_aspect_ratio filter out
    non-rectangular blobs (e.g. a hand) without requiring a perfect outline.
    """
    best_corners, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_fraction * frame_area or area > max_area_fraction * frame_area:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < 1 or rect_h < 1:
            continue

        fill_ratio = area / (rect_w * rect_h)
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
        if fill_ratio < min_fill_ratio or aspect_ratio > max_aspect_ratio:
            continue

        if area > best_area:
            best_area = area
            best_corners = cv2.boxPoints(rect)

    return best_corners


def find_plate_corners(
    frame: np.ndarray,
    min_area_fraction: float = 0.03,
    max_area_fraction: float = 0.9,
    min_fill_ratio: float = 0.5,
    max_aspect_ratio: float = 2.5,
) -> np.ndarray | None:
    """Edge-based detection: return ordered 4 corners of the largest
    rectangle-ish contour in the Canny edge map, or None. Needs one mostly
    unbroken outline around the whole plate - text, solder highlights, and
    components on a busy real part can fragment that outline, making this
    method intermittent. Prefer find_plate_corners_by_color when a plate
    color has been calibrated (see sample_plate_color)."""
    edges = get_edge_map(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    best_corners = _largest_rectangle_ish_contour(
        contours, frame_area, min_area_fraction, max_area_fraction, min_fill_ratio, max_aspect_ratio
    )
    if best_corners is None:
        return None
    return order_points(best_corners)


def sample_plate_color(
    frame: np.ndarray, center: tuple[int, int] | None = None, sample_half: int = 20
) -> tuple[list[float], list[float], list[float]]:
    """Median HSV in a small box (default: frame center) plus a tolerance
    margin, for calibrating find_plate_corners_by_color against this
    specific board under this specific lighting. Returns (lower, upper,
    raw_median) - the caller can sanity-check raw_median (e.g. flag a low
    saturation as "this might be background/skin, not the board") before
    committing it, since a bad sample here silently breaks all detection
    until someone notices and re-diagnoses it."""
    h, w = frame.shape[:2]
    cx, cy = center if center else (w // 2, h // 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[max(cy - sample_half, 0) : cy + sample_half, max(cx - sample_half, 0) : cx + sample_half]
    median = np.median(roi.reshape(-1, 3), axis=0)
    h_med, s_med, v_med = median
    # Tolerance is deliberately tighter than a first pass at this (+-80 on
    # S/V): that was wide enough to also match a grey sweater in the same
    # shot, since a fixed +-80 window barely constrains anything when the
    # sampled saturation/value themselves aren't near the extremes. A
    # saturation floor is added too - most fabric backgrounds read
    # noticeably less saturated than a glossy PCB, so this is the strongest
    # single lever against matching them, without hard-coding an absolute
    # threshold that could reject a legitimately duller board.
    lower = [max(0, h_med - 10), max(0, s_med - 40, s_med * 0.6), max(0, v_med - 50)]
    upper = [min(179, h_med + 10), min(255, s_med + 40), min(255, v_med + 50)]
    return lower, upper, median.tolist()


def find_plate_corners_by_color(
    frame: np.ndarray,
    hsv_lower: list[float],
    hsv_upper: list[float],
    min_area_fraction: float = 0.03,
    max_area_fraction: float = 0.9,
    min_fill_ratio: float = 0.35,
    max_aspect_ratio: float = 2.5,
) -> np.ndarray | None:
    """Color-based detection: threshold for the plate's own (calibrated)
    color, close small gaps (text, solder pads, holes) with morphology, and
    take the largest resulting blob. Far more stable than edge-tracing for a
    part with one dominant, distinctive color, since it needs a filled
    region rather than one continuous outline."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    # The 4 mounting grommets aren't board-colored, so they punch
    # non-board-colored notches right at the board's corners - a 7x7
    # kernel only bridges gaps a few pixels wide (reach ~= iterations *
    # (kernel-1)/2), nowhere near a real grommet's size once the board
    # fills a meaningful chunk of a modern camera frame. That let the
    # detected shape wobble frame to frame depending on how much of each
    # notch happened to close, which is what "the box curves / cuts
    # between the green things" was. A much larger close kernel bridges
    # the whole grommet reliably; open with a small kernel afterward just
    # to clean up thin noise, not to re-open what was just bridged.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    best_corners = _largest_rectangle_ish_contour(
        contours, frame_area, min_area_fraction, max_area_fraction, min_fill_ratio, max_aspect_ratio
    )
    if best_corners is None:
        return None
    return order_points(best_corners)


def find_plate_corners_by_grommets(
    frame: np.ndarray,
    hsv_lower: list[float],
    hsv_upper: list[float],
    min_area_fraction: float = 0.0004,
    max_area_fraction: float = 0.05,
    min_circularity: float = 0.55,
    outward_push_factor: float = 1.3,
) -> np.ndarray | None:
    """Detect the plate by finding its 4 mounting grommets and connecting
    their centers, instead of tracing the board's own body color.

    The board's own color forms an irregular blob once the grommets (not
    board-colored) punch notches out of it right at the corners - every
    fix so far (bigger closing kernel, looser shape thresholds) has been
    compensating for that irregularity, and it still wobbled with real
    hand tremor. A grommet by itself is a small, simple, consistently
    circular blob with no such problem - four clean landmarks are a much
    more stable basis for the plate's corners than one irregular outline.

    Requires at least 4 blobs above min_circularity (4*pi*area/perimeter^2,
    1.0 = a perfect circle) in the expected size range; if more than 4
    qualify, keeps the 4 closest in size to the group's median area (real
    grommets are made the same size; a stray similarly-colored blob
    elsewhere usually isn't). Each grommet's centroid is pushed outward
    from the quad's own center by its own radius (deliberately a bit past
    outward_push_factor=1.3, not 1.0) since the grommet's centroid sits at
    the hole in its middle, while the board's actual corner is beyond the
    ring's outer edge, not at its center.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_fraction * frame_area or area > max_area_fraction * frame_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        candidates.append({"center": np.array([cx, cy], dtype=np.float32), "radius": radius, "area": area})

    if len(candidates) < 4:
        return None

    if len(candidates) > 4:
        median_area = np.median([c["area"] for c in candidates])
        candidates.sort(key=lambda c: abs(c["area"] - median_area))
        candidates = candidates[:4]

    centers = np.array([c["center"] for c in candidates], dtype=np.float32)
    avg_radius = float(np.mean([c["radius"] for c in candidates]))

    ordered = order_points(centers)
    quad_center = ordered.mean(axis=0)
    pushed = []
    for pt in ordered:
        direction = pt - quad_center
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-3 else direction
        pushed.append(pt + direction * avg_radius * outward_push_factor)
    return np.array(pushed, dtype=np.float32)


def rectify_plate(
    frame: np.ndarray,
    corners: np.ndarray,
    board_px: int = 500,
    top_margin_fraction: float = 0.18,
    bottom_margin_fraction: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Warp the plate to an upright board_px x board_px square, with extra
    strips above and below (sized as a fraction of the board's own height)
    where connector wires exit. Doing wire detection in this straightened
    space sidesteps a real-world problem the previous perpendicular-offset
    math had: a hand-held plate is rarely perfectly level, and on a visibly
    tilted plate that offset landed off to one side instead of straight
    down from each pad.

    Returns (warped_bgr, inverse_transform, top_margin_px, board_px, bottom_margin_px).
    """
    top_left, top_right, bottom_right, bottom_left = corners

    left_vec = bottom_left - top_left
    right_vec = bottom_right - top_right
    left_len = np.linalg.norm(left_vec)
    right_len = np.linalg.norm(right_vec)
    left_dir = left_vec / (left_len + 1e-6)
    right_dir = right_vec / (right_len + 1e-6)

    top_margin_px = int(board_px * top_margin_fraction)
    bottom_margin_px = int(board_px * bottom_margin_fraction)
    total_px = top_margin_px + board_px + bottom_margin_px

    ext_top_left = top_left - left_dir * (left_len * top_margin_fraction)
    ext_top_right = top_right - right_dir * (right_len * top_margin_fraction)
    ext_bottom_left = bottom_left + left_dir * (left_len * bottom_margin_fraction)
    ext_bottom_right = bottom_right + right_dir * (right_len * bottom_margin_fraction)

    src = np.array([ext_top_left, ext_top_right, ext_bottom_right, ext_bottom_left], dtype="float32")
    dst = np.array(
        [[0, 0], [board_px, 0], [board_px, total_px], [0, total_px]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    # BORDER_REPLICATE: when the margin extrapolates past the actual camera
    # frame (common once the board fills most of the shot), repeat the
    # nearest real edge pixel instead of the default hard black fill - a
    # solid black region was getting misread as the (very loosely defined)
    # "black" wire color.
    warped = cv2.warpPerspective(frame, transform, (board_px, total_px), borderMode=cv2.BORDER_REPLICATE)
    inverse_transform = np.linalg.inv(transform)
    return warped, inverse_transform, top_margin_px, board_px, bottom_margin_px


def locate_color_cluster_x_range(
    strip_bgr: np.ndarray,
    color_ranges: dict,
    colors: list[str],
    min_column_fraction: float = 0.06,
    exclude_hsv_range: tuple[list[float], list[float]] | None = None,
    exclude_near_edge: str = "top",
    exclude_near_edge_fraction: float = 0.3,
) -> tuple[int, int, int] | None:
    """Within a horizontal strip, find the x-range containing any of the
    given known colors - this is what replaces "assume wires are evenly
    spread across the full board width", which put sample points over bare
    background/skin whenever the wires only occupied part of that width.
    Also returns the row (y) where the match is densest across that range,
    since the wires don't necessarily run the full height of the strip
    (they can end, bend, or braid partway down) - sampling at a fixed
    fraction of the strip's height would miss them just as easily as
    assuming even spacing across the full width did.

    exclude_hsv_range (typically the calibrated plate color) is only
    applied within a band near exclude_near_edge - see the matching note
    on find_wire_slots_by_color for why (a wire color can legitimately
    resemble the board's own color away from the board seam; excluding it
    everywhere caused a real bug).
    """
    hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color in colors:
        for lower, upper in color_ranges.get(color, []):
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    if exclude_hsv_range is not None:
        exclude_lower, exclude_upper = exclude_hsv_range
        raw_exclude_mask = cv2.inRange(hsv, np.array(exclude_lower), np.array(exclude_upper))
        exclude_mask = _mask_near_edge(raw_exclude_mask, exclude_near_edge, exclude_near_edge_fraction)
        mask &= cv2.bitwise_not(exclude_mask)

    # A real wire is many pixels wide; a thin (few-px) sliver is almost
    # always an anti-aliasing/interpolation seam at a geometric boundary
    # (e.g. warpPerspective blending board and background at a tilted
    # corner) rather than an actual wire, and such blended pixels are often
    # dark/desaturated enough to false-match the loose "black" bucket.
    # Opening erodes those thin seams away while leaving solid wire blobs intact.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    column_hits = mask.sum(axis=0) / 255
    threshold = strip_bgr.shape[0] * min_column_fraction
    matched_columns = np.where(column_hits > threshold)[0]
    if len(matched_columns) == 0:
        return None
    x_min, x_max = int(matched_columns.min()), int(matched_columns.max())

    row_hits = mask[:, x_min : x_max + 1].sum(axis=1)
    best_y = int(np.argmax(row_hits))
    return x_min, x_max, best_y


def build_solder_reach_mask(
    hsv: np.ndarray,
    solder_range: list[list[list[float]]],
    top_band_fraction: float = 0.2,
    x_dilate_px: int = 20,
) -> np.ndarray:
    """A wire is always physically anchored at a solder joint - a stray
    color match elsewhere (a mounting grommet, background, skin) never is,
    regardless of how close its hue happens to land to a wire color's
    range under some lighting. This finds solder-colored (bright,
    low-to-moderate saturation, metallic) pixels in the top band of the
    strip - where the joints actually are, right at the board edge - and
    marks the full column below each one as "reachable", with a small
    horizontal dilation so a wire doesn't have to sit in the exact same
    pixel column as its joint. Intersecting a color mask with this before
    counting blobs rejects anything not descending from a real joint.
    """
    h, w = hsv.shape[:2]
    top_band = hsv[: max(1, int(h * top_band_fraction)), :]
    solder_mask = np.zeros(top_band.shape[:2], dtype=np.uint8)
    for lower, upper in solder_range:
        solder_mask |= cv2.inRange(top_band, np.array(lower), np.array(upper))

    has_solder_col = solder_mask.any(axis=0)
    if x_dilate_px > 0:
        has_solder_col = (
            cv2.dilate(has_solder_col.astype(np.uint8).reshape(1, -1), np.ones((1, 2 * x_dilate_px + 1), np.uint8))
            .reshape(-1)
            .astype(bool)
        )

    reach_mask = np.zeros((h, w), dtype=np.uint8)
    reach_mask[:, has_solder_col] = 255
    return reach_mask


def _color_blobs(
    hsv: np.ndarray,
    color_ranges: dict,
    color: str,
    exclude_mask: np.ndarray | None,
    min_pixel_count: int,
    solder_reach_mask: np.ndarray | None = None,
) -> list[dict]:
    """All distinct blobs of one color in an HSV image, sorted left to
    right, each above min_pixel_count.

    The opening kernel is deliberately small: a wire squeezed into a
    narrow gap between two tightly-twisted neighbors can be only a few
    pixels wide where it's actually visible, and a larger kernel (this
    used 7x7) erases a sliver that thin before it's even measured -
    verified live, where a real yellow wire in exactly that situation
    disappeared entirely and reported NOT FOUND. A 3x3 kernel still
    removes single-pixel noise without erasing a genuine thin sliver.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges.get(color, []):
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    if exclude_mask is not None:
        mask &= cv2.bitwise_not(exclude_mask)
    if solder_reach_mask is not None:
        mask &= solder_reach_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [
        {"pixel_count": int(stats[i, cv2.CC_STAT_AREA]), "centroid": (int(centroids[i][0]), int(centroids[i][1]))}
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_pixel_count
    ]
    blobs.sort(key=lambda b: b["centroid"][0])
    return blobs


def _mask_near_edge(mask: np.ndarray, near_edge: str, fraction: float) -> np.ndarray:
    """Zero out everything except a band near one edge of the mask."""
    height = mask.shape[0]
    band = max(1, int(height * fraction))
    limited = np.zeros_like(mask)
    if near_edge == "top":
        limited[:band, :] = mask[:band, :]
    else:
        limited[-band:, :] = mask[-band:, :]
    return limited


def find_wire_slots_by_color(
    strip_bgr: np.ndarray,
    wire_slots: list[dict],
    color_ranges: dict,
    exclude_hsv_range: tuple[list[float], list[float]] | None = None,
    min_pixel_count: int = 60,
    require_solder_reach: bool = False,
    exclude_near_edge: str = "top",
    exclude_near_edge_fraction: float = 0.3,
) -> list[dict]:
    """For each expected color, find every distinct blob of it in the strip
    (not just the single largest) and hand them out left-to-right to the
    slots that expect that color, in slot order - instead of assuming the
    wires sit at evenly-spaced x positions.

    Real hookup wire bends and drifts as it hangs, so at any fixed sampling
    row the wires are rarely at the x positions even spacing would predict -
    verified against an actual reference photo, where only the two
    outermost wires landed correctly and the three middle ones missed
    entirely. Searching per-color directly sidesteps that. A harness can
    also repeat a color (e.g. two black wires) - matching only "the
    biggest blob of black" per slot would collapse both slots onto the
    same wire, so blobs are enumerated and assigned in left-to-right
    order among just the slots that need that color, preserving the
    ability to tell "first black" from "second black".

    Centroid x order is still checked against the expected slot order
    afterward (by the caller), so a left-right swap is still caught.

    require_solder_reach requires every candidate blob to sit in a column
    descending from a solder-colored region near the top of the strip
    (see build_solder_reach_mask) - i.e. actually anchored to a joint, not
    just a coincidental color match somewhere else in frame (a mounting
    grommet, background, skin). Needs a "solder" entry in color_ranges;
    silently has no effect without one.

    exclude_hsv_range (typically the calibrated plate color) is only
    applied within a band near exclude_near_edge (the strip's edge
    adjacent to the board - "top" for a strip below the board, "bottom"
    for a strip above it), not the whole strip. Board-color leakage into
    the strip only happens right at that seam (rectification blending
    board and margin together there); applying the exclusion everywhere
    caused a real bug verified against a live photo, where a genuine teal
    wire's hue happened to fall inside the board's own calibrated color
    range and got excluded everywhere it appeared, not just near the seam.

    Returns one dict per slot, in original slot order: {slot, pad_label,
    expected_color, found, matched_color, centroid, pixel_count}.
    """
    hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    exclude_mask = None
    if exclude_hsv_range is not None:
        exclude_lower, exclude_upper = exclude_hsv_range
        raw_exclude_mask = cv2.inRange(hsv, np.array(exclude_lower), np.array(exclude_upper))
        exclude_mask = _mask_near_edge(raw_exclude_mask, exclude_near_edge, exclude_near_edge_fraction)

    solder_reach_mask = None
    if require_solder_reach and color_ranges.get("solder"):
        solder_reach_mask = build_solder_reach_mask(hsv, color_ranges["solder"])

    color_to_slot_indices: dict[str, list[int]] = {}
    for idx, slot in enumerate(wire_slots):
        color_to_slot_indices.setdefault(slot["expected_color"], []).append(idx)

    matches: list[dict | None] = [None] * len(wire_slots)
    for color, slot_indices in color_to_slot_indices.items():
        blobs = _color_blobs(hsv, color_ranges, color, exclude_mask, min_pixel_count, solder_reach_mask)
        # A wire can fragment into several disconnected blobs (another wire
        # crossing in front, a twist briefly hiding part of it) - the real
        # wire is still the largest of those fragments, so pick the top-N
        # by size first (N = number of slots wanting this color), then
        # order only those N left-to-right for assignment. Sorting by x
        # first and taking the first N would hand a slot a tiny leftmost
        # fragment instead of the actual wire.
        largest_first = sorted(blobs, key=lambda b: -b["pixel_count"])[: len(slot_indices)]
        largest_first.sort(key=lambda b: b["centroid"][0])
        for slot_idx, blob in zip(slot_indices, largest_first):
            matches[slot_idx] = {"matched_color": color, **blob}

    # alt_color fallback: only for slots still unmatched, and only among
    # blobs not already claimed by another slot via alt matching.
    claimed_alt_centroids: set[tuple[int, int]] = set()
    for idx, slot in enumerate(wire_slots):
        if matches[idx] is not None or not slot.get("alt_color"):
            continue
        blobs = _color_blobs(hsv, color_ranges, slot["alt_color"], exclude_mask, min_pixel_count, solder_reach_mask)
        for blob in blobs:
            if blob["centroid"] not in claimed_alt_centroids:
                matches[idx] = {"matched_color": slot["alt_color"], **blob}
                claimed_alt_centroids.add(blob["centroid"])
                break

    results = []
    for idx, slot in enumerate(wire_slots):
        match = matches[idx]
        results.append(
            {
                "slot": slot["slot"],
                "pad_label": slot.get("pad_label", slot["slot"]),
                "expected_color": slot["expected_color"],
                "found": match is not None,
                "matched_color": match["matched_color"] if match else None,
                "centroid": match["centroid"] if match else None,
                "pixel_count": match["pixel_count"] if match else 0,
            }
        )
    return results


def is_monotonic(xs: list[float]) -> bool:
    """True if xs is sorted strictly in one direction (increasing or
    decreasing) - accepting either direction tolerates a harness
    photographed rotated/mirrored relative to how its left-to-right order
    was defined, while a real swapped-wire defect still fails either way
    (any single swap breaks monotonicity in both directions)."""
    increasing = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    decreasing = all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1))
    return increasing or decreasing


def evaluate_wire_results(results: list[dict], check_order: bool = True) -> bool:
    """Overall pass/fail from find_wire_slots_by_color's output.

    check_order=False skips the left-to-right ordering check - meaningful
    for a straight/parallel harness, but not for a twisted pair: twisted
    wires spiral and swap left-right position along their length by
    design, so two same-colored wires from a twisted pair can legitimately
    measure out of x-order depending on exactly where each blob was
    sampled. For those harnesses, color presence and count is the
    meaningful check, not position.

    Order is accepted in either direction (strictly increasing OR
    strictly decreasing x) - verified against a real photo where the
    board was rotated/mirrored relative to how the harness's left-to-right
    order was originally defined, and the (perfectly correctly wired)
    sequence measured monotonically decreasing instead. A real swapped-wire
    defect still fails either way: any single swap breaks monotonicity in
    both directions, so this only tolerates a globally mirrored photo, not
    an actual wiring mistake.
    """
    if not all(r["found"] for r in results):
        return False
    if not check_order:
        return True
    centroids_x = [r["centroid"][0] for r in results if r["found"]]
    return is_monotonic(centroids_x)
