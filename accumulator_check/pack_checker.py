"""Battery pack terminal-orientation checker ("accumulator check").

Separate, much simpler algorithm from the wire/solder inspection system in
app/ - different dataset, different physical part, no shared code or spec.

The photo is a top-down shot of a 6x6 grid of cylindrical cells (36
total) in a 3D-printed holder, taped at the 4 corners. Each cell's
visible top is one of two distinct terminal ends:
  - "+"  a SMALL silver button surrounded by a THICK/dark-teal ring
  - "-"  a LARGE silver disc surrounded by a THIN/bright-mint ring
Reverse-engineered from the user's own reference photos (dataset/correct)
and confirmed by the user (2026-09-14): a correctly-assembled pack
alternates +/- strictly BY COLUMN - every row reads +,-,+,-,+,- (or the
mirror -,+,-,+,-,+; which absolute polarity a given photo shows doesn't
matter, only that each row is internally consistent with its own columns
and every column agrees with itself down all 6 rows). The defect this
catches is a single cell seated with the wrong terminal facing up,
breaking that column's otherwise-uniform sign.

Calibration status (2026-09-14, against the user's own dataset of 4
correct / 34 defect / 8 defect_optional photos): the alternation RULE is
confirmed against all 4 "correct" photos. Per-cell classification went
through two generations before landing here:

  1. Color/brightness alone (ring-hue area minus metal-brightness
     fraction, both within the cell's own circle) - 2 to 5 of 36 cells
     falsely flagged per "correct" photo. Traced two real causes, not
     just noise: the Hough-detected radius balloons into the black
     corner tape for a few specific cells (diluting the sample), and
     cells farther from the frame's optical center catch a different
     specular-highlight angle that shifts brightness/color independent
     of which terminal is actually up. Neither was fixable by adjusting
     the sampling radius.
  2. Adding TEXTURE fixed most of it: the "+" button has a visibly
     grooved/machined surface, the "-" disc is comparatively smooth - a
     Laplacian-variance measurement over the cell's own central patch
     (_texture_score) picks this up and is barely affected by the
     lighting/reflection issues that hurt the color signal, because it's
     a LOCAL contrast measure, not an absolute color or brightness level.
     Combined as (1.5 * z-scored texture) + (1.0 * z-scored color) per
     photo (weights swept 1.0-3.0 against the same 4 photos; 1.5 was the
     minimum), false flags on "correct" photos dropped to 1 total across
     all 4 (was 15). The two signals fail on different cells, which is
     why combining them (rather than replacing one with the other) is
     what closed most of the gap.

  3. Confirmed against a SECOND real photo (also a user-reported false
     FAIL on a known-good pack, 2026-09-14) that the remaining failure
     mode is a corner cell, every time: the diagonal corner tape casts a
     shadow across part of that cell's disc, which _color_score misreads
     (texture reads it fine) - same class of cause as (1), now nailed
     down to exactly where it happens. Tried three more targeted fixes on
     the MAIN score (a shared median radius, capping color's z-score
     outliers, a per-cell relative-brightness threshold, an inward-
     shifted sampling center) plus blending a LAB chroma signal into
     every cell's score - none of those closed it without breaking a
     different cell elsewhere.
  4. First fix tried: L*a*b* chroma (the B channel), which is far less
     shadow-sensitive than brightness/saturation - the second case's
     shadowed corner had chroma matching its OWN column's other 4
     members much more closely than the opposing column's, used as a
     targeted cross-check only on already-flagged corners (not blended
     into every cell's score - tried that too, made things worse
     broadly). Closed that case (1 -> 0 false flags on "correct" photos),
     but a THIRD real photo (also user-reported, 2026-09-15) turned up a
     shadowed corner where chroma's own margin was too close to call
     (3.1 vs 2.6, noise-level) - chroma didn't reliably generalize.
  5. What actually closed it: the SAME cross-check structure, but using
     TEXTURE (_texture_score) instead of chroma - texture had already
     read every shadowed corner correctly across all three real photos
     tested (chroma only got 1 of 2 conclusively), which makes sense
     causally: the tape's shadow changes a corner's apparent BRIGHTNESS,
     not its physical surface, so texture is exactly the kind of signal a
     shadow shouldn't touch. Confirmed this also correctly leaves a REAL
     corner defect alone rather than erasing it: a genuinely swapped
     corner cell is an actually different physical surface, so its
     texture should resemble the OPPOSING columns, not its own, and the
     override requires the corner's texture to side with its own column
     to fire at all.

Every corner cell still gets marked low_confidence regardless of the
above (see classify_pack) - this cross-check has only been run against
three real photos so far, nowhere near as thoroughly as the main signal.

A correctly-alternating 6x6 pack is always 18 "+" / 18 "-" overall, so
plus_count/minus_count (see check_pack) is a cheap independent sanity
total, though not yet proven trustworthy on its own either - recheck it
if you rely on it before more real photos have been run through this.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

EXPECTED_ROWS = 6
EXPECTED_COLS = 6

# Hough circle params tuned against the reference dataset's photo scale
# (~2560x1920, cell radius ~85-125px). param2 (accumulator threshold) is
# swept rather than fixed - verified against the dataset that a single
# fixed value finds exactly 36 circles on some photos but 35 or 37 on
# others (a weak or doubled edge response), and a stricter/looser
# threshold recovers exactly 36 on most of those without touching
# anything else.
_HOUGH_PARAM2_SWEEP = (40, 35, 45, 30, 50, 25)
_MIN_RADIUS = 85
_MAX_RADIUS = 130
_MIN_DIST = 170


def _reject_isolated_circles(circles: np.ndarray) -> np.ndarray:
    """Drop any detected circle whose nearest OTHER detected circle is
    much farther away than typical. Every real cell has a real neighbor
    at roughly the grid's own pitch (measured ~195-250px in the reference
    dataset), while a false positive from background clutter has no real
    neighbor nearby at all - confirmed 2026-09-15 against a real photo
    where a white power cable curving through the background got picked
    up as an extra circle 733px from its nearest neighbor, versus every
    real cell's 194-248px, and that one photo failed detection outright
    (37 circles, never exactly 36) until this filter was added. The
    threshold is relative to THIS photo's own median nearest-neighbor
    distance, not a fixed px count, so it isn't tied to one specific
    camera distance/zoom.
    """
    if len(circles) < 2:
        return circles
    coords = circles[:, :2]
    nearest_neighbor_dist = np.array(
        [np.min(np.hypot(*(coords - coords[i]).T)[np.arange(len(coords)) != i]) for i in range(len(coords))]
    )
    threshold = np.median(nearest_neighbor_dist) * 1.8
    return circles[nearest_neighbor_dist <= threshold]


def detect_cell_grid(image: np.ndarray) -> tuple[dict[tuple[int, int], tuple[float, float, float]] | None, str | None]:
    """Locate all 36 cells and assign each a (row, col) index, tolerant of
    camera tilt/skew: rows are found by 1D k-means on y (not a fixed pitch
    grid, which broke under a few degrees of tilt - verified against the
    dataset that a global 2-axis lattice fit aliased to the wrong column
    for some photos), then within each row the 6 members are just sorted
    left-to-right and assigned column 0..5 by RANK, not by absolute x
    position - this sidesteps skew entirely since it only compares
    circles already known to be in the same row.

    Returns (grid, None) on success - grid maps (row, col) -> (x, y,
    radius) in the image's own pixel coordinates - or (None, reason) if
    circle detection didn't find a clean 6x6 grid.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 7)

    circles = None
    for param2 in _HOUGH_PARAM2_SWEEP:
        found = cv2.HoughCircles(
            gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=_MIN_DIST,
            param1=60, param2=param2, minRadius=_MIN_RADIUS, maxRadius=_MAX_RADIUS,
        )
        if found is None:
            continue
        filtered = _reject_isolated_circles(found[0])
        if len(filtered) == EXPECTED_ROWS * EXPECTED_COLS:
            circles = filtered
            break
    if circles is None:
        return None, "could not find exactly 36 cell circles in this photo"

    ys = circles[:, 1].astype(np.float32).reshape(-1, 1)
    _, labels, centers = cv2.kmeans(
        ys, EXPECTED_ROWS, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
        10, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=EXPECTED_ROWS)
    if not np.all(counts == EXPECTED_COLS):
        return None, f"rows didn't split into 6 groups of 6 (got {counts.tolist()})"

    row_order = np.argsort(centers.flatten())
    grid: dict[tuple[int, int], tuple[float, float, float]] = {}
    for row_rank, cluster_id in enumerate(row_order):
        members = sorted(
            (tuple(circles[j]) for j in range(len(circles)) if labels[j] == cluster_id),
            key=lambda m: m[0],
        )
        for col, (x, y, r) in enumerate(members):
            grid[(row_rank, col)] = (float(x), float(y), float(r))
    return grid, None


# How much weight _combined_scores gives the texture signal relative to
# the color signal (color always weight 1.0) - swept 1.0-3.0 against the
# reference dataset's 4 "correct" photos, picked as the minimum weight
# that already reached that sweep's best result (1 false flag total
# across all 4, down from 15 at weight 1.0 i.e. color-only-strength);
# higher weights started re-introducing false flags on different cells.
_TEXTURE_WEIGHT = 1.5


def _color_score(hsv: np.ndarray, x: float, y: float, r: float) -> float:
    """Higher = more likely "+" (thick ring, small button), lower = more
    likely "-" (thin ring, large disc), from color/brightness alone.
    Combines two sub-signals rather than either alone - verified against
    the dataset that either used by itself misreads some cells (a
    metallic-brightness reading is thrown off by shadow/reflection on the
    disc; a ring-hue-area reading alone is noisier for cells nearer the
    frame's center, likely a perspective effect on apparent ring width)
    but their difference is more consistently separated per-photo, since
    the two sub-signals' own errors aren't the same shape. Still not
    reliable alone - see _texture_score and the module docstring.
    """
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(full_mask, (int(x), int(y)), int(r * 0.9), 255, -1)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    ring_mask = ((hue >= 40) & (hue <= 100) & (sat >= 30)).astype(np.uint8) * 255
    ring_mask = cv2.bitwise_and(ring_mask, full_mask)
    ring_fraction = np.count_nonzero(ring_mask) / max(1, np.count_nonzero(full_mask))

    inner_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(inner_mask, (int(x), int(y)), int(r * 0.85), 255, -1)
    metal_mask = ((sat < 70) & (val > 90)).astype(np.uint8) * 255
    metal_mask = cv2.bitwise_and(metal_mask, inner_mask)
    metal_fraction = np.count_nonzero(metal_mask) / max(1, np.count_nonzero(inner_mask))

    return ring_fraction - metal_fraction


def _texture_score(gray: np.ndarray, x: float, y: float, r: float) -> float:
    """Higher = more likely "+": the "+" terminal's small button has a
    visibly grooved/machined surface (real local pixel-to-pixel contrast),
    while the "-" terminal's large disc is comparatively smooth - Laplacian
    variance over a central patch (0.55r, well inside the button/disc,
    never touching the ring) picks this up directly. Added after
    color-only scoring left several cells falsely flagged on "correct"
    photos - this signal is barely affected by the lighting/reflection
    issues that hurt color, since it's a LOCAL contrast measure, not an
    absolute brightness or hue level (see module docstring for the before/
    after)."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r * 0.55), 255, -1)
    x0, y0 = max(0, int(x - r)), max(0, int(y - r))
    x1, y1 = int(x + r), int(y + r)
    crop_gray = gray[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop_gray.size == 0:
        return 0.0
    laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F, ksize=3)
    values = laplacian[crop_mask > 0]
    return float(values.var()) if values.size else 0.0


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    return (arr - arr.mean()) / (arr.std() + 1e-9)


def _combined_scores(
    image: np.ndarray, items: list[tuple[tuple[int, int], tuple[float, float, float]]]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-photo z-scored blend of _texture_score and _color_score (see
    _TEXTURE_WEIGHT) - z-scoring each signal separately (not just summing
    raw values) matters because they live on completely different scales
    (texture is a Laplacian variance in the thousands, color is a
    fraction difference in [-1, 1]) and, more importantly, because the
    absolute scale of EITHER signal varies photo to photo (lighting,
    exposure) the same way _color_score alone did - see classify_pack for
    why clustering happens per-photo rather than against a fixed cutoff.

    Returns (combined, texture_z) - the raw z-scored texture signal is
    also returned on its own for classify_pack's corner cross-check (see
    there), which needs texture in isolation, not blended with color.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    texture = _zscore([_texture_score(gray, x, y, r) for _, (x, y, r) in items])
    color = _zscore([_color_score(hsv, x, y, r) for _, (x, y, r) in items])
    return _TEXTURE_WEIGHT * texture + color, texture


def classify_pack(image: np.ndarray, grid: dict[tuple[int, int], tuple[float, float, float]]) -> dict[str, Any]:
    """Classify every cell as "+"/"-", using per-PHOTO (not a fixed global
    threshold) 2-means clustering on _combined_scores (texture + color) -
    verified necessary against the dataset: absolute brightness/color/
    texture varies enough between photos (lighting, exposure) that a
    fixed cutoff misclassified whole columns on some photos while working
    on others, whereas the two classes are still cleanly bimodal within
    any single photo.

    A cell's own "sign" is compared against what its column SHOULD be
    under the whole pack's best-fit alternating template (see
    expected_by_column below), not just against its own column's raw
    majority - the confirmed rule tolerates either global starting
    polarity (a pack can be seated either way), but does NOT tolerate two
    ADJACENT columns sharing a sign. An earlier version of this function
    only checked a cell against its own column's majority, which caught a
    single mis-seated cell fine but had a real blind spot (2026-09-16,
    user-reported): a column whose EVERY cell agreed with each other but
    whose majority itself broke the alternation (e.g. two neighboring
    columns both "+") was invisible to that check, since nothing there
    ever compared one column against another.

    low_confidence lists cells whose score sits close to the midpoint
    between this photo's two cluster centers - per the calibration note
    in this module's docstring, these are worth flagging for a human
    look even when they don't end up disagreeing with their column,
    since the same weak-signal positions are where real misclassifications
    concentrated in testing.
    """
    items = list(grid.items())
    combined, texture_z = _combined_scores(image, items)
    scores = combined.reshape(-1, 1).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        scores, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
        10, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()
    centers = centers.flatten()
    plus_cluster = int(np.argmax(centers))
    midpoint = float(np.mean(centers))
    half_gap = abs(centers[0] - centers[1]) / 2

    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for (rc, _), label, score in zip(items, labels, scores.flatten()):
        sign = "+" if label == plus_cluster else "-"
        margin = abs(float(score) - midpoint) / half_gap if half_gap > 1e-6 else 0.0
        cells[rc] = {"sign": sign, "score": float(score), "margin": round(margin, 2)}

    # The 4 physical corner cells sit right where the diagonal corner
    # tape crosses closest to the pack - confirmed against three separate
    # real photos (2026-09-14 through 2026-09-16, all user-reported) that
    # this specific tape proximity casts a shadow across part of a corner
    # cell's disc, skewing _color_score toward a false reading even
    # though _texture_score reads that same cell correctly. Tried three
    # targeted fixes on the color signal itself (a shared median radius,
    # capping color's z-score outliers, a per-cell relative-brightness
    # threshold) - none resolved it, so corners are excluded from
    # col_majority below rather than trusted at face value, and separately
    # cross-checked against their own column via texture further down.
    corner_cells = {(0, 0), (0, EXPECTED_COLS - 1), (EXPECTED_ROWS - 1, 0), (EXPECTED_ROWS - 1, EXPECTED_COLS - 1)}

    col_signs: dict[int, list[str]] = {c: [] for c in range(EXPECTED_COLS)}
    for (row, col), info in cells.items():
        if (row, col) in corner_cells:
            continue
        col_signs[col].append(info["sign"])
    col_majority = {c: max(set(signs), key=signs.count) for c, signs in col_signs.items()}

    # Cross-check EVERY corner (not just ones already disagreeing with
    # the column below - see why this matters next) against its own
    # column using TEXTURE ALONE (not blended with color, unlike the main
    # score) - confirmed against three real photos (2026-09-14 through
    # 2026-09-16) that texture keeps reading a shadowed corner correctly
    # even when color is a strong outlier there, so a corner's own
    # texture_z compared against its column's other rows vs. the
    # opposing-sign columns' texture_z resolves exactly the case
    # _combined_scores gets wrong. (A first attempt at this cross-check
    # used LAB chroma instead - it resolved one real case but was
    # ambiguous, not clearly either way, on a second; texture wasn't
    # ambiguous on either.) This is also theoretically why texture should
    # be trustworthy here specifically: the tape's shadow changes a
    # corner's apparent BRIGHTNESS, not its physical surface - the
    # button's groove pattern - so a genuinely swapped corner cell (a
    # real defect, not a shadow artifact) should still show texture
    # matching the OPPOSING columns, not its own, and this override would
    # correctly decline to fire for it.
    #
    # Applying this to EVERY corner, not only ones _combined_scores
    # already flagged, closed a real gap (2026-09-16, user-reported): a
    # corner whose bad color reading happened to coincidentally match
    # what the alternating pattern expected slipped through as a false
    # PASS, because it was never a "mismatch" to begin with, so the
    # earlier version of this cross-check (which only ran on already-
    # flagged corners) never got a chance to look at it. Comparing every
    # corner's texture against its own (corner-excluded, so uncontaminated
    # by this exact problem) column majority regardless of the corner's
    # current pass/fail status is what catches that.
    texture_z_map = {rc: float(texture_z[i]) for i, (rc, _) in enumerate(items)}
    for row, col in sorted(corner_cells):
        other_rows_in_col = [r for r in range(EXPECTED_ROWS) if (r, col) not in corner_cells]
        own_column_texture = float(np.mean([texture_z_map[(r, col)] for r in other_rows_in_col]))
        corner_texture = texture_z_map[(row, col)]
        opposing_texture_samples = [
            texture_z_map[(r, c)]
            for c in range(EXPECTED_COLS) if col_majority[c] != col_majority[col]
            for r in range(EXPECTED_ROWS) if (r, c) not in corner_cells
        ]
        if not opposing_texture_samples:
            continue
        opposing_texture = float(np.mean(opposing_texture_samples))
        if abs(corner_texture - own_column_texture) < abs(corner_texture - opposing_texture):
            cells[(row, col)]["sign"] = col_majority[col]
            cells[(row, col)]["texture_override"] = True

    # Fit the 6 observed (corner-excluded, texture-corrected) column-
    # majorities against BOTH possible perfect alternating templates
    # (starting "+" or starting "-") and keep whichever the majority of
    # columns already agrees with. This is what preserves "either global
    # polarity is fine" while still catching a column whose OWN majority
    # breaks the alternation with its neighbor - expected_by_column, not
    # col_majority, is what every cell's sign (corners included) gets
    # checked against below.
    template_plus_first = {c: ("+" if c % 2 == 0 else "-") for c in range(EXPECTED_COLS)}
    template_minus_first = {c: ("-" if c % 2 == 0 else "+") for c in range(EXPECTED_COLS)}
    agree_plus_first = sum(1 for c in range(EXPECTED_COLS) if col_majority[c] == template_plus_first[c])
    agree_minus_first = EXPECTED_COLS - agree_plus_first
    expected_by_column = template_plus_first if agree_plus_first >= agree_minus_first else template_minus_first

    mismatches = []
    low_confidence = []
    for (row, col), info in cells.items():
        info["expected"] = expected_by_column[col]
        info["pass"] = info["sign"] == expected_by_column[col]
        if not info["pass"]:
            mismatches.append((row, col))
        if info["margin"] < 0.4 or (row, col) in corner_cells:
            low_confidence.append((row, col))

    # Plain +/- tally across all 36 cells - independent of the per-cell
    # mismatch check above, but not actually a separate possible failure
    # mode given EXPECTED_ROWS/EXPECTED_COLS are both even and
    # expected_by_column is a strict alternation: zero mismatches already
    # mathematically guarantees 18/18. Reported explicitly anyway (a
    # requirement in its own right, 2026-09-16) since it's a much simpler
    # thing for a caller to check/display than walking the full mismatch
    # list, and it stays meaningful as an independent cross-check if this
    # grid's dimensions ever change.
    plus_count = sum(1 for info in cells.values() if info["sign"] == "+")
    minus_count = len(cells) - plus_count
    counts_balanced = plus_count == minus_count

    return {
        "cells": cells,
        "column_majority": col_majority,
        "expected_by_column": expected_by_column,
        "mismatches": mismatches,
        "low_confidence": low_confidence,
        "plus_count": plus_count,
        "minus_count": minus_count,
        "counts_balanced": counts_balanced,
        "pass": len(mismatches) == 0 and counts_balanced,
    }


def check_pack(image_path: str) -> dict[str, Any]:
    """Entry point: load a photo, find the 36-cell grid, classify every
    cell, and report any that disagree with their own column - see module
    docstring for the confirmed rule and this checker's calibration
    status."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    grid, error = detect_cell_grid(image)
    if grid is None:
        return {"pass": None, "error": error, "cells": {}, "mismatches": [], "low_confidence": []}

    result = classify_pack(image, grid)
    result["grid"] = grid
    return result
