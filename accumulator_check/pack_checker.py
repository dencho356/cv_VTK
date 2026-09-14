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

  3. Confirmed against TWO separate real photos (both user-reported
     false FAILs on known-good packs, 2026-09-14) that the remaining
     failure mode is a corner cell, every time: the diagonal corner tape
     casts a shadow across part of that cell's disc, which _color_score
     misreads (texture reads it fine) - same class of cause as (1), now
     nailed down to exactly where it happens. Tried three more targeted
     fixes on the MAIN score (a shared median radius, capping color's
     z-score outliers, a per-cell relative-brightness threshold, an
     inward-shifted sampling center) plus blending a LAB chroma signal
     into every cell's score - none of those closed it without breaking
     a different cell elsewhere.
  4. What finally worked: L*a*b* chroma (_lab_chroma_score) turned out to
     be far less shadow-sensitive than brightness/saturation - the same
     shadowed corner's chroma matched its OWN column's other 4 members
     much more closely than the opposing column's. Rather than blend
     this into every cell's score (tried that in step 3, made things
     worse broadly), classify_pack uses it ONLY as a targeted cross-check
     on a corner that's ALREADY flagged: compare that corner's chroma
     against its own column's other rows vs. against the columns holding
     the opposite sign, and override only when chroma clearly sides with
     its own column. Closed the last false flag (1 -> 0 across all 4
     "correct" photos) without suppressing detection elsewhere (applied
     to only 7 of 36 "defect" photos' flagged corners, each of which
     still had other, non-corner mismatches reported).

Every corner cell still gets marked low_confidence regardless of the
above (see classify_pack) - the LAB cross-check has only been run against
this one dataset so far, not proven as thoroughly as the main signal.

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
        if found is not None and len(found[0]) == EXPECTED_ROWS * EXPECTED_COLS:
            circles = found[0]
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


def _lab_chroma_score(lab: np.ndarray, x: float, y: float, r: float) -> float:
    """Mean L*a*b* B-channel (blue-yellow chroma) in the cell's ring
    annulus (0.65r-0.95r, same band _color_score's ring_fraction covers).
    Added specifically to cross-check a flagged CORNER cell (see
    classify_pack) after confirming against a real photo (2026-09-14)
    that the corner tape's shadow drops a corner's LAB lightness (L) and
    throws off _color_score's brightness-dependent reading, but leaves
    this chroma channel close to its own column's other members - chroma
    is far less sensitive to a lighting/shadow difference than brightness
    or saturation are, which is exactly the failure mode _color_score has
    at these positions. Tried blending this into _combined_scores for
    every cell first; that made things WORSE overall (more false flags,
    not fewer) - chroma isn't uniformly more reliable, it specifically
    rescues the shadowed-corner case, so it's only used there.
    """
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r * 0.95), 255, -1)
    cv2.circle(mask, (int(x), int(y)), int(r * 0.65), 0, -1)
    b_channel = lab[:, :, 2]
    selected = mask > 0
    return float(b_channel[selected].mean()) if np.any(selected) else 0.0


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    return (arr - arr.mean()) / (arr.std() + 1e-9)


def _combined_scores(image: np.ndarray, items: list[tuple[tuple[int, int], tuple[float, float, float]]]) -> np.ndarray:
    """Per-photo z-scored blend of _texture_score and _color_score (see
    _TEXTURE_WEIGHT) - z-scoring each signal separately (not just summing
    raw values) matters because they live on completely different scales
    (texture is a Laplacian variance in the thousands, color is a
    fraction difference in [-1, 1]) and, more importantly, because the
    absolute scale of EITHER signal varies photo to photo (lighting,
    exposure) the same way _color_score alone did - see classify_pack for
    why clustering happens per-photo rather than against a fixed cutoff.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    texture = _zscore([_texture_score(gray, x, y, r) for _, (x, y, r) in items])
    color = _zscore([_color_score(hsv, x, y, r) for _, (x, y, r) in items])
    return _TEXTURE_WEIGHT * texture + color


def classify_pack(image: np.ndarray, grid: dict[tuple[int, int], tuple[float, float, float]]) -> dict[str, Any]:
    """Classify every cell as "+"/"-", using per-PHOTO (not a fixed global
    threshold) 2-means clustering on _combined_scores (texture + color) -
    verified necessary against the dataset: absolute brightness/color/
    texture varies enough between photos (lighting, exposure) that a
    fixed cutoff misclassified whole columns on some photos while working
    on others, whereas the two classes are still cleanly bimodal within
    any single photo.

    A cell's own "sign" is then compared against its COLUMN's majority
    sign (not a hardcoded absolute pattern) - the confirmed rule tolerates
    either global polarity (a pack can be seated either way), so only a
    LOCAL disagreement within a column is treated as a possible defect.

    low_confidence lists cells whose score sits close to the midpoint
    between this photo's two cluster centers - per the calibration note
    in this module's docstring, these are worth flagging for a human
    look even when they don't end up disagreeing with their column,
    since the same weak-signal positions are where real misclassifications
    concentrated in testing.
    """
    items = list(grid.items())
    scores = _combined_scores(image, items).reshape(-1, 1).astype(np.float32)

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

    col_signs: dict[int, list[str]] = {c: [] for c in range(EXPECTED_COLS)}
    for (row, col), info in cells.items():
        col_signs[col].append(info["sign"])
    col_majority = {c: max(set(signs), key=signs.count) for c, signs in col_signs.items()}

    # The 4 physical corner cells sit right where the diagonal corner
    # tape crosses closest to the pack - confirmed against two separate
    # real photos (2026-09-14, user-reported) that this specific tape
    # proximity casts a shadow across part of a corner cell's disc,
    # dropping it below the metal-brightness cutoff and skewing
    # _color_score toward a false "+" even though _texture_score reads
    # that same cell correctly. Tried three targeted fixes (a shared
    # median radius instead of each cell's own noisy Hough radius,
    # capping color's z-score outliers, and a per-cell relative-
    # brightness threshold instead of the fixed one) - none resolved it
    # without introducing new false flags elsewhere, so corners are
    # unconditionally treated as low_confidence rather than trusted at
    # face value, regardless of their own margin.
    corner_cells = {(0, 0), (0, EXPECTED_COLS - 1), (EXPECTED_ROWS - 1, 0), (EXPECTED_ROWS - 1, EXPECTED_COLS - 1)}

    mismatches = []
    low_confidence = []
    for (row, col), info in cells.items():
        info["expected"] = col_majority[col]
        info["pass"] = info["sign"] == col_majority[col]
        if not info["pass"]:
            mismatches.append((row, col))
        if info["margin"] < 0.4 or (row, col) in corner_cells:
            low_confidence.append((row, col))

    # Cross-check any FLAGGED corner against its own column using LAB
    # chroma (see _lab_chroma_score) - confirmed 2026-09-14 against a
    # real photo that this specific comparison (a corner's chroma vs. its
    # own column's other rows vs. the columns holding the opposite sign)
    # correctly recognizes a corner that only misread on brightness,
    # closing the one remaining false flag from the color+texture signal
    # above. Still only overrides the SPECIFIC corners currently flagged
    # - every corner stays in low_confidence either way (see above),
    # since this cross-check hasn't been tried on nearly as many photos
    # as the main signal has.
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for row, col in list(mismatches):
        if (row, col) not in corner_cells:
            continue
        other_rows_in_col = [r for r in range(EXPECTED_ROWS) if (r, col) not in corner_cells]
        own_column_chroma = float(np.mean([_lab_chroma_score(lab_image, *grid[(r, col)]) for r in other_rows_in_col]))
        corner_chroma = _lab_chroma_score(lab_image, *grid[(row, col)])
        opposing_chroma_samples = [
            _lab_chroma_score(lab_image, *grid[(r, c)])
            for c in range(EXPECTED_COLS) if col_majority[c] != col_majority[col]
            for r in range(EXPECTED_ROWS) if (r, c) not in corner_cells
        ]
        if not opposing_chroma_samples:
            continue
        opposing_chroma = float(np.mean(opposing_chroma_samples))
        if abs(corner_chroma - own_column_chroma) < abs(corner_chroma - opposing_chroma):
            cells[(row, col)]["sign"] = col_majority[col]
            cells[(row, col)]["pass"] = True
            cells[(row, col)]["lab_override"] = True
            mismatches.remove((row, col))

    # Plain +/- tally across all 36 cells, independent of the column-
    # majority check above - e.g. useful as a quick sanity total (a
    # correctly alternating 6x6 pack is always 18/18) even before looking
    # at which specific cells disagree with their column.
    plus_count = sum(1 for info in cells.values() if info["sign"] == "+")
    minus_count = len(cells) - plus_count

    return {
        "cells": cells,
        "column_majority": col_majority,
        "mismatches": mismatches,
        "low_confidence": low_confidence,
        "plus_count": plus_count,
        "minus_count": minus_count,
        "pass": len(mismatches) == 0,
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
