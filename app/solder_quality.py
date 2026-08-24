"""Heuristic solder JOINT QUALITY scorer (good/bad), separate from
count_solder_joints (app/solder_checker.py, which only finds WHERE the
joints are). Calibrated 2026-08-24 against exactly 2 real macro photos
supplied by hand (1 confirmed-bad joint, 1 photo of 3 confirmed-good
joints) - this is a first pass to validate the approach works at all,
not a production-ready classifier. Every threshold below is a rough
midpoint between that single bad example and those three good ones;
expect to retune once more real examples exist (see spec.json's own
note that PatchCore, trained on 100-300 real good-joint crops, is the
intended long-term replacement for hand-tuned thresholds like these -
same reasoning as everywhere else in this codebase that a fixed
threshold calibrated against one small real sample breaks on a new one).

Measured features, real photo evidence (2026-08-24):
                    solidity   circularity   lap_var   dark_frac
  bad (n=1)           0.662       0.257        934       0.034
  good (n=3)        0.82-0.91   0.42-0.65   2964-3147   0.004-0.008

solidity/circularity: a lumpy, poorly-wetted joint has visible
concavities and an irregular silhouette a clean dome doesn't.
lap_var (Laplacian variance of grayscale, inside the joint's own
silhouette): counter to the naive "cold joint = visually grainy = high-
frequency" guess, the REAL bad example measured LOWER than every real
good example - a clean dome's sharp curved-metal specular facets
produce more local contrast than the bad joint's duller, more diffusely
lit surface did. Trust the measurement, not the guess.
dark_frac: fraction of the joint's own silhouette that's near-black
(voids, oxidation, deep shadow in a crevice) - the bad example had 4-8x
more of this than any good one.
"""
from __future__ import annotations

import cv2
import numpy as np

# See module docstring for the real-photo evidence these were set from.
METAL_V_THRESHOLD = 120
MIN_COMPONENT_AREA_PX = 300
SOLIDITY_MIN = 0.75
CIRCULARITY_MIN = 0.35
LAP_VAR_MIN = 1800.0
DARK_FRAC_MAX = 0.02
# Below this bbox diagonal (px), a joint is treated as "too small to
# grade" rather than run through the shape/texture checks below - the
# calibration evidence covers a joint filling ~300-840px (dedicated
# macro close-ups) and separately a whole-plate photo where every real,
# confirmed-good joint measured only ~15-70px and EVERY ONE of them got
# wrongly flagged suspect (verified 2026-08-24: solidity/circularity/
# lap_var all degrade toward noise once there are only a few dozen px of
# actual joint to measure - not a quality signal at that scale, just
# insufficient resolution). This floor is a rough interpolation between
# those two known points, not itself independently measured - there's no
# real photo yet at, say, 150-250px to confirm exactly where grading
# starts being reliable rather than just no-longer-obviously-broken.
MIN_GRADEABLE_BBOX_DIAG_PX = 150
# component's own bounding-box diagonal is treated as a satellite/solder-
# ball defect, not an unrelated object elsewhere in the crop - verified
# against the real bad example, where the satellite sat well within this
# reach of the main joint.
SATELLITE_REACH_DIAGONALS = 2.5
# ...but only below this fraction of the main component's own area - a
# real solder-ball satellite is a small fragment (the real bad example's
# was 0.6% of its joint's area), whereas a joint crop that also catches
# part of the connector's own plastic housing above it produces a much
# BIGGER bright blob nearby (24-47% of the joint's area on 3 real good
# crops) that isn't a solder defect at all, just an imperfectly centered
# crop. This ratio is what tells the two apart.
SATELLITE_MAX_AREA_FRACTION = 0.15
# ...and at least this big in absolute terms - verified against a real
# good crop that a small incidental reflective fleck elsewhere in a busy
# real PCB background (a via edge, unrelated to the joint at all) is
# only 350-400px, well under the real bad example's actual 1212px
# satellite ball. Below this floor there's too little evidence in either
# direction to call it a defect rather than scene clutter.
SATELLITE_MIN_AREA_PX = 700


def _component_metrics(labels: np.ndarray, v: np.ndarray, gray: np.ndarray, index: int) -> dict:
    ys, xs = np.where(labels == index)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    mask = labels[y0 : y1 + 1, x0 : x1 + 1] == index
    area = int(mask.sum())

    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0

    lap = cv2.Laplacian(gray[y0 : y1 + 1, x0 : x1 + 1], cv2.CV_64F)
    lap_var = float(lap[mask].var()) if mask.any() else 0.0
    dark_frac = float((v[y0 : y1 + 1, x0 : x1 + 1][mask] < 60).sum()) / max(1, area)

    bbox_diag = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
    cx, cy = x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2
    return {
        "area": area,
        "solidity": solidity,
        "circularity": circularity,
        "lap_var": lap_var,
        "dark_frac": dark_frac,
        "bbox_diag": bbox_diag,
        "center": (cx, cy),
    }


def assess_joint_quality(crop_bgr: np.ndarray) -> dict:
    """Grade ONE joint's already-cropped, close-up, in-focus image as
    good/suspect and say why - see module docstring for how these
    thresholds were set and how little evidence backs them so far.

    Expects a crop centered on a single joint (e.g. from count_solder_
    joints' (x,y) plus a small margin) - a crop containing several
    joints, or a wide shot with a lot of background, will have its
    largest metallic component picked as "the joint" and everything
    else ignored, which is unlikely to be what's wanted.

    Returns {verdict, reasons: [...], metrics: {...}}. verdict is
    "good", "suspect", or "no_joint_found" (no metallic region at all
    in the crop) - with a "_low_confidence" suffix on good/suspect when
    the joint measured under MIN_GRADEABLE_BBOX_DIAG_PX in THIS crop (see
    that constant's own comment - still scored, not skipped, but not
    proven meaningful at that size yet either). reasons lists every rule
    that fired, not just the first one, so an operator can see the
    actual evidence.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2]

    metal = (v >= METAL_V_THRESHOLD).astype(np.uint8) * 255
    metal = cv2.morphologyEx(metal, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(metal, connectivity=8)

    components = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA_PX]
    if not components:
        return {"verdict": "no_joint_found", "reasons": ["no metallic region found in crop"], "metrics": {}}

    components.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    main_index = components[0]
    main = _component_metrics(labels, v, gray, main_index)

    # Below MIN_GRADEABLE_BBOX_DIAG_PX the checks below are run anyway
    # rather than skipped outright - 2026-08-24: with only a resolution
    # ESTIMATE for the actual production camera (not a real photo at
    # that resolution yet), refusing to score at all leaves nothing to
    # look at today. low_confidence marks the verdict as unproven at
    # this size rather than hiding it - a real photo at the eventual
    # production resolution/framing is still what actually answers
    # whether these numbers mean anything at that scale, not this flag.
    low_confidence = main["bbox_diag"] < MIN_GRADEABLE_BBOX_DIAG_PX

    reasons = []
    if main["solidity"] < SOLIDITY_MIN:
        reasons.append(f"irregular silhouette (solidity {main['solidity']:.2f} < {SOLIDITY_MIN})")
    if main["circularity"] < CIRCULARITY_MIN:
        reasons.append(f"not dome-shaped (circularity {main['circularity']:.2f} < {CIRCULARITY_MIN})")
    if main["lap_var"] < LAP_VAR_MIN:
        reasons.append(f"lacks a clean dome's sharp specular facets (texture {main['lap_var']:.0f} < {LAP_VAR_MIN:.0f})")
    if main["dark_frac"] > DARK_FRAC_MAX:
        reasons.append(f"voids/oxidation ({main['dark_frac']:.1%} dark pixels > {DARK_FRAC_MAX:.0%})")

    for other_index in components[1:]:
        other = _component_metrics(labels, v, gray, other_index)
        dist = float(np.hypot(*(np.array(other["center"]) - np.array(main["center"]))))
        close_enough = dist <= SATELLITE_REACH_DIAGONALS * main["bbox_diag"]
        right_sized = SATELLITE_MIN_AREA_PX <= other["area"] <= SATELLITE_MAX_AREA_FRACTION * main["area"]
        if close_enough and right_sized:
            reasons.append(f"separate solder blob nearby ({other['area']}px, {dist:.0f}px away) - possible solder ball")
            break  # one satellite is enough to flag; no need to enumerate every fleck

    verdict = "suspect" if reasons else "good"
    if low_confidence:
        verdict = f"{verdict}_low_confidence"
        reasons.append(
            f"low confidence: joint is only {main['bbox_diag']:.0f}px across in this crop "
            f"(<{MIN_GRADEABLE_BBOX_DIAG_PX:.0f}px) - not yet validated to mean anything at this size"
        )
    metrics = {k: v for k, v in main.items() if k != "center"} | {"center": list(main["center"])}
    return {"verdict": verdict, "reasons": reasons, "metrics": metrics}
