"""Combines the wire check and solder check into one backend response (Step 6).

This function is the one piece of logic shared unchanged between phase 1
(folder read) and phase 2 (live camera + HTTP endpoint) — only the caller
changes between phases, never this.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.harness_photo import check_harness_photo, count_harness_solder_joints
from app.solder_checker import check_solder
from app.wire_checker import check_wires

SPEC_PATH = Path(__file__).resolve().parent.parent / "spec.json"
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "results.jsonl"


def _load_thresholds() -> dict[str, float]:
    with open(SPEC_PATH) as f:
        return json.load(f)["confidence_thresholds"]


def classify(confidence: float, thresholds: dict[str, float]) -> str:
    if confidence >= thresholds["pass_at_or_above"]:
        return "pass"
    if confidence < thresholds["fail_below"]:
        return "fail"
    return "uncertain"


def inspect_image(image_path: str, unit_id: str, harness_type: str = "default") -> dict[str, Any]:
    thresholds = _load_thresholds()
    defects: list[str] = []

    try:
        wire_result = check_wires(image_path, harness_type=harness_type)
    except (RuntimeError, FileNotFoundError) as exc:
        wire_result = {"pass": None, "confidence": 0.0, "wires": [], "error": str(exc)}
    if wire_result["pass"] is False:
        defects.extend(
            f"wire slot {w['slot']}: expected {w['expected_color']}, got {w['detected_color']}"
            for w in wire_result.get("wires", [])
            if not w["pass"]
        )
        wire_count = wire_result.get("wire_count")
        if wire_count and wire_count["enforced"] and not wire_count["match"]:
            defects.append(f"wire count mismatch: found {wire_count['found']}, expected {wire_count['expected']}")

    solder_result = check_solder(image_path)
    if solder_result.get("status") == "untrained":
        defects.append("solder check untrained: no result available yet")

    scored = [r["confidence"] for r in (wire_result, solder_result) if r.get("confidence") is not None]
    combined_confidence = sum(scored) / len(scored) if scored else 0.0

    result = classify(combined_confidence, thresholds)
    if wire_result.get("pass") is None or solder_result.get("pass") is None:
        result = "uncertain"

    response = {
        "unit_id": unit_id,
        "result": result,
        "confidence": round(combined_confidence, 3),
        "wire_check": wire_result,
        "solder_check": solder_result,
        "defects": defects,
    }

    _log_result(image_path, response)
    return response


def inspect_harness_photos(
    upper_image_path: str,
    bottom_image_path: str,
    unit_id: str,
    upper_harness_type: str = "power_connector",
    bottom_harness_type: str = "default",
) -> dict[str, Any]:
    """Two-photo counterpart to inspect_image: one dedicated close-up
    photo per connector (an "upper" photo for the harness whose wires
    exit upward, a "bottom" photo for the one whose wires hang down)
    instead of one whole-plate photo split into strips after
    rectification (see app/harness_photo.py for why no plate/grommet
    detection or perspective warp is needed here). Solder-joint QUALITY
    is still the same untrained PatchCore-shaped placeholder (check_solder,
    Step 5 in the plan) inspect_image uses - only joint count/location
    (count_harness_solder_joints) is a real, working check here.
    """
    thresholds = _load_thresholds()
    defects: list[str] = []

    photos = [
        ("upper", upper_image_path, upper_harness_type),
        ("bottom", bottom_image_path, bottom_harness_type),
    ]
    harness_results: dict[str, Any] = {}
    for label, path, harness_type in photos:
        try:
            wire_result = check_harness_photo(path, harness_type=harness_type)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            wire_result = {"pass": None, "confidence": 0.0, "wires": [], "error": str(exc)}
        if wire_result["pass"] is False:
            defects.extend(
                f"{label} ({harness_type}) wire slot {w['slot']}: expected {w['expected_color']}, got {w['detected_color']}"
                for w in wire_result.get("wires", [])
                if not w["pass"]
            )
            wire_count = wire_result.get("wire_count")
            if wire_count and wire_count["enforced"] and not wire_count["match"]:
                defects.append(
                    f"{label} wire count mismatch: found {wire_count['found']}, expected {wire_count['expected']}"
                )

        try:
            joint_result = count_harness_solder_joints(path, harness_type=harness_type)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            joint_result = {"count": 0, "joints": [], "error": str(exc)}

        harness_results[label] = {
            "harness_type": harness_type,
            "wire_check": wire_result,
            "joint_check": joint_result,
        }

    solder_result = check_solder(upper_image_path)
    if solder_result.get("status") == "untrained":
        defects.append(
            "solder quality check untrained: no result available yet "
            "(joint count/location above is separate and working)"
        )

    scored = [
        r["wire_check"]["confidence"] for r in harness_results.values() if r["wire_check"].get("confidence") is not None
    ]
    combined_confidence = sum(scored) / len(scored) if scored else 0.0
    result = classify(combined_confidence, thresholds)
    if any(r["wire_check"].get("pass") is None for r in harness_results.values()):
        result = "uncertain"

    response = {
        "unit_id": unit_id,
        "result": result,
        "confidence": round(combined_confidence, 3),
        "upper": harness_results["upper"],
        "bottom": harness_results["bottom"],
        "defects": defects,
    }

    _log_result(f"{upper_image_path} | {bottom_image_path}", response)
    return response


def _log_result(image_path: str, response: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        **response,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
