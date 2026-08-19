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


def _log_result(image_path: str, response: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        **response,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
