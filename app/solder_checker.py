"""Solder joint quality checker (Step 5 of the plan).

Placeholder until enough good-joint images exist to train an anomaly
detector (recommended: PatchCore via Anomalib, see spec.json thresholds).
Kept structurally identical to what the trained version will return, so
main.py never needs to change when the real model is dropped in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "solder_patchcore.ckpt"


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
