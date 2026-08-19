"""Phase 1 entry point: reads images from a folder instead of a live camera
(Step 2), runs the shared inspection pipeline, and prints/logs the result
(Steps 6-7). Swap this script for the camera + HTTP loop in phase 2 —
app/inspect.py itself does not change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inspect import inspect_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase 1 inspection over a folder of images.")
    parser.add_argument(
        "--folder",
        default="dataset/incoming",
        help="Folder of images to process (default: dataset/incoming)",
    )
    parser.add_argument("--harness-type", default="default")
    args = parser.parse_args()

    folder = Path(args.folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.exists() else []

    if not images:
        print(f"No images found in {folder}. Drop test photos there and re-run.")
        return

    for image_path in images:
        response = inspect_image(str(image_path), unit_id=image_path.stem, harness_type=args.harness_type)
        print(f"{image_path.name}: {response['result']} (confidence={response['confidence']})")
        if response["defects"]:
            for defect in response["defects"]:
                print(f"  - {defect}")


if __name__ == "__main__":
    main()
