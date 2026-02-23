# -*- coding: utf-8 -*-
"""
scripts/inspect_data.py
=======================
Ground-truth inspector: pulls a real training image for a given label and
compares its pixel statistics + color convention against your debug drawing.

Usage (from project root)
-------------------------
    py -3.12 scripts/inspect_data.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLASS_MAPPING  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data" / "raw" / "uTHCD_b(80-20-split)" / "80-20-split" / "train-test-classwise" / "train"
TARGET_LABEL_ID = 47                                     # ர (Ra) — index in 156-class system


def inspect_ground_truth() -> None:
    target_char  = CLASS_MAPPING.get(TARGET_LABEL_ID, "?")
    folder_path  = DATA_DIR / target_char                # train split uses Tamil chars as folder names

    print("\n" + "=" * 55)
    print(f"  Ground Truth Inspector — label {TARGET_LABEL_ID} = '{target_char}'")
    print("=" * 55)
    print(f"  Dataset folder : {folder_path}")

    if not folder_path.exists():
        print(f"\n[FAIL] Folder not found: {folder_path}")
        print("       Edit DATA_DIR in this script to the correct path.")
        return

    images = sorted(folder_path.glob("*.bmp")) + sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpg"))
    if not images:
        print(f"[FAIL] No images found in {folder_path}")
        return

    # Pick a few samples for a more robust picture
    samples = random.sample(images, min(5, len(images)))
    print(f"\n  Total images in folder : {len(images)}")
    print(f"  Inspecting {len(samples)} random samples:\n")

    for img_path in samples:
        img      = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mean_val = float(np.mean(img))
        bg       = "WHITE bg / BLACK text" if mean_val > 127 else "BLACK bg / WHITE text"
        print(f"    {img_path.name:<30}  mean={mean_val:>6.1f}  →  {bg}")

    # Save one representative sample for visual comparison
    rep_img = cv2.imread(str(samples[0]), cv2.IMREAD_GRAYSCALE)
    out_path = PROJECT_ROOT / "debug_ground_truth.png"
    cv2.imwrite(str(out_path), rep_img)
    print(f"\n  [SAVED] Training sample → {out_path.name}")

    # Compare against debug_best_view.png if it exists
    debug_path = PROJECT_ROOT / "debug_best_view.png"
    if debug_path.exists():
        dbg      = cv2.imread(str(debug_path), cv2.IMREAD_GRAYSCALE)
        dbg_mean = float(np.mean(dbg))
        rep_mean = float(np.mean(rep_img))

        dbg_bg = "WHITE bg" if dbg_mean > 127 else "BLACK bg"
        rep_bg = "WHITE bg" if rep_mean > 127 else "BLACK bg"

        print(f"\n  Comparison:")
        print(f"    debug_best_view.png   mean={dbg_mean:>6.1f}  →  {dbg_bg}")
        print(f"    debug_ground_truth.png mean={rep_mean:>6.1f}  →  {rep_bg}")

        if (dbg_mean > 127) == (rep_mean > 127):
            print("\n  [OK] Both images share the same color convention.")
        else:
            print("\n  [MISMATCH] Color conventions differ!")
            print("             The model sees inverted images at inference.")
            print("             Fix: remove or flip the auto-invert logic in preprocess.py.")
    else:
        print(f"\n  [SKIP] '{debug_path.name}' not found — run auto_tune.py first.")

    print("\n  Open both PNGs side-by-side and check:")
    print("    1. Same background color?")
    print("    2. Does the training 'ர' shape match what you drew?")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    inspect_ground_truth()
