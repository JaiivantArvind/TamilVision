# -*- coding: utf-8 -*-
"""
scripts/sanity_check.py
=======================
Validates the CLASS_MAPPING and model checkpoint independently of any drawing.

Usage (from project root)
-------------------------
    py -3.12 scripts/sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLASS_MAPPING, DEVICE, NUM_CLASSES, TAMIL_CLASSES  # noqa: E402
from model import TamilVision                                              # noqa: E402

CHAR_TO_CHECK = "ர"


def run_sanity_check() -> None:
    print("\n" + "=" * 50)
    print("  MODEL SANITY CHECK")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. Verify CLASS_MAPPING entry for the target character
    # ------------------------------------------------------------------
    found_id: int | None = None
    for k, v in CLASS_MAPPING.items():
        if v == CHAR_TO_CHECK:
            found_id = k
            break

    if found_id is not None:
        print(f"[OK] '{CHAR_TO_CHECK}' found in CLASS_MAPPING → index {found_id}")
    else:
        print(f"[FAIL] '{CHAR_TO_CHECK}' NOT found in CLASS_MAPPING values.")
        print("       First 10 entries:")
        for k, v in list(CLASS_MAPPING.items())[:10]:
            print(f"         '{k}' → '{v}'")
        return

    # ------------------------------------------------------------------
    # 2. Cross-check TAMIL_CLASSES list directly
    # ------------------------------------------------------------------
    try:
        list_idx = TAMIL_CLASSES.index(CHAR_TO_CHECK)
        print(f"[OK] TAMIL_CLASSES[{list_idx}] = '{TAMIL_CLASSES[list_idx]}'")
    except ValueError:
        print(f"[FAIL] '{CHAR_TO_CHECK}' not found in TAMIL_CLASSES list.")

    # Print neighbours so we can visually confirm ordering
    print(f"\n  Neighbours in TAMIL_CLASSES around index {list_idx}:")
    for i in range(max(0, list_idx - 3), min(len(TAMIL_CLASSES), list_idx + 4)):
        marker = "  <<<" if i == list_idx else ""
        print(f"    [{i:>3}] {TAMIL_CLASSES[i]}{marker}")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print()
    if not MODEL_PATH.exists():
        print(f"[FAIL] Checkpoint not found: {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model = TamilVision(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        val_acc = checkpoint.get("val_acc", "N/A")
        print(f"[OK] Model loaded — val_acc in checkpoint: {val_acc}")
    except Exception as exc:
        print(f"[FAIL] Model loading failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 4. Load debug_best_view.png and run inference, printing Top-10
    # ------------------------------------------------------------------
    debug_img_path = PROJECT_ROOT / "debug_best_view.png"
    if not debug_img_path.exists():
        print(f"\n[SKIP] '{debug_img_path.name}' not found — skipping inference test.")
        print("       Run auto_tune.py first to generate it.")
    else:
        from PIL import Image
        img       = Image.open(debug_img_path).convert("L")
        arr       = np.array(img, dtype=np.float32) / 255.0
        arr       = (arr - 0.5) / 0.5
        tensor    = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs      = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            sorted_idx = np.argsort(probs)[::-1]

        target_rank = int(np.where(sorted_idx == found_id)[0][0]) + 1
        target_conf = probs[found_id] * 100

        print(f"\n  Inference on '{debug_img_path.name}':")
        print(f"  Target '{CHAR_TO_CHECK}' (label {found_id}) — rank #{target_rank}, conf {target_conf:.2f}%")
        print(f"\n  Top-10 predictions:")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            char   = CLASS_MAPPING.get(idx, "?")
            marker = "  <<<" if idx == found_id else ""
            print(f"    #{rank:>2}  [{idx:>3}] {char:<6}  {probs[idx]*100:.2f}%{marker}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    run_sanity_check()
