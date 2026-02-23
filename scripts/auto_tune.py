# -*- coding: utf-8 -*-
"""
scripts/auto_tune.py
====================
Advanced diagnostic grid search — finds the best preprocessing combination
by tracking the *rank* of the target character across all 156 classes.

Usage (from project root)
-------------------------
    py -3.12 scripts/auto_tune.py

Output
------
  • Live table showing predicted label, confidence, and TARGET rank per combo.
  • Saves ``debug_best_view.png`` for the combo where target ranked highest.
  • Saves ``debug_winner.png`` if target reaches rank #1.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLASS_MAPPING, DEVICE, NUM_CLASSES  # noqa: E402
from model import TamilVision                              # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_LABEL_ID: int = 47                               # 'ர' (Ra) — index in 156-class system
TEST_IMAGE: str      = str(PROJECT_ROOT / "debug_last_inference.png")
MODEL_PATH           = PROJECT_ROOT / "models" / "best_model.pth"

# ---------------------------------------------------------------------------
# Search grid (focused: thicker strokes, moderate padding)
# ---------------------------------------------------------------------------
BLURS     = [0, 0.5, 1.0]
DILATIONS = [1, 2, 3, 4]
PADDINGS  = [40, 60, 80]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model() -> TamilVision:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = TamilVision(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image_path: str, blur: float, dilation: int, padding: int) -> Image.Image:
    # 1. Load grayscale via cv2
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 2. Auto-invert: white text on black background
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # 3. Dilation
    if dilation > 0:
        kernel    = np.ones((3, 3), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=dilation)

    img = Image.fromarray(img_array)

    # 4. Smart crop
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 5. Square padding
    w, h    = img.size
    max_dim = max(w, h) + padding
    padded  = Image.new("L", (max_dim, max_dim), 0)
    padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

    # 6. Resize
    final = padded.resize((128, 128), Image.Resampling.LANCZOS)

    # 7. Blur
    if blur > 0:
        final = final.filter(ImageFilter.GaussianBlur(radius=blur))

    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grid_search() -> None:
    if not Path(TEST_IMAGE).exists():
        print(f"[ERROR] Test image not found: {TEST_IMAGE}")
        print("  Run a prediction first to generate debug_last_inference.png")
        sys.exit(1)

    model = load_model()
    target_char = CLASS_MAPPING.get(TARGET_LABEL_ID, "?")
    total = len(BLURS) * len(DILATIONS) * len(PADDINGS)

    print(f"\n{'='*80}")
    print(f"  Tamil Preprocessing Diagnostic Grid Search")
    print(f"  Target : {target_char} (label {TARGET_LABEL_ID})")
    print(f"  Grid   : {total} combinations")
    print(f"{'='*80}")
    print(f"\n{'Blur':<6} {'Dil':<4} {'Pad':<4} | {'Pred':<6} {'Conf%':<7} | "
          f"{'Target Rank':<13} {'Target Conf%':<13} | Top-3 Guesses")
    print("-" * 80)

    best_rank   = 999
    best_config = None

    for blur, dil, pad in itertools.product(BLURS, DILATIONS, PADDINGS):
        img = preprocess(TEST_IMAGE, blur, dil, pad)

        tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs       = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            top5_idx    = np.argsort(probs)[::-1][:5]
            sorted_all  = np.argsort(probs)[::-1]

        target_rank = int(np.where(sorted_all == TARGET_LABEL_ID)[0][0]) + 1
        target_conf = probs[TARGET_LABEL_ID] * 100
        pred_label  = top5_idx[0]
        pred_conf   = probs[pred_label] * 100
        pred_char   = CLASS_MAPPING.get(pred_label, "?")

        top3 = "  ".join(
            f"{CLASS_MAPPING.get(i,'?')}({int(probs[i]*100)}%)"
            for i in top5_idx[:3]
        )

        print(f"{blur:<6} {dil:<4} {pad:<4} | {pred_char:<6} {int(pred_conf):<7} | "
              f"#{target_rank:<12} {target_conf:<13.1f} | {top3}")

        if target_rank < best_rank:
            best_rank   = target_rank
            best_config = (blur, dil, pad)
            img.save(PROJECT_ROOT / "debug_best_view.png")

        if target_rank == 1:
            img.save(PROJECT_ROOT / "debug_winner.png")
            print(f"\n\033[92mSUCCESS! Blur={blur}, Dil={dil}, Pad={pad} "
                  f"→ {target_char} at {target_conf:.1f}%\033[0m")
            break

    print(f"\n{'='*80}")
    print(f"  Best rank achieved : #{best_rank}")
    print(f"  Best config        : blur={best_config[0]}, dilation={best_config[1]}, padding={best_config[2]}")
    print(f"  Check debug_best_view.png to see what the model saw.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_grid_search()

