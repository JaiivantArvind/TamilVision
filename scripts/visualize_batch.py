# -*- coding: utf-8 -*-
"""
visualize_batch.py
==================
Quick sanity-check: loads one training batch, renders a grid, saves it to
``debug_batch.png``, and prints the first 5 labels with their Tamil characters.

Usage
-----
    python scripts/visualize_batch.py
"""

import sys
from pathlib import Path

# Allow imports from src/ regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import torchvision.utils as vutils

from config import CLASS_MAPPING
from dataset import get_data_loaders

TRAIN_DIR  = PROJECT_ROOT / "data" / "raw" / "uTHCD_b(80-20-split)" / "80-20-split" / "train-test-classwise" / "train"
TEST_DIR   = PROJECT_ROOT / "data" / "raw" / "uTHCD_b(80-20-split)" / "80-20-split" / "train-test-classwise" / "test"
OUTPUT_IMG = PROJECT_ROOT / "debug_batch.png"


def main() -> None:
    # -----------------------------------------------------------------------
    # Load one batch
    # -----------------------------------------------------------------------
    train_loader, _ = get_data_loaders(str(TRAIN_DIR), str(TEST_DIR), batch_size=32)

    images, labels = next(iter(train_loader))  # (32, 1, 128, 128)

    # -----------------------------------------------------------------------
    # Build image grid
    # -----------------------------------------------------------------------
    grid = vutils.make_grid(images, nrow=8, padding=4, normalize=False)

    # Un-normalise:  [-1, 1]  →  [0, 1]
    grid = grid * 0.5 + 0.5

    # Tensor (C, H, W) → numpy (H, W, C) for matplotlib
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # -----------------------------------------------------------------------
    # Plot and save
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(grid_np, cmap="gray")
    ax.axis("off")
    ax.set_title("Training batch — 32 Tamil characters (un-normalised)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_IMG), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid → {OUTPUT_IMG}")

    # -----------------------------------------------------------------------
    # Print first 5 labels + Tamil characters
    # -----------------------------------------------------------------------
    print("\nFirst 5 samples in batch:")
    print(f"  {'Idx':>4}  {'Label':>6}  Tamil")
    print("  " + "-" * 20)
    for i in range(5):
        lbl   = labels[i].item()
        tamil = CLASS_MAPPING.get(lbl, "?")
        print(f"  {i:>4}  {lbl:>6}  {tamil}")


if __name__ == "__main__":
    main()
