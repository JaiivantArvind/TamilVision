# -*- coding: utf-8 -*-
"""
train.py
========
Robust training script for 156-class Tamil character recognition.

Features
--------
  • Mixed-precision training (AMP) via GradScaler — optimised for GTX 1650
  • AdamW + CosineAnnealingLR scheduler
  • CrossEntropyLoss with label smoothing (handles visually similar characters)
  • Top-1 and Top-3 validation accuracy
  • Best-model checkpoint saved to models/best_model.pth

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import BATCH_SIZE, DEVICE, EPOCHS, NUM_CLASSES
from dataset import get_data_loaders
from model import get_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data" / "raw"
    / "uTHCD_b(80-20-split)" / "80-20-split"
    / "train-test-classwise"
)
TRAIN_DIR       = _SPLIT_ROOT / "train"
TEST_DIR        = _SPLIT_ROOT / "test"
CHECKPOINT_DIR  = PROJECT_ROOT / "models"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
LR           = 1e-3   # AdamW does not benefit from linear LR scaling; keep original value
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3     # linear warm-up before cosine annealing


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> float:
    """Run one full training epoch.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, labels in pbar:
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed-precision forward pass
            with autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(images)
                loss   = criterion(logits, labels)

            # Scaled backward + optimiser step (with gradient clipping)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"  CUDA OUT OF MEMORY — batch size {loader.batch_size} is too large.\n"
                f"  Lower BATCH_SIZE in config.py (try {loader.batch_size // 2}).\n"
                f"{'='*60}"
            ) from None

    return running_loss / len(loader)


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float]:
    """Run validation and compute loss, Top-1 and Top-3 accuracy.

    Returns:
        ``(val_loss, top1_acc, top3_acc)`` — all as percentages (0–100).
    """
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    total        = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Valid", leave=False, unit="batch")
        for images, labels in pbar:
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device, enabled=(device == "cuda")):
                    logits = model(images)
                    loss   = criterion(logits, labels)

                running_loss += loss.item()

                # Top-1
                preds_top1 = logits.argmax(dim=1)
                top1_correct += (preds_top1 == labels).sum().item()

                # Top-3
                preds_top3 = logits.topk(k=min(3, NUM_CLASSES), dim=1).indices
                top3_correct += (preds_top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

                total += labels.size(0)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"  CUDA OUT OF MEMORY — batch size {loader.batch_size} is too large.\n"
                    f"  Lower BATCH_SIZE in config.py (try {loader.batch_size // 2}).\n"
                    f"{'='*60}"
                ) from None

    val_loss  = running_loss / len(loader)
    top1_acc  = 100.0 * top1_correct / total
    top3_acc  = 100.0 * top3_correct / total
    return val_loss, top1_acc, top3_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Tamil Character Recognition — Training")
    print("=" * 60)

    # --- Data ---
    train_loader, val_loader = get_data_loaders(
        train_dir   = str(TRAIN_DIR),
        val_dir     = str(TEST_DIR),
        batch_size  = BATCH_SIZE,
        num_workers = 8,
        pin_memory  = True,
    )

    # --- Model ---
    model = get_model(DEVICE)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    # --- Optimiser ---
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # --- Scheduler: 3-epoch linear warm-up → cosine anneal to 0 ---
    # Warm-up prevents the high initial LR from destroying pretrained weights.
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,          # start at LR * 0.1 = 1e-4
        end_factor   = 1.0,          # ramp to full LR
        total_iters  = WARMUP_EPOCHS,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    # --- AMP scaler (no-op on CPU) ---
    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # --- Training loop ---
    best_val_acc = 0.0
    print()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:>2}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        val_loss, top1_acc, top3_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        # --- Print epoch summary ---
        print(
            f"  Epoch {epoch:>2}/{EPOCHS}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Val Acc (Top-1): {top1_acc:.2f}%"
            f" | Val Acc (Top-3): {top3_acc:.2f}%"
            f" | LR: {current_lr:.2e}"
        )

        # --- Save best checkpoint ---
        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            torch.save(
                {
                    "epoch"     : epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_acc"   : top1_acc,
                    "val_loss"  : val_loss,
                },
                BEST_MODEL_PATH,
            )
            print(f"  ✓ New best model saved  (Val Acc: {top1_acc:.2f}%)")

        print()

    print("=" * 60)
    print(f"Training complete.  Best Val Acc: {best_val_acc:.2f}%")
    print(f"Checkpoint: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
