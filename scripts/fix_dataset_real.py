# -*- coding: utf-8 -*-
"""
fix_dataset_real.py
====================
Processes the Mendeley "tamil-handwritten-character-carpus" (2023) dataset.

Raw layout expected
-------------------
data/raw/
    1/       ← folder for class index 1  →  maps to label '000'
    2/       ← folder for class index 2  →  maps to label '001'
    ...
    247/     ← folder for class index 247 → maps to label '246'

Output layout produced
----------------------
data/processed/
    000/     ← Tamil character அ
    001/     ← Tamil character ஆ
    ...
    246/     ← Tamil character னௌ

Each image is:
  • Converted to grayscale
  • Binarised with Otsu's threshold (white character, black background)
  • Square-padded to preserve aspect ratio
  • Resized to 128 × 128 px
  • Saved as PNG

Usage
-----
    # Dry-run (no files written, just a report)
    python scripts/fix_dataset_real.py --dry-run

    # Full processing with 8 worker threads
    python scripts/fix_dataset_real.py --workers 8

    # Custom paths
    python scripts/fix_dataset_real.py --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import sys
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root so the script works from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLASS_MAPPING, IMG_SIZE, NUM_CLASSES  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image-processing helpers
# ---------------------------------------------------------------------------

def _square_pad(img: np.ndarray) -> np.ndarray:
    """Pad a grayscale image to a square by adding black borders."""
    h, w = img.shape[:2]
    side = max(h, w)
    pad_top    = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left   = (side - w) // 2
    pad_right  = side - w - pad_left
    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )


def process_image(
    src_path: Path,
    dst_path: Path,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Load, clean, and save a single image.

    Returns:
        (success, message) — message is non-empty only on failure.
    """
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Cannot read: {src_path}"

        # Otsu's binarisation — white character on black background
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Square-pad → resize
        padded  = _square_pad(binary)
        resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        if not dry_run:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_path), resized)

        return True, ""

    except Exception as exc:  # noqa: BLE001
        return False, f"{src_path} → {exc}"


# ---------------------------------------------------------------------------
# Dataset walker
# ---------------------------------------------------------------------------

def build_task_list(
    raw_dir: Path,
    out_dir: Path,
) -> list[tuple[Path, Path]]:
    """Collect (src_path, dst_path) pairs for every image in data/raw.

    Mendeley folders are named 1…247.  We map:
        folder N  →  label ID = str(N - 1).zfill(3)
    """
    tasks: list[tuple[Path, Path]] = []
    missing_folders: list[int] = []

    for folder_num in range(1, NUM_CLASSES + 1):
        label_id = str(folder_num - 1).zfill(3)
        src_folder = raw_dir / str(folder_num)

        if not src_folder.is_dir():
            missing_folders.append(folder_num)
            continue

        dst_folder = out_dir / label_id
        tamil_char = CLASS_MAPPING.get(label_id, "?")

        image_files = [
            p for p in src_folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]

        for src_img in image_files:
            dst_img = dst_folder / (src_img.stem + ".png")
            tasks.append((src_img, dst_img))

        log.debug("Folder %3d → '%s' (%s) : %d image(s)",
                  folder_num, label_id, tamil_char, len(image_files))

    if missing_folders:
        log.warning("Missing raw folders (skipped): %s", missing_folders)

    return tasks


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    raw_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
    workers: int = 4,
) -> None:
    log.info("Raw dir  : %s", raw_dir)
    log.info("Out dir  : %s", out_dir)
    log.info("Dry run  : %s", dry_run)
    log.info("Workers  : %d", workers)

    if not raw_dir.is_dir():
        log.error("Raw directory not found: %s", raw_dir)
        sys.exit(1)

    tasks = build_task_list(raw_dir, out_dir)
    if not tasks:
        log.error("No images found under %s", raw_dir)
        sys.exit(1)

    log.info("Total images to process: %d", len(tasks))

    success_count = 0
    failure_count = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_image, src, dst, dry_run): (src, dst)
            for src, dst in tasks
        }

        with tqdm(total=len(futures), unit="img", desc="Processing") as pbar:
            for future in as_completed(futures):
                ok, msg = future.result()
                if ok:
                    success_count += 1
                else:
                    failure_count += 1
                    failures.append(msg)
                pbar.update(1)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Processed  : %d", success_count)
    log.info("  Failed     : %d", failure_count)
    if dry_run:
        log.info("  (Dry-run — no files were written)")
    else:
        log.info("  Output dir : %s", out_dir)
    if failures:
        log.warning("Failed files:")
        for msg in failures[:20]:
            log.warning("  %s", msg)
        if len(failures) > 20:
            log.warning("  … and %d more.", len(failures) - 20)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Mendeley Tamil dataset → 128×128 PNGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw", type=Path,
        default=PROJECT_ROOT / "data" / "raw"
                / "tamil-handwritten-character-carpus"
                / "tamil-alphabets-raw"
                / "tamil-alphabets-raw",
        help="Path to raw dataset root (contains folders 1–247)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Destination directory for processed images",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and validate without writing any files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_dataset(
        raw_dir=args.raw,
        out_dir=args.out,
        dry_run=args.dry_run,
        workers=args.workers,
    )
