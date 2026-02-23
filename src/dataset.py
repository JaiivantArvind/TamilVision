# -*- coding: utf-8 -*-
"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for Tamil character recognition.

Usage
-----
    from dataset import get_data_loaders

    train_loader, val_loader = get_data_loaders(
        data_dir   = "data/processed",
        batch_size = 32,
        val_split  = 0.1,
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import FOLDER_TO_CLASS, BATCH_SIZE, IMG_SIZE
from preprocess import ImagePreprocessor


class TamilDataset(Dataset):
    """Dataset for the uTHCD Tamil handwritten character corpus.

    Expected directory layout::

        root_dir/
            அ/          ← Tamil character string as folder name
                0001_0.bmp
                ...
            ஆ/
            ...

    Supports both ``.bmp`` (uTHCD raw) and ``.png`` images.

    Args:
        root_dir:     Path to the train or test split directory.
        transform:    Optional callable applied to each PIL image before it is
                      returned.  When ``None`` the raw PIL image is returned.
        cache_in_ram: If ``True`` (default), load every image into a shared-memory
                      ``torch.uint8`` tensor on first use and save a ``.pt`` cache
                      file beside the split directory so subsequent runs skip
                      the disk scan entirely.
    """

    def __init__(self, root_dir: str | Path, transform=None, cache_in_ram: bool = True) -> None:
        self.root_dir: Path = Path(root_dir)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir.resolve()}"
            )

        # Collect every BMP/PNG under root_dir (one level: root/label/file)
        self.image_paths: list[Path] = sorted(
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in {".bmp", ".png"} and p.is_file()
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No .bmp/.png files found under {self.root_dir.resolve()}."
            )

        # Folder name is Tamil character string → integer class index
        self.labels: list[int] = [
            FOLDER_TO_CLASS[path.parent.name] for path in self.image_paths
        ]

        # ── Shared-memory cache ──────────────────────────────────────────
        # Images are stored as a (N, IMG_SIZE, IMG_SIZE) uint8 tensor in
        # shared memory so ALL DataLoader worker processes read the same
        # physical RAM pages — zero memory duplication, no pickling overhead.
        # A .pt file next to the split dir avoids rebuilding on every run.
        self._cache: torch.Tensor | None = None
        if cache_in_ram:
            cache_file = self.root_dir.parent / f".cache_{self.root_dir.name}.pt"
            loaded = False
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, weights_only=True)
                    if len(cached) == len(self.image_paths):
                        self._cache = cached
                        print(f"  [{self.root_dir.name}] Cache loaded  ({len(self._cache):,} images)")
                        loaded = True
                    else:
                        print(f"  [{self.root_dir.name}] Cache stale — rebuilding …")
                except Exception:
                    print(f"  [{self.root_dir.name}] Cache unreadable — rebuilding …")

            if not loaded:
                self._cache = self._build_cache()
                torch.save(self._cache, cache_file)
                print(f"  [{self.root_dir.name}] Cache saved → {cache_file.name}")

            # Move into shared memory: workers access the same physical pages,
            # no per-worker copy is made when the dataset is pickled by DataLoader.
            self._cache.share_memory_()

    def _build_cache(self) -> torch.Tensor:
        """Load every image in parallel and return a (N, H, W) uint8 tensor."""
        def _load(p: Path) -> np.ndarray:
            return np.array(
                Image.open(p).convert("L").resize(
                    (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )

        print(f"  [{self.root_dir.name}] Building cache ({len(self.image_paths):,} images) …")
        with ThreadPoolExecutor(max_workers=8) as pool:
            arrays = list(tqdm(
                pool.map(_load, self.image_paths),
                total=len(self.image_paths),
                desc=f"    {self.root_dir.name}",
                unit="img",
                leave=False,
            ))
        return torch.from_numpy(np.stack(arrays))  # (N, IMG_SIZE, IMG_SIZE) uint8

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._cache is not None:
            # Convert shared-memory uint8 slice back to PIL (zero-copy view → numpy → PIL)
            image = Image.fromarray(self._cache[idx].numpy())
        else:
            image = Image.open(self.image_paths[idx]).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[idx]

    # ------------------------------------------------------------------
    # Human-readable helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TamilDataset(root='{self.root_dir}', "
            f"images={len(self)}, "
            f"classes={len(set(self.labels))})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_data_loaders(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation ``DataLoader`` objects from pre-split dirs.

    The uTHCD dataset is already split into train and test folders; no
    random splitting is performed here.

    Args:
        train_dir:   Root directory of the training split.
        val_dir:     Root directory of the validation/test split.
        batch_size:  Images per mini-batch (default: ``BATCH_SIZE`` from config).
        num_workers: Parallel workers for data loading (default: 0).
        pin_memory:  Pin host memory for faster CPU→GPU transfers (default: True).

    Returns:
        ``(train_loader, val_loader)`` — both are
        :class:`torch.utils.data.DataLoader` instances.

    Raises:
        FileNotFoundError: If either directory does not exist.
        RuntimeError: If no images are found inside a directory.
    """
    preprocessor = ImagePreprocessor()

    train_dataset = TamilDataset(root_dir=train_dir, transform=preprocessor.get_train_transforms())
    val_dataset   = TamilDataset(root_dir=val_dir,   transform=preprocessor.get_valid_transforms())

    pin = pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = pin,
        drop_last          = True,
        persistent_workers = num_workers > 0,
        prefetch_factor    = 4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin,
        persistent_workers = num_workers > 0,
        prefetch_factor    = 4 if num_workers > 0 else None,
    )

    print(
        f"Dataset loaded — train: {len(train_dataset):,}  "
        f"val: {len(val_dataset):,}  "
        f"batch: {batch_size}  workers: {num_workers}"
    )

    return train_loader, val_loader
