# -*- coding: utf-8 -*-
"""
create_readable_dataset.py
==========================
Copies the uTHCD numbered class folders (000–155) into a human-readable
directory structure named  uTHCD_readable/  where each folder is renamed to
the format:  <ID>_<TamilCharacter>   e.g.  000_அ, 001_ஆ, …, 155_நௌ

Mapping source
--------------
  src/config.py  →  TAMIL_CLASSES list (247 entries, first 156 = uTHCD IDs 0-155)

Usage
-----
  python scripts/create_readable_dataset.py [--src <path>] [--dst <path>]

  --src : Root that contains the numbered sub-folders 000..155.
          Defaults to the train-test-classwise/Train folder of the 70-30 split.
  --dst : Parent directory where uTHCD_readable/ will be created.
          Defaults to  data/raw/uTHCD_readable  relative to the repo root.
"""

import argparse
import shutil
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate repo root and extract TAMIL_CLASSES from src/config.py via ast
# (avoids executing `import torch` which may not be available)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → repo root

import ast as _ast

def _load_tamil_classes(config_path: Path) -> list:
    """Parse TAMIL_CLASSES from config.py without executing the module."""
    src = config_path.read_text(encoding="utf-8")
    tree = _ast.parse(src)
    # Collect all list literals assigned to _VOWELS, _AYUDHA, _CONSONANTS, _UYIRMEI
    lists: dict = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name) and t.id in (
                    "_VOWELS", "_AYUDHA", "_CONSONANTS", "_UYIRMEI"
                ):
                    lists[t.id] = _ast.literal_eval(node.value)
    return (
        lists["_VOWELS"]
        + lists["_AYUDHA"]
        + lists["_CONSONANTS"]
        + lists["_UYIRMEI"]
    )

_CONFIG_PATH = REPO_ROOT / "src" / "config.py"
TAMIL_CLASSES = _load_tamil_classes(_CONFIG_PATH)

# uTHCD uses only the first 156 labels (classes 0-155)
UTHCD_NUM_CLASSES = 156
UTHCD_CLASSES = TAMIL_CLASSES[:UTHCD_NUM_CLASSES]

# ---------------------------------------------------------------------------
# Helper: sanitise a Tamil string so it is safe as part of a folder name
# ---------------------------------------------------------------------------
_UNSAFE_CHARS = set(r'\/:*?"<>|')


def safe_folder_part(char: str) -> str:
    """Return a filesystem-safe version of a Tamil character string.

    Steps:
      1. NFC-normalise (canonical decomposition + canonical composition).
      2. Strip any characters that are illegal in Windows/Linux folder names.
      3. If the result is empty (shouldn't happen for Tamil), fall back to
         the hex codepoints joined with '+'.
    """
    normalised = unicodedata.normalize("NFC", char)
    cleaned = "".join(c for c in normalised if c not in _UNSAFE_CHARS)
    if not cleaned:
        cleaned = "+".join(f"U{ord(c):04X}" for c in normalised)
    return cleaned


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_SRC = (
    REPO_ROOT
    / "data" / "raw"
    / "uTHCD_a(70-30-split)" / "70-30-split"
    / "train-test-classwise" / "Train"
)
DEFAULT_DST = REPO_ROOT / "data" / "raw" / "uTHCD_readable"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy uTHCD numbered folders into readable Tamil-named folders."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source root containing folders 000–155. Default: {DEFAULT_SRC}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination root (uTHCD_readable will be created here). Default: {DEFAULT_DST}",
    )
    args = parser.parse_args()

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()

    if not src_root.exists():
        print(f"[ERROR] Source directory not found:\n  {src_root}", file=sys.stderr)
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Source : {src_root}")
    print(f"Dest   : {dst_root}")
    print(f"Classes: {UTHCD_NUM_CLASSES}  (IDs 0 – {UTHCD_NUM_CLASSES - 1})\n")

    skipped, copied, missing = 0, 0, 0

    for class_id, tamil_char in enumerate(UTHCD_CLASSES):
        folder_id = f"{class_id:03d}"               # zero-padded: "000", "001", …
        safe_char = safe_folder_part(tamil_char)
        new_name  = f"{folder_id}_{safe_char}"      # e.g. "000_அ"

        src_folder = src_root / folder_id
        dst_folder = dst_root / new_name

        if not src_folder.exists():
            print(f"  [MISSING] {folder_id}  →  {new_name}  (source not found)")
            missing += 1
            continue

        if dst_folder.exists():
            print(f"  [SKIP]    {folder_id}  →  {new_name}  (already exists)")
            skipped += 1
            continue

        shutil.copytree(src_folder, dst_folder)
        print(f"  [OK]      {folder_id}  →  {new_name}  ({len(list(dst_folder.iterdir()))} files)")
        copied += 1

    print(
        f"\nDone.  Copied: {copied}  |  Skipped: {skipped}  |  Missing: {missing}"
    )


if __name__ == "__main__":
    main()
