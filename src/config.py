# -*- coding: utf-8 -*-
"""
Tamil Character Recognition — Configuration
============================================
Defines the full 156-character uTHCD Tamil label space, model
hyper-parameters, and convenience label-lookup helpers.
"""

import torch

# ---------------------------------------------------------------------------
# 156 uTHCD classes — ordered to match the official dataset groupings
# ---------------------------------------------------------------------------

# 13: Vowels + Aytham  (உயிர் எழுத்துக்கள் + ஆயுத எழுத்து)
_VOWELS_AYTHAM = [
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ',
    'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'ஃ',
]

# 23: Pure consonants / Mei  (மெய் எழுத்துக்கள் — includes Grantha)
_PURE_CONSONANTS = [
    'க்', 'ங்', 'ச்', 'ஞ்', 'ட்', 'ண்', 'த்', 'ந்', 'ப்', 'ம்', 'ய்', 'ர்',
    'ல்', 'வ்', 'ழ்', 'ள்', 'ற்', 'ன்', 'ஜ்', 'ஷ்', 'ஸ்', 'ஹ்', 'க்ஷ்',
]

# 23: Base consonants  (includes Grantha)
_BASE_CONSONANTS = [
    'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ',
    'ழ', 'ள', 'ற', 'ன', 'ஜ', 'ஷ', 'ஸ', 'ஹ', 'க்ஷ',
]

# 23: 'i' series / ி
_I_SERIES = [
    'கி', 'ஙி', 'சி', 'ஞி', 'டி', 'ணி', 'தி', 'நி', 'பி', 'மி', 'யி', 'ரி',
    'லி', 'வி', 'ழி', 'ளி', 'றி', 'னி', 'ஜி', 'ஷி', 'ஸி', 'ஹி', 'க்ஷி',
]

# 23: 'ii' series / ீ
_II_SERIES = [
    'கீ', 'ஙீ', 'சீ', 'ஞீ', 'டீ', 'ணீ', 'தீ', 'நீ', 'பீ', 'மீ', 'யீ', 'ரீ',
    'லீ', 'வீ', 'ழீ', 'ளீ', 'றீ', 'னீ', 'ஜீ', 'ஷீ', 'ஸீ', 'ஹீ', 'க்ஷீ',
]

# 23: 'u' series / ு
_U_SERIES = [
    'கு', 'ஙு', 'சு', 'ஞு', 'டு', 'ணு', 'து', 'நு', 'பு', 'மு', 'யு', 'ரு',
    'லு', 'வு', 'ழு', 'ளு', 'று', 'னு', 'ஜு', 'ஷு', 'ஸு', 'ஹு', 'க்ஷு',
]

# 23: 'uu' series / ூ
_UU_SERIES = [
    'கூ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ', 'ரூ',
    'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ', 'னூ', 'ஜூ', 'ஷூ', 'ஸூ', 'ஹூ', 'க்ஷூ',
]

# 5: Standalone vowel-marker / modifier shapes
_MODIFIERS = ['ா', 'ெ', 'ே', 'ை', 'ௌ']

# ---------------------------------------------------------------------------
# Master label list  — 13+23+23+23+23+23+23+5 = 156 entries
# ---------------------------------------------------------------------------
TAMIL_CLASSES: list[str] = (
    _VOWELS_AYTHAM
    + _PURE_CONSONANTS
    + _BASE_CONSONANTS
    + _I_SERIES
    + _II_SERIES
    + _U_SERIES
    + _UU_SERIES
    + _MODIFIERS
)

assert len(TAMIL_CLASSES) == 156, (
    f"Expected 156 Tamil classes, got {len(TAMIL_CLASSES)}"
)

# ---------------------------------------------------------------------------
# Folder-name → class-index mapping
# Tamil character strings are used directly as folder names in uTHCD.
# ---------------------------------------------------------------------------
FOLDER_TO_CLASS: dict[str, int] = {
    char: idx for idx, char in enumerate(TAMIL_CLASSES)
}

# ---------------------------------------------------------------------------
# Index → character mapping  (kept for inference / display use)
# ---------------------------------------------------------------------------
CLASS_MAPPING: dict[int, str] = {
    idx: char for idx, char in enumerate(TAMIL_CLASSES)
}

# ---------------------------------------------------------------------------
# Hyper-parameters & hardware
# ---------------------------------------------------------------------------
IMG_SIZE: int = 128
BATCH_SIZE: int = 256         # try 256; if VRAM OOM, drop to 128
NUM_CLASSES: int = 156
EPOCHS: int = 30
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------

def get_label(idx: int) -> str:
    """Return the Tamil character for a given class index (0–155).

    Args:
        idx: Integer class index produced by the model.

    Returns:
        The corresponding Tamil character string.

    Raises:
        IndexError: If *idx* is outside the valid range [0, NUM_CLASSES).
    """
    if not (0 <= idx < NUM_CLASSES):
        raise IndexError(
            f"Index {idx} is out of range. Valid range: 0 – {NUM_CLASSES - 1}."
        )
    return TAMIL_CLASSES[idx]
