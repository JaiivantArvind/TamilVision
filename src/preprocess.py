# -*- coding: utf-8 -*-
"""
preprocess.py
=============
Robust preprocessing module for Tamil Character Recognition.

Provides:
  • ``ImagePreprocessor`` — class with train and validation transform pipelines.
  • ``process_image_for_inference`` — standalone function for single-image inference.

Usage
-----
    from preprocess import ImagePreprocessor, process_image_for_inference

    preprocessor = ImagePreprocessor()

    # In a Dataset.__getitem__
    train_tensor = preprocessor.get_train_transforms()(pil_image)
    valid_tensor = preprocessor.get_valid_transforms()(pil_image)

    # For a single inference call
    tensor = process_image_for_inference(image_bytes)  # (1, 1, 128, 128)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFilter

from config import IMG_SIZE

# Normalisation constants: maps [0, 1] → [-1, 1]
_MEAN = [0.5]
_STD  = [0.5]


class ImagePreprocessor:
    """Factory for torchvision transform pipelines.

    Instantiate once and call :meth:`get_train_transforms` or
    :meth:`get_valid_transforms` to obtain a ``Compose`` pipeline.
    """

    def get_train_transforms(self) -> T.Compose:
        """Return the augmented training pipeline.

        Augmentations applied (in order):
          1. Resize to ``IMG_SIZE × IMG_SIZE``
          2. RandomRotation ±20° — simulates tilted handwriting
          3. RandomAffine (translate 15 %, scale 80–120 %) — size / position variance
          4. RandomPerspective (p=0.5, distortion=0.3) — camera-angle variation
          5. ElasticTransform (p=0.5) — deforms strokes to cover different writing styles
          6. ToTensor + Normalize to [-1, 1]

        Returns:
            :class:`torchvision.transforms.Compose`
        """
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomRotation(degrees=20),                              # tilted writing
            T.RandomAffine(degrees=0, translate=(0.15, 0.15),
                           scale=(0.8, 1.2)),                          # size variance
            T.RandomPerspective(distortion_scale=0.3, p=0.5),         # camera angles
            T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),  # stroke shape (costly — apply sparingly)
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

    def get_valid_transforms(self) -> T.Compose:
        """Return the deterministic validation / test pipeline.

        No augmentation is applied — only resize and normalise.

        Returns:
            :class:`torchvision.transforms.Compose`
        """
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])


# ---------------------------------------------------------------------------
# Standalone inference helper
# ---------------------------------------------------------------------------

def process_image_for_inference(image_bytes) -> torch.Tensor:
    """Robust OpenCV preprocessing — handles canvas drawings, uploads, and transparent PNGs.

    Input types handled
    -------------------
    • **Canvas drawing** — white strokes on a solid black background.
    • **Uploaded image** — black strokes on a white background, any aspect
      ratio, possibly with JPG compression noise.
    • **Transparent PNG** — RGBA; detects stroke brightness and composites onto
      the opposite-colour background (black bg for white strokes, white bg for
      black strokes) so the character is always distinguishable.

    Pipeline
    --------
    1.  Decode with ``cv2.IMREAD_UNCHANGED`` — preserves the alpha channel.
        BGRA: composite onto white via alpha, then convert to grayscale.
        BGR: convert to grayscale directly.
    2.  Otsu's binary threshold — eliminates JPG noise, forces pure black/white.
    3.  Bulletproof background check — ``np.mean(img) > 127`` detects a
        predominantly light image and inverts it so the background is black
        and the character stroke is white.  Works for all input types.
    4.  Median blur (3x3) — removes isolated single-pixel JPG artefacts before
        bounding-box detection without blurring stroke edges.
    5.  Bounding-box crop — ``cv2.findNonZero`` + ``cv2.boundingRect`` finds the
        white strokes and discards surrounding black space.
    6.  Aspect-ratio-preserving square pad — pad the shorter axis with black
        pixels so the crop becomes a perfect square without stretching.
    7.  Uniform border padding (``_PAD`` px black) — ensures the glyph never
        touches the image edges after resizing.
    8.  Final invert (``cv2.bitwise_not``) — flip to black strokes on white
        background, matching the training data format.
    9.  Gaussian blur (3x3, sigma=1.5) — softens hard digital edges to match
        the scanned-ink texture of the training corpus.
    10. Resize to ``IMG_SIZE x IMG_SIZE`` (Lanczos).
    11. Debug save -> ``debug_last_inference.png``.
    12. Normalise to [-1, 1] and add batch + channel dims -> (1, 1, H, W).

    Args:
        image_bytes: File-like object (e.g. ``io.BytesIO``) containing the raw
                     image data sent by the frontend.

    Returns:
        Float32 tensor of shape ``(1, 1, IMG_SIZE, IMG_SIZE)`` in ``[-1, 1]``.

    Raises:
        ValueError: If the image cannot be decoded or the canvas is blank.
    """
    _PAD = 15    # uniform border (px) added around the squared glyph

    # 1. Decode → grayscale numpy array (preserve alpha channel if present)
    raw = np.frombuffer(image_bytes.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image — ensure the file is a valid PNG/JPG.")

    # Handle transparent PNGs (BGRA)
    # Detect whether strokes are light or dark, then composite onto the
    # opposite-colour background so strokes are always distinguishable.
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        gray  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        opaque = alpha > 0.5
        # Light strokes (white canvas) → composite onto black; dark strokes → onto white
        bg_val = 0.0 if (opaque.any() and np.mean(gray[opaque]) > 127) else 255.0
        img = (gray * alpha + bg_val * (1.0 - alpha)).astype(np.uint8)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Otsu's binary threshold — removes JPG noise, forces pure black/white
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Bulletproof background check — use overall mean brightness.
    #    If the image is predominantly light (uploaded scan / white bg), invert
    #    so both input types normalise to: white character on black background.
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    # img now has: black background, white character strokes

    # 4. Median blur — removes isolated single-pixel JPG artefacts before
    #    bounding-box detection without blurring stroke edges significantly.
    img = cv2.medianBlur(img, 3)

    # 5. Bounding-box crop — locate white (non-zero) stroke pixels
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("Canvas appears to be blank — no strokes detected.")
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]

    # 6. Aspect-ratio-preserving square pad — pad the shorter axis with black
    h, w    = img.shape
    max_dim = max(h, w)
    square  = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_off   = (max_dim - h) // 2
    x_off   = (max_dim - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = img
    img = square

    # 7. Uniform black border so the glyph never touches the image edges
    img = cv2.copyMakeBorder(
        img, _PAD, _PAD, _PAD, _PAD,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )

    # 8. Final invert → black strokes on white background (training data format)
    img = cv2.bitwise_not(img)

    # 9. Gaussian blur — softens hard digital edges to match scanned-ink texture
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.5)

    # 10. Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # 11. Debug: save exactly what the model will see
    _debug_path = Path(__file__).resolve().parent.parent / "debug_last_inference.png"
    cv2.imwrite(str(_debug_path), img)

    # 12. Normalise to [-1, 1] and shape to (1, 1, IMG_SIZE, IMG_SIZE)
    tensor = torch.from_numpy(img).float() / 255.0   # [0, 1]
    tensor = (tensor - _MEAN[0]) / _STD[0]           # [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(0)           # (1, 1, IMG_SIZE, IMG_SIZE)
