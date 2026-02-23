# -*- coding: utf-8 -*-
"""
app/main.py
===========
Production-ready FastAPI application for TamilVision — 156-class Tamil
handwritten character recognition.

Endpoints
---------
  GET  /          → health-check / status
  POST /predict   → upload image as multipart/form-data, receive Top-3 Tamil
                    character predictions with actual character strings and
                    confidence scores.

Run (from project root)
-----------------------
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    (Use full path if needed: C:/Users/Savag/AppData/Local/Programs/Python/Python312/python.exe)
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import CLASS_MAPPING, DEVICE, NUM_CLASSES
from model import TamilVision
from preprocess import process_image_for_inference

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"

# ---------------------------------------------------------------------------
# Global model handle (populated at startup)
# ---------------------------------------------------------------------------
_model: TamilVision | None = None
_best_val_acc: str = "N/A"


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup (modern FastAPI pattern)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained TamilVision weights before the first request."""
    global _model, _best_val_acc

    if not CHECKPOINT_PATH.exists():
        print(
            f"[WARNING] Checkpoint not found at {CHECKPOINT_PATH}. "
            "The /predict endpoint will be unavailable until a model is trained."
        )
    else:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            _model = TamilVision(num_classes=NUM_CLASSES)
            _model.load_state_dict(checkpoint["model_state"])
            _model.to(DEVICE)
            _model.eval()
            val_acc = checkpoint.get("val_acc", None)
            _best_val_acc = f"{val_acc:.2f}%" if val_acc is not None else "N/A"
            print(f"[TamilVision] Model loaded — device: {DEVICE} | best val acc: {_best_val_acc}")
        except Exception as exc:
            print(
                f"[ERROR] Failed to load checkpoint: {exc}\n"
                "        The /predict endpoint will be unavailable. "
                "Delete models/best_model.pth and retrain if the checkpoint is stale."
            )
            _model = None

    yield  # application runs here

    # Teardown (optional cleanup)
    _model = None
    print("[TamilVision] Model unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "TamilVision API",
    description = "156-class Tamil handwritten character recognition",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health-check endpoint."""
    return {
        "status"  : "TamilVision API Online",
        "accuracy": _best_val_acc,
        "device"  : DEVICE,
        "classes" : NUM_CLASSES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the Tamil character in an uploaded image.

    Accepts any image format supported by PIL (PNG, JPG, BMP, …) sent as
    ``multipart/form-data`` with a ``file`` field.

    The preprocessing pipeline (in ``preprocess.process_image_for_inference``)
    handles the domain gap between the frontend canvas (white ink on black) and
    the training data (black ink on white):

    1. Grayscale conversion.
    2. Color inversion — white-on-black → black-on-white.
    3. Bounding-box crop to remove empty margins.
    4. Uniform white padding + resize to 128 × 128.
    5. Gaussian blur to match scanned-ink texture.
    6. Normalize to [-1, 1] and add batch dimension → [1, 1, 128, 128].

    Returns the Top-3 predictions ordered by confidence, each containing
    ``predicted_character`` (actual Tamil string), ``confidence``, and
    ``label_id``.
    """
    if _model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Train the model first (python src/train.py).",
        )

    # --- Read & preprocess ---
    try:
        raw_bytes = await file.read()
        tensor    = process_image_for_inference(io.BytesIO(raw_bytes))  # (1, 1, 128, 128)
        tensor    = tensor.to(DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    # --- Inference ---
    with torch.no_grad():
        logits               = _model(tensor)                         # (1, 156)
        probs                = F.softmax(logits, dim=1)               # (1, 156)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    # --- Build response ---
    predictions = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        predictions.append(
            {
                "predicted_character": CLASS_MAPPING.get(idx, "?"),
                "confidence"         : round(prob, 6),
                "label_id"           : idx,
            }
        )

    return {"predictions": predictions}


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
