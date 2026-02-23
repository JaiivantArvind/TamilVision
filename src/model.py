# -*- coding: utf-8 -*-
"""
model.py
========
Custom MobileNetV3-Small adapted for 156-class Tamil character recognition.

Key modifications over the stock ImageNet model:
  1. First conv layer changed from 3-channel (RGB) → 1-channel (grayscale)
     by summing the pretrained RGB weights across the channel dimension,
     preserving the learned low-level feature detectors.
  2. Final classifier head replaced with a new Linear(1024 → num_classes)
     layer initialised with Kaiming-uniform weights.

Usage
-----
    from model import get_model
    from config import DEVICE, NUM_CLASSES

    model = get_model(DEVICE)            # prints param count, moves to device
    logits = model(batch_tensor)         # (B, 156)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, DEVICE


class TamilVision(nn.Module):
    """MobileNetV3-Small adapted for grayscale Tamil character recognition.

    Args:
        num_classes: Number of output classes (default: 156).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Base: MobileNetV3-Small with ImageNet pretrained weights
        # ---------------------------------------------------------------
        self.model = models.mobilenet_v3_small(weights="DEFAULT")

        # ---------------------------------------------------------------
        # Modification 1 — first conv: 3 channels → 1 channel (grayscale)
        # ---------------------------------------------------------------
        original_conv: nn.Conv2d = self.model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = original_conv.out_channels,
            kernel_size  = original_conv.kernel_size,
            stride       = original_conv.stride,
            padding      = original_conv.padding,
            bias         = original_conv.bias is not None,
        )

        # Transfer pretrained knowledge: sum RGB filters → single-channel filter.
        # This preserves learned edge/texture detectors rather than discarding them.
        with torch.no_grad():
            new_conv.weight.copy_(
                original_conv.weight.sum(dim=1, keepdim=True)
            )
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.model.features[0][0] = new_conv

        # ---------------------------------------------------------------
        # Modification 2 — classifier head: 1000 classes → num_classes
        # ---------------------------------------------------------------
        in_features: int = self.model.classifier[3].in_features

        new_head = nn.Linear(in_features, num_classes)
        nn.init.kaiming_uniform_(new_head.weight, nonlinearity="relu")
        nn.init.zeros_(new_head.bias)

        self.model.classifier[3] = new_head

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(B, 1, H, W)`` in ``[-1, 1]``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_model(device: str = DEVICE) -> TamilVision:
    """Instantiate :class:`TamilVision`, print parameter stats, and move to *device*.

    Args:
        device: Target device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Model instance on *device*, ready for training or inference.
    """
    model = TamilVision(num_classes=NUM_CLASSES).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model    : TamilVision (MobileNetV3-Small, 1-ch input, {NUM_CLASSES} classes)")
    print(f"Device   : {device}")
    print(f"Params   : {total:,} total  |  {trainable:,} trainable")

    return model
