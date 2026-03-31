"""UNet ResNet34 supervised baseline using segmentation-models-pytorch."""

from pathlib import Path
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class UNetBaseline(nn.Module):
    """smp.Unet with ResNet34 ImageNet-pretrained encoder.

    Loss: 0.5 * DiceLoss(sigmoid=True) + 0.5 * BCEWithLogitsLoss

    Args:
        encoder_name: Encoder backbone name (default: resnet34).
        encoder_weights: Pretrained weights source (default: imagenet).
        in_channels: Number of input image channels.
        num_classes: Number of output segmentation classes.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # raw logits — loss applies sigmoid
        )
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor, shape (B, 3, H, W).

        Returns:
            Logit mask tensor, shape (B, 1, H, W).
        """
        return self.model(x)

    def compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Args:
            logits: Raw model output, shape (B, 1, H, W).
            masks: Binary ground-truth masks, shape (B, 1, H, W), values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        return 0.5 * self.dice_loss(logits, masks) + 0.5 * self.bce_loss(logits, masks)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run inference and return binary mask.

        Args:
            x: Input tensor, shape (B, 3, H, W).
            threshold: Sigmoid threshold for binarisation.

        Returns:
            Binary mask tensor, shape (B, 1, H, W), values in {0, 1}.
        """
        self.eval()
        with autocast(enabled=torch.cuda.is_available()):
            logits = self.forward(x)
        return (torch.sigmoid(logits) > threshold).float()
