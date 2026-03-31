"""EfficientNet-B0 lesion localizer: predicts bounding box from image alone."""

from typing import Tuple

import timm
import torch
import torch.nn as nn


class LesionLocalizer(nn.Module):
    """EfficientNet-B0 with a bbox regression head.

    Architecture:
      - EfficientNet-B0 feature extractor (timm, num_classes=0 → returns features)
      - Global average pooling
      - Linear head → 4 outputs (x0, y0, x1, y1) normalised to [0, 1] via sigmoid

    Loss: SmoothL1Loss on normalised bbox coords.

    Args:
        pretrained: Load ImageNet pretrained weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # returns feature vector, not logits
        )
        feature_dim = self.backbone.num_features
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # normalise to [0, 1]
        )
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor, shape (B, 3, H, W). Normalised to ImageNet stats.

        Returns:
            Bbox tensor, shape (B, 4), values in [0, 1] representing
            (x0, y0, x1, y1) normalised to image dimensions.
        """
        features = self.backbone(x)  # (B, feature_dim)
        return self.bbox_head(features)  # (B, 4)

    def compute_loss(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        """Compute SmoothL1 regression loss on normalised bbox coords.

        Args:
            pred_bbox: Predicted bbox, shape (B, 4), values in [0, 1].
            gt_bbox: Ground-truth bbox, shape (B, 4), values in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(pred_bbox, gt_bbox)

    @torch.no_grad()
    def predict_bbox_pixels(self, x: torch.Tensor, image_size: int) -> torch.Tensor:
        """Predict bbox in pixel coordinates.

        Args:
            x: Input image tensor, shape (B, 3, H, W).
            image_size: Square image dimension (assumes H == W == image_size).

        Returns:
            Bbox tensor, shape (B, 4), values in [0, image_size].
        """
        self.eval()
        norm_bbox = self.forward(x)
        return norm_bbox * image_size


def mask_to_bbox(mask: torch.Tensor, padding: int = 10, image_size: int = 512) -> torch.Tensor:
    """Derive a tight bounding box from a binary mask.

    Used to generate ground-truth bbox targets for localizer training,
    and to produce the UNREALISTIC GT bbox prompt for SAM evaluation.

    Args:
        mask: Binary mask tensor, shape (H, W) or (1, H, W), values in {0, 1}.
        padding: Pixel padding added to each side of the tight bbox.
        image_size: Image dimension used for clamping.

    Returns:
        Normalised bbox tensor, shape (4,), values in [0, 1]: (x0, y0, x1, y1).
    """
    mask = mask.squeeze().float()
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return torch.zeros(4)

    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values

    x0 = max(0, int(x_min) - padding)
    y0 = max(0, int(y_min) - padding)
    x1 = min(image_size, int(x_max) + padding)
    y1 = min(image_size, int(y_max) + padding)

    return torch.tensor([x0 / image_size, y0 / image_size, x1 / image_size, y1 / image_size])
