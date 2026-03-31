"""GradCAM-derived bounding box prompt from the lesion localizer.

Extracts GradCAM activation from the final conv layer of EfficientNet-B0,
thresholds at 0.5, and returns the bounding box of the activation region.

REALISTIC — no ground truth used.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExtractor:
    """Extract GradCAM heatmap from a target layer of a CNN.

    Hooks are registered before the forward pass and removed after to
    avoid memory leaks.

    Args:
        model: The CNN model (LesionLocalizer backbone).
        target_layer: The nn.Module layer to hook (typically the last conv block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._handles = []

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def _remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def compute(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Compute GradCAM heatmap for a single image.

        Args:
            image_tensor: Normalised image tensor, shape (1, 3, H, W).
                          Must be on the same device as the model.

        Returns:
            GradCAM heatmap as a float32 numpy array, shape (H, W), values in [0, 1].
            Resized to match input image spatial dimensions.
        """
        self._register_hooks()
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        try:
            # Forward — use sum of bbox outputs as the scalar target for gradients
            features = self.model.backbone(image_tensor)
            bbox_pred = self.model.bbox_head(features)
            score = bbox_pred.sum()

            self.model.zero_grad()
            score.backward()

            if self._gradients is None or self._activations is None:
                raise RuntimeError("GradCAM hooks did not fire — check target_layer.")

            # Pool gradients across spatial dims
            weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
            cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
            cam = F.relu(cam)

            # Normalise to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)

            # Resize to input image spatial dims
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            return cam.squeeze().cpu().numpy()  # (H, W)

        finally:
            self._remove_hooks()
            self._gradients = None
            self._activations = None


def gradcam_to_bbox(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    padding: int = 10,
    image_size: int = 512,
) -> np.ndarray:
    """Convert a GradCAM heatmap to a bounding box.

    REALISTIC — no GT information used.

    Args:
        heatmap: Float32 heatmap, shape (H, W), values in [0, 1].
        threshold: Activation threshold for binarisation.
        padding: Pixel padding added around the activation region.
        image_size: Image dimension used for clamping.

    Returns:
        Bbox array [x0, y0, x1, y1] in pixel coords, shape (4,).
        Returns a central default box if no activation region found.
    """
    binary = (heatmap > threshold).astype(np.uint8)
    coords = np.argwhere(binary > 0)

    if coords.size == 0:
        # Fallback: return a central box covering 50% of the image
        q = image_size // 4
        return np.array([q, q, image_size - q, image_size - q], dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    x0 = max(0, x_min - padding)
    y0 = max(0, y_min - padding)
    x1 = min(image_size, x_max + padding)
    y1 = min(image_size, y_max + padding)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def get_gradcam_bbox(
    localizer,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
    image_size: int = 512,
) -> np.ndarray:
    """Full pipeline: image tensor → GradCAM → bbox.

    Args:
        localizer: LesionLocalizer model with EfficientNet-B0 backbone.
        image_tensor: Normalised image tensor, shape (1, 3, H, W).
        threshold: GradCAM threshold for binarisation.
        image_size: Used for bbox clamping.

    Returns:
        Bbox array [x0, y0, x1, y1] in pixel coords, shape (4,).
    """
    # Hook the last conv block of EfficientNet-B0
    target_layer = localizer.backbone.blocks[-1]
    extractor = GradCAMExtractor(localizer, target_layer)
    heatmap = extractor.compute(image_tensor)
    return gradcam_to_bbox(heatmap, threshold=threshold, image_size=image_size)
