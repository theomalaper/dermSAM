"""SAM / MedSAM inference with all four prompt strategies.

All GT-derived prompt strategies are labelled UNREALISTIC — they use ground-truth
information unavailable at deployment. Only auto_bbox and gradcam_bbox are REALISTIC/DEPLOYABLE.

Prompt derivation logic lives here only — never duplicated elsewhere.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# SAM expects RGB uint8 numpy arrays — never pass tensors or float arrays


def _load_sam_predictor(checkpoint: Path, model_type: str = "vit_h"):
    """Load a SAM predictor from checkpoint.

    Args:
        checkpoint: Path to .pth checkpoint file.
        model_type: SAM model type string ('vit_h', 'vit_b', etc.).

    Returns:
        SamPredictor instance with image encoder loaded.
    """
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(str(checkpoint), map_location=device)
    sam.load_state_dict(state_dict)
    sam.eval()
    sam.to(device)
    return SamPredictor(sam)


def _load_medsam_predictor(checkpoint: Path):
    """Load a MedSAM predictor (vit_b architecture).

    MedSAM checkpoint is vit_b — never pass to vit_h registry.

    Args:
        checkpoint: Path to medsam_vit_b.pth.

    Returns:
        SamPredictor wrapping MedSAM model.
    """
    from segment_anything import SamPredictor, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    medsam = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(str(checkpoint), map_location=device)
    medsam.load_state_dict(state_dict)
    medsam.eval()
    medsam.to(device)
    return SamPredictor(medsam)


def predict_with_point_prompt(
    predictor,
    image: np.ndarray,
    point_xy: Tuple[float, float],
) -> np.ndarray:
    """Run SAM inference with a single foreground point prompt.

    UNREALISTIC — uses GT-derived centroid. For upper-bound evaluation only.

    Args:
        predictor: SamPredictor instance (already loaded).
        image: RGB uint8 numpy array, shape (H, W, 3).
        point_xy: (x, y) pixel coordinates of the prompt point.

    Returns:
        Binary mask numpy array, shape (H, W), dtype uint8.
    """
    predictor.set_image(image)
    input_point = np.array([[point_xy[0], point_xy[1]]])
    input_label = np.array([1])  # foreground
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]
    return best_mask.astype(np.uint8)


def predict_with_bbox_prompt(
    predictor,
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> np.ndarray:
    """Run SAM / MedSAM inference with a bounding box prompt.

    Can be UNREALISTIC (GT bbox) or REALISTIC (auto bbox from localizer).
    The caller is responsible for labelling which is which.

    Args:
        predictor: SamPredictor instance.
        image: RGB uint8 numpy array, shape (H, W, 3).
        bbox_xyxy: Bbox in [x0, y0, x1, y1] pixel format, shape (4,).

    Returns:
        Binary mask numpy array, shape (H, W), dtype uint8.
    """
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=bbox_xyxy[None, :],  # (1, 4)
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]
    return best_mask.astype(np.uint8)


def gt_centroid_prompt(mask: np.ndarray) -> Tuple[float, float]:
    """Derive (x, y) centroid from ground-truth mask.

    UNREALISTIC — GT information unavailable at deployment.

    Args:
        mask: Binary mask, shape (H, W), values in {0, 1} or {0, 255}.

    Returns:
        (x, y) centroid pixel coordinates.
    """
    coords = np.argwhere(mask > 0)  # (N, 2) in (row, col) = (y, x)
    centroid_yx = coords.mean(axis=0)
    return float(centroid_yx[1]), float(centroid_yx[0])  # return as (x, y)


def gt_bbox_prompt(mask: np.ndarray, padding: int = 10) -> np.ndarray:
    """Derive tight bounding box from ground-truth mask.

    UNREALISTIC — GT information unavailable at deployment.

    Args:
        mask: Binary mask, shape (H, W).
        padding: Pixel padding on each side.

    Returns:
        Bbox array [x0, y0, x1, y1] in pixel coords, shape (4,).
    """
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask.shape
    x0 = max(0, x_min - padding)
    y0 = max(0, y_min - padding)
    x1 = min(w, x_max + padding)
    y1 = min(h, y_max + padding)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def auto_bbox_prompt(
    localizer,
    image_tensor: torch.Tensor,
    image_size: int = 512,
) -> np.ndarray:
    """Derive bbox from the EfficientNet localizer (no GT used).

    REALISTIC/DEPLOYABLE — the core contribution of this pipeline.

    Args:
        localizer: LesionLocalizer model in eval mode.
        image_tensor: Normalised image tensor, shape (1, 3, H, W).
        image_size: Image dimension to convert normalised coords to pixels.

    Returns:
        Bbox array [x0, y0, x1, y1] in pixel coords, shape (4,).
    """
    device = next(localizer.parameters()).device
    image_tensor = image_tensor.to(device)
    localizer.eval()
    with torch.no_grad():
        norm_bbox = localizer(image_tensor).squeeze().cpu().numpy()
    return (norm_bbox * image_size).astype(np.float32)
