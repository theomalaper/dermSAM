"""Shared utilities: seeding, AMP helpers, metric computation."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler


def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_scaler(enabled: bool = True) -> GradScaler:
    """Create a GradScaler for AMP training.

    Args:
        enabled: If False, returns a no-op scaler (CPU / non-AMP runs).

    Returns:
        GradScaler instance.
    """
    return GradScaler("cuda", enabled=enabled)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute Dice coefficient between predicted and target binary masks.

    Operates on CPU numpy arrays; expects predictions to be logits or probabilities.

    Args:
        pred: Predicted mask tensor, shape (H, W) or (1, H, W). Logits or probabilities.
        target: Ground-truth binary mask tensor, shape (H, W) or (1, H, W).
        threshold: Binarisation threshold applied to pred.
        eps: Smoothing term to avoid division by zero.

    Returns:
        Scalar Dice coefficient in [0, 1].
    """
    pred = pred.detach().cpu().float().squeeze()
    target = target.detach().cpu().float().squeeze()
    pred_bin = (torch.sigmoid(pred) > threshold).float() if pred.requires_grad or pred.max() > 1.0 else (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return float((2.0 * intersection + eps) / (pred_bin.sum() + target.sum() + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute Intersection-over-Union (Jaccard index) for binary masks.

    Args:
        pred: Predicted mask tensor. Logits or probabilities.
        target: Ground-truth binary mask tensor.
        threshold: Binarisation threshold.
        eps: Smoothing term.

    Returns:
        Scalar IoU in [0, 1].
    """
    pred = pred.detach().cpu().float().squeeze()
    target = target.detach().cpu().float().squeeze()
    pred_bin = (torch.sigmoid(pred) > threshold).float() if pred.max() > 1.0 else (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def hausdorff95(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute 95th-percentile Hausdorff distance between predicted and GT masks.

    Args:
        pred: Predicted mask tensor. Logits or probabilities.
        target: Ground-truth binary mask tensor.
        threshold: Binarisation threshold.

    Returns:
        HD95 in pixels. Returns 0.0 if either mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred.detach().cpu().float().squeeze().numpy()
    target = target.detach().cpu().float().squeeze().numpy()
    pred_bin = (pred > threshold).astype(np.uint8)

    if pred_bin.sum() == 0 or target.sum() == 0:
        return 0.0

    dt_pred = distance_transform_edt(1 - pred_bin)
    dt_target = distance_transform_edt(1 - target.astype(np.uint8))

    hd_p2t = dt_target[pred_bin == 1]
    hd_t2p = dt_pred[target == 1]
    return float(np.percentile(np.concatenate([hd_p2t, hd_t2p]), 95))


def bbox_iou(box_pred: np.ndarray, box_gt: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in [x0, y0, x1, y1] format.

    Args:
        box_pred: Predicted bbox, shape (4,).
        box_gt: Ground-truth bbox, shape (4,).

    Returns:
        Scalar IoU in [0, 1].
    """
    x0 = max(box_pred[0], box_gt[0])
    y0 = max(box_pred[1], box_gt[1])
    x1 = min(box_pred[2], box_gt[2])
    y1 = min(box_pred[3], box_gt[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union = area_pred + area_gt - inter
    return float(inter / union) if union > 0 else 0.0


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_dice: float,
) -> None:
    """Save a full training checkpoint.

    Args:
        path: File path to save the checkpoint (.pth).
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        scheduler: LR scheduler whose state_dict to save.
        epoch: Current epoch number.
        best_val_dice: Best validation Dice seen so far.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_dice": best_val_dice,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None, device: str = "cpu"):
    """Load a checkpoint into model (and optionally optimizer/scheduler).

    Args:
        path: Path to the .pth checkpoint file.
        model: Model to load weights into.
        optimizer: If provided, load optimizer state.
        scheduler: If provided, load scheduler state.
        device: Device string for map_location.

    Returns:
        Tuple (epoch, best_val_dice).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_dice", 0.0)
