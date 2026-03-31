"""Prompt degradation analysis: Dice vs bbox perturbation magnitude."""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import ISICDataset, get_val_transforms
from src.models.localizer import LesionLocalizer
from src.models.sam_inference import (
    _load_medsam_predictor,
    auto_bbox_prompt,
    gt_bbox_prompt,
    predict_with_bbox_prompt,
)
from src.utils import bbox_iou, dice_coefficient, load_checkpoint, set_seed


def perturb_bbox(bbox: np.ndarray, offset: int, image_size: int) -> np.ndarray:
    """Apply symmetric expansion/shift perturbation to a bounding box.

    Expands the box outward by `offset` pixels on all sides.

    Args:
        bbox: Original bbox [x0, y0, x1, y1] in pixel coords.
        offset: Number of pixels to expand each side.
        image_size: Image dimension used for clamping.

    Returns:
        Perturbed bbox [x0, y0, x1, y1], clamped to image bounds.
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - offset)
    y0 = max(0, y0 - offset)
    x1 = min(image_size, x1 + offset)
    y1 = min(image_size, y1 + offset)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def run_sensitivity(
    test_csv: Path,
    medsam_ckpt: Path,
    localizer_ckpt: Path,
    offsets: List[int],
    output_csv: Path,
    output_fig: Path,
    image_size: int = 1024,
    n_samples: int = 200,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run prompt degradation analysis across perturbation levels.

    For each offset value, degraded GT bboxes are fed to MedSAM and Dice is measured.
    The auto-prompt result is plotted as a red dot on the resulting curve.

    Args:
        test_csv: Path to test split CSV.
        medsam_ckpt: Path to medsam_vit_b.pth.
        localizer_ckpt: Path to best localizer checkpoint.
        offsets: List of pixel perturbation offsets to evaluate.
        output_csv: Path to save results CSV.
        output_fig: Path to save degradation curve figure.
        image_size: SAM image size.
        n_samples: Number of test samples to use (subset for speed).
        device: Device string.

    Returns:
        DataFrame with columns [offset, dice_mean, dice_std].
    """
    set_seed(42)

    medsam_predictor = _load_medsam_predictor(medsam_ckpt)
    localizer = LesionLocalizer(pretrained=False).to(device)
    load_checkpoint(localizer_ckpt, localizer, device=device)
    localizer.eval()

    ds = ISICDataset(
        test_csv,
        get_val_transforms(image_size, sam_mode=True),
        sam_mode=True,
        image_size=image_size,
    )
    ds_loc = ISICDataset(
        test_csv,
        get_val_transforms(512),
        sam_mode=False,
        image_size=512,
    )

    indices = list(range(min(n_samples, len(ds))))
    results = []

    for offset in offsets:
        dice_scores = []
        for i in indices:
            sample = ds[i]
            image_rgb = sample["image"]
            mask_np = sample["mask"]
            if mask_np.sum() == 0:
                continue
            gt_box = gt_bbox_prompt((mask_np > 0.5).astype(np.uint8), padding=10)
            degraded_box = perturb_bbox(gt_box, offset=offset, image_size=image_size)
            pred = predict_with_bbox_prompt(medsam_predictor, image_rgb, degraded_box)
            dice_scores.append(
                dice_coefficient(torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(mask_np))
            )
        results.append({"offset": offset, "dice_mean": np.mean(dice_scores), "dice_std": np.std(dice_scores)})
        print(f"  offset={offset:4d}px | Dice {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")

    df = pd.DataFrame(results)

    # Auto-prompt result (for the red dot)
    auto_dice_scores = []
    for i in indices:
        sample_sam = ds[i]
        sample_loc = ds_loc[i]
        mask_np = sample_sam["mask"]
        if mask_np.sum() == 0:
            continue
        img_tensor = sample_loc["image"].unsqueeze(0).to(device)
        auto_box = auto_bbox_prompt(localizer, img_tensor, image_size=512)
        auto_box_sam = (auto_box * (image_size / 512)).astype(np.float32)
        pred = predict_with_bbox_prompt(medsam_predictor, sample_sam["image"], auto_box_sam)
        auto_dice_scores.append(
            dice_coefficient(torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(mask_np))
        )
    auto_dice_mean = np.mean(auto_dice_scores)

    # --- Save CSV ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # --- Plot degradation curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["offset"], df["dice_mean"], "b-o", label="GT bbox + perturbation")
    ax.fill_between(
        df["offset"],
        df["dice_mean"] - df["dice_std"],
        df["dice_mean"] + df["dice_std"],
        alpha=0.2, color="blue",
    )
    ax.axhline(auto_dice_mean, color="red", linestyle="--", alpha=0.5)
    ax.scatter(
        [0], [auto_dice_mean], color="red", s=120, zorder=5,
        label=f"Auto-prompt (localizer) Dice={auto_dice_mean:.3f}",
    )
    ax.set_xlabel("Bbox perturbation (pixels expanded per side)")
    ax.set_ylabel("Dice coefficient")
    ax.set_title("MedSAM Prompt Sensitivity — Deployment Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensitivity curve saved to {output_fig}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    p.add_argument("--medsam-ckpt", type=Path, default=Path("checkpoints/medsam_vit_b.pth"))
    p.add_argument("--localizer-ckpt", type=Path, default=Path("checkpoints/best_localizer.pth"))
    p.add_argument("--offsets", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/prompt_sensitivity.csv"))
    p.add_argument("--output-fig", type=Path, default=Path("outputs/figures/prompt_sensitivity.png"))
    p.add_argument("--n-samples", type=int, default=200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_sensitivity(
        test_csv=args.test_csv,
        medsam_ckpt=args.medsam_ckpt,
        localizer_ckpt=args.localizer_ckpt,
        offsets=args.offsets,
        output_csv=args.output_csv,
        output_fig=args.output_fig,
        n_samples=args.n_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
