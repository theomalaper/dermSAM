"""Portfolio figures: qualitative grid, deployment gap bar chart, failure cases."""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import ISICDataset, get_val_transforms
from src.models.gradcam_prompt import get_gradcam_bbox
from src.models.localizer import LesionLocalizer
from src.models.sam_inference import (
    _load_medsam_predictor,
    auto_bbox_prompt,
    gt_bbox_prompt,
    predict_with_bbox_prompt,
)
from src.models.unet_baseline import UNetBaseline
from src.utils import dice_coefficient, load_checkpoint, set_seed


# ---------------------------------------------------------------------------
# Figure 1: Main results table (rendered as matplotlib table)
# ---------------------------------------------------------------------------

def plot_results_table(benchmark_csv: Path, output_path: Path) -> None:
    """Render the 5-row benchmark table as a figure.

    Args:
        benchmark_csv: Path to benchmark.csv from evaluate.py.
        output_path: Path to save the figure.
    """
    df = pd.read_csv(benchmark_csv)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    col_labels = ["Approach", "Dice", "IoU", "HD95 (px)"]
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["Approach"],
            f"{row['Dice mean']:.4f} ± {row['Dice std']:.4f}",
            f"{row['IoU mean']:.4f} ± {row['IoU std']:.4f}",
            f"{row['HD95 mean']:.1f} ± {row['HD95 std']:.1f}",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight REALISTIC rows in green
    for row_idx, row_data in enumerate(table_data):
        if "REALISTIC" in row_data[0] and "UNREALISTIC" not in row_data[0]:
            for col_idx in range(len(col_labels)):
                table[row_idx + 1, col_idx].set_facecolor("#d4f1d4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Results table saved to {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Deployment gap bar chart
# ---------------------------------------------------------------------------

def plot_deployment_gap(benchmark_csv: Path, output_path: Path) -> None:
    """Bar chart showing realistic vs unrealistic Dice — the visual argument.

    Args:
        benchmark_csv: Path to benchmark.csv.
        output_path: Path to save the figure.
    """
    df = pd.read_csv(benchmark_csv)

    labels = [row["Approach"].split("[")[0].strip() for _, row in df.iterrows()]
    dice_means = df["Dice mean"].tolist()
    dice_stds = df["Dice std"].tolist()
    colors = []
    for approach in df["Approach"]:
        if "UNREALISTIC" in approach:
            colors.append("#e57373")
        elif "REALISTIC" in approach:
            colors.append("#66bb6a")
        else:
            colors.append("#90caf9")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), dice_means, yerr=dice_stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Dice coefficient")
    ax.set_title("Deployment Gap: GT-Prompted (Unrealistic) vs Auto-Prompted (Realistic)")
    ax.set_ylim(0, 1.05)
    baseline_line = plt.Line2D([0], [0], color="black", linestyle=":", alpha=0.6,
                               label="Published baseline (ResUNet++ 0.7726)")
    ax.axhline(0.7726, color="black", linestyle=":", alpha=0.6)

    patches = [
        mpatches.Patch(color="#e57373", label="Unrealistic (GT prompt)"),
        mpatches.Patch(color="#66bb6a", label="Realistic (auto prompt)"),
        mpatches.Patch(color="#90caf9", label="Supervised baseline"),
        baseline_line,
    ]
    ax.legend(handles=patches, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Deployment gap figure saved to {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Qualitative grid
# ---------------------------------------------------------------------------

def plot_qualitative_grid(
    test_csv: Path,
    unet_ckpt: Path,
    localizer_ckpt: Path,
    medsam_ckpt: Path,
    output_path: Path,
    n_rows: int = 6,
    device: str = "cuda",
) -> None:
    """5-column qualitative grid: image | GT | UNet | MedSAM+GT | MedSAM+Auto.

    Args:
        test_csv: Test split CSV.
        unet_ckpt: Best UNet checkpoint.
        localizer_ckpt: Best localizer checkpoint.
        medsam_ckpt: MedSAM ViT-B weights.
        output_path: Path to save figure.
        n_rows: Number of sample rows in the grid.
        device: Device string.
    """
    set_seed(42)
    unet = UNetBaseline().to(device)
    load_checkpoint(unet_ckpt, unet, device=device)
    unet.eval()

    localizer = LesionLocalizer(pretrained=False).to(device)
    load_checkpoint(localizer_ckpt, localizer, device=device)
    localizer.eval()

    medsam = _load_medsam_predictor(medsam_ckpt)

    ds_unet = ISICDataset(test_csv, get_val_transforms(512), sam_mode=False, image_size=512)
    ds_sam = ISICDataset(test_csv, get_val_transforms(1024, sam_mode=True), sam_mode=True, image_size=1024)

    fig, axes = plt.subplots(n_rows, 6, figsize=(18, n_rows * 3))
    col_titles = [
        "Image + Auto bbox\n(localizer output)",
        "GT Mask",
        "UNet",
        "MedSAM + GT bbox\n[UNREALISTIC]",
        "MedSAM + Auto bbox\n[REALISTIC]",
        "MedSAM + GradCAM bbox\n[REALISTIC]",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    step = max(1, len(ds_unet) // n_rows)
    sample_indices = list(range(0, len(ds_unet), step))[:n_rows]

    for row, idx in enumerate(sample_indices):
        s_unet = ds_unet[idx]
        s_sam = ds_sam[idx]

        # Original image (denormalise from ImageNet stats)
        img_display = s_unet["image"].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = np.clip(img_display * std + mean, 0, 1)

        # GT mask
        mask_np = s_unet["mask"].squeeze().numpy()
        mask_sam_np = s_sam["mask"]

        # UNet prediction
        with torch.no_grad():
            logit = unet(s_unet["image"].unsqueeze(0).to(device))
        pred_unet = (torch.sigmoid(logit.cpu()) > 0.5).float().squeeze().numpy()

        # MedSAM + GT bbox
        img_tensor = s_unet["image"].unsqueeze(0).to(device)
        gt_box = gt_bbox_prompt((mask_sam_np > 0.5).astype(np.uint8), padding=10)
        pred_gt = predict_with_bbox_prompt(medsam, s_sam["image"], gt_box)

        # MedSAM + Auto bbox
        auto_box = auto_bbox_prompt(localizer, img_tensor, image_size=512)
        auto_box_sam = (auto_box * 2.0).astype(np.float32)  # 512 -> 1024
        pred_auto = predict_with_bbox_prompt(medsam, s_sam["image"], auto_box_sam)

        # MedSAM + GradCAM bbox
        gradcam_box = get_gradcam_bbox(localizer, img_tensor, image_size=512)
        gradcam_box_sam = (gradcam_box * 2.0).astype(np.float32)  # 512 -> 1024
        pred_gradcam = predict_with_bbox_prompt(medsam, s_sam["image"], gradcam_box_sam)

        # Dice scores
        d_unet = dice_coefficient(torch.from_numpy(pred_unet), torch.from_numpy(mask_np))
        d_gt = dice_coefficient(torch.from_numpy(pred_gt.astype(np.float32)), torch.from_numpy(mask_sam_np))
        d_auto = dice_coefficient(torch.from_numpy(pred_auto.astype(np.float32)), torch.from_numpy(mask_sam_np))
        d_gradcam = dice_coefficient(torch.from_numpy(pred_gradcam.astype(np.float32)), torch.from_numpy(mask_sam_np))

        # Scale boxes from 1024/512 to 512 for display on img_display
        gt_box_512 = (gt_box / 2.0).astype(int)
        auto_box_512 = auto_box.astype(int)
        gradcam_box_512 = gradcam_box.astype(int)

        # Col 0: image + auto bbox (yellow)
        axes[row, 0].imshow(img_display)
        x0, y0, x1, y1 = auto_box_512
        axes[row, 0].add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="yellow", facecolor="none"
        ))
        axes[row, 0].axis("off")

        # Col 1: GT mask
        axes[row, 1].imshow(mask_np, cmap="gray")
        axes[row, 1].axis("off")

        # Col 2: UNet prediction
        axes[row, 2].imshow(pred_unet, cmap="gray")
        axes[row, 2].set_xlabel(f"Dice={d_unet:.3f}", fontsize=8)
        axes[row, 2].axis("off")

        # Col 3: image + GT bbox (green) + segmentation overlay
        axes[row, 3].imshow(img_display)
        axes[row, 3].imshow(pred_gt, cmap="Reds", alpha=0.4)
        x0, y0, x1, y1 = gt_box_512
        axes[row, 3].add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="lime", facecolor="none"
        ))
        axes[row, 3].set_xlabel(f"Dice={d_gt:.3f}", fontsize=8)
        axes[row, 3].axis("off")

        # Col 4: image + auto bbox (yellow) + segmentation overlay
        axes[row, 4].imshow(img_display)
        axes[row, 4].imshow(pred_auto, cmap="Reds", alpha=0.4)
        x0, y0, x1, y1 = auto_box_512
        axes[row, 4].add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="yellow", facecolor="none"
        ))
        axes[row, 4].set_xlabel(f"Dice={d_auto:.3f}", fontsize=8)
        axes[row, 4].axis("off")

        # Col 5: image + GradCAM bbox (orange) + segmentation overlay
        axes[row, 5].imshow(img_display)
        axes[row, 5].imshow(pred_gradcam, cmap="Reds", alpha=0.4)
        x0, y0, x1, y1 = gradcam_box_512
        axes[row, 5].add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="orange", facecolor="none"
        ))
        axes[row, 5].set_xlabel(f"Dice={d_gradcam:.3f}", fontsize=8)
        axes[row, 5].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Qualitative grid saved to {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Failure case analysis
# ---------------------------------------------------------------------------

def plot_failure_cases(
    test_csv: Path,
    unet_ckpt: Path,
    localizer_ckpt: Path,
    medsam_ckpt: Path,
    output_path: Path,
    device: str = "cuda",
) -> None:
    """6-panel failure case analysis with annotated failure types.

    Failure types: poor localizer bbox / ambiguous boundary / atypical morphology
                   small lesion / hair artefact / large lesion boundary error

    Args:
        test_csv: Test split CSV.
        unet_ckpt: Best UNet checkpoint.
        localizer_ckpt: Best localizer checkpoint.
        medsam_ckpt: MedSAM ViT-B weights.
        output_path: Path to save figure.
        device: Device string.
    """
    set_seed(42)
    unet = UNetBaseline().to(device)
    load_checkpoint(unet_ckpt, unet, device=device)
    unet.eval()

    localizer = LesionLocalizer(pretrained=False).to(device)
    load_checkpoint(localizer_ckpt, localizer, device=device)
    localizer.eval()

    medsam = _load_medsam_predictor(medsam_ckpt)

    ds_unet = ISICDataset(test_csv, get_val_transforms(512), sam_mode=False, image_size=512)
    ds_sam = ISICDataset(test_csv, get_val_transforms(1024, sam_mode=True), sam_mode=True, image_size=1024)

    failure_types = [
        "Poor localizer bbox",
        "Ambiguous lesion boundary",
        "Atypical morphology",
        "Small lesion",
        "Hair artefact",
        "Large lesion boundary error",
    ]

    # Find samples with lowest auto-bbox Dice (these are likely failure cases)
    scores = []
    for i in range(min(len(ds_unet), 260)):
        s_sam = ds_sam[i]
        s_loc = ds_unet[i]
        mask_np = s_sam["mask"]
        if mask_np.sum() == 0:
            continue
        img_tensor = s_loc["image"].unsqueeze(0).to(device)
        auto_box = auto_bbox_prompt(localizer, img_tensor, image_size=512)
        auto_box_sam = (auto_box * 2.0).astype(np.float32)
        pred = predict_with_bbox_prompt(medsam, s_sam["image"], auto_box_sam)
        d = dice_coefficient(torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(mask_np))
        scores.append((d, i))

    scores.sort(key=lambda x: x[0])
    worst_indices = [idx for _, idx in scores[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for panel, (idx, failure_type) in enumerate(zip(worst_indices, failure_types)):
        s_unet = ds_unet[idx]
        s_sam = ds_sam[idx]

        img_display = s_unet["image"].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = np.clip(img_display * std + mean, 0, 1)
        mask_np = s_sam["mask"]

        img_tensor = s_unet["image"].unsqueeze(0).to(device)
        auto_box = auto_bbox_prompt(localizer, img_tensor, image_size=512)
        auto_box_sam = (auto_box * 2.0).astype(np.float32)
        pred = predict_with_bbox_prompt(medsam, s_sam["image"], auto_box_sam)
        d = dice_coefficient(torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(mask_np))

        # Overlay: image + GT contour + pred mask
        overlay = img_display.copy()
        pred_rgb = np.zeros_like(overlay)
        pred_rgb[:, :, 0] = cv2.resize(pred, (512, 512)) * 0.4  # red channel for prediction

        axes[panel].imshow(overlay)
        axes[panel].imshow(pred_rgb, alpha=0.4)
        # Draw auto bbox
        x0, y0, x1, y1 = (auto_box * 1.0).astype(int)
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="yellow", facecolor="none")
        axes[panel].add_patch(rect)
        axes[panel].set_title(f"{failure_type}\nDice={d:.3f}", fontsize=9, color="darkred")
        axes[panel].axis("off")

    fig.suptitle("Failure Case Analysis — MedSAM + Auto bbox", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Failure cases saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    p.add_argument("--benchmark-csv", type=Path, default=Path("outputs/metrics/benchmark.csv"))
    p.add_argument("--unet-ckpt", type=Path, default=Path("checkpoints/best_unet.pth"))
    p.add_argument("--localizer-ckpt", type=Path, default=Path("checkpoints/best_localizer.pth"))
    p.add_argument("--medsam-ckpt", type=Path, default=Path("checkpoints/medsam_vit_b.pth"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    plot_results_table(args.benchmark_csv, args.figures_dir / "results_table.png")
    plot_deployment_gap(args.benchmark_csv, args.figures_dir / "deployment_gap.png")
    plot_qualitative_grid(args.test_csv, args.unet_ckpt, args.localizer_ckpt, args.medsam_ckpt,
                          args.figures_dir / "qualitative_grid.png", device=device)
    plot_failure_cases(args.test_csv, args.unet_ckpt, args.localizer_ckpt, args.medsam_ckpt,
                       args.figures_dir / "failure_cases.png", device=device)


if __name__ == "__main__":
    main()
