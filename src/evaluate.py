"""Full benchmark: evaluate all 7 approaches on the test set and write results CSV."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.dataset import ISICDataset, get_val_transforms
from src.models.gradcam_prompt import get_gradcam_bbox
from src.models.localizer import LesionLocalizer
from src.models.medsam_finetune import MedSAMFinetune
from src.models.sam_inference import (
    _load_medsam_predictor,
    _load_sam_predictor,
    auto_bbox_prompt,
    gt_bbox_prompt,
    gt_centroid_prompt,
    predict_with_bbox_prompt,
    predict_with_point_prompt,
)
from src.models.unet_baseline import UNetBaseline
from src.utils import (
    bbox_iou,
    dice_coefficient,
    hausdorff95,
    iou_score,
    load_checkpoint,
    set_seed,
)


def _infer_finetuned_medsam(
    model: MedSAMFinetune,
    image_rgb_np: np.ndarray,
    bbox_np: np.ndarray,
    device: str,
    image_size: int = 1024,
) -> np.ndarray:
    """Run a single inference pass with the fine-tuned MedSAM model.

    Args:
        model: MedSAMFinetune model (eval mode, on device).
        image_rgb_np: uint8 numpy array, shape (H, W, 3).
        bbox_np: Bbox in pixel coords [x0, y0, x1, y1], shape (4,).
        device: Device string.
        image_size: Expected spatial size (must match model's 1024).

    Returns:
        Binary mask as float32 numpy array, shape (image_size, image_size).
    """
    img_t = (
        torch.from_numpy(image_rgb_np.astype("float32") / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    box_t = torch.from_numpy(bbox_np.astype("float32")).unsqueeze(0).to(device)
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        logit = model(img_t, box_t)  # (1, 1, 256, 256)
    pred = torch.sigmoid(logit.cpu()).squeeze()  # (256, 256)
    pred_up = F.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return (pred_up > 0.5).float().numpy()


def run_benchmark(
    test_csv: Path,
    unet_ckpt: Path,
    localizer_ckpt: Path,
    medsam_ckpt: Path,
    sam_ckpt: Path,
    output_csv: Path,
    finetuned_medsam_ckpt: Optional[Path] = None,
    image_size_unet: int = 512,
    image_size_sam: int = 1024,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate all approaches on the test set.

    Approaches:
      1. UNet ResNet34 — supervised baseline
      2. SAM ViT-H zero-shot + GT centroid prompt         [UNREALISTIC]
      3. MedSAM ViT-B zero-shot + GT bbox prompt          [UNREALISTIC]
      4. MedSAM ViT-B zero-shot + auto bbox               [REALISTIC]
      5. MedSAM ViT-B zero-shot + GradCAM bbox            [REALISTIC]
      6. MedSAM ViT-B fine-tuned + GT bbox prompt         [UNREALISTIC] (if ckpt provided)
      7. MedSAM ViT-B fine-tuned + auto bbox              [REALISTIC]   (if ckpt provided)

    Args:
        test_csv: Path to test split CSV.
        unet_ckpt: Path to best UNet checkpoint.
        localizer_ckpt: Path to best localizer checkpoint.
        medsam_ckpt: Path to medsam_vit_b.pth weights.
        sam_ckpt: Path to sam_vit_h_4b8939.pth weights.
        output_csv: Path to write benchmark results CSV.
        finetuned_medsam_ckpt: Path to best_medsam.pth (fine-tuned). If None, rows 6/7 skipped.
        image_size_unet: Resize for UNet inference.
        image_size_sam: Resize for SAM/MedSAM inference.
        device: Device string.

    Returns:
        DataFrame with per-approach mean/std Dice, IoU, HD95.
    """
    set_seed(42)

    # --- Load models ---
    unet = UNetBaseline().to(device)
    load_checkpoint(unet_ckpt, unet, device=device)
    unet.eval()

    localizer = LesionLocalizer(pretrained=False).to(device)
    load_checkpoint(localizer_ckpt, localizer, device=device)
    localizer.eval()

    medsam_predictor = _load_medsam_predictor(medsam_ckpt)
    sam_predictor = _load_sam_predictor(sam_ckpt, model_type="vit_h")

    # Fine-tuned MedSAM (optional — rows 6 & 7)
    finetuned_medsam = None
    if finetuned_medsam_ckpt is not None and finetuned_medsam_ckpt.exists():
        finetuned_medsam = MedSAMFinetune(medsam_ckpt).to(device)
        load_checkpoint(finetuned_medsam_ckpt, finetuned_medsam, device=device)
        finetuned_medsam.eval()
        print(f"Loaded fine-tuned MedSAM from {finetuned_medsam_ckpt}")
    else:
        print("No fine-tuned MedSAM checkpoint provided — skipping rows 6 & 7")

    # --- Datasets ---
    ds_unet = ISICDataset(test_csv, get_val_transforms(image_size_unet), sam_mode=False, image_size=image_size_unet)
    ds_sam = ISICDataset(test_csv, get_val_transforms(image_size_sam, sam_mode=True), sam_mode=True, image_size=image_size_sam)
    ds_localizer = ISICDataset(test_csv, get_val_transforms(image_size_unet), sam_mode=False, image_size=image_size_unet)

    rows = {name: {"dice": [], "iou": [], "hd95": []} for name in [
        "unet", "sam_gt_centroid", "medsam_gt_bbox", "medsam_auto_bbox", "medsam_gradcam_bbox",
        "medsam_ft_gt_bbox", "medsam_ft_auto_bbox",
    ]}
    localizer_bbox_ious = []

    n = len(ds_unet)
    for i in range(n):
        if i % 50 == 0:
            print(f"  [{i}/{n}]")

        sample_unet = ds_unet[i]
        sample_sam = ds_sam[i]
        sample_loc = ds_localizer[i]

        mask_gt = sample_unet["mask"]  # (1, H, W) float32 tensor
        mask_gt_np = mask_gt.squeeze().numpy()
        orig_h, orig_w = sample_unet["original_size"]

        # ---------- 1. UNet ----------
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            logit = unet(sample_unet["image"].unsqueeze(0).to(device))
        import torch.nn.functional as F
        pred_unet = torch.sigmoid(logit.cpu()).squeeze()
        # Upsample to original size for metric computation
        pred_unet_orig = F.interpolate(
            pred_unet.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze()
        mask_gt_orig = torch.from_numpy(
            __import__("cv2").resize(mask_gt_np, (orig_w, orig_h), interpolation=__import__("cv2").INTER_NEAREST)
        )
        rows["unet"]["dice"].append(dice_coefficient(pred_unet_orig, mask_gt_orig))
        rows["unet"]["iou"].append(iou_score(pred_unet_orig, mask_gt_orig))
        rows["unet"]["hd95"].append(hausdorff95(pred_unet_orig, mask_gt_orig))

        # ---------- 2. SAM ViT-H + GT centroid [UNREALISTIC] ----------
        image_rgb = sample_sam["image"]  # uint8 numpy
        mask_sam_np = sample_sam["mask"]
        cx, cy = gt_centroid_prompt((mask_sam_np > 0.5).astype(np.uint8))
        pred_sam_centroid = predict_with_point_prompt(sam_predictor, image_rgb, (cx, cy))
        pred_t = torch.from_numpy(pred_sam_centroid.astype(np.float32))
        mask_t = torch.from_numpy(mask_sam_np)
        rows["sam_gt_centroid"]["dice"].append(dice_coefficient(pred_t, mask_t))
        rows["sam_gt_centroid"]["iou"].append(iou_score(pred_t, mask_t))
        rows["sam_gt_centroid"]["hd95"].append(hausdorff95(pred_t, mask_t))

        # ---------- 3. MedSAM + GT bbox [UNREALISTIC] ----------
        gt_box = gt_bbox_prompt((mask_sam_np > 0.5).astype(np.uint8), padding=10)
        pred_medsam_gt = predict_with_bbox_prompt(medsam_predictor, image_rgb, gt_box)
        pred_t = torch.from_numpy(pred_medsam_gt.astype(np.float32))
        rows["medsam_gt_bbox"]["dice"].append(dice_coefficient(pred_t, mask_t))
        rows["medsam_gt_bbox"]["iou"].append(iou_score(pred_t, mask_t))
        rows["medsam_gt_bbox"]["hd95"].append(hausdorff95(pred_t, mask_t))

        # ---------- 4. MedSAM + auto bbox [REALISTIC] ----------
        img_tensor = sample_loc["image"].unsqueeze(0).to(device)
        auto_box = auto_bbox_prompt(localizer, img_tensor, image_size=image_size_unet)
        # Scale auto box from unet_size to sam_size
        scale = image_size_sam / image_size_unet
        auto_box_sam = (auto_box * scale).astype(np.float32)
        pred_medsam_auto = predict_with_bbox_prompt(medsam_predictor, image_rgb, auto_box_sam)
        pred_t = torch.from_numpy(pred_medsam_auto.astype(np.float32))
        rows["medsam_auto_bbox"]["dice"].append(dice_coefficient(pred_t, mask_t))
        rows["medsam_auto_bbox"]["iou"].append(iou_score(pred_t, mask_t))
        rows["medsam_auto_bbox"]["hd95"].append(hausdorff95(pred_t, mask_t))
        # Log localizer bbox quality
        gt_box_unet = gt_bbox_prompt((mask_gt_np > 0.5).astype(np.uint8), padding=10)
        localizer_bbox_ious.append(bbox_iou(auto_box, gt_box_unet))

        # ---------- 5. MedSAM + GradCAM bbox [REALISTIC] ----------
        gradcam_box = get_gradcam_bbox(localizer, img_tensor, image_size=image_size_unet)
        gradcam_box_sam = (gradcam_box * scale).astype(np.float32)
        pred_medsam_gradcam = predict_with_bbox_prompt(medsam_predictor, image_rgb, gradcam_box_sam)
        pred_t = torch.from_numpy(pred_medsam_gradcam.astype(np.float32))
        rows["medsam_gradcam_bbox"]["dice"].append(dice_coefficient(pred_t, mask_t))
        rows["medsam_gradcam_bbox"]["iou"].append(iou_score(pred_t, mask_t))
        rows["medsam_gradcam_bbox"]["hd95"].append(hausdorff95(pred_t, mask_t))

        # ---------- 6 & 7. Fine-tuned MedSAM [UNREALISTIC + REALISTIC] ----------
        if finetuned_medsam is not None:
            # 6. Fine-tuned + GT bbox [UNREALISTIC]
            pred_ft_gt = _infer_finetuned_medsam(
                finetuned_medsam, image_rgb, gt_box, device, image_size=image_size_sam
            )
            pred_t = torch.from_numpy(pred_ft_gt)
            rows["medsam_ft_gt_bbox"]["dice"].append(dice_coefficient(pred_t, mask_t))
            rows["medsam_ft_gt_bbox"]["iou"].append(iou_score(pred_t, mask_t))
            rows["medsam_ft_gt_bbox"]["hd95"].append(hausdorff95(pred_t, mask_t))

            # 7. Fine-tuned + auto bbox [REALISTIC]
            pred_ft_auto = _infer_finetuned_medsam(
                finetuned_medsam, image_rgb, auto_box_sam, device, image_size=image_size_sam
            )
            pred_t = torch.from_numpy(pred_ft_auto)
            rows["medsam_ft_auto_bbox"]["dice"].append(dice_coefficient(pred_t, mask_t))
            rows["medsam_ft_auto_bbox"]["iou"].append(iou_score(pred_t, mask_t))
            rows["medsam_ft_auto_bbox"]["hd95"].append(hausdorff95(pred_t, mask_t))

    # --- Aggregate ---
    labels = {
        "unet": "UNet ResNet34",
        "sam_gt_centroid": "SAM ViT-H zero-shot + GT centroid [UNREALISTIC]",
        "medsam_gt_bbox": "MedSAM ViT-B zero-shot + GT bbox [UNREALISTIC]",
        "medsam_auto_bbox": "MedSAM ViT-B zero-shot + Auto bbox [REALISTIC]",
        "medsam_gradcam_bbox": "MedSAM ViT-B zero-shot + GradCAM bbox [REALISTIC]",
        "medsam_ft_gt_bbox": "MedSAM ViT-B fine-tuned + GT bbox [UNREALISTIC]",
        "medsam_ft_auto_bbox": "MedSAM ViT-B fine-tuned + Auto bbox [REALISTIC]",
    }
    results = []
    for key, label in labels.items():
        d = rows[key]
        if not d["dice"]:  # fine-tuned rows skipped if no checkpoint
            continue
        results.append({
            "Approach": label,
            "Dice mean": np.mean(d["dice"]),
            "Dice std": np.std(d["dice"]),
            "IoU mean": np.mean(d["iou"]),
            "IoU std": np.std(d["iou"]),
            "HD95 mean": np.mean(d["hd95"]),
            "HD95 std": np.std(d["hd95"]),
        })

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nBenchmark results saved to {output_csv}")
    print(df.to_string(index=False))
    print(f"\nLocalizer bbox IoU: {np.mean(localizer_bbox_ious):.4f} ± {np.std(localizer_bbox_ious):.4f}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    p.add_argument("--unet-ckpt", type=Path, default=Path("checkpoints/best_unet.pth"))
    p.add_argument("--localizer-ckpt", type=Path, default=Path("checkpoints/best_localizer.pth"))
    p.add_argument("--medsam-ckpt", type=Path, default=Path("checkpoints/medsam_vit_b.pth"))
    p.add_argument("--sam-ckpt", type=Path, default=Path("checkpoints/sam_vit_h_4b8939.pth"))
    p.add_argument("--finetuned-medsam-ckpt", type=Path, default=Path("checkpoints/best_medsam.pth"))
    p.add_argument("--output", type=Path, default=Path("outputs/metrics/benchmark.csv"))
    p.add_argument("--all", action="store_true", default=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(
        test_csv=args.test_csv,
        unet_ckpt=args.unet_ckpt,
        localizer_ckpt=args.localizer_ckpt,
        medsam_ckpt=args.medsam_ckpt,
        sam_ckpt=args.sam_ckpt,
        output_csv=args.output,
        finetuned_medsam_ckpt=args.finetuned_medsam_ckpt,
        device=device,
    )


if __name__ == "__main__":
    main()
