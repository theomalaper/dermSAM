"""Unified training entry point for UNet, Localizer, and MedSAM fine-tune."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.dataset import get_train_transforms, get_val_transforms, make_loader
from src.models.localizer import LesionLocalizer, mask_to_bbox
from src.models.medsam_finetune import MedSAMFinetune
from src.models.sam_inference import gt_bbox_prompt
from src.models.unet_baseline import UNetBaseline
from src.utils import dice_coefficient, load_checkpoint, save_checkpoint, set_seed


# ---------------------------------------------------------------------------
# Per-model training steps
# ---------------------------------------------------------------------------

def train_epoch_unet(
    model: UNetBaseline,
    loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    accum_steps: int = 1,
) -> float:
    """Run one training epoch for UNet.

    Args:
        model: UNetBaseline model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device string.
        accum_steps: Gradient accumulation steps.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images)
            loss = model.compute_loss(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


def train_epoch_localizer(
    model: LesionLocalizer,
    loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    image_size: int = 512,
) -> float:
    """Run one training epoch for the lesion localizer.

    Args:
        model: LesionLocalizer model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device string.
        image_size: Image size used to normalise GT bbox coords.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]

        # Derive GT bbox from mask (normalised to [0, 1])
        gt_bboxes = torch.stack([mask_to_bbox(m, image_size=image_size) for m in masks]).to(device)

        with autocast("cuda", enabled=scaler.is_enabled()):
            pred_bboxes = model(images)
            loss = model.compute_loss(pred_bboxes, gt_bboxes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_medsam(
    model: MedSAMFinetune,
    loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    accum_steps: int = 4,
    clip_grad: float = 1.0,
    image_size: int = 1024,
) -> float:
    """Run one training epoch for MedSAM decoder fine-tuning.

    Args:
        model: MedSAMFinetune model.
        loader: Training DataLoader (sam_mode=True, image_size=1024).
        optimizer: AdamW optimizer on trainable params only.
        scaler: GradScaler for AMP.
        device: Device string.
        accum_steps: Gradient accumulation steps (effective batch = batch * accum_steps).
        clip_grad: Max norm for gradient clipping.
        image_size: SAM expects 1024x1024 input.

    Returns:
        Mean training loss over the epoch.
    """
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # SAM mode: images are uint8 numpy arrays — convert to float tensors
        images_np = batch["image"]  # list of (H, W, 3) uint8
        masks_np = batch["mask"]    # list of (H, W) float32

        def _to_numpy(x):
            return x.numpy() if isinstance(x, torch.Tensor) else x

        images = torch.stack([
            torch.from_numpy(_to_numpy(img).astype("float32") / 255.0).permute(2, 0, 1)
            if _to_numpy(img).ndim == 3
            else torch.from_numpy(_to_numpy(img).astype("float32") / 255.0)
            for img in images_np
        ]).to(device)

        masks = torch.stack([
            torch.from_numpy(_to_numpy(m)).unsqueeze(0) if not isinstance(m, torch.Tensor)
            else m.unsqueeze(0) if m.ndim == 2 else m
            for m in masks_np
        ]).to(device)

        # Derive GT bbox for prompt (UNREALISTIC — training only, GT available)
        gt_bboxes = []
        for m in masks_np:
            m_np = _to_numpy(m)
            bbox = gt_bbox_prompt((m_np > 0.5).astype("uint8"), padding=10)
            gt_bboxes.append(torch.from_numpy(bbox))
        bbox_tensor = torch.stack(gt_bboxes).to(device)

        # Resize GT masks to 256x256 for loss (SAM decoder output size)
        masks_256 = F.interpolate(masks, size=(256, 256), mode="nearest")

        with autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images, bbox_tensor)
            loss = model.compute_loss(logits, masks_256) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device: str, model_type: str = "unet") -> float:
    """Run validation and return mean Dice coefficient.

    Args:
        model: Model to evaluate.
        loader: Validation DataLoader.
        device: Device string.
        model_type: One of 'unet', 'localizer', 'medsam'.

    Returns:
        Mean Dice coefficient over the validation set.
    """
    import torch.nn.functional as F

    model.eval()
    dice_scores = []

    for batch in loader:
        if model_type == "localizer":
            # No Dice for localizer — return placeholder
            return -1.0

        images = batch["image"].to(device)
        masks = batch["mask"]

        with autocast("cuda", enabled=torch.cuda.is_available()):
            if model_type == "unet":
                logits = model(images)
                preds = torch.sigmoid(logits).cpu()
            elif model_type == "medsam":
                # Placeholder — full MedSAM val needs bbox input
                continue

        for pred, mask in zip(preds, masks):
            dice_scores.append(dice_coefficient(pred, mask))

    return float(sum(dice_scores) / len(dice_scores)) if dice_scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UNet / Localizer / MedSAM")
    p.add_argument("--model", choices=["unet", "localizer", "medsam"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--medsam-checkpoint", type=Path, default=Path("checkpoints/medsam_vit_b.pth"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    sam_mode = args.model == "medsam"
    img_size = 1024 if sam_mode else args.image_size

    train_loader = make_loader(
        args.splits_dir / "train.csv",
        get_train_transforms(img_size, sam_mode=sam_mode),
        batch_size=args.batch_size,
        shuffle=True,
        sam_mode=sam_mode,
        image_size=img_size,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        args.splits_dir / "val.csv",
        get_val_transforms(img_size, sam_mode=sam_mode),
        batch_size=args.batch_size,
        shuffle=False,
        sam_mode=sam_mode,
        image_size=img_size,
        num_workers=args.num_workers,
    )

    # --- Model & optimizer ---
    if args.model == "unet":
        model = UNetBaseline().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == "localizer":
        model = LesionLocalizer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == "medsam":
        model = MedSAMFinetune(args.medsam_checkpoint).to(device)
        optimizer = optim.AdamW(
            model.trainable_parameters(), lr=args.lr, weight_decay=1e-4
        )

    # --- Scheduler ---
    if args.scheduler == "plateau":
        plateau_mode = "min" if args.model == "localizer" else "max"
        scheduler = ReduceLROnPlateau(optimizer, mode=plateau_mode, patience=5, factor=0.5)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- AMP ---
    scaler = GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())

    # --- Resume ---
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume is not None:
        start_epoch, best_val_dice = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}, best val Dice {best_val_dice:.4f}")

    # --- wandb ---
    wandb.init(project="melanoma-sam", config=vars(args), name=f"{args.model}_lr{args.lr}")

    # --- Training loop ---
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        if args.model == "unet":
            train_loss = train_epoch_unet(model, train_loader, optimizer, scaler, device, args.grad_accum)
        elif args.model == "localizer":
            train_loss = train_epoch_localizer(model, train_loader, optimizer, scaler, device, img_size)
        elif args.model == "medsam":
            train_loss = train_epoch_medsam(
                model, train_loader, optimizer, scaler, device,
                accum_steps=args.grad_accum, clip_grad=args.clip_grad, image_size=img_size,
            )

        val_dice = validate(model, val_loader, device, model_type=args.model)

        if args.scheduler == "plateau":
            scheduler.step(train_loss if args.model == "localizer" else val_dice)
        else:
            scheduler.step()

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_dice": val_dice,
                   "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch+1}/{args.epochs} | loss {train_loss:.4f} | val_dice {val_dice:.4f}")

        # --- Checkpoint ---
        # Localizer uses loss (lower=better); others use val_dice (higher=better)
        if args.model == "localizer":
            is_best = train_loss < -best_val_dice  # store negative loss in best_val_dice slot
            if epoch == 0 or train_loss < getattr(main, "_best_localizer_loss", float("inf")):
                main._best_localizer_loss = train_loss
                patience_counter = 0
                save_checkpoint(
                    args.checkpoints_dir / f"best_{args.model}.pth",
                    model, optimizer, scheduler, epoch + 1, train_loss,
                )
                print(f"  New best loss: {train_loss:.4f} — saved best_{args.model}.pth")
            else:
                patience_counter += 1
        else:
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_counter = 0
                save_checkpoint(
                    args.checkpoints_dir / f"best_{args.model}.pth",
                    model, optimizer, scheduler, epoch + 1, best_val_dice,
                )
                print(f"  New best: {best_val_dice:.4f} — saved best_{args.model}.pth")
            else:
                patience_counter += 1

        # Also save latest
        save_checkpoint(
            args.checkpoints_dir / f"last_{args.model}.pth",
            model, optimizer, scheduler, epoch + 1, best_val_dice,
        )

        # --- Early stopping ---
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {args.early_stopping_patience} epochs)")
            break

    wandb.finish()
    print(f"Training complete. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
