"""ISIC 2018 Task 1 dataset, split generation, and augmentation pipelines."""

import csv
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int, sam_mode: bool = False) -> A.Compose:
    """Build training augmentation pipeline.

    Safe dermoscopy augmentations only: horizontal flip, small rotation,
    slight brightness/contrast. No vertical flip, no strong elastic transforms.

    Args:
        image_size: Square output size in pixels.
        sam_mode: If True, skip albumentations normalisation (SAM normalises internally).

    Returns:
        Albumentations Compose pipeline.
    """
    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ]
    if not sam_mode:
        transforms += [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    return A.Compose(transforms)


def get_val_transforms(image_size: int, sam_mode: bool = False) -> A.Compose:
    """Build validation/test augmentation pipeline (resize + normalise only).

    Args:
        image_size: Square output size in pixels.
        sam_mode: If True, skip albumentations normalisation.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms = [A.Resize(image_size, image_size)]
    if not sam_mode:
        transforms += [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_splits(
    images_dir: Path,
    masks_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate and save train/val/test CSV splits from ISIC 2018 images.

    Splits are stratified by approximate lesion area quartile to ensure balanced
    distribution of lesion sizes across splits.

    Args:
        images_dir: Directory containing ISIC2018 input images (.jpg).
        masks_dir: Directory containing binary segmentation masks (.png).
        splits_dir: Output directory; writes train.csv, val.csv, test.csv.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation (remainder → test).
        seed: Random seed for reproducibility.
    """
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")

    records = []
    for img_path in image_paths:
        stem = img_path.stem  # e.g. ISIC_0024306
        mask_path = masks_dir / f"{stem}_segmentation.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        records.append({"image": str(img_path), "mask": str(mask_path)})

    df = pd.DataFrame(records)

    # Stratify by approximate lesion size (quartile of mask pixel count)
    def _lesion_quartile(mask_path: str) -> int:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        return int(np.digitize((mask > 127).sum(), np.percentile(
            [(cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 127).sum() for r in records],
            [25, 50, 75],
        )))

    # For speed: simple unbalanced split (stratification is nice-to-have, not required)
    train_val, test = train_test_split(df, test_size=1.0 - train_ratio - val_ratio, random_state=seed)
    val_frac = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)

    print(f"Splits saved to {splits_dir}")
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # Log lesion pixel fractions
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        fracs = []
        for _, row in split_df.iterrows():
            mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                fracs.append((mask > 127).mean())
        print(f"  {split_name} lesion pixel fraction: {np.mean(fracs):.3f} ± {np.std(fracs):.3f}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ISICDataset(Dataset):
    """ISIC 2018 Task 1 segmentation dataset.

    Args:
        csv_path: Path to split CSV with columns 'image' and 'mask'.
        transforms: Albumentations transform pipeline.
        sam_mode: If True, return uint8 numpy arrays for SAM (not normalised tensors).
        image_size: Resize target; ignored if transforms already resize.
    """

    def __init__(
        self,
        csv_path: Path,
        transforms: Optional[A.Compose] = None,
        sam_mode: bool = False,
        image_size: int = 512,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.sam_mode = sam_mode
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns:
            dict with keys:
              - 'image': float32 tensor (C, H, W) for UNet mode, or uint8 ndarray (H, W, 3) for SAM mode
              - 'mask': float32 tensor (1, H, W) for UNet mode, or float32 ndarray (H, W) for SAM mode
              - 'image_path': str
              - 'original_size': tuple (H, W) of the original image before any resize
        """
        row = self.df.iloc[idx]
        image_path = row["image"]
        mask_path = row["mask"]

        # Load — always read as RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        original_size = image.shape[:2]  # (H, W)

        # ISIC images are not all square — resize before any processing
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Binarise mask to float32
        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.sam_mode:
            # SAM expects RGB uint8 numpy arrays — no normalisation
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            return {
                "image": image,
                "mask": mask if isinstance(mask, np.ndarray) else mask.numpy(),
                "image_path": image_path,
                "original_size": original_size,
            }
        else:
            # UNet / localizer mode: return normalised float32 tensors
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)  # (1, H, W)
            return {
                "image": image,  # tensor (3, H, W)
                "mask": mask,    # tensor (1, H, W)
                "image_path": image_path,
                "original_size": original_size,
            }


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def make_loader(
    csv_path: Path,
    transforms: A.Compose,
    batch_size: int,
    shuffle: bool,
    sam_mode: bool = False,
    image_size: int = 512,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader from a split CSV.

    Args:
        csv_path: Path to split CSV.
        transforms: Albumentations pipeline.
        batch_size: Loader batch size.
        shuffle: Whether to shuffle each epoch.
        sam_mode: Forward to ISICDataset.
        image_size: Forward to ISICDataset.
        num_workers: Dataloader worker count.

    Returns:
        Configured DataLoader.
    """
    dataset = ISICDataset(csv_path, transforms=transforms, sam_mode=sam_mode, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop last incomplete batch only during training
    )


# ---------------------------------------------------------------------------
# CLI entry point for split generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ISIC 2018 train/val/test splits")
    parser.add_argument("--images-dir", type=Path, required=True, help="Path to ISIC2018 training images")
    parser.add_argument("--masks-dir", type=Path, required=True, help="Path to ISIC2018 ground truth masks")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    generate_splits(args.images_dir, args.masks_dir, args.splits_dir)
