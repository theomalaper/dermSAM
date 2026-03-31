# Architecture & Design Decisions

## The Deployment Gap Problem

Published SAM/MedSAM papers report results using ground-truth-derived prompts
(GT centroids, GT bounding boxes). These are unavailable at clinical deployment time.
This project makes that gap explicit and proposes a two-stage pipeline that closes it.

## Two-Stage Pipeline

### Stage 1 — Lesion Localizer
- EfficientNet-B0 (timm, ImageNet pretrained) with a 4-output bbox regression head
- Sigmoid-activated outputs → normalised coords in [0, 1]
- Loss: SmoothL1 on normalised bbox coords derived from GT masks
- Alternative prompt: GradCAM from final conv block → threshold → bbox

### Stage 2 — MedSAM Inference
- ViT-B architecture loaded from MedSAM checkpoint (bowang-lab HuggingFace)
- Image encoder frozen; mask decoder trainable during fine-tuning
- Accepts bbox prompt from Stage 1 — no GT used at inference

## Why Not MedSAM2?
MedSAM2 (based on SAM2) is designed for video sequences and 3D volumes (CT/MRI stacks).
For single 2D dermoscopy images, it offers no advantage and adds installation complexity.
MedSAM (ViT-B, 2024 Nature Communications) is the appropriate choice for ISIC 2018.

## Model Selection Rationale

| Model | Why |
|---|---|
| UNet + ResNet34 | Standard smp baseline; fast to train; well-established on ISIC |
| SAM ViT-H | Largest SAM model for strongest upper-bound zero-shot result |
| MedSAM ViT-B | Medical-domain fine-tuned; fits 16GB VRAM; Nature Communications reference |
| EfficientNet-B0 | Lightweight; fast inference; good ImageNet features for bbox regression |

## Loss Function
Segmentation: 0.5 × DiceLoss(sigmoid=True) + 0.5 × BCEWithLogitsLoss
- Dice handles class imbalance (lesion pixels ~15–30% of image)
- BCE provides stable gradients early in training

## Training Decisions
- AdamW for MedSAM (ViT architecture) vs Adam for CNN models
- Gradient clipping (max_norm=1.0) on MedSAM decoder to prevent exploding gradients
- Gradient accumulation (×4) for MedSAM to simulate batch_size=16 on 16GB VRAM
- AMP (autocast + GradScaler) throughout — required for ViT models

## Metrics
- Dice: primary segmentation quality
- IoU: standard complement to Dice
- HD95: 95th-percentile Hausdorff distance — clinically relevant for excision margin planning
- Localizer bbox IoU: diagnostic metric to assess prompt quality before MedSAM sees it

## Prompt Sensitivity Analysis
GT bbox is synthetically degraded (expanded by N pixels per side) and MedSAM Dice is
measured at each level. The auto-prompt result is plotted on this curve as a red dot,
showing where realistic deployment lands relative to the upper bound.
