# Skin Lesion Segmentation — Closing the SAM Deployment Gap

## Project Overview
Portfolio project with a specific research contribution: benchmarking SAM/MedSAM on melanoma
segmentation (ISIC 2018) while addressing the **deployment gap** — the fact that published SAM
papers use ground-truth-derived prompts that are unavailable in real clinical settings.

**Core contribution:** A two-stage automatic pipeline — a lightweight lesion localizer generates
prompts for MedSAM without any ground-truth, making the system actually deployable.

**The comparison story (5 rows in the final results table):**
1. UNet ResNet34 — supervised baseline
2. SAM ViT-H zero-shot + GT centroid prompt — unrealistic upper bound
3. MedSAM ViT-B zero-shot + GT bbox prompt — unrealistic upper bound
4. MedSAM ViT-B + **auto bbox** (from localizer) — our realistic pipeline
5. MedSAM ViT-B + **GradCAM prompt** (from localizer activations) — promptless variant

**Stack:** Python 3.10, PyTorch 2.x, segment-anything, segmentation-models-pytorch,
albumentations, timm, MONAI, Gradio, wandb, matplotlib, scikit-learn

## Repo Structure
```
melanoma-sam/
├── CLAUDE.md
├── data/
│   ├── ISIC2018_Task1_Training_Input/        # raw images (gitignored)
│   ├── ISIC2018_Task1_GroundTruth/           # binary masks (gitignored)
│   └── splits/                               # train/val/test CSVs (committed)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── dataset.py                # ISICDataset, split logic, augmentation
│   ├── models/
│   │   ├── unet_baseline.py      # smp.Unet wrapper + training step
│   │   ├── localizer.py          # EfficientNet-B0 bbox/heatmap localizer
│   │   ├── sam_inference.py      # SAM prompt strategies (GT centroid, GT bbox, auto)
│   │   ├── medsam_finetune.py    # frozen encoder, trainable decoder fine-tuning
│   │   └── gradcam_prompt.py     # GradCAM -> bbox prompt extraction
│   ├── train.py                  # unified training entry point
│   ├── evaluate.py               # full benchmark across all 5 approaches
│   ├── prompt_sensitivity.py     # prompt degradation analysis
│   └── visualise.py              # qualitative figures, failure cases
├── notebooks/
│   └── results.ipynb             # portfolio-facing narrative and figures
├── app/
│   └── demo.py                   # Gradio: upload image -> auto segmentation, no clicking
├── checkpoints/                  # gitignored
├── outputs/                      # figures, CSVs — committed
│   ├── figures/
│   └── metrics/
├── docs/
│   ├── architecture.md           # design decisions, model choices, ablations
│   └── results_log.md            # running metric log per experiment
├── tests/
├── requirements.txt
└── README.md
```

## Commands
```bash
# Environment
conda activate melanoma-sam

# Step 1 — train UNet baseline
python src/train.py --model unet --epochs 30 --lr 1e-4 --batch-size 16 --scheduler plateau --amp

# Step 2 — train lesion localizer
python src/train.py --model localizer --epochs 20 --lr 1e-4 --batch-size 32 --scheduler plateau --amp

# Step 3 — fine-tune MedSAM decoder
python src/train.py --model medsam --epochs 20 --lr 1e-4 --freeze-encoder --batch-size 4 --scheduler cosine --amp --grad-accum 4 --clip-grad 1.0

# Step 4 — full benchmark (all 5 approaches on test set)
python src/evaluate.py --all --output outputs/metrics/benchmark.csv

# Step 5 — prompt degradation analysis
python src/prompt_sensitivity.py --offsets 0 10 25 50 100 200

# Launch demo
python app/demo.py

# Tests
pytest tests/ -v
```

## The Two-Stage Pipeline — How It Works

### Stage 1: Lesion Localizer
- Architecture: EfficientNet-B0 (timm) with a bbox regression head (4 outputs: x0, y0, x1, y1,
  sigmoid-scaled to image dims)
- Trained with SmoothL1 loss on bbox coords derived from GT masks
- Also extract GradCAM heatmap from final conv layer -> threshold at 0.5 -> get bbox
- This gives two auto-prompt variants: direct bbox regression, and GradCAM-derived bbox

### Stage 2: MedSAM Inference
- Auto bbox from localizer -> fed directly to MedSAM predictor
- Evaluate vs GT bbox to quantify how much prompt degradation hurts downstream Dice

### Key Analysis — Prompt Degradation Curve
prompt_sensitivity.py takes GT bbox and artificially degrades it (expand/shift by N pixels),
measuring Dice at each level. Gives a "tolerance curve" showing MedSAM's robustness to imperfect
prompts. The auto-prompt performance is plotted as a dot on this curve — showing where realistic
deployment lands relative to the theoretical upper bound.

## Key Conventions

**Data:**
- Split is fixed at 80/10/10, seeded at 42, stored in data/splits/ CSVs — never regenerate
- ISIC 2018 Task 1: 2594 training images → ~2075 train / 259 val / 260 test (stratified shuffle split, seed 42)
- Masks are binary float32: 0.0 or 1.0
- Images: UNet uses 512x512, SAM uses 1024x1024 — ISICDataset accepts a `sam_mode: bool` flag
- Log lesion pixel fraction per split (expect ~15-30%) for class imbalance reporting
- Safe dermoscopy augmentations: HorizontalFlip, rotation ±15°, slight brightness/contrast jitter. Avoid vertical flip, strong elastic transforms (distort lesion boundaries), and aggressive colour jitter.

**Models:**
- SAM image encoder always frozen — only mask decoder + prompt encoder trainable
- Use vit_b for MedSAM (fits 16GB VRAM); vit_h for SAM zero-shot eval only (inference only)
- UNet encoder: resnet34, imagenet pretrained
- Localizer: efficientnet_b0 from timm, imagenet pretrained, num_classes=0 + custom bbox head
- Loss (segmentation): 0.5 * DiceLoss(sigmoid=True) + 0.5 * BCEWithLogitsLoss
- Loss (localizer): SmoothL1Loss on normalised bbox coords [0, 1]

**Prompts — critical methodology:**
- GT centroid prompt: np.argwhere(mask > 0).mean(axis=0) — always labelled UNREALISTIC in code comments
- GT bbox prompt: tight bbox from GT mask + 10px padding — always labelled UNREALISTIC in code comments
- Auto bbox: output of localizer, no GT used — labelled REALISTIC/DEPLOYABLE
- GradCAM bbox: threshold GradCAM at 0.5, bounding box of activation region — labelled REALISTIC
- All prompt derivation logic lives in sam_inference.py only — never duplicated elsewhere

**Metrics:**
- Primary: Dice coefficient
- Secondary: IoU, Hausdorff distance HD95 (boundary quality — clinically relevant for excision planning)
- Also report: localizer bbox IoU (quality of auto-prompts before MedSAM sees them)
- Always report mean +/- std over test set, not just mean
- Compute metrics on original-resolution masks — upsample prediction to original dims before Dice/HD95, never on resized output
- Published baseline to beat: ResUNet++ Dice ~0.7726 (Jha et al. 2019)
- Optional TTA for final reported numbers only: horizontal flip + average → free ~0.5% Dice; never use during development

**Training:**
- LR scheduling: UNet + Localizer use `ReduceLROnPlateau(patience=5, factor=0.5)`; MedSAM fine-tune uses linear warmup (first 5% of steps) + cosine decay
- Optimizer: Adam for UNet/EfficientNet; `AdamW(lr=1e-4, weight_decay=1e-4)` for MedSAM (ViT architecture)
- Mixed precision: always use `torch.cuda.amp.autocast()` + `GradScaler` — required for ViT models; SAM ViT-H at 1024x1024 will OOM without it
- Gradient clipping for MedSAM decoder: `clip_grad_norm_(params, max_norm=1.0)` — transformers are prone to exploding gradients
- Gradient accumulation for MedSAM: `accum_steps=4` → effective batch of 16, same VRAM cost as batch_size=4
- Early stopping: patience=10 epochs on val Dice for all models

**Reproducibility:**
- `set_seed(42)` must also set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` — without these, cudnn picks non-deterministic conv algorithms

**Experiment tracking:**
- All runs logged to wandb, project name `melanoma-sam`
- Checkpoint naming: {model}_{lr}_{epoch}_{val_dice:.4f}.pth
- Checkpoint dict format: `{"model_state_dict": ..., "optimizer_state_dict": ..., "scheduler_state_dict": ..., "epoch": N, "best_val_dice": X}` — save full dict, not just weights
- Best checkpoint = highest val Dice

**Code style:**
- Type hints on all function signatures
- Docstrings: one-line summary + Args + Returns on all public functions
- No logic in notebooks — notebooks import from src/ only
- Single set_seed(42) at top of every script entry point

## Figures to Produce (Portfolio-Facing)

1. **Main results table** — 5-row benchmark, Dice / IoU / HD95 per approach
2. **Deployment gap figure** — bar chart: realistic (auto-prompt) vs unrealistic (GT-prompt) Dice.
   This is the visual argument. The gap between rows 3 and 4 is the contribution.
3. **Prompt degradation curve** — Dice vs bbox perturbation magnitude (x-axis: pixels of shift/expansion),
   with the auto-prompt result plotted as a red dot showing where realistic deployment lands
4. **Qualitative grid** — 5 columns: image | GT mask | UNet | MedSAM+GT bbox | MedSAM+Auto bbox
5. **Failure case analysis** — 6 cases annotated by failure type:
   poor localizer bbox / ambiguous lesion boundary / atypical morphology / small lesion / hair artefact

## Gradio Demo
The demo must be fully automatic — user uploads an image, gets segmentation back with no clicking.
This is the proof of concept for the deployable pipeline.
Internal flow: image -> localizer -> auto bbox -> MedSAM -> mask overlay returned.
Show the auto-generated bbox as a rectangle overlay alongside the final segmentation.

## Gotchas
- SAM expects RGB uint8 numpy arrays before predictor.set_image() — never pass tensors or float arrays
- Albumentations normalisation is for UNet only — SAM handles its own normalisation internally
- ISIC mask filenames have _segmentation suffix that image filenames do not — handle in dataset.py
- MedSAM checkpoint is vit_b architecture — passing to vit_h registry silently loads wrong weights
- EfficientNet with num_classes=0 returns features, not logits — add bbox head explicitly
- GradCAM hooks must be registered before forward pass and removed after to avoid memory leaks
- ISIC images vary in resolution (not all square) — resize before any processing, not after

## References
- MedSAM: Ma et al. 2024, Nature Communications — medsam_vit_b.pth from bowang-lab HuggingFace
- SAM: Kirillov et al. 2023, ICCV — sam_vit_h_4b8939.pth from Meta
- ISIC 2018 baseline (ResUNet++): Jha et al. 2019 — Dice 0.7726 to beat
- Deployment gap framing and design decisions: see docs/architecture.md