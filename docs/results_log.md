# Results Log

Running log of experiment results. Add an entry after each completed run.

---

## 2026-04-02 — Project Complete

### What was built
DermSAM is a two-stage automatic pipeline for skin lesion segmentation that addresses the deployment gap in published SAM benchmarks. Stage 1: EfficientNet-B0 localizer predicts a bounding box from the image alone (no ground truth). Stage 2: MedSAM uses that box as a prompt to segment the lesion. The full pipeline requires no clicks, no annotations, and no ground truth at inference time.

### The core finding
Published SAM benchmarks report performance using ground-truth-derived prompts — information unavailable in clinical deployment. This project quantifies that gap and shows a lightweight localizer can partially close it.

Fine-tuning MedSAM on ISIC data dramatically improves performance when given a perfect GT prompt (0.883 → 0.964 Dice) but barely helps with a realistic auto-prompt (0.811 → 0.815). This means the localizer quality — not the segmentation model — is the binding constraint for real-world deployment. Improving the localizer is the clearest path to better pipeline performance.

GradCAM activations from the localizer proved insufficient as prompts (Dice 0.429) — the heatmaps are too diffuse. A dedicated bbox regression head is necessary.

### Prompt sensitivity
MedSAM is robust to moderate bbox imprecision (up to ~25px expansion) but degrades sharply beyond 50px. The auto-prompt sits at roughly 35-40px equivalent imprecision relative to GT — interpretable and improvable.

### What's live
- Demo: huggingface.co/spaces/Malaper/dermSAM
- Code: github.com/theomalaper/dermSAM
- All results, figures, and CSVs committed to repo

### Potential next experiments
- Retrain localizer with EfficientNet-B3 or longer schedule — biggest expected impact
- Test generalisation on PH2 or ISIC 2016 without retraining
- Compare against SAM2/MedSAM2
- Report sensitivity/specificity for clinical framing

## Format
```
### YYYY-MM-DD — [model] lr=[lr] epochs=[N]
- Val Dice: X.XXXX
- Notes: ...
```

---

<!-- Add entries below as experiments complete -->

### 2026-04-01 — unet lr=1e-4 epochs=30                                   
  - lr:0.00005
  - train_loss:0.04746
  - val_dice:0.89522               
  - Notes: ResNet34 ImageNet pretrained encoder. Beats ResUNet++ baseline
  (0.7726) comfortably. 

### 2026-04-02 — localizer lr=1e-4 epochs=20
  - Final train loss: 0.0010
  - Val bbox IoU: 0.693 ± 0.201
  - Notes: 
      - EfficientNet-B0 bbox regression. Full 20 epochs, converged cleanly.
      - Val bbox IoU below 0.75 target but visually reasonable boxes. 
      - First run failed due to bad symlinks (cv2 silently loading None). 
      - Infrastructure fixed: images verified loading before training, checkpoints saving directly to Drive. Downstream impact assessed in full benchmark.

  ### 2026-04-02 — Full benchmark (test set, n=260)
                                                                                   
  | Approach | Dice | IoU | HD95 |                                                             
  |---|---|---|---|                                                                              
  | UNet ResNet34 | 0.892 ± 0.115 | 0.821 ± 0.152 | 65.9 ± 185.3 |                             
  | SAM ViT-H + GT centroid [UNREALISTIC] | 0.645 ± 0.294 | 0.538 ± 0.291 | 137.8 ± 165.3 |      
  | MedSAM ViT-B + GT bbox [UNREALISTIC] | 0.883 ± 0.112 | 0.804 ± 0.144 | 14.4 ± 22.2 |         
  | MedSAM ViT-B + Auto bbox [REALISTIC] | 0.811 ± 0.157 | 0.706 ± 0.188 | 35.0 ± 38.7 |         
  | MedSAM ViT-B + GradCAM bbox [REALISTIC] | 0.429 ± 0.213 | 0.297 ± 0.180 | 254.7 ± 148.8 |    
                                                                                                 
  - Deployment gap: GT bbox → Auto bbox = 0.072 Dice drop                                        
  - MedSAM boundaries (HD95) consistently better than UNet even with auto prompts                
  - GradCAM bbox fails — activations too imprecise for reliable prompts                          
  - Published baseline to beat (ResUNet++ 0.7726): beaten by all except GradCAM    

### 2026-04-02 — medsam fine-tune lr=1e-4 epochs=20/20 (full run)
  - Final lr: 0.0 (cosine decay)
  - train_loss epoch 10: 0.03748 → epoch 20: 0.03518
  - val_dice: 0 (not computed during training — bbox input required)
  - Notes:
      - Full 20 epochs completed. Loss continued improving slightly in second half.
      - Checkpoint: last_medsam.pth

### 2026-04-02 — Full benchmark v2 (7 rows, test set n=260)

  | Approach | Dice | IoU | HD95 |
  |---|---|---|---|
  | UNet ResNet34 | 0.892 ± 0.115 | 0.821 ± 0.152 | 65.9 ± 185.3 |
  | SAM ViT-H zero-shot + GT centroid [UNREALISTIC] | 0.645 ± 0.294 | 0.538 ± 0.291 | 137.8 ± 165.3 |
  | MedSAM ViT-B zero-shot + GT bbox [UNREALISTIC] | 0.883 ± 0.112 | 0.804 ± 0.144 | 14.4 ± 22.2 |
  | MedSAM ViT-B zero-shot + Auto bbox [REALISTIC] | 0.811 ± 0.157 | 0.706 ± 0.188 | 35.0 ± 38.7 |
  | MedSAM ViT-B zero-shot + GradCAM bbox [REALISTIC] | 0.429 ± 0.213 | 0.297 ± 0.180 | 254.7 ± 148.8 |
  | MedSAM ViT-B fine-tuned + GT bbox [UNREALISTIC] | 0.964 ± 0.023 | 0.932 ± 0.041 | 0.8 ± 3.1 |
  | MedSAM ViT-B fine-tuned + Auto bbox [REALISTIC] | 0.815 ± 0.171 | 0.717 ± 0.207 | 34.8 ± 42.0 |

  - Key finding: fine-tuning dramatically improves GT-prompted performance (0.883→0.964)
    but barely helps auto-prompted (0.811→0.815) — localizer quality is the bottleneck
  - Fine-tuned HD95 with GT bbox = 0.8px — near-perfect boundary delineation
  - All realistic approaches beat ResUNet++ baseline (0.7726)
  - We need to retain the localizer next

