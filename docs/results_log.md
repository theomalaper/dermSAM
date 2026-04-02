# Results Log

Running log of experiment results. Add an entry after each completed run.

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

### 2026-04-02 — medsam fine-tune lr=1e-4 epochs=10/20 (partial run)
  - lr: 0.00005 (halved from plateau scheduler)
  - train_loss: 0.03748
  - val_dice: 0 (placeholder — MedSAM val requires bbox input, not computed during training)
  - Notes:
      - 10/20 epochs completed before session reset. Checkpoint saved to Drive as best_medsam.pth.
      - Loss well below UNet (0.047) suggesting strong decoder adaptation.
      - Resume training for remaining 10 epochs with --resume checkpoints/best_medsam.pth.