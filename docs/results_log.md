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