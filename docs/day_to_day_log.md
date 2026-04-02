  ## Day 1 — Setup & first results

  ---
  Most SAM/MedSAM papers report impressive segmentation numbers. What they don't
  mention is those numbers use ground-truth prompts — the model is essentially told
  where the lesion is before being asked to segment it. That's not how a clinic works.

  I'm calling this the deployment gap, and this project is my attempt to close it.

  The idea: a two-stage pipeline where a lightweight EfficientNet-B0 localizer predicts
   a bounding box from the image alone, which then feeds into MedSAM as a prompt. No
  ground truth. No human clicking. Fully automatic.

  To quantify the gap, I'm running 5 approaches on ISIC 2018 melanoma segmentation —
  from a supervised UNet baseline, through unrealistic GT-prompted SAM upper bounds,
  down to the realistic auto-prompted pipeline. The gap between rows 3 and 4 in that
  table is the contribution.

  Today: built the full scaffold — data pipeline, all five models, training loop with
  AMP, gradient accumulation, wandb tracking, and a Gradio demo. Everything running on
  a free Colab T4.

  First result: UNet baseline epoch 1 — val Dice 0.827, already above the published
  ResUNet++ baseline of 0.7726. Training overnight.

  Tomorrow: the lesion localizer.


 ## Day 2 Summary                                         
                                         
  Retrained the EfficientNet-B0 lesion localizer after  
  fixing infrastructure issues from yesterday (nested   
  repo, incomplete unzip, bad checkpoint). Clean run    
  this time:                                          
                       
  - Trained 20 epochs, final train loss: 0.0010         
  - Val bbox IoU: 0.693 ± 0.201
  - Boxes look visually reasonable — roughly covering   
  the lesion in most cases                              
  - Below the 0.75 IoU target but acceptable; will
  assess downstream impact in the full benchmark        
                                                      
  Key debug lessons: always verify images load with cv2 
  before training, always save checkpoints directly to
  Drive, always verify symlinks resolve correctly before
   starting a run.                                    
                       
  MedSAM fine-tuning kicked off overnight.


## Day 3 — Benchmark, Figures, Demo

MedSAM fine-tuning completed — 20 epochs, final train loss 0.035. Then ran the full
7-row benchmark on the test set (n=260). The headline numbers:

  - UNet ResNet34:                       Dice 0.892
  - MedSAM zero-shot + GT bbox [UNREAL]: Dice 0.883
  - MedSAM zero-shot + Auto bbox [REAL]: Dice 0.811
  - MedSAM fine-tuned + GT bbox [UNREAL]:Dice 0.964
  - MedSAM fine-tuned + Auto bbox [REAL]:Dice 0.815
  - GradCAM bbox [REAL]:                 Dice 0.429

The most interesting finding: fine-tuning dramatically improves performance when given
a perfect GT prompt (0.883 → 0.964) but barely helps with a realistic auto-prompt
(0.811 → 0.815). The localizer — not the segmentation model — is the binding
constraint for real-world performance. Improving the localizer is the clearest next
step.

The prompt sensitivity curve showed MedSAM is robust to moderate imprecision (up to
~25px) but degrades sharply beyond 50px. The auto-prompt sits at roughly 35-40px
equivalent imprecision — interpretable and improvable.

All four portfolio figures generated. README written. Results notebook written.
Gradio demo deployed to HuggingFace Spaces (huggingface.co/spaces/Malaper/dermSAM) —
upload an image, get a segmentation back, no clicking required.

Project is complete as a portfolio piece. Next experiments if time allows: stronger
localizer, second dataset for generalisation testing.