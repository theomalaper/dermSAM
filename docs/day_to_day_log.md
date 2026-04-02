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