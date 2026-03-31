"""MedSAM fine-tuning: frozen ViT-B encoder, trainable mask decoder."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp


class MedSAMFinetune(nn.Module):
    """MedSAM with frozen image encoder and trainable mask decoder.

    Architecture: SAM ViT-B loaded from MedSAM checkpoint.
    Only mask_decoder (and prompt_encoder) parameters are trained.
    Image encoder is frozen throughout.

    Loss: 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss (same as UNet baseline)

    Args:
        checkpoint: Path to medsam_vit_b.pth.
    """

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from segment_anything import sam_model_registry

        # MedSAM checkpoint is vit_b — never pass to vit_h registry
        self.sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))

        # Freeze image encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode image + bbox prompt, decode mask.

        Args:
            image: Batch of images, shape (B, 3, 1024, 1024). SAM expects 1024x1024.
            bbox: Batch of bboxes, shape (B, 4) in pixel coords [x0, y0, x1, y1].

        Returns:
            Logit mask tensor, shape (B, 1, 256, 256).
            Note: SAM decoder outputs 256x256 — upsample to original size for metric computation.
        """
        B = image.shape[0]
        device = image.device

        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image)  # (B, 256, 64, 64)

        masks_list = []
        for i in range(B):
            # Prompt encoder expects box in (1, 4) format
            box = bbox[i].unsqueeze(0)  # (1, 4)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
            low_res_logit, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks_list.append(low_res_logit)  # (1, 1, 256, 256)

        return torch.cat(masks_list, dim=0)  # (B, 1, 256, 256)

    def compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Args:
            logits: Raw decoder output, shape (B, 1, 256, 256).
            masks: Binary GT masks resized to 256x256, shape (B, 1, 256, 256).

        Returns:
            Scalar loss tensor.
        """
        return 0.5 * self.dice_loss(logits, masks) + 0.5 * self.bce_loss(logits, masks)

    def trainable_parameters(self):
        """Return only the trainable (non-frozen) parameters.

        Returns:
            Iterator of trainable parameters.
        """
        return (p for p in self.parameters() if p.requires_grad)
