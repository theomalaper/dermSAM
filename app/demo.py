"""Gradio demo: upload image → automatic segmentation, no clicking required.

Internal flow: image → localizer → auto bbox → MedSAM → mask overlay returned.
Shows auto-generated bbox as a rectangle overlay alongside final segmentation.
"""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import gradio as gr
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.models.localizer import LesionLocalizer
from src.models.sam_inference import _load_medsam_predictor, auto_bbox_prompt, predict_with_bbox_prompt
from src.utils import load_checkpoint, set_seed

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

_localizer: LesionLocalizer = None
_medsam_predictor = None
_device: str = "cpu"

_unet_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_models(localizer_ckpt: Path, medsam_ckpt: Path, device: str = "cpu") -> None:
    """Load localizer and MedSAM models into global state.

    Args:
        localizer_ckpt: Path to best localizer checkpoint.
        medsam_ckpt: Path to medsam_vit_b.pth.
        device: Device string.
    """
    global _localizer, _medsam_predictor, _device
    _device = device

    _localizer = LesionLocalizer(pretrained=False).to(device)
    load_checkpoint(localizer_ckpt, _localizer, device=device)
    _localizer.eval()

    _medsam_predictor = _load_medsam_predictor(medsam_ckpt)
    print("Models loaded.")


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def segment_image(image: np.ndarray):
    """Full auto pipeline: image → localizer → MedSAM → annotated output.

    Args:
        image: RGB uint8 numpy array from Gradio upload, shape (H, W, 3).

    Returns:
        Tuple of (segmentation overlay, bbox overlay) both as RGB uint8 arrays.
    """
    if _localizer is None or _medsam_predictor is None:
        raise gr.Error("Models not loaded. Run demo.py with --localizer-ckpt and --medsam-ckpt.")

    # --- Stage 1: Localizer → auto bbox ---
    # Resize to 512x512 for localizer
    orig_h, orig_w = image.shape[:2]
    img_512 = cv2.resize(image, (512, 512))
    augmented = _unet_transforms(image=img_512)
    img_tensor = augmented["image"].unsqueeze(0).to(_device)

    auto_box_512 = auto_bbox_prompt(_localizer, img_tensor, image_size=512)

    # Scale bbox to 1024x1024 for MedSAM
    auto_box_1024 = (auto_box_512 * 2.0).astype(np.float32)

    # --- Stage 2: MedSAM → mask ---
    img_1024 = cv2.resize(image, (1024, 1024))
    mask = predict_with_bbox_prompt(_medsam_predictor, img_1024, auto_box_1024)

    # Resize mask back to original resolution
    mask_orig = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # --- Build output overlays ---
    # 1. Segmentation overlay: green mask on original image
    seg_overlay = image.copy()
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = mask_orig * 180  # green channel
    seg_overlay = cv2.addWeighted(seg_overlay, 0.7, green_mask, 0.5, 0)

    # Draw contour
    contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_overlay, contours, -1, (0, 255, 0), 2)

    # 2. Bbox overlay: show auto-generated bbox on 512x512 image
    bbox_overlay = img_512.copy()
    x0, y0, x1, y1 = auto_box_512.astype(int)
    cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 200, 0), 2)
    cv2.putText(bbox_overlay, "Auto bbox", (x0, max(y0 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    return seg_overlay, bbox_overlay


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks interface.

    Returns:
        gr.Blocks application.
    """
    examples_dir = Path(__file__).parent / "examples"
    example_images = sorted(examples_dir.glob("*.jpg")) + sorted(examples_dir.glob("*.png"))
    examples = [[str(img)] for img in example_images] if example_images else None

    with gr.Blocks(title="DermSAM — Automatic Lesion Segmentation") as demo:
        gr.Markdown("""
        # DermSAM — Automatic Melanoma Segmentation
        Upload a dermoscopy image. The pipeline automatically localises the lesion and
        segments it — **no clicking required**.

        **How it works:**
        1. EfficientNet-B0 localizer predicts a bounding box (no ground truth used)
        2. MedSAM ViT-B uses that box as a prompt to produce the final segmentation
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input dermoscopy image", type="numpy")
                run_btn = gr.Button("Segment", variant="primary")
            with gr.Column():
                seg_output = gr.Image(label="Segmentation overlay")
                bbox_output = gr.Image(label="Auto-generated bbox (localizer output)")

        run_btn.click(fn=segment_image, inputs=[input_image], outputs=[seg_output, bbox_output])

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_image],
                label="Example dermoscopy images — click to load",
            )

        gr.Markdown("""
        ---
        **Disclaimer:** Research prototype only. Not validated for clinical use.
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    p = argparse.ArgumentParser(description="Launch DermSAM Gradio demo")
    p.add_argument("--localizer-ckpt", type=Path, default=Path("checkpoints/best_localizer.pth"))
    p.add_argument("--medsam-ckpt", type=Path, default=Path("checkpoints/medsam_vit_b.pth"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", default=False)
    args = p.parse_args()

    load_models(args.localizer_ckpt, args.medsam_ckpt, args.device)
    interface = build_interface()
    interface.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
