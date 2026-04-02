"""HuggingFace Spaces entry point — loads models and launches the Gradio demo."""

import os
from pathlib import Path

# HuggingFace Spaces serves from the repo root — add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.demo import build_interface, load_models

LOCALIZER_CKPT = Path("checkpoints/best_localizer.pth")
MEDSAM_CKPT = Path("checkpoints/medsam_vit_b.pth")

load_models(LOCALIZER_CKPT, MEDSAM_CKPT, device="cpu")
demo = build_interface()
demo.launch()
