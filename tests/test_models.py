"""Smoke tests: forward pass shape checks for all models."""

import numpy as np
import pytest
import torch

from src.models.localizer import LesionLocalizer, mask_to_bbox
from src.models.unet_baseline import UNetBaseline


def test_unet_forward_shape():
    model = UNetBaseline()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 1, 512, 512)


def test_unet_loss_computes():
    model = UNetBaseline()
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    masks = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss = model.compute_loss(logits, masks)
    assert loss.item() > 0


def test_localizer_forward_shape():
    model = LesionLocalizer(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 4)
    # Sigmoid output should be in [0, 1]
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_mask_to_bbox_normalised():
    mask = torch.zeros(512, 512)
    mask[100:300, 150:400] = 1.0
    bbox = mask_to_bbox(mask, padding=0, image_size=512)
    assert bbox.shape == (4,)
    assert (bbox >= 0).all() and (bbox <= 1).all()
    # x0 < x1, y0 < y1
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


def test_mask_to_bbox_empty():
    mask = torch.zeros(512, 512)
    bbox = mask_to_bbox(mask)
    assert bbox.sum().item() == pytest.approx(0.0)
