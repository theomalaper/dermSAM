"""Smoke tests for utility functions."""

import numpy as np
import pytest
import torch

from src.utils import bbox_iou, dice_coefficient, hausdorff95, iou_score, set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.rand(10)
    set_seed(42)
    b = torch.rand(10)
    assert torch.allclose(a, b)


def test_dice_perfect():
    mask = torch.ones(256, 256)
    assert dice_coefficient(mask * 10, mask) == pytest.approx(1.0, abs=1e-4)


def test_dice_zero():
    pred = torch.zeros(256, 256)
    gt = torch.ones(256, 256)
    assert dice_coefficient(pred, gt) < 0.01


def test_iou_perfect():
    mask = torch.ones(256, 256)
    assert iou_score(mask * 10, mask) == pytest.approx(1.0, abs=1e-4)


def test_bbox_iou_perfect():
    box = np.array([10, 10, 100, 100], dtype=np.float32)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([20, 20, 30, 30], dtype=np.float32)
    assert bbox_iou(a, b) == 0.0


def test_hausdorff_same_mask():
    mask = torch.zeros(64, 64)
    mask[20:40, 20:40] = 1.0
    hd = hausdorff95(mask * 10, mask)
    assert hd == pytest.approx(0.0, abs=1e-3)
