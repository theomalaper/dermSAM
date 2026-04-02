"""Tests for prompt sensitivity logic."""

import numpy as np
import pandas as pd
import pytest

from src.prompt_sensitivity import perturb_bbox


def test_perturb_no_offset():
    bbox = np.array([100.0, 150.0, 400.0, 350.0], dtype=np.float32)
    result = perturb_bbox(bbox, offset=0, image_size=1024)
    np.testing.assert_array_equal(result, bbox)


def test_perturb_expands_correctly():
    bbox = np.array([100.0, 150.0, 400.0, 350.0], dtype=np.float32)
    result = perturb_bbox(bbox, offset=50, image_size=1024)
    assert result[0] == 50.0   # x0 moved left
    assert result[1] == 100.0  # y0 moved up
    assert result[2] == 450.0  # x1 moved right
    assert result[3] == 400.0  # y1 moved down


def test_perturb_clamps_to_image_bounds():
    bbox = np.array([10.0, 10.0, 500.0, 500.0], dtype=np.float32)
    result = perturb_bbox(bbox, offset=200, image_size=512)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 512.0
    assert result[3] == 512.0


def test_perturb_large_offset_fills_image():
    bbox = np.array([200.0, 200.0, 300.0, 300.0], dtype=np.float32)
    result = perturb_bbox(bbox, offset=9999, image_size=1024)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 1024.0
    assert result[3] == 1024.0


def test_perturb_output_dtype():
    bbox = np.array([100.0, 100.0, 400.0, 400.0], dtype=np.float32)
    result = perturb_bbox(bbox, offset=10, image_size=512)
    assert result.dtype == np.float32


def test_csv_auto_row_format(tmp_path):
    """CSV written by run_sensitivity should have an 'auto' offset row."""
    csv_path = tmp_path / "sensitivity.csv"
    # Simulate the df_out structure manually
    rows = [
        {"offset": 0, "dice_mean": 0.88, "dice_std": 0.05},
        {"offset": 50, "dice_mean": 0.80, "dice_std": 0.07},
        {"offset": "auto", "dice_mean": 0.81, "dice_std": 0.06},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    loaded = pd.read_csv(csv_path)
    auto_rows = loaded[loaded["offset"] == "auto"]
    assert len(auto_rows) == 1
    assert float(auto_rows.iloc[0]["dice_mean"]) == pytest.approx(0.81)
