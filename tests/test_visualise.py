"""Tests for visualise.py — pure-data figures only (no GPU required)."""

from pathlib import Path

import pandas as pd
import pytest

from src.visualise import plot_deployment_gap, plot_results_table


@pytest.fixture
def benchmark_csv(tmp_path):
    """Write a minimal benchmark CSV matching evaluate.py output format."""
    rows = [
        {"Approach": "UNet ResNet34",
         "Dice mean": 0.892, "Dice std": 0.115,
         "IoU mean": 0.821, "IoU std": 0.152,
         "HD95 mean": 65.9, "HD95 std": 185.3},
        {"Approach": "MedSAM ViT-B zero-shot + GT bbox [UNREALISTIC]",
         "Dice mean": 0.883, "Dice std": 0.112,
         "IoU mean": 0.804, "IoU std": 0.144,
         "HD95 mean": 14.4, "HD95 std": 22.2},
        {"Approach": "MedSAM ViT-B zero-shot + Auto bbox [REALISTIC]",
         "Dice mean": 0.811, "Dice std": 0.157,
         "IoU mean": 0.706, "IoU std": 0.188,
         "HD95 mean": 35.0, "HD95 std": 38.7},
    ]
    path = tmp_path / "benchmark.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_results_table_creates_file(benchmark_csv, tmp_path):
    out = tmp_path / "results_table.png"
    plot_results_table(benchmark_csv, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_deployment_gap_creates_file(benchmark_csv, tmp_path):
    out = tmp_path / "deployment_gap.png"
    plot_deployment_gap(benchmark_csv, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_results_table_creates_parent_dirs(benchmark_csv, tmp_path):
    out = tmp_path / "nested" / "dir" / "results_table.png"
    plot_results_table(benchmark_csv, out)
    assert out.exists()


def test_deployment_gap_creates_parent_dirs(benchmark_csv, tmp_path):
    out = tmp_path / "nested" / "dir" / "deployment_gap.png"
    plot_deployment_gap(benchmark_csv, out)
    assert out.exists()


def test_deployment_gap_handles_all_row_types(tmp_path):
    """All three colour categories (unrealistic/realistic/baseline) handled without error."""
    rows = [
        {"Approach": "UNet ResNet34",
         "Dice mean": 0.89, "Dice std": 0.1,
         "IoU mean": 0.82, "IoU std": 0.1,
         "HD95 mean": 60.0, "HD95 std": 10.0},
        {"Approach": "SAM ViT-H zero-shot + GT centroid [UNREALISTIC]",
         "Dice mean": 0.65, "Dice std": 0.2,
         "IoU mean": 0.54, "IoU std": 0.2,
         "HD95 mean": 130.0, "HD95 std": 50.0},
        {"Approach": "MedSAM ViT-B fine-tuned + Auto bbox [REALISTIC]",
         "Dice mean": 0.84, "Dice std": 0.1,
         "IoU mean": 0.74, "IoU std": 0.1,
         "HD95 mean": 25.0, "HD95 std": 15.0},
    ]
    csv_path = tmp_path / "bench.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out = tmp_path / "gap.png"
    plot_deployment_gap(csv_path, out)
    assert out.exists()
