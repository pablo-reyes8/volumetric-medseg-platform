from pathlib import Path

import numpy as np

from src.mlops.drift import build_reference_profile, evaluate_reference_drift, save_profile, load_profile
from tests.utils import write_nifti


def test_drift_profile_detects_distribution_shift(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rng = np.random.default_rng(42)

    for index in range(3):
        write_nifti(
            baseline_dir / f"case_{index:03d}.nii.gz",
            rng.normal(0.2, 0.05, size=(8, 8, 8)).astype(np.float32),
        )
        write_nifti(
            candidate_dir / f"case_{index:03d}.nii.gz",
            rng.normal(0.8, 0.05, size=(8, 8, 8)).astype(np.float32),
        )

    baseline = build_reference_profile(baseline_dir, dataset_version="2026.03.21")
    report = evaluate_reference_drift(candidate_dir, baseline, mean_shift_threshold=0.03)
    path = save_profile(report, tmp_path / "drift_report.json")
    loaded = load_profile(path)

    assert baseline["num_volumes"] == 3
    assert report["status"] == "drift_detected"
    assert loaded["status"] == "drift_detected"
