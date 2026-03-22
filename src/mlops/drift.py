import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import ks_2samp

from data.ingestion import list_nifti_files
from data.preprocessing import load_nifti_volume


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_values(volume: np.ndarray, max_values: int = 32768) -> np.ndarray:
    flattened = volume.reshape(-1)
    if flattened.size <= max_values:
        return flattened.astype(np.float32)
    stride = max(1, flattened.size // max_values)
    return flattened[::stride][:max_values].astype(np.float32)


def _normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    total = max(1.0, float(histogram.sum()))
    return histogram / total


def population_stability_index(expected_histogram: np.ndarray, actual_histogram: np.ndarray, eps: float = 1e-6) -> float:
    expected = _normalize_histogram(expected_histogram) + eps
    actual = _normalize_histogram(actual_histogram) + eps
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference_profile(
    volume_paths: Sequence[Path | str] | Path | str,
    dataset_version: str = "unknown",
    clip_percentiles=(1, 99),
    bins: int = 32,
    max_values_per_volume: int = 32768,
    histogram_edges: Sequence[float] | None = None,
) -> Dict[str, object]:
    if isinstance(volume_paths, (str, Path)):
        paths = list_nifti_files(volume_paths)
    else:
        paths = [Path(path) for path in volume_paths]
    if not paths:
        raise ValueError("No se encontraron volúmenes para construir el perfil de drift.")

    samples: List[np.ndarray] = []
    shapes: List[tuple[int, int, int]] = []
    spacings: List[tuple[float, float, float]] = []
    for path in paths:
        volume, _, spacing = load_nifti_volume(path)
        samples.append(_sample_values(volume, max_values=max_values_per_volume))
        shapes.append(tuple(int(dim) for dim in volume.shape))
        spacings.append(tuple(float(value) for value in spacing[:3]))

    stacked_samples = np.concatenate(samples)
    if histogram_edges is None:
        bin_edges = np.histogram_bin_edges(stacked_samples, bins=bins)
        if len(np.unique(bin_edges)) < 2:
            center = float(stacked_samples.mean()) if stacked_samples.size else 0.0
            bin_edges = np.array([center - 1.0, center + 1.0], dtype=np.float32)
    else:
        bin_edges = np.asarray(histogram_edges, dtype=np.float32)
    histogram, bin_edges = np.histogram(stacked_samples, bins=bin_edges)

    return {
        "schema_version": "1.0",
        "dataset_version": dataset_version,
        "created_at_utc": _now_utc(),
        "num_volumes": len(paths),
        "clip_percentiles": list(clip_percentiles),
        "intensity_sample": stacked_samples.tolist(),
        "histogram": histogram.tolist(),
        "bin_edges": bin_edges.tolist(),
        "summary": {
            "intensity_mean": float(stacked_samples.mean()),
            "intensity_std": float(stacked_samples.std()),
            "shape_mean": [round(float(np.mean(axis)), 3) for axis in zip(*shapes)],
            "spacing_mean": [round(float(np.mean(axis)), 5) for axis in zip(*spacings)],
        },
    }


def evaluate_reference_drift(
    volume_paths: Sequence[Path | str] | Path | str,
    baseline_profile: Dict[str, object],
    ks_threshold: float = 0.12,
    psi_threshold: float = 0.2,
    mean_shift_threshold: float = 0.08,
    std_shift_threshold: float = 0.08,
) -> Dict[str, object]:
    candidate_profile = build_reference_profile(
        volume_paths=volume_paths,
        dataset_version="candidate",
        clip_percentiles=tuple(baseline_profile["clip_percentiles"]),
        bins=len(baseline_profile["histogram"]),
        histogram_edges=baseline_profile["bin_edges"],
    )

    baseline_sample = np.asarray(baseline_profile["intensity_sample"], dtype=np.float32)
    candidate_sample = np.asarray(candidate_profile["intensity_sample"], dtype=np.float32)
    baseline_hist = np.asarray(baseline_profile["histogram"], dtype=np.float32)
    candidate_hist = np.asarray(candidate_profile["histogram"], dtype=np.float32)

    ks_statistic = float(ks_2samp(baseline_sample, candidate_sample).statistic)
    psi_value = population_stability_index(baseline_hist, candidate_hist)
    mean_shift = abs(
        float(candidate_profile["summary"]["intensity_mean"]) - float(baseline_profile["summary"]["intensity_mean"])
    )
    std_shift = abs(
        float(candidate_profile["summary"]["intensity_std"]) - float(baseline_profile["summary"]["intensity_std"])
    )
    shape_shift = [
        abs(float(candidate) - float(baseline))
        for candidate, baseline in zip(candidate_profile["summary"]["shape_mean"], baseline_profile["summary"]["shape_mean"])
    ]

    checks = {
        "ks_statistic": {"value": ks_statistic, "threshold": ks_threshold, "drift": ks_statistic > ks_threshold},
        "psi": {"value": psi_value, "threshold": psi_threshold, "drift": psi_value > psi_threshold},
        "mean_shift": {
            "value": mean_shift,
            "threshold": mean_shift_threshold,
            "drift": mean_shift > mean_shift_threshold,
        },
        "std_shift": {
            "value": std_shift,
            "threshold": std_shift_threshold,
            "drift": std_shift > std_shift_threshold,
        },
    }
    drift_detected = any(entry["drift"] for entry in checks.values())
    return {
        "status": "drift_detected" if drift_detected else "ok",
        "created_at_utc": _now_utc(),
        "baseline_dataset_version": baseline_profile["dataset_version"],
        "candidate_profile": candidate_profile["summary"],
        "baseline_profile": baseline_profile["summary"],
        "checks": checks,
        "shape_shift": shape_shift,
    }


def save_profile(profile: Dict[str, object], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def load_profile(path: Path | str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
