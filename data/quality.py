import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - fallback for lightweight smoke environments
    class _ValidationError:
        def __init__(self, message: str):
            self.message = message

    class Draft202012Validator:  # type: ignore[no-redef]
        def __init__(self, schema):
            self.schema = schema

        def iter_errors(self, payload):
            return list(_iter_schema_errors(payload, self.schema))

    def _type_matches(value, expected_type) -> bool:
        expected = expected_type if isinstance(expected_type, list) else [expected_type]
        for item in expected:
            if item == "null" and value is None:
                return True
            if item == "object" and isinstance(value, dict):
                return True
            if item == "array" and isinstance(value, list):
                return True
            if item == "string" and isinstance(value, str):
                return True
            if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
                return True
            if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if item == "boolean" and isinstance(value, bool):
                return True
        return False

    def _iter_schema_errors(payload, schema, path=""):
        expected_type = schema.get("type")
        if expected_type and not _type_matches(payload, expected_type):
            yield _ValidationError(f"{path or 'value'} is not of type {expected_type}")
            return

        if isinstance(payload, dict):
            for required_key in schema.get("required", []):
                if required_key not in payload:
                    yield _ValidationError(f"{path + '.' if path else ''}{required_key} is a required property")
            for key, child_schema in schema.get("properties", {}).items():
                if key in payload:
                    yield from _iter_schema_errors(payload[key], child_schema, f"{path + '.' if path else ''}{key}")

        if isinstance(payload, list) and "items" in schema:
            for index, item in enumerate(payload):
                yield from _iter_schema_errors(item, schema["items"], f"{path}[{index}]")

        if "minimum" in schema and isinstance(payload, (int, float)) and payload < schema["minimum"]:
            yield _ValidationError(f"{path or 'value'} is less than the minimum of {schema['minimum']}")

from data.ingestion import list_nifti_files, pair_image_and_mask_files
from data.preprocessing import load_nifti_volume, quick_subsample_stats


def _load_yaml(path: Path | str) -> Dict[str, object]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _coerce_allowed_labels(allowed_labels: Optional[Dict[object, object]]) -> Optional[set[int]]:
    if allowed_labels is None:
        return None
    return {int(key) for key in allowed_labels.keys()}


def detect_ignored_sidecar_files(directory: Path | str) -> List[str]:
    path = Path(directory)
    return sorted(file_path.name for file_path in path.iterdir() if file_path.is_file() and file_path.name.startswith("._"))


def validate_volume_pair(
    image_path: Path | str,
    label_path: Path | str,
    allowed_labels: Optional[Dict[object, object]] = None,
) -> Dict[str, object]:
    image_file = Path(image_path)
    label_file = Path(label_path)
    issues: List[Dict[str, object]] = []

    if image_file.stat().st_size == 0:
        issues.append({"severity": "error", "rule": "no_zero_byte_files", "message": "image file is empty"})
    if label_file.stat().st_size == 0:
        issues.append({"severity": "error", "rule": "no_zero_byte_files", "message": "label file is empty"})
    if issues:
        return {"case_id": image_file.name, "issues": issues}

    image, _, image_spacing = load_nifti_volume(image_file)
    label, _, label_spacing = load_nifti_volume(label_file, dtype=np.int16)

    if image.shape != label.shape:
        issues.append(
            {
                "severity": "error",
                "rule": "paired_image_label_shapes_match",
                "message": f"shape mismatch {image.shape} vs {label.shape}",
            }
        )

    image_stats = quick_subsample_stats(image_file)
    label_stats = quick_subsample_stats(label_file)
    if not image_stats["finite_ok"]:
        issues.append({"severity": "error", "rule": "finite_voxels", "message": "image contains NaN or inf values"})
    if not label_stats["finite_ok"]:
        issues.append({"severity": "error", "rule": "finite_voxels", "message": "label contains NaN or inf values"})

    if any(spacing <= 0 for spacing in image_spacing) or any(spacing <= 0 for spacing in label_spacing):
        issues.append({"severity": "error", "rule": "positive_voxel_spacing", "message": "invalid voxel spacing"})

    unique_labels = {int(value) for value in np.unique(label)}
    allowed = _coerce_allowed_labels(allowed_labels)
    if allowed is not None and not unique_labels.issubset(allowed):
        issues.append(
            {
                "severity": "error",
                "rule": "allowed_label_values_only",
                "message": f"unexpected labels {sorted(unique_labels - allowed)}",
            }
        )
    if unique_labels == {0}:
        issues.append(
            {
                "severity": "warning",
                "rule": "non_empty_foreground_labels",
                "message": "label mask contains background only",
            }
        )

    return {
        "case_id": image_file.name.replace(".nii.gz", "").replace(".nii", ""),
        "issues": issues,
        "image_shape": tuple(int(dim) for dim in image.shape),
        "label_shape": tuple(int(dim) for dim in label.shape),
        "image_spacing": tuple(float(value) for value in image_spacing[:3]),
        "label_spacing": tuple(float(value) for value in label_spacing[:3]),
        "labels_present": sorted(unique_labels),
    }


def generate_quality_report(
    images_dir: Path | str,
    labels_dir: Path | str,
    allowed_labels: Optional[Dict[object, object]] = None,
    dataset_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> Dict[str, object]:
    pairs = pair_image_and_mask_files(images_dir, labels_dir)
    case_reports = [validate_volume_pair(image, label, allowed_labels=allowed_labels) for image, label in pairs]
    issue_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    failed_pairs = 0
    label_presence: Counter[int] = Counter()

    for case_report in case_reports:
        if case_report["issues"]:
            failed_pairs += 1
        for issue in case_report["issues"]:
            issue_counter[str(issue["rule"])] += 1
            severity_counter[str(issue["severity"])] += 1
        for label in case_report.get("labels_present", []):
            label_presence[int(label)] += 1

    ignored_files = {
        "images_dir": detect_ignored_sidecar_files(images_dir),
        "labels_dir": detect_ignored_sidecar_files(labels_dir),
    }
    passed_pairs = len(case_reports) - failed_pairs
    status = "pass" if severity_counter.get("error", 0) == 0 else "fail"
    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "status": status,
        "total_pairs": len(case_reports),
        "passed_pairs": passed_pairs,
        "failed_pairs": failed_pairs,
        "rules": dict(issue_counter),
        "severity": dict(severity_counter),
        "ignored_sidecars": ignored_files,
        "labels_present_counts": dict(sorted(label_presence.items())),
        "issues": [case_report for case_report in case_reports if case_report["issues"]],
    }


def validate_json_schema(payload: object, schema_path: Path | str) -> List[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [error.message for error in validator.iter_errors(payload)]


def validate_manifest_schema(manifest: Dict[str, object], schema_path: Path | str) -> List[str]:
    return validate_json_schema(manifest, schema_path)


def validate_registry_schema(registry: Dict[str, object], schema_path: Path | str) -> List[str]:
    return validate_json_schema(registry, schema_path)


def validate_task04_dataset_layout(dataset_root: Path | str, contract_path: Path | str) -> Dict[str, object]:
    contract = _load_yaml(contract_path)
    dataset_path = Path(dataset_root)
    required_entries = set(contract.get("expected_structure", {}).get("required_top_level", []))
    present_entries = {path.name for path in dataset_path.iterdir()}
    missing = sorted(required_entries - present_entries)
    extra_sidecars = sorted(name for name in present_entries if name.startswith("._"))

    return {
        "dataset_root": str(dataset_path),
        "missing_entries": missing,
        "ignored_sidecars": extra_sidecars,
        "status": "pass" if not missing else "fail",
    }
