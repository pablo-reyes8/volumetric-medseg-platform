import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from data.ingestion import pair_image_and_mask_files, sha256_file
from data.preprocessing import summarize_volume


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean_shape(shape_summaries: List[tuple[int, ...]]) -> List[float]:
    if not shape_summaries:
        return []
    dims = zip(*shape_summaries)
    return [round(sum(dim_values) / len(shape_summaries), 3) for dim_values in dims]


def build_dataset_manifest(
    images_dir: Path | str,
    labels_dir: Path | str,
    dataset_name: str,
    version: str,
    source_url: Optional[str] = None,
    notes: Optional[str] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    image_dir_path = Path(images_dir)
    label_dir_path = Path(labels_dir)
    pairs = pair_image_and_mask_files(image_dir_path, label_dir_path)

    records: List[Dict[str, object]] = []
    image_shapes: List[tuple[int, ...]] = []
    label_shapes: List[tuple[int, ...]] = []

    for image_path, label_path in pairs:
        image_summary = summarize_volume(image_path)
        label_summary = summarize_volume(label_path)
        if tuple(image_summary["shape"]) != tuple(label_summary["shape"]):
            raise ValueError(f"Imagen y mascara con shapes distintos para {image_path.name}")

        image_shapes.append(tuple(image_summary["shape"]))
        label_shapes.append(tuple(label_summary["shape"]))
        records.append(
            {
                "case_id": image_path.stem.replace(".nii", ""),
                "image": {
                    "path": str(image_path),
                    "sha256": sha256_file(image_path),
                    "summary": image_summary,
                },
                "label": {
                    "path": str(label_path),
                    "sha256": sha256_file(label_path),
                    "summary": label_summary,
                },
            }
        )

    manifest = {
        "schema_version": "1.0",
        "dataset_name": dataset_name,
        "version": version,
        "created_at_utc": _now_utc(),
        "source_url": source_url,
        "notes": notes,
        "images_dir": str(image_dir_path),
        "labels_dir": str(label_dir_path),
        "total_pairs": len(records),
        "summary": {
            "mean_image_shape": _mean_shape(image_shapes),
            "mean_label_shape": _mean_shape(label_shapes),
            "unique_image_shapes": sorted({tuple(shape) for shape in image_shapes}),
            "unique_label_shapes": sorted({tuple(shape) for shape in label_shapes}),
        },
        "records": records,
        "extra_metadata": extra_metadata or {},
    }
    return manifest


def save_dataset_manifest(manifest: Dict[str, object], output_path: Path | str) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def load_dataset_manifest(path: Path | str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_dataset_registry(
    manifest: Dict[str, object],
    registry_path: Path | str,
    manifest_path: Optional[Path | str] = None,
) -> Path:
    destination = Path(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        registry = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    else:
        registry = {}

    datasets = registry.get("datasets", [])
    filtered = [
        entry
        for entry in datasets
        if not (
            entry.get("dataset_name") == manifest["dataset_name"] and entry.get("version") == manifest["version"]
        )
    ]
    filtered.append(
        {
            "dataset_name": manifest["dataset_name"],
            "version": manifest["version"],
            "created_at_utc": manifest["created_at_utc"],
            "total_pairs": manifest["total_pairs"],
            "manifest_path": str(manifest_path) if manifest_path else None,
            "images_dir": manifest["images_dir"],
            "labels_dir": manifest["labels_dir"],
        }
    )

    registry["datasets"] = sorted(filtered, key=lambda entry: (entry["dataset_name"], entry["version"]))
    destination.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return destination
