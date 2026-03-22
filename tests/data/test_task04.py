from pathlib import Path

import json

from data.task04 import load_task04_source_metadata, validate_task04_dataset_json, validate_task04_source_contract


def test_task04_source_metadata_matches_known_dataset_contract():
    metadata = load_task04_source_metadata()

    assert metadata["dataset_name"] == "task04_hippocampus"
    assert metadata["dataset_json_name"] == "Hippocampus"
    assert metadata["splits"]["training_labeled"] == 260
    assert metadata["splits"]["test_unlabeled"] == 130
    assert metadata["labels"]["1"] == "Anterior"


def test_task04_source_metadata_is_aligned_with_contract():
    report = validate_task04_source_contract(load_task04_source_metadata())

    assert report["status"] == "pass"
    assert report["issues"] == []


def test_task04_dataset_json_validation_accepts_expected_metadata(tmp_path: Path):
    dataset_json = {
        "name": "Hippocampus",
        "description": "Left and right hippocampus segmentation",
        "licence": "CC-BY-SA 4.0",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "numTraining": 260,
        "numTest": 130,
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset_json), encoding="utf-8")

    report = validate_task04_dataset_json(path, source_metadata=load_task04_source_metadata())

    assert report["status"] == "pass"
    assert report["schema_errors"] == []
    assert report["issues"] == []
