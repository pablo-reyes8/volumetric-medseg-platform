import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional

import yaml

from data.ingestion import download_file
from data.quality import generate_quality_report, validate_json_schema, validate_manifest_schema, validate_task04_dataset_layout
from data.versioning import build_dataset_manifest, save_dataset_manifest, update_dataset_registry


TASK04_SOURCE_PATH = Path("data/sources/task04_hippocampus.yaml")
TASK04_CONTRACT_PATH = Path("data/contracts/task04_hippocampus.contract.yaml")
TASK04_MANIFEST_SCHEMA_PATH = Path("data/contracts/dataset_manifest.schema.json")
TASK04_DATASET_JSON_SCHEMA_PATH = Path("data/contracts/task04_dataset_json.schema.json")


def load_task04_source_metadata(path: Path | str = TASK04_SOURCE_PATH) -> Dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_task04_contract(path: Path | str = TASK04_CONTRACT_PATH) -> Dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _safe_extract_tar(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar_file:
        destination_root = destination.resolve()
        for member in tar_file.getmembers():
            member_path = (destination / member.name).resolve()
            if member_path != destination_root and destination_root not in member_path.parents:
                raise ValueError(f"Tar member fuera del destino permitido: {member.name}")
        tar_file.extractall(destination)
    return destination


def download_task04_archive(
    output_dir: Path | str = "data/external/task04_hippocampus",
    archive_name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    source_metadata = load_task04_source_metadata()
    archive_url = source_metadata["official_source"]["archive_url"]
    file_name = archive_name or source_metadata["official_source"].get("archive_filename", "Task04_Hippocampus.tar")
    destination = Path(output_dir) / file_name
    return download_file(archive_url, destination, overwrite=overwrite)


def extract_task04_archive(
    archive_path: Path | str,
    raw_dir: Path | str = "data/raw/task04_hippocampus",
) -> Path:
    destination = Path(raw_dir)
    _safe_extract_tar(Path(archive_path), destination)
    source_metadata = load_task04_source_metadata()
    root_dir = source_metadata["archive_layout"]["root_dir"]
    return destination / root_dir


def standardize_task04_layout(
    extracted_root: Path | str,
    processed_dir: Path | str,
    overwrite: bool = False,
) -> Path:
    source_root = Path(extracted_root)
    destination_root = Path(processed_dir)
    destination_root.mkdir(parents=True, exist_ok=True)

    for entry_name in ["imagesTr", "labelsTr", "imagesTs", "dataset.json"]:
        source_entry = source_root / entry_name
        destination_entry = destination_root / entry_name
        if destination_entry.exists() and not overwrite:
            continue
        if source_entry.is_dir():
            if destination_entry.exists():
                shutil.rmtree(destination_entry)
            shutil.copytree(source_entry, destination_entry)
        elif source_entry.is_file():
            shutil.copy2(source_entry, destination_entry)
        else:
            raise FileNotFoundError(f"No se encontro {entry_name} dentro de {source_root}")
    return destination_root


def validate_task04_dataset_json(dataset_json_path: Path | str, source_metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    expected = source_metadata or load_task04_source_metadata()
    payload = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
    schema_errors = validate_json_schema(payload, TASK04_DATASET_JSON_SCHEMA_PATH)
    issues = []

    expected_labels = expected.get("labels", {})
    expected_modality = expected.get("modality", {})
    expected_training = expected.get("splits", {}).get("training_labeled")
    expected_test = expected.get("splits", {}).get("test_unlabeled")
    expected_tensor_size = expected.get("tensor_image_size")
    expected_name = expected.get("dataset_json_name")
    expected_license = expected.get("license")

    if expected_name and payload.get("name") != expected_name:
        issues.append("name in dataset.json does not match the registered source metadata")
    if payload.get("labels") != expected_labels:
        issues.append("labels in dataset.json do not match the registered source metadata")
    if payload.get("modality") != expected_modality:
        issues.append("modality in dataset.json does not match the registered source metadata")
    if payload.get("tensorImageSize") != expected_tensor_size:
        issues.append("tensorImageSize in dataset.json does not match the expected dimensionality")
    if payload.get("numTraining") != expected_training:
        issues.append("numTraining in dataset.json does not match the expected split size")
    if payload.get("numTest") != expected_test:
        issues.append("numTest in dataset.json does not match the expected split size")
    if expected_license and payload.get("licence") not in {None, expected_license}:
        issues.append("licence in dataset.json does not match the registered source metadata")

    return {
        "status": "pass" if not issues and not schema_errors else "fail",
        "dataset_name": payload.get("name"),
        "description": payload.get("description"),
        "tensorImageSize": payload.get("tensorImageSize"),
        "labels": payload.get("labels"),
        "modality": payload.get("modality"),
        "numTraining": payload.get("numTraining"),
        "numTest": payload.get("numTest"),
        "schema_errors": schema_errors,
        "issues": issues,
    }


def validate_task04_source_contract(
    source_metadata: Optional[Dict[str, object]] = None,
    contract_path: Path | str = TASK04_CONTRACT_PATH,
) -> Dict[str, object]:
    metadata = source_metadata or load_task04_source_metadata()
    contract = load_task04_contract(contract_path)
    issues = []

    if metadata.get("labels") != contract.get("label_schema"):
        issues.append("source metadata labels do not match the declared contract label schema")
    if metadata.get("tensor_image_size") != contract.get("modality", {}).get("tensor_image_size"):
        issues.append("source metadata tensor image size does not match the contract")
    if metadata.get("modality", {}).get("0") != contract.get("modality", {}).get("primary"):
        issues.append("source metadata modality does not match the contract primary modality")

    source_top_level = sorted(metadata.get("archive_layout", {}).get("required_top_level", []))
    contract_top_level = sorted(contract.get("expected_structure", {}).get("required_top_level", []))
    if source_top_level != contract_top_level:
        issues.append("source metadata archive layout does not match the contract required structure")

    return {
        "status": "pass" if not issues else "fail",
        "dataset_name": metadata.get("dataset_name"),
        "contract_path": str(contract_path),
        "issues": issues,
    }


def prepare_task04_dataset(
    dataset_version: str,
    base_dir: Path | str = "data",
    manifest_output_path: Optional[Path | str] = None,
    registry_path: Optional[Path | str] = None,
    overwrite_archive: bool = False,
    overwrite_processed: bool = False,
) -> Dict[str, object]:
    base_path = Path(base_dir)
    archive_path = download_task04_archive(base_path / "external" / "task04_hippocampus", overwrite=overwrite_archive)
    raw_root = extract_task04_archive(archive_path, base_path / "raw" / "task04_hippocampus" / dataset_version)
    processed_root = standardize_task04_layout(
        raw_root,
        base_path / "processed" / "task04_hippocampus" / dataset_version,
        overwrite=overwrite_processed,
    )

    source_metadata = load_task04_source_metadata()
    source_contract_report = validate_task04_source_contract(source_metadata=source_metadata)
    dataset_layout_report = validate_task04_dataset_layout(processed_root, TASK04_CONTRACT_PATH)
    quality_report = generate_quality_report(
        processed_root / "imagesTr",
        processed_root / "labelsTr",
        allowed_labels=source_metadata["labels"],
        dataset_name=source_metadata["dataset_name"],
        dataset_version=dataset_version,
    )
    dataset_json_report = validate_task04_dataset_json(processed_root / "dataset.json", source_metadata=source_metadata)

    if source_contract_report["status"] != "pass":
        raise ValueError(f"Metadata fuente y contrato inconsistentes: {source_contract_report['issues']}")
    if dataset_layout_report["status"] != "pass":
        raise ValueError(f"Layout Task04 invalido: {dataset_layout_report['missing_entries']}")
    if dataset_json_report["status"] != "pass":
        raise ValueError(
            "dataset.json invalido para Task04: "
            f"schema_errors={dataset_json_report['schema_errors']} issues={dataset_json_report['issues']}"
        )
    if quality_report["status"] != "pass":
        raise ValueError(f"Quality gate fallido para Task04: {quality_report['issues'][:3]}")

    manifest = build_dataset_manifest(
        images_dir=processed_root / "imagesTr",
        labels_dir=processed_root / "labelsTr",
        dataset_name=source_metadata["dataset_name"],
        version=dataset_version,
        source_url=source_metadata["official_source"]["archive_url"],
        notes="Prepared from official Task04_Hippocampus archive.",
        extra_metadata={
            "processed_root": str(processed_root),
            "images_test_dir": str(processed_root / "imagesTs"),
            "dataset_json_path": str(processed_root / "dataset.json"),
            "source_contract_report": source_contract_report,
            "layout_report": dataset_layout_report,
            "dataset_json_report": dataset_json_report,
        },
        source_metadata=source_metadata,
        data_contract_path=TASK04_CONTRACT_PATH,
        quality_report=quality_report,
    )
    schema_errors = validate_manifest_schema(manifest, TASK04_MANIFEST_SCHEMA_PATH)
    if schema_errors:
        raise ValueError(f"Manifest invalido contra el schema: {schema_errors}")

    manifest_path = Path(manifest_output_path) if manifest_output_path else base_path / "manifests" / f"task04_hippocampus_{dataset_version}.json"
    registry_output_path = Path(registry_path) if registry_path else base_path / "registry" / "datasets.yaml"
    save_dataset_manifest(manifest, manifest_path)
    update_dataset_registry(manifest, registry_output_path, manifest_path=manifest_path)

    return {
        "archive_path": str(archive_path),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "source_contract_report": source_contract_report,
        "layout_report": dataset_layout_report,
        "quality_report": quality_report,
        "dataset_json_report": dataset_json_report,
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_output_path),
    }
