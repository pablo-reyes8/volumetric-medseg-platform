# Data Layout and Versioning

This repository now treats data as a first-class project asset instead of hiding it under `src/`.

## Directory Contract
- `data/raw/`: immutable raw drops exactly as received from the source.
- `data/external/`: third-party downloads or archives before unpacking.
- `data/interim/`: temporary normalized or reindexed artifacts.
- `data/processed/`: training-ready volumes and masks.
- `data/manifests/`: JSON manifests generated from a concrete dataset version.
- `data/registry/datasets.yaml`: dataset registry with explicit semantic versions.

## Versioning Workflow
1. Download or place the raw dataset under `data/raw/<dataset_name>/<version>/`.
2. Generate a manifest with `data.versioning.build_dataset_manifest(...)`.
3. Save the manifest in `data/manifests/`.
4. Update `data/registry/datasets.yaml` with `data.versioning.update_dataset_registry(...)`.
5. Log the manifest as an artifact in MLflow when training.

## Why This Matters
- Every training run can point to a concrete dataset version.
- The registry documents what was used, where it lives, and how many paired volumes it contains.
- Drift baselines can be derived from a manifest instead of from ad hoc folders.
