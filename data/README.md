# DataOps Layout

This folder is the project's DataOps layer. It defines where data lives, how it is versioned, what contract it must satisfy, and how quality is checked before training or deployment baselines are created.

## Directory Contract
- `data/raw/`: immutable raw drops exactly as received from the source archive.
- `data/external/`: downloaded third-party archives before extraction.
- `data/interim/`: temporary or review-stage data products.
- `data/processed/`: standardized training-ready layout.
- `data/manifests/`: versioned JSON manifests with content hashes and quality metadata.
- `data/registry/datasets.yaml`: catalog of dataset versions used by the project.
- `data/contracts/`: explicit schemas and dataset contracts.
- `data/sources/`: source-of-truth metadata for official datasets.

## Dataset in Scope
The current canonical dataset is **Medical Segmentation Decathlon Task04 Hippocampus**.

Source metadata is stored in `data/sources/task04_hippocampus.yaml`.

Current contract:
- modality: `MRI`
- dimensionality: `3D`
- labels: `0=background`, `1=anterior`, `2=posterior`
- training pairs: `260`
- test volumes: `130`
- required archive layout: `dataset.json`, `imagesTr`, `labelsTr`, `imagesTs`

## Data Contract
The data contract is stored in `data/contracts/task04_hippocampus.contract.yaml`.

It defines:
- expected top-level directories,
- label schema,
- required quality rules,
- manifest schema location,
- registry location.

The manifest schema is defined in `data/contracts/dataset_manifest.schema.json`.

## Quality Gates
`data/quality.py` currently checks:
- zero-byte files,
- finite voxel values,
- image/label shape match,
- positive voxel spacing,
- allowed label values only,
- empty foreground masks,
- ignored macOS sidecar files such as `._dataset.json`.

Every manifest can embed a `quality_report` so the version registry documents both lineage and quality state.

## Versioning Workflow
1. Download the official archive into `data/external/`.
2. Extract the original structure into `data/raw/`.
3. Standardize the layout into `data/processed/<dataset>/<version>/`.
4. Run quality validation.
5. Generate a manifest with hashes and metadata.
6. Register the dataset version in `data/registry/datasets.yaml`.
7. Log the manifest into MLflow during training.

## Task04 End-to-End CLI
```bash
python scripts/run_task04_dataops.py \
  --dataset-version 2026.03.21
```

This workflow will:
- download `Task04_Hippocampus.tar`,
- extract the raw archive,
- copy the canonical folders into `data/processed/`,
- validate structure and quality,
- generate the manifest,
- update the dataset registry.

## Why This Matters
- Training runs point to an explicit dataset version instead of a floating folder.
- Every dataset version is hashable, reviewable and schema-bound.
- Quality failures surface before model training.
- Drift baselines can be built from a registered dataset version, not from an ad hoc sample.
