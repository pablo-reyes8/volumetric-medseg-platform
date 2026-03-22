# DataOps Layout

This folder is the project's DataOps layer. It makes the dataset explicit as an asset with lineage, contracts, schemas, quality gates and versioned registration before training or deployment.

## Directory Contract
- `data/raw/`: immutable raw drops exactly as received from the source archive.
- `data/external/`: downloaded third-party archives before extraction.
- `data/interim/`: temporary or review-stage data products.
- `data/processed/`: standardized training-ready layout under `data/processed/<dataset>/<version>/`.
- `data/manifests/`: versioned JSON manifests with hashes, summaries and quality metadata.
- `data/registry/datasets.yaml`: versioned catalog of approved dataset builds.
- `data/contracts/`: explicit schemas and dataset contracts.
- `data/sources/`: source-of-truth metadata for official datasets.

## Canonical Dataset
The current canonical dataset is **Medical Segmentation Decathlon Task04 Hippocampus**.

Source-of-truth metadata lives in `data/sources/task04_hippocampus.yaml` and captures:
- the official download URL,
- the archive filename,
- the official challenge and citation references,
- modality and dimensionality,
- the official label map from `dataset.json`,
- expected split sizes.

The official archive flow reproduced by the code is equivalent to:

```bash
curl -L "https://drive.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C" -o "Task04_Hippocampus.tar"
tar -xf Task04_Hippocampus.tar
```

For Task04 the registered contract is:
- modality: `MRI`
- tensor image size: `3D`
- labels: `0=background`, `1=Anterior`, `2=Posterior`
- training pairs: `260`
- test volumes: `130`
- required archive layout: `dataset.json`, `imagesTr`, `labelsTr`, `imagesTs`

## Schemas and Contracts
The folder is governed by explicit schemas instead of loose conventions:
- `data/contracts/task04_hippocampus.contract.yaml`: operational contract for Task04.
- `data/contracts/task04_dataset_json.schema.json`: schema for the official MSD `dataset.json`.
- `data/contracts/dataset_manifest.schema.json`: schema for generated dataset manifests.
- `data/contracts/dataset_registry.schema.json`: schema for the dataset registry.

The Task04 pipeline validates three layers before registration:
1. Source metadata and contract alignment.
2. Archive layout and official `dataset.json`.
3. Pair-level and dataset-level quality gates.

## Quality Gates
`data/quality.py` validates:
- zero-byte files,
- finite voxel values,
- image and label shape match,
- positive voxel spacing,
- allowed label values only,
- empty foreground masks,
- ignored macOS sidecar files such as `._dataset.json`,
- JSON-schema conformance for manifests and registry entries.

Every manifest embeds a `quality_report`, and the registry stores the resulting `quality_status` for traceability.

## Versioning and Registry
Versioning is explicit and calendar- or semver-based at the dataset level.

The expected lifecycle is:
1. Download the official archive into `data/external/`.
2. Extract the untouched source layout into `data/raw/`.
3. Standardize it into `data/processed/task04_hippocampus/<version>/`.
4. Validate the source metadata, contract, layout and `dataset.json`.
5. Run dataset quality checks over every training pair.
6. Generate a manifest with file hashes, summary statistics and provenance.
7. Register the dataset version in `data/registry/datasets.yaml`.
8. Log the manifest and contract artifacts into MLflow during training.

Registry entries capture:
- dataset name and version,
- creation timestamp,
- manifest path,
- processed image and label roots,
- quality status,
- source URL,
- contract path.

## Main Entry Points
Task04 end-to-end preparation:

```bash
python scripts/run_task04_dataops.py \
  --dataset-version 2026.03.21
```

Generic manifest and registry generation:

```bash
python scripts/run_data_registry.py \
  --images-dir data/processed/task04_hippocampus/2026.03.21/imagesTr \
  --labels-dir data/processed/task04_hippocampus/2026.03.21/labelsTr \
  --dataset-name task04_hippocampus \
  --version 2026.03.21 \
  --manifest-out data/manifests/task04_hippocampus_2026.03.21.json
```

## Why This Matters
- Training runs point to an explicit dataset version instead of a floating folder.
- The dataset is schema-bound and reviewable from source drop to manifest.
- Quality failures surface before training and before drift baselines are created.
- The registry becomes the audit trail that links data version, quality state and downstream MLflow runs.
