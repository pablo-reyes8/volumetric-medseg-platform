# UNet3D Segmentation Suite

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/unet3d-medseg)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/unet3d-medseg)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/unet3d-medseg)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/unet3d-medseg)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/unet3d-medseg?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/unet3d-medseg?style=social)

Production-minded 3D medical image segmentation built around a PyTorch UNet3D. This repository is not just a training script: it includes a FastAPI inference service, a Streamlit review console, Dockerized local deployment, test coverage, explicit DataOps contracts, and MLOps hooks for tracking, monitoring, drift checks, retraining and rollback decisions.

## Quick Navigation
- [Visual Results](#visual-results)
- [Project Snapshot](#project-snapshot)
- [Quickstart](#quickstart)
- [API Surface](#api-surface)
- [Data and MLOps](#data-and-mlops)
- [Repository Structure](#repository-structure)

## Project Snapshot
- **Model**: `UNet3D` for volumetric medical segmentation.
- **Framework**: `PyTorch`.
- **Data format**: `.nii` and `.nii.gz` NIfTI volumes.
- **Serving layer**: `FastAPI` with schemas, health probes, model metadata and binary mask download.
- **Review interface**: `Streamlit` for slice inspection, overlays, histograms and inference comparison.
- **Ops baseline**: test suite, CLI scripts, split Docker images, environment-based configuration and runtime monitoring.
- **Data foundation**: source metadata, contracts, manifests, registry and quality checks for the canonical dataset.

## Why This Repo Works As A Portfolio Project
- It solves a real 3D medical imaging task instead of a 2D toy demo.
- It shows the full path from training to serving to human review.
- The API looks like a deployable product, not a notebook wrapper.
- The data layer is explicit, versioned and contract-driven.
- The MLOps layer already includes MLflow integration, drift analysis and operational policies.
- The repo is easy to run locally through scripts, tests and Docker.

> Start with the outputs. The section below is intentionally near the top because the visual evidence is one of the strongest parts of the project.


<h2 align="center"> Some Visual Results</h2>

<table align="center">
  <!-- Top row: smaller, context metrics -->
  <tr>
    <td align="center" width="50%">
      <b>IoU by class (validation)</b><br/>
      <img src="experiments/IoU.png" width="360" alt="IoU by class (validation)">
    </td>
    <td align="center" width="50%">
      <b>IoU across volume (validation)</b><br/>
      <img src="experiments/IoU%20image%20z%20axis.png" width="360" alt="IoU across volume (validation)">
    </td>
  </tr>

  <!-- Second row: larger, main qualitative results -->
  <tr>
    <td align="center" colspan="2">
      <b>Sample predictions</b><br/>
      <img src="experiments/model%20predictions.png" width="820" alt="Sample predictions">
    </td>
  </tr>

  <tr>
    <td align="center" colspan="2">
      <b>Error overlays</b><br/>
      <img src="experiments/overlay%20errors.png" width="720" alt="Error overlays">
    </td>
  </tr>
</table>

## Quickstart
### 1. Create the local environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the API
```bash
python scripts/run_api.py \
  --reload \
  --model-path /absolute/path/to/unet3d_best.pt
```

### 3. Run the Streamlit review app
```bash
python scripts/run_app.py \
  --api-url http://localhost:8000 \
  --model-path /absolute/path/to/unet3d_best.pt
```

### 4. Run the tests
```bash
python scripts/run_tests.py tests/data tests/mlops tests/api --verbose
```

### 5. Run the local Docker stack
```bash
python scripts/run_docker.py build
python scripts/run_docker.py up --detach
```

This starts:
- `api` on `http://localhost:8000`
- `streamlit` on `http://localhost:8501`

Stop the stack with:
```bash
python scripts/run_docker.py down
```

## End-to-End Workflow
```text
NIfTI volume
   ->
preprocessing
   - percentile clipping
   - min-max normalization
   - padding to multiples of 16
   ->
UNet3D inference
   ->
predicted mask
   ->
FastAPI JSON response / NIfTI download / Streamlit visual QA
```

## API Surface
OpenAPI docs are available at `http://localhost:8000/docs` and `http://localhost:8000/redoc`.

### Platform and health
- `GET /` returns service discovery metadata.
- `GET /health/live` exposes liveness.
- `GET /health/ready` validates readiness, including model/checkpoint status.
- `GET /api/v1/config` returns the active runtime configuration.

### Model management
- `GET /api/v1/model` returns model metadata from `model.yaml` and runtime settings.
- `POST /api/v1/model/reload` reloads the checkpoint without restarting the API.

### Inference endpoints
- `POST /api/v1/predictions` returns JSON inference metadata, runtime breakdown, spacing, orientation and histogram information.
- `POST /api/v1/predictions/download` returns the predicted mask as `.nii.gz`.
- `POST /v1/predict` remains available as a legacy compatibility route.

Example JSON inference:
```bash
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "accept: application/json" \
  -F "file=@/path/to/volume.nii.gz"
```

Example mask download:
```bash
curl -X POST "http://localhost:8000/api/v1/predictions/download" \
  -F "file=@/path/to/volume.nii.gz" \
  -o mask.nii.gz
```

## Streamlit Review App
The review app supports two main workflows:
- **API deployment mode**: calls the live FastAPI backend, checks readiness and downloads the predicted mask.
- **Local checkpoint mode**: runs the same inference service locally for quick validation on the same machine.

Each inference session includes:
- slice-by-slice browsing on any axis,
- image, mask and overlay visualization,
- runtime metadata and request identifiers,
- class histograms and prediction metadata,
- direct `.nii.gz` download for the predicted mask.

## Docker Deployment
The repository ships separate images for the backend and the review interface.

Raw image builds:
```bash
docker build -f docker/Dockerfile.api -t unet3d-medseg-api .
docker build -f docker/Dockerfile.streamlit -t unet3d-medseg-streamlit .
```

## Configuration
Environment variables use the `UNET3D_` prefix. The most relevant ones are:
- `UNET3D_MODEL_PATH`
- `UNET3D_DEVICE`
- `UNET3D_DEFAULT_THRESHOLD`
- `UNET3D_PAD_MULTIPLE`
- `UNET3D_CLIP_PERCENTILES`
- `UNET3D_ALLOW_ORIGINS`
- `UNET3D_PRELOAD_MODEL`

See `src/api/settings.py` for the full runtime configuration contract.

## Data and MLOps
### Data foundation
The project treats data as a first-class asset through the root `data/` package:
- `data/raw/` for immutable source drops.
- `data/external/` for downloaded archives.
- `data/interim/` for temporary or review-stage artifacts.
- `data/processed/` for standardized training-ready volumes.
- `data/manifests/` for JSON manifests with hashes and lineage.
- `data/registry/datasets.yaml` for registered dataset versions.

For **Medical Segmentation Decathlon Task04 Hippocampus**, the repo already includes:
- source metadata in `data/sources/task04_hippocampus.yaml`,
- a data contract in `data/contracts/task04_hippocampus.contract.yaml`,
- a schema for the official `dataset.json`,
- a schema for dataset manifests,
- a schema for the dataset registry,
- quality validation and version registration pipelines.

Task04 preparation:
```bash
python scripts/run_task04_dataops.py \
  --dataset-version 2026.03.21
```

Manual manifest registration:
```bash
python scripts/run_data_registry.py \
  --images-dir data/processed/task04_hippocampus/2026.03.21/imagesTr \
  --labels-dir data/processed/task04_hippocampus/2026.03.21/labelsTr \
  --dataset-name task04_hippocampus \
  --version 2026.03.21 \
  --manifest-out data/manifests/task04_hippocampus_2026.03.21.json
```

### MLflow tracking and packaging
`src/mlops/mlflow_tracking.py` wraps the training loop without replacing it. Each run can log:
- hyperparameters and tags,
- train and validation metrics per epoch,
- best checkpoint artifacts,
- dataset manifests,
- `model.yaml`,
- packaging manifests,
- serving files and Docker packaging files,
- data contracts and source metadata.

### Drift, retraining and rollback
The project does not reduce operations to drift alone.

It already includes:
- drift baselining and evaluation for deployment batches,
- periodic retraining policies,
- KPI-driven retraining decisions,
- rollback rules for runtime regressions,
- runtime monitoring endpoints exposed by the API.

Drift baseline example:
```bash
python scripts/run_drift_check.py baseline \
  --images-dir data/processed/task04_hippocampus/2026.03.21/imagesTr \
  --dataset-version 2026.03.21 \
  --output-path data/manifests/task04_hippocampus_drift_baseline.json
```

### Deployment monitoring
The serving layer explicitly tracks:
- latency,
- throughput,
- error rate,
- per-endpoint behavior,
- CPU and memory usage,
- GPU memory usage when available,
- estimated cost per 1000 requests.

Useful monitoring endpoints:
- `GET /api/v1/monitoring/runtime`
- `GET /api/v1/monitoring/policy`
- `GET /api/v1/monitoring/retraining-assessment`

For the operational summary, see `docs/mlops_playbook.md`.

## Repository Structure
```text
├─ app/                     # Streamlit review console
├─ data/                    # Dataset layout, manifests, registry, ingestion and preprocessing
├─ docker/                  # Dedicated Dockerfiles for API and Streamlit
├─ experiments/             # Qualitative and quantitative results
├─ requirements/            # Dependency profiles: base, api, app, dev
├─ scripts/                 # CLI helpers for API, app, tests, Docker, data and drift
├─ src/
│  ├─ api/                  # FastAPI app, schemas, settings, inference service
│  ├─ mlops/                # MLflow tracking and deployment drift evaluation
│  ├─ model/                # UNet3D architecture and blocks
│  ├─ model_inference.py/   # Analysis and qualitative utilities
│  └─ training/             # Training loop and metrics
├─ tests/                   # API, data and MLOps smoke tests
├─ docker-compose.yml       # Multi-service local stack
├─ model.yaml               # Model card and serving metadata
├─ requirements.txt         # Full development dependencies
└─ Dockerfile               # Backward-compatible API image build
```

## Testing
The `tests/` suite currently covers:
- settings validation,
- dataset pairing, loading and manifest versioning,
- dataset quality contracts and Task04 source metadata,
- MLflow tracking hooks, retraining policy logic and deployment drift evaluation,
- inference-service metadata and padding behavior,
- FastAPI routes for health, metadata, reload, prediction and file validation.

This gives the repository a solid engineering baseline before adding CI/CD, model registries, cloud deployment and full production observability.

## License
MIT License. You are free to use, modify, and distribute with attribution and without warranty.

## Support
For issues or improvements, open a ticket or reach the maintainers listed in `model.yaml`.
