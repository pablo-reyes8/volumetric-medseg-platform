# UNet3D Segmentation Suite

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/unet3d-medseg)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/unet3d-medseg)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/unet3d-medseg)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/unet3d-medseg)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/unet3d-medseg?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/unet3d-medseg?style=social)

Portfolio-ready stack for 3D medical image segmentation with a UNet3D backbone. This repository goes beyond a training notebook: it packages the model as a usable product with a versioned FastAPI service, a Streamlit review console, test coverage, Dockerized services, and CLI entrypoints for reproducible local workflows.

## Why This Project Is Strong
- **Real volumetric segmentation**: 3D UNet trained for medical volumes in NIfTI format, not 2D toy data.
- **Inference pipeline that looks like production**: schema-driven FastAPI endpoints, health probes, model metadata, and reload support.
- **Human-in-the-loop UI**: Streamlit app for remote or local inference, overlay inspection, histogram review, and mask download.
- **Operational baseline in place**: dedicated Docker images, `docker-compose.yml`, environment-driven configuration, and a `tests/` suite.
- **Data and monitoring are explicit**: root-level `data/` package, dataset manifests, registry versioning, MLflow hooks, and deployment drift checks.
- **Clear extensibility path**: model card in `model.yaml`, CLI scripts in `scripts/`, and a clean separation between `training`, `api`, `data`, and `mlops`.

## Architecture
```
Input volume (.nii/.nii.gz)
        |
        v
  Preprocessing
  - percentile clipping
  - min-max normalization
  - padding to multiples of 16
        |
        v
     UNet3D
        |
        v
  Predicted mask
  - JSON metadata via FastAPI
  - NIfTI download via FastAPI
  - visual QA via Streamlit
```

## Repository Structure
```
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

## Model Card Snapshot
- **Model**: `unet3d-segmentation`
- **Framework**: PyTorch
- **Task**: 3D medical segmentation over NIfTI volumes
- **Input contract**: single-channel volume with percentile clipping `(1, 99)` and padding to multiples of `16`
- **Output contract**: NIfTI mask with `3` classes by default
- **Primary checkpoint path**: `artifacts/unet3d_best.pt`
- **Serving metadata source**: `model.yaml`

## Quickstart
### 1. Local environment
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

### 3. Run the Streamlit app
```bash
python scripts/run_app.py \
  --api-url http://localhost:8000 \
  --model-path /absolute/path/to/unet3d_best.pt
```

### 4. Run tests
```bash
python scripts/run_tests.py tests/data tests/mlops tests/api --verbose
```

## API Surface
OpenAPI docs are available at `http://localhost:8000/docs` and `http://localhost:8000/redoc`.

### Monitoring and platform
- `GET /` returns service discovery info.
- `GET /health/live` exposes liveness.
- `GET /health/ready` exposes readiness and checkpoint/model checks.
- `GET /api/v1/config` returns the active runtime config.

### Model management
- `GET /api/v1/model` returns model metadata derived from `model.yaml` and runtime settings.
- `POST /api/v1/model/reload` forces a checkpoint reload without restarting the service.

### Inference
- `POST /api/v1/predictions` returns structured JSON with runtime breakdown, orientation, spacing, histogram, and threshold used.
- `POST /api/v1/predictions/download` returns the predicted mask as `.nii.gz`.
- `POST /v1/predict` remains available as a legacy compatibility endpoint.

Example JSON inference:
```bash
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "accept: application/json" \
  -F "file=@/path/to/volume.nii.gz"
```

Example binary download:
```bash
curl -X POST "http://localhost:8000/api/v1/predictions/download" \
  -F "file=@/path/to/volume.nii.gz" \
  -o mask.nii.gz
```

## Streamlit Review App
The app supports two workflows:
- **API Deployment**: checks API readiness, reads live model metadata, requests inference, and downloads the resulting mask.
- **Local Checkpoint**: runs the same inference service locally for checkpoint validation on the same machine.

Each run includes:
- slice-by-slice inspection over any axis,
- image/mask/overlay visualization,
- runtime metrics and request identifiers,
- class histograms and raw prediction metadata,
- direct `.nii.gz` download for the predicted mask.

## Docker Deployment
This repo now ships separate images for the backend and the review UI.

### Compose workflow
```bash
python scripts/run_docker.py build
python scripts/run_docker.py up --detach
```

This starts:
- `api` on `http://localhost:8000`
- `streamlit` on `http://localhost:8501`

To stop the stack:
```bash
python scripts/run_docker.py down
```

### Raw Docker builds
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

See `src/api/settings.py` for the full configuration contract.

## Data and MLOps
### Explicit dataset layout
The project now promotes data to a first-class asset in the root `data/` package:
- `data/raw/` for immutable source drops,
- `data/external/` for downloaded archives,
- `data/interim/` for temporary transformations,
- `data/processed/` for training-ready volumes,
- `data/manifests/` for JSON dataset manifests,
- `data/registry/datasets.yaml` for versioned dataset registration.

Create and register a dataset manifest:
```bash
python scripts/run_data_registry.py \
  --images-dir data/processed/hippocampus/2026.03.21/imagesTr \
  --labels-dir data/processed/hippocampus/2026.03.21/labelsTr \
  --dataset-name hippocampus \
  --version 2026.03.21 \
  --manifest-out data/manifests/hippocampus_2026.03.21.json
```

### MLflow training wrapper
`src/mlops/mlflow_tracking.py` adds a thin wrapper around the existing PyTorch training loop instead of replacing it. The wrapper logs:
- training params and tags,
- per-epoch train/validation metrics,
- best checkpoint artifacts,
- dataset manifests,
- `model.yaml` as training metadata.

### Data drift for deployment
`src/mlops/drift.py` builds a baseline profile from reference NIfTI volumes and evaluates candidate volumes using:
- KS statistic,
- population stability index (PSI),
- mean/std shift,
- average shape shift.

Build a drift baseline and evaluate a candidate batch:
```bash
python scripts/run_drift_check.py baseline \
  --images-dir data/processed/hippocampus/2026.03.21/imagesTr \
  --dataset-version 2026.03.21 \
  --output-path data/manifests/hippocampus_drift_baseline.json

python scripts/run_drift_check.py evaluate \
  --images-dir data/interim/deployment_batch \
  --baseline-path data/manifests/hippocampus_drift_baseline.json
```

## Testing
The `tests/` folder currently covers:
- settings validation,
- dataset pairing, loading and manifest versioning,
- MLflow tracking hooks and deployment drift evaluation,
- inference-service metadata and padding behavior,
- FastAPI routes for health, metadata, reload, prediction, and file validation.

This gives the project a real engineering baseline before moving into MLOps concerns such as CI/CD, registries, monitoring, and deployment automation.

## Visual Results


<h2 align="center">Visual Results</h2>

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

## License
MIT License. You are free to use, modify, and distribute with attribution and without warranty.

## Support
For issues or improvements, open a ticket or reach the maintainers listed in `model.yaml`.
