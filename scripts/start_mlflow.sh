#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/mlflow/artifacts

mlflow server \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --default-artifact-root artifacts/mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000
