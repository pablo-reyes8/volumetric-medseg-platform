#!/usr/bin/env bash
set -euo pipefail

docker compose --profile airflow up airflow-init
