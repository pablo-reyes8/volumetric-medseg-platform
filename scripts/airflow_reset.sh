#!/usr/bin/env bash
set -euo pipefail

docker compose --profile airflow down --volumes
rm -rf airflow/logs/*
