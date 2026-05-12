#!/usr/bin/env bash
set -euo pipefail

airflow connections add mlflow_local --conn-type http --conn-host http://mlflow --conn-port 5000 || true
airflow connections add api_local --conn-type http --conn-host http://api --conn-port 8000 || true
airflow connections add prometheus_local --conn-type http --conn-host http://prometheus --conn-port 9090 || true
airflow connections add grafana_local --conn-type http --conn-host http://grafana --conn-port 3000 || true
