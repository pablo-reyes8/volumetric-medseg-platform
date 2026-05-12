PYTHON ?= python
PIP ?= $(PYTHON) -m pip
COMPOSE ?= docker compose

.PHONY: setup install lint format test test-unit test-api test-mlops test-data test-airflow train serve-api serve-app build docker-up docker-down mlflow-up airflow-up airflow-down airflow-logs airflow-load-vars airflow-init-connections airflow-list-dags airflow-test-dataops airflow-test-train airflow-test-monitoring airflow-test-demo airflow-reset-local prometheus-up clean-artifacts check-env smoke-docker promote rollback mlops-demo

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements/dev.txt

install:
	$(PIP) install -r requirements/dev.txt

lint:
	$(PYTHON) -m compileall src data scripts tests

format:
	$(PYTHON) -m compileall src data scripts tests

test:
	$(PYTHON) scripts/run_tests.py tests/data tests/mlops tests/api

test-unit:
	$(PYTHON) scripts/run_tests.py tests

test-api:
	$(PYTHON) scripts/run_tests.py tests/api

test-mlops:
	$(PYTHON) scripts/run_tests.py tests/mlops

test-data:
	$(PYTHON) scripts/run_tests.py tests/data

test-airflow:
	$(PYTHON) scripts/run_tests.py tests/airflow

train:
	$(PYTHON) -m src.training.train_unet

serve-api:
	$(PYTHON) scripts/run_api.py --reload

serve-app:
	$(PYTHON) scripts/run_app.py

build:
	$(PYTHON) scripts/run_docker.py build

docker-up:
	$(COMPOSE) --profile all up -d api streamlit

docker-down:
	$(COMPOSE) down

mlflow-up:
	$(COMPOSE) --profile mlflow up -d mlflow

airflow-up:
	$(COMPOSE) --profile airflow up -d airflow-postgres airflow-init airflow-webserver airflow-scheduler

airflow-down:
	$(COMPOSE) --profile airflow down

airflow-logs:
	$(COMPOSE) logs -f airflow-webserver airflow-scheduler

airflow-load-vars:
	$(COMPOSE) exec airflow-webserver airflow variables import airflow/include/airflow_variables.example.json

airflow-init-connections:
	$(COMPOSE) exec airflow-webserver bash scripts/airflow_init_connections.sh

airflow-list-dags:
	$(COMPOSE) exec airflow-webserver airflow dags list

airflow-test-dataops:
	$(COMPOSE) exec airflow-webserver airflow dags test dataset_validation_registration_dag $$(date +%Y-%m-%d)

airflow-test-train:
	$(COMPOSE) exec airflow-webserver airflow dags test train_evaluate_promote_dag $$(date +%Y-%m-%d)

airflow-test-monitoring:
	$(COMPOSE) exec airflow-webserver airflow dags test monitoring_drift_retraining_dag $$(date +%Y-%m-%d)

airflow-test-demo:
	$(COMPOSE) exec airflow-webserver airflow dags test local_full_lifecycle_demo_dag $$(date +%Y-%m-%d)

airflow-reset-local:
	$(COMPOSE) --profile airflow down --volumes
	rm -rf airflow/logs/* artifacts/airflow/*

prometheus-up:
	$(COMPOSE) --profile monitoring up -d prometheus grafana

check-env:
	$(PYTHON) scripts/check_environment.py

smoke-docker:
	$(PYTHON) scripts/smoke_test_docker_stack.py

promote:
	$(PYTHON) scripts/promote_model.py --candidate-version $(VERSION) --require-eval-pass --write-deployment-record

rollback:
	$(PYTHON) scripts/rollback_model.py --to previous

mlops-demo:
	$(PYTHON) scripts/run_full_local_lifecycle_demo.py --dry-run

clean-artifacts:
	rm -rf artifacts/mlflow artifacts/reports/* artifacts/predictions/* artifacts/feedback/*
