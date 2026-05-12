PYTHON ?= python
PIP ?= $(PYTHON) -m pip
COMPOSE ?= docker compose

.PHONY: setup install lint format test test-unit test-api test-mlops test-data train serve-api serve-app build docker-up docker-down mlflow-up airflow-up airflow-down prometheus-up clean-artifacts check-env smoke-docker promote rollback

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

clean-artifacts:
	rm -rf artifacts/mlflow artifacts/reports/* artifacts/predictions/* artifacts/feedback/*
