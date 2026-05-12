# Local Airflow Orchestration

This folder contains the local-first Airflow orchestration layer for the volumetric medical segmentation lifecycle. Airflow coordinates repository scripts and artifacts; model training, evaluation, registry, drift, and policy logic stay in the project modules and CLIs.

## DAGs

| DAG | Purpose | Main outputs |
|---|---|---|
| `dataset_validation_registration_dag` | Validate dataset paths, contract, data quality, manifest, and registry state. | `data/manifests/`, `artifacts/reports/data_quality/` |
| `train_evaluate_promote_dag` | Run CPU-safe smoke training or candidate training, evaluate, gate, package, register, and branch promote/reject. | `artifacts/models/<version>/`, `artifacts/registry/` |
| `monitoring_drift_retraining_dag` | Summarize runtime, prediction, feedback, drift, and retraining policy signals. | `artifacts/reports/retraining/`, `artifacts/reports/drift/` |
| `local_full_lifecycle_demo_dag` | Prove the complete local MLOps lifecycle without heavy training. | `artifacts/reports/airflow/full_lifecycle_demo_*.md` |

The DAG files include lightweight fallback specs so `pytest tests/airflow` can validate the task graph without installing a full Airflow runtime.

Run Airflow locally:

```bash
make airflow-up
make airflow-load-vars
make airflow-init-connections
make airflow-list-dags
```

UI:

```text
http://localhost:8080
```

Default local credentials are `airflow` / `airflow`.

Run DAG smoke checks inside the Airflow container:

```bash
make airflow-test-dataops
make airflow-test-train
make airflow-test-monitoring
make airflow-test-demo
```
