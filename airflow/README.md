# Local Airflow Orchestration

This folder contains local-first DAGs for the volumetric medical segmentation lifecycle. The DAG files are importable without a local Airflow install so smoke tests can validate task shape in lightweight environments.

Run Airflow locally:

```bash
make airflow-up
```

UI:

```text
http://localhost:8080
```

Default local credentials are `airflow` / `airflow`.
