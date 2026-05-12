from pathlib import Path

from airflow.plugins.medseg_airflow.paths import dag_run_output_dir, ensure_artifact_dirs, safe_run_id


def test_airflow_path_helpers_create_expected_directories(tmp_path: Path):
    paths = ensure_artifact_dirs(tmp_path / "artifacts")
    assert paths["dag_runs"].exists()
    assert paths["registry_history"].exists()

    output = dag_run_output_dir("demo", "manual:2026-05-12+test", tmp_path / "artifacts")
    assert output.exists()
    assert safe_run_id("manual:2026-05-12+test") == "manual_2026-05-12_test"

