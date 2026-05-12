#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, now_utc, read_yaml, write_json, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic CPU-safe smoke training artifact.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/training/local_unet3d_smoke.yaml")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--model-version", default="smoke")
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    output_dir = __import__("pathlib").Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_yaml(args.config)
    checkpoint = output_dir / "checkpoint.pt"
    checkpoint.write_bytes(f"smoke-checkpoint:{args.model_version}:{now_utc()}".encode("utf-8"))
    write_yaml(output_dir / "training_config.yaml", config)
    metrics = config.get("metrics", {})
    mlflow_run = add_lineage(
        {
            "tracking_uri": "local-smoke",
            "experiment_name": "unet3d-medseg",
            "run_id": f"smoke-{args.model_version}",
            "artifact_uri": str(output_dir),
            "status": "completed",
            "metrics": metrics,
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="train_candidate",
    )
    write_json(output_dir / "mlflow_run.json", mlflow_run)
    training_report = add_lineage(
        {
            "status": "completed",
            "mode": "smoke",
            "model_version": args.model_version,
            "checkpoint_path": str(checkpoint),
            "training_config": args.config,
            "dataset_manifest": args.dataset_manifest,
            "metrics": metrics,
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="train_candidate",
    )
    write_json(output_dir / "training_report.json", training_report)
    print(str(checkpoint))


if __name__ == "__main__":
    main()

