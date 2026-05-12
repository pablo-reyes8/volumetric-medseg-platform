#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from mlops_cli_utils import add_lineage, now_utc, read_yaml, write_json, write_yaml


def _train_tiny_mlp(seed: int = 42, epochs: int = 25) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = ((x[:, 0] + x[:, 1] * 0.5 - x[:, 2] * 0.25) > 0).astype(np.float32).reshape(-1, 1)
    w1 = rng.normal(scale=0.1, size=(4, 6)).astype(np.float32)
    b1 = np.zeros((1, 6), dtype=np.float32)
    w2 = rng.normal(scale=0.1, size=(6, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)
    lr = 0.1

    for _ in range(epochs):
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad_logits = (probs - y) / len(x)
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0, keepdims=True)
        grad_hidden = grad_logits @ w2.T * (1.0 - hidden**2)
        grad_w1 = x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0, keepdims=True)
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

    hidden = np.tanh(x @ w1 + b1)
    probs = 1.0 / (1.0 + np.exp(-(hidden @ w2 + b2)))
    predictions = (probs >= 0.5).astype(np.float32)
    accuracy = float((predictions == y).mean())
    loss = float(-(y * np.log(probs + 1e-7) + (1 - y) * np.log(1 - probs + 1e-7)).mean())
    return {
        "mlp_accuracy": round(accuracy, 6),
        "mlp_loss": round(loss, 6),
        "mlp_parameters": int(w1.size + b1.size + w2.size + b2.size),
    }


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
    mlp_metrics = _train_tiny_mlp(seed=int(config.get("seed", 42)))
    checkpoint.write_bytes(f"smoke-mlp-checkpoint:{args.model_version}:{mlp_metrics}:{now_utc()}".encode("utf-8"))
    write_yaml(output_dir / "training_config.yaml", config)
    metrics = {**config.get("metrics", {}), **mlp_metrics}
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
