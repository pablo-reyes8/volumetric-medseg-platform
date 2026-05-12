#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlops_cli_utils import now_utc, read_yaml, write_json
from src.mlops.model_registry import LocalModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a local model package as a candidate.")
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    package_path = Path(args.model_package)
    package = read_yaml(package_path)
    version = str(package["model_version"])
    registry = LocalModelRegistry(args.artifact_root)
    record = registry.register_model_package(package_path.parent, stage="challenger", model_version=version)

    candidate_record = {
        "status": "candidate_registered",
        "model_version": version,
        "package_path": record["package_path"],
        "manifest_path": record["manifest_path"],
        "registered_at": now_utc(),
        "mlflow_run_id": package.get("mlflow_run_id"),
        "dataset_manifest_hash": package.get("dataset_manifest_hash"),
    }
    write_json(Path(args.artifact_root) / "registry" / "candidates" / f"{version}.json", candidate_record)
    history = Path(args.artifact_root) / "registry" / "history" / "candidate_events.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as file_obj:
        import json

        file_obj.write(json.dumps(candidate_record, sort_keys=True) + "\n")
    print(record)


if __name__ == "__main__":
    main()

