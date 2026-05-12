#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, read_json, write_json
from data.versioning import load_dataset_manifest, update_dataset_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a validated dataset version.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--registry-path", default="data/registry/datasets.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest_path)
    quality = read_json(args.quality_report)
    manifest["quality_report"] = manifest.get("quality_report") or {}
    manifest["quality_report"]["status"] = "pass" if quality.get("status") == "passed" else "fail"
    registry_path = update_dataset_registry(manifest, args.registry_path, manifest_path=args.manifest_path)
    report = add_lineage(
        {
            "status": "registered",
            "dataset_name": args.dataset_name,
            "dataset_version": args.dataset_version,
            "manifest_path": args.manifest_path,
            "quality_report": args.quality_report,
            "registry_path": str(registry_path),
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="register_dataset_version",
    )
    if args.output:
        write_json(args.output, report)
    else:
        print(report)


if __name__ == "__main__":
    main()
