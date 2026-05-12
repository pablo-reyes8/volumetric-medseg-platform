#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlops_cli_utils import add_lineage, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset contract against local data paths.")
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    contract = read_yaml(args.contract_path)
    failures = []
    warnings = []
    for label, path in [("images_dir", Path(args.images_dir)), ("labels_dir", Path(args.labels_dir)), ("contract_path", Path(args.contract_path))]:
        if not path.exists():
            failures.append(f"{label} does not exist: {path}")
    if not contract:
        failures.append("contract is empty")
    allowed_labels = contract.get("label_schema", {})
    if not allowed_labels:
        warnings.append("contract has no label_schema")

    report = add_lineage(
        {
            "status": "passed" if not failures else "failed",
            "contract_path": args.contract_path,
            "images_dir": args.images_dir,
            "labels_dir": args.labels_dir,
            "failures": failures,
            "warnings": warnings,
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="validate_dataset_contract",
    )
    write_json(args.output, report)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()

