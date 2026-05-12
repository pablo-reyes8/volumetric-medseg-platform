#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, read_yaml, write_json
from data.quality import generate_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local NIfTI data quality checks.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    contract = read_yaml(args.contract_path)
    quality = generate_quality_report(
        args.images_dir,
        args.labels_dir,
        allowed_labels=contract.get("label_schema"),
    )
    issues = quality.get("issues", [])
    failures = [issue for case in issues for issue in case.get("issues", []) if issue.get("severity") == "error"]
    warnings = [issue for case in issues for issue in case.get("issues", []) if issue.get("severity") == "warning"]
    report = add_lineage(
        {
            "status": "passed" if quality.get("status") == "pass" else "failed",
            "n_images": quality.get("total_pairs", 0),
            "n_labels": quality.get("total_pairs", 0),
            "n_pairs": quality.get("total_pairs", 0),
            "n_failures": len(failures),
            "n_warnings": len(warnings),
            "failures": failures,
            "warnings": warnings,
            "raw_quality_report": quality,
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="run_data_quality_checks",
    )
    write_json(args.output, report)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
