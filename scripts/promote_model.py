#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_cli_utils import PROJECT_ROOT  # noqa: F401 - ensures repo root is importable
from src.mlops.model_registry import LocalModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a local challenger model package to champion.")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--require-eval-pass", action="store_true")
    parser.add_argument("--write-deployment-record", action="store_true")
    parser.add_argument("--promoted-by", default="local_user")
    parser.add_argument("--reason", default="manual_promotion")
    args = parser.parse_args()

    registry = LocalModelRegistry(args.artifact_root)
    record = registry.promote_challenger_to_champion(
        candidate_version=args.candidate_version,
        require_eval_pass=args.require_eval_pass,
        promoted_by=args.promoted_by,
        reason=args.reason,
    )
    root = Path(args.artifact_root)
    (root / "registry" / "champions").mkdir(parents=True, exist_ok=True)
    (root / "registry" / "history").mkdir(parents=True, exist_ok=True)
    champion_record = root / "registry" / "champion.json"
    champion_record.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    (root / "registry" / "champions" / f"{args.candidate_version}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (root / "registry" / "history" / "promotion_history.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, sort_keys=True) + "\n")
    deployment_record = Path(str(record["active_model_path"])) / "deployment_record.json"
    deployment_record.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Promoted {record['active_model_version']} to champion at {record['active_model_path']}")


if __name__ == "__main__":
    main()
