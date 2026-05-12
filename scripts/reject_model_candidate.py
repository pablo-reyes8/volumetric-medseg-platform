#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_cli_utils import now_utc, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject a local model candidate and write registry history.")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    record = {
        "status": "rejected",
        "model_version": args.candidate_version,
        "reason": args.reason,
        "rejected_at": now_utc(),
    }
    root = Path(args.artifact_root)
    write_json(root / "registry" / "rejected" / f"{args.candidate_version}.json", record)
    history = root / "registry" / "history" / "rejection_history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, sort_keys=True) + "\n")
    print(record)


if __name__ == "__main__":
    main()

