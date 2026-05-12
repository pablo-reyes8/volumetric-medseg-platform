#!/usr/bin/env python3
from __future__ import annotations

import argparse

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
    print(f"Promoted {record['active_model_version']} to champion at {record['active_model_path']}")


if __name__ == "__main__":
    main()
