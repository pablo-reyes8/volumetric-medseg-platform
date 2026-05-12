#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.mlops.model_registry import LocalModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback the local champion model to the previous champion.")
    parser.add_argument("--to", default="previous", choices=["previous"])
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--promoted-by", default="local_user")
    parser.add_argument("--reason", default="rollback")
    args = parser.parse_args()

    registry = LocalModelRegistry(args.artifact_root)
    record = registry.rollback_to_previous_champion(promoted_by=args.promoted_by, reason=args.reason)
    print(f"Rolled back to {record['active_model_version']} at {record['active_model_path']}")


if __name__ == "__main__":
    main()
