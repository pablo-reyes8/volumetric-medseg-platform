#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import read_json, write_json
from data.quality import validate_json_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a JSON file against a JSON schema.")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    errors = validate_json_schema(read_json(args.json), args.schema)
    report = {"status": "passed" if not errors else "failed", "schema": args.schema, "json": args.json, "errors": errors}
    if args.output:
        write_json(args.output, report)
    else:
        print(report)
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
