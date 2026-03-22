#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download, prepare and validate MSD Task04 Hippocampus.")
    parser.add_argument("--dataset-version", required=True, help="Version label for the prepared dataset.")
    parser.add_argument("--base-dir", default="data", help="Base data directory.")
    parser.add_argument("--manifest-out", help="Optional explicit manifest output path.")
    parser.add_argument("--registry-path", help="Optional explicit registry YAML path.")
    parser.add_argument("--overwrite-archive", action="store_true", help="Redownload the archive even if it exists.")
    parser.add_argument("--overwrite-processed", action="store_true", help="Replace the processed dataset layout.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.task04 import prepare_task04_dataset

    report = prepare_task04_dataset(
        dataset_version=args.dataset_version,
        base_dir=args.base_dir,
        manifest_output_path=args.manifest_out,
        registry_path=args.registry_path,
        overwrite_archive=args.overwrite_archive,
        overwrite_processed=args.overwrite_processed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
