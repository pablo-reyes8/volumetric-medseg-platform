#!/usr/bin/env python3
import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dataset manifest and update the registry.")
    parser.add_argument("--images-dir", required=True, help="Directory with input NIfTI volumes.")
    parser.add_argument("--labels-dir", required=True, help="Directory with mask NIfTI volumes.")
    parser.add_argument("--dataset-name", required=True, help="Dataset logical name.")
    parser.add_argument("--version", required=True, help="Semantic or calendar dataset version.")
    parser.add_argument("--manifest-out", required=True, help="Output JSON manifest path.")
    parser.add_argument("--registry-path", default="data/registry/datasets.yaml", help="Registry YAML path.")
    parser.add_argument("--source-url", help="Optional source URL for the dataset.")
    parser.add_argument("--notes", help="Optional free-form notes.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.versioning import build_dataset_manifest, save_dataset_manifest, update_dataset_registry
    from data.quality import validate_manifest_schema

    manifest = build_dataset_manifest(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        dataset_name=args.dataset_name,
        version=args.version,
        source_url=args.source_url,
        notes=args.notes,
    )
    schema_errors = validate_manifest_schema(manifest, "data/contracts/dataset_manifest.schema.json")
    if schema_errors:
        raise ValueError(f"Manifest invalido contra el schema: {schema_errors}")
    manifest_path = save_dataset_manifest(manifest, args.manifest_out)
    registry_path = update_dataset_registry(manifest, args.registry_path, manifest_path=manifest_path)
    print(f"Manifest saved to {manifest_path}")
    print(f"Registry updated at {registry_path}")


if __name__ == "__main__":
    main()
