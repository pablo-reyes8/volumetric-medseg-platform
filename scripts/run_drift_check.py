#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or evaluate a medical-volume drift profile.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Create a baseline drift profile from a directory.")
    baseline.add_argument("--images-dir", required=True, help="Directory with baseline NIfTI volumes.")
    baseline.add_argument("--dataset-version", required=True, help="Dataset version attached to this baseline.")
    baseline.add_argument("--output-path", required=True, help="Output JSON profile path.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate candidate volumes against a baseline profile.")
    evaluate.add_argument("--images-dir", required=True, help="Directory with candidate NIfTI volumes.")
    evaluate.add_argument("--baseline-path", required=True, help="Existing baseline profile JSON.")
    evaluate.add_argument("--output-path", help="Optional output JSON report path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.mlops.drift import build_reference_profile, evaluate_reference_drift, load_profile, save_profile

    if args.command == "baseline":
        profile = build_reference_profile(args.images_dir, dataset_version=args.dataset_version)
        save_profile(profile, args.output_path)
        print(f"Baseline profile saved to {args.output_path}")
        return

    baseline = load_profile(args.baseline_path)
    report = evaluate_reference_drift(args.images_dir, baseline)
    if args.output_path:
        save_profile(report, args.output_path)
        print(f"Drift report saved to {args.output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
