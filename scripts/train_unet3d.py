#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet3D or run the deterministic smoke trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", default="candidate")
    parser.add_argument("--smoke", action="store_true", help="Run CPU-safe smoke training instead of real training.")
    args = parser.parse_args()

    if not args.smoke:
        raise SystemExit("Real training is intentionally not wired into Airflow CI. Re-run with --smoke locally.")
    command = [
        sys.executable,
        "scripts/train_unet3d_smoke.py",
        "--config",
        args.config,
        "--dataset-manifest",
        args.dataset_manifest,
        "--output-dir",
        args.output_dir,
        "--model-version",
        args.model_version,
    ]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()

