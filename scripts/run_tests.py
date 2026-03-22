#!/usr/bin/env python3
import argparse

try:
    from scripts.common import exit_with_command, python_module_command
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from common import exit_with_command, python_module_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UNet3D test suite.")
    parser.add_argument("paths", nargs="*", default=["tests"], help="Specific test paths to run.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pytest output.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure.")
    parser.add_argument("--cov", action="store_true", help="Enable lightweight coverage over src/api.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = python_module_command("pytest", *args.paths)
    if args.verbose:
        command.append("-vv")
    if args.fail_fast:
        command.append("-x")
    if args.cov:
        command.extend(["--cov=src/api", "--cov-report=term-missing"])

    exit_with_command(command)


if __name__ == "__main__":
    main()
