#!/usr/bin/env python3
import argparse

try:
    from scripts.common import exit_with_command, python_module_command
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from common import exit_with_command, python_module_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UNet3D FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding for uvicorn.")
    parser.add_argument("--port", default="8000", help="Port for uvicorn.")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for local development.")
    parser.add_argument("--model-path", help="Checkpoint path exported as UNET3D_MODEL_PATH.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug"], help="Uvicorn log level.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = python_module_command("uvicorn", "src.api.main:app", "--host", args.host, "--port", args.port)
    if args.reload:
        command.append("--reload")
    if args.log_level:
        command.extend(["--log-level", args.log_level])

    exit_with_command(
        command,
        extra_env={
            "UNET3D_MODEL_PATH": args.model_path,
            "UNET3D_DEVICE": args.device,
        },
    )


if __name__ == "__main__":
    main()
