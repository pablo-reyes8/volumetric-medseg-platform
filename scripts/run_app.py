#!/usr/bin/env python3
import argparse

try:
    from scripts.common import PROJECT_ROOT, exit_with_command, python_module_command
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from common import PROJECT_ROOT, exit_with_command, python_module_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UNet3D Streamlit application.")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding for Streamlit.")
    parser.add_argument("--port", default="8501", help="Port for Streamlit.")
    parser.add_argument("--api-url", help="Base URL for the FastAPI service.")
    parser.add_argument("--model-path", help="Checkpoint path for local inference mode.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = python_module_command(
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app" / "streamlit_app.py"),
        "--server.address",
        args.host,
        "--server.port",
        args.port,
        "--browser.gatherUsageStats",
        "false",
    )

    exit_with_command(
        command,
        extra_env={
            "UNET3D_API_URL": args.api_url,
            "UNET3D_MODEL_PATH": args.model_path,
        },
    )


if __name__ == "__main__":
    main()
