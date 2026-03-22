#!/usr/bin/env python3
import argparse

try:
    from scripts.common import PROJECT_ROOT, exit_with_command
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from common import PROJECT_ROOT, exit_with_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the UNet3D Docker stack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one or all services.")
    build.add_argument("--service", choices=["api", "streamlit"], help="Optional single service to build.")

    up = subparsers.add_parser("up", help="Start the docker-compose stack.")
    up.add_argument("--service", choices=["api", "streamlit"], help="Optional single service to start.")
    up.add_argument("--detach", action="store_true", help="Run containers in detached mode.")

    down = subparsers.add_parser("down", help="Stop and remove the docker-compose stack.")
    down.add_argument("--volumes", action="store_true", help="Remove attached named volumes.")

    logs = subparsers.add_parser("logs", help="Tail compose logs.")
    logs.add_argument("--service", choices=["api", "streamlit"], help="Optional single service log stream.")
    logs.add_argument("--follow", action="store_true", help="Follow logs.")
    return parser


def compose_command(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), *args]


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build":
        command = compose_command("build")
        if args.service:
            command.append(args.service)
        exit_with_command(command)

    if args.command == "up":
        command = compose_command("up")
        if args.detach:
            command.append("-d")
        if args.service:
            command.append(args.service)
        exit_with_command(command)

    if args.command == "down":
        command = compose_command("down")
        if args.volumes:
            command.append("--volumes")
        exit_with_command(command)

    command = compose_command("logs")
    if args.follow:
        command.append("-f")
    if args.service:
        command.append(args.service)
    exit_with_command(command)


if __name__ == "__main__":
    main()
