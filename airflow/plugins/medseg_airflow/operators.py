from __future__ import annotations


def repo_bash(command: str) -> str:
    return f"cd /workspace && {command}"


def script_command(script_name: str, *args: str) -> str:
    rendered_args = " ".join(str(arg) for arg in args if arg is not None)
    return repo_bash(f"python scripts/{script_name} {rendered_args}".strip())

