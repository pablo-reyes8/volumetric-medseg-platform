import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_env(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    root = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{python_path}" if python_path else root
    if extra:
        env.update({key: value for key, value in extra.items() if value is not None})
    return env


def run_command(command: list[str], extra_env: Optional[Mapping[str, str]] = None) -> int:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=build_env(extra_env), check=False)
    return completed.returncode


def exit_with_command(command: list[str], extra_env: Optional[Mapping[str, str]] = None) -> None:
    raise SystemExit(run_command(command, extra_env=extra_env))


def python_module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]
