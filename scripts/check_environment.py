#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_import(module_name: str) -> Dict[str, object]:
    try:
        module = importlib.import_module(module_name)
        return {"name": module_name, "ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:  # pylint: disable=broad-except
        return {"name": module_name, "ok": False, "error": str(exc)}


def main() -> None:
    modules = ["torch", "fastapi", "mlflow", "nibabel", "jsonschema", "prometheus_client"]
    directories = ["data", "artifacts", "artifacts/registry", "artifacts/models", "artifacts/reports"]
    directory_report = []
    for relative_path in directories:
        path = PROJECT_ROOT / relative_path
        path.mkdir(parents=True, exist_ok=True)
        directory_report.append({"path": relative_path, "exists": path.exists()})

    report = {
        "python": {
            "version": platform.python_version(),
            "ok": platform.python_version_tuple() >= ("3", "10", "0"),
        },
        "imports": [_check_import(module_name) for module_name in modules],
        "directories": directory_report,
        "env_file": {"path": ".env", "exists": (PROJECT_ROOT / ".env").exists()},
    }
    failed = []
    if not report["python"]["ok"]:
        failed.append("python>=3.10")
    failed.extend(item["name"] for item in report["imports"] if not item["ok"])
    failed.extend(item["path"] for item in directory_report if not item["exists"])
    report["status"] = "pass" if not failed else "fail"
    report["failed_checks"] = failed
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
