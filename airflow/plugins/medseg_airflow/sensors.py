from __future__ import annotations

from pathlib import Path


def artifact_exists(path: Path | str) -> bool:
    return Path(path).exists()

