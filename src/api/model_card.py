from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_model_card(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}

    return payload if isinstance(payload, dict) else {}


def extract_maintainers(model_card: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_maintainers = model_card.get("metadata", {}).get("maintainers", [])
    maintainers: List[Dict[str, str]] = []

    for item in raw_maintainers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        email = str(item.get("email", "")).strip()
        if not name:
            continue
        entry = {"name": name}
        if email:
            entry["email"] = email
        maintainers.append(entry)

    return maintainers
