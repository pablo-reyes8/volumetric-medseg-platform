from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.mlops.evaluation_report import is_evaluation_passed, load_evaluation_report


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git may be unavailable in packaged runtimes
        return "unknown"


class LocalModelRegistry:
    def __init__(self, artifact_root: Path | str = "artifacts"):
        self.artifact_root = Path(artifact_root)
        self.registry_root = self.artifact_root / "registry"
        self.models_root = self.artifact_root / "models"
        self.models_yaml = self.registry_root / "models.yaml"
        self.deployments_yaml = self.registry_root / "deployments.yaml"
        for path in [
            self.registry_root,
            self.models_root / "champion",
            self.models_root / "challenger",
            self.models_root / "archive",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _read_yaml(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _write_yaml(self, path: Path, payload: Dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        return path

    def _package_dir(self, stage: str, version: str) -> Path:
        return self.models_root / stage / version

    def register_model_package(
        self,
        package_dir: Path | str,
        stage: str = "challenger",
        model_version: Optional[str] = None,
    ) -> Dict[str, object]:
        source = Path(package_dir)
        manifest_path = source / "model_package.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing model package manifest: {manifest_path}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        version = str(model_version or manifest.get("model_version"))
        if not version:
            raise ValueError("model_version is required in the manifest or method argument")
        if stage not in {"challenger", "champion"}:
            raise ValueError("stage must be 'challenger' or 'champion'")

        destination = self._package_dir(stage, version)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

        registry = self._read_yaml(self.models_yaml)
        registry.setdefault("schema_version", "1.0")
        registry["updated_at_utc"] = _now_utc()
        registry.setdefault("packages", {})
        registry["packages"][version] = {
            "model_version": version,
            "stage": stage,
            "package_path": str(destination),
            "manifest_path": str(destination / "model_package.yaml"),
            "registered_at_utc": _now_utc(),
            "mlflow_run_id": manifest.get("mlflow_run_id"),
            "dataset_manifest_hash": manifest.get("dataset_manifest_hash"),
            "status": manifest.get("status", "candidate"),
        }
        if stage == "champion":
            registry["champion_version"] = version
        else:
            registry["latest_challenger_version"] = version
        self._write_yaml(self.models_yaml, registry)
        return registry["packages"][version]

    def build_model_package_manifest(
        self,
        model_name: str,
        model_version: str,
        checkpoint_path: Path | str,
        training_config_path: Path | str,
        dataset_manifest_path: Path | str,
        mlflow_run_id: str,
        model_card_path: Path | str = "model.yaml",
        evaluation_report_path: Path | str | None = None,
        metrics: Optional[Dict[str, float]] = None,
        status: str = "candidate",
    ) -> Dict[str, object]:
        training_config = Path(training_config_path)
        dataset_manifest = Path(dataset_manifest_path)
        checkpoint = Path(checkpoint_path)
        evaluation_path = Path(evaluation_report_path) if evaluation_report_path else None
        return {
            "model_name": model_name,
            "model_version": model_version,
            "created_at": _now_utc(),
            "git_commit": _git_commit(),
            "training_config_path": str(training_config),
            "training_config_hash": _sha256(training_config) if training_config.exists() else None,
            "dataset_manifest_path": str(dataset_manifest),
            "dataset_manifest_hash": _sha256(dataset_manifest) if dataset_manifest.exists() else None,
            "mlflow_run_id": mlflow_run_id,
            "checkpoint_path": str(checkpoint),
            "model_card_path": str(model_card_path),
            "evaluation_report_path": str(evaluation_path) if evaluation_path else None,
            "metrics": metrics or {},
            "status": status,
        }

    def write_model_package_manifest(self, manifest: Dict[str, object], package_dir: Path | str) -> Path:
        destination = Path(package_dir)
        destination.mkdir(parents=True, exist_ok=True)
        return self._write_yaml(destination / "model_package.yaml", manifest)

    def get_champion(self) -> Optional[Dict[str, object]]:
        registry = self._read_yaml(self.models_yaml)
        version = registry.get("champion_version")
        if not version:
            return None
        return registry.get("packages", {}).get(version)

    def get_challenger(self, version: Optional[str] = None) -> Optional[Dict[str, object]]:
        registry = self._read_yaml(self.models_yaml)
        selected = version or registry.get("latest_challenger_version")
        if not selected:
            return None
        return registry.get("packages", {}).get(selected)

    def promote_challenger_to_champion(
        self,
        candidate_version: str,
        require_eval_pass: bool = False,
        promoted_by: str = "local_user",
        reason: str = "manual_promotion",
    ) -> Dict[str, object]:
        challenger = self.get_challenger(candidate_version)
        if not challenger:
            raise ValueError(f"Challenger version not found: {candidate_version}")
        source = Path(str(challenger["package_path"]))
        manifest = yaml.safe_load((source / "model_package.yaml").read_text(encoding="utf-8")) or {}
        evaluation_report_path = manifest.get("evaluation_report_path")
        if require_eval_pass:
            if not evaluation_report_path:
                raise ValueError("Promotion requires an evaluation report")
            if not is_evaluation_passed(load_evaluation_report(evaluation_report_path)):
                raise ValueError("Evaluation gates did not pass")

        registry = self._read_yaml(self.models_yaml)
        previous_version = registry.get("champion_version")
        if previous_version:
            previous_path = self._package_dir("champion", str(previous_version))
            if previous_path.exists():
                archive_path = self.models_root / "archive" / str(previous_version)
                if archive_path.exists():
                    shutil.rmtree(archive_path)
                shutil.move(str(previous_path), str(archive_path))

        champion_path = self._package_dir("champion", candidate_version)
        if champion_path.exists():
            shutil.rmtree(champion_path)
        shutil.copytree(source, champion_path)

        registry.setdefault("packages", {})
        registry["packages"][candidate_version] = {
            **registry["packages"].get(candidate_version, {}),
            "stage": "champion",
            "package_path": str(champion_path),
            "manifest_path": str(champion_path / "model_package.yaml"),
            "promoted_at_utc": _now_utc(),
        }
        registry["champion_version"] = candidate_version
        registry["previous_champion_version"] = previous_version
        registry["updated_at_utc"] = _now_utc()
        self._write_yaml(self.models_yaml, registry)
        return self.write_deployment_record(
            active_model_version=candidate_version,
            active_model_path=str(champion_path),
            previous_model_version=previous_version,
            promoted_by=promoted_by,
            reason=reason,
            mlflow_run_id=manifest.get("mlflow_run_id"),
            dataset_manifest_hash=manifest.get("dataset_manifest_hash"),
            evaluation_report_path=evaluation_report_path,
        )

    def rollback_to_previous_champion(self, promoted_by: str = "local_user", reason: str = "rollback") -> Dict[str, object]:
        registry = self._read_yaml(self.models_yaml)
        current_version = registry.get("champion_version")
        previous_version = registry.get("previous_champion_version")
        if not previous_version:
            raise ValueError("No previous champion version is available for rollback")

        archived = self.models_root / "archive" / str(previous_version)
        if not archived.exists():
            raise FileNotFoundError(f"Archived champion package not found: {archived}")

        current_path = self._package_dir("champion", str(current_version))
        if current_path.exists():
            archive_current = self.models_root / "archive" / str(current_version)
            if archive_current.exists():
                shutil.rmtree(archive_current)
            shutil.move(str(current_path), str(archive_current))

        restored_path = self._package_dir("champion", str(previous_version))
        if restored_path.exists():
            shutil.rmtree(restored_path)
        shutil.copytree(archived, restored_path)

        registry["champion_version"] = previous_version
        registry["previous_champion_version"] = current_version
        registry.setdefault("packages", {})
        registry["packages"][str(previous_version)] = {
            **registry["packages"].get(str(previous_version), {}),
            "stage": "champion",
            "package_path": str(restored_path),
            "manifest_path": str(restored_path / "model_package.yaml"),
            "rolled_back_at_utc": _now_utc(),
        }
        self._write_yaml(self.models_yaml, registry)
        manifest = yaml.safe_load((restored_path / "model_package.yaml").read_text(encoding="utf-8")) or {}
        return self.write_deployment_record(
            active_model_version=str(previous_version),
            active_model_path=str(restored_path),
            previous_model_version=str(current_version) if current_version else None,
            promoted_by=promoted_by,
            reason=reason,
            mlflow_run_id=manifest.get("mlflow_run_id"),
            dataset_manifest_hash=manifest.get("dataset_manifest_hash"),
            evaluation_report_path=manifest.get("evaluation_report_path"),
        )

    def write_deployment_record(
        self,
        active_model_version: str,
        active_model_path: str,
        previous_model_version: Optional[str],
        promoted_by: str,
        reason: str,
        mlflow_run_id: Optional[str] = None,
        dataset_manifest_hash: Optional[str] = None,
        evaluation_report_path: Optional[str] = None,
    ) -> Dict[str, object]:
        deployments = self._read_yaml(self.deployments_yaml)
        deployments.setdefault("schema_version", "1.0")
        deployments.setdefault("history", [])
        record = {
            "active_model_version": active_model_version,
            "active_model_path": active_model_path,
            "previous_model_version": previous_model_version,
            "promoted_at": _now_utc(),
            "promoted_by": promoted_by,
            "reason": reason,
            "mlflow_run_id": mlflow_run_id,
            "dataset_manifest_hash": dataset_manifest_hash,
            "evaluation_report_path": evaluation_report_path,
            "git_commit": _git_commit(),
        }
        deployments.update(record)
        deployments["history"].append(record)
        self._write_yaml(self.deployments_yaml, deployments)
        return record
