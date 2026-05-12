from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


@dataclass
class DagSpec:
    dag_id: str
    task_ids: List[str]
    dependencies: List[Tuple[str, str]] = field(default_factory=list)


def build_dag_spec(dag_id: str, task_ids: Iterable[str], dependencies: Iterable[Tuple[str, str]]) -> DagSpec:
    return DagSpec(dag_id=dag_id, task_ids=list(task_ids), dependencies=list(dependencies))
