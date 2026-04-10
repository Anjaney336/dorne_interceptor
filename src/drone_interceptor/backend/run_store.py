from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class RunRecord:
    run_id: str
    schema_version: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class FileRunStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def create_run(self, kind: str, status: str, config: dict[str, Any], seed: int | None = None) -> RunRecord:
        now = _now_iso()
        record = RunRecord(
            run_id=f"run_{uuid4().hex[:12]}",
            schema_version="1.0",
            kind=str(kind),
            status=str(status),
            created_at=now,
            updated_at=now,
            config=dict(config),
            seed=seed,
        )
        self._write(record)
        return record

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RunRecord:
        record = self.get_run(run_id)
        if status is not None:
            record.status = str(status)
        if metrics is not None:
            record.metrics.update(metrics)
        if artifacts is not None:
            record.artifacts.update(artifacts)
        if validation is not None:
            record.validation.update(validation)
        if error is not None:
            record.error = str(error)
        record.updated_at = _now_iso()
        self._write(record)
        return record

    def list_runs(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for path in sorted(self._root.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append(RunRecord(**payload))
            except Exception:
                continue
        return records

    def get_run(self, run_id: str) -> RunRecord:
        path = self._path_for(run_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord(**payload)

    def list_artifacts(self, run_id: str) -> dict[str, Any]:
        return dict(self.get_run(run_id).artifacts)

    def _write(self, record: RunRecord) -> None:
        self._path_for(record.run_id).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    def _path_for(self, run_id: str) -> Path:
        return self._root / f"{run_id}.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["FileRunStore", "RunRecord"]
