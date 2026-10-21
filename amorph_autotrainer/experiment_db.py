from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    fit_seconds REAL,
    predict_latency_ms REAL,
    validation_metrics_json TEXT,
    test_metrics_json TEXT,
    statistical_gate_json TEXT,
    cost_json TEXT,
    artifact_path TEXT,
    error TEXT,
    PRIMARY KEY (run_id, strategy)
);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at_utc);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_sha256);
"""


class ExperimentDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def record(self, row: dict[str, Any]) -> None:
        keys = [
            "run_id", "strategy", "created_at_utc", "dataset_sha256", "config_json",
            "status", "fit_seconds", "predict_latency_ms", "validation_metrics_json",
            "test_metrics_json", "statistical_gate_json", "cost_json", "artifact_path", "error",
        ]
        values = [row.get(key) for key in keys]
        placeholders = ",".join("?" for _ in keys)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO experiments ({','.join(keys)}) VALUES ({placeholders})",
                values,
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id,strategy,created_at_utc,dataset_sha256,status,fit_seconds,"
                "predict_latency_ms,validation_metrics_json,test_metrics_json,statistical_gate_json,"
                "cost_json,artifact_path,error FROM experiments ORDER BY created_at_utc DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
