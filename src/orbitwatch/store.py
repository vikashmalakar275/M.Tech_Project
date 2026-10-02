from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from orbitwatch.evidence import InvestigationReport


class InvestigationStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS investigations ("
                "id INTEGER PRIMARY KEY, evidence_id TEXT NOT NULL, run_id TEXT NOT NULL,"
                "channel TEXT NOT NULL, engine TEXT NOT NULL, created_at TEXT NOT NULL,"
                "report_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tool_audit ("
                "id INTEGER PRIMARY KEY, tool TEXT NOT NULL, arguments_json TEXT NOT NULL,"
                "created_at TEXT NOT NULL, outcome TEXT NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, report: InvestigationReport) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO investigations "
                "(evidence_id,run_id,channel,engine,created_at,report_json) VALUES (?,?,?,?,?,?)",
                (
                    report.evidence.evidence_id,
                    report.evidence.run_id,
                    str(report.evidence.facts["channel"]),
                    report.engine,
                    report.generated_at,
                    report.model_dump_json(),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 100:
            raise ValueError("History limit must be between 1 and 100.")
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,evidence_id,run_id,channel,engine,created_at FROM investigations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def audit(self, tool: str, arguments: dict, outcome: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO tool_audit (tool,arguments_json,created_at,outcome) VALUES (?,?,?,?)",
                (
                    tool,
                    json.dumps(arguments, allow_nan=False),
                    datetime.now(UTC).isoformat(),
                    outcome,
                ),
            )
