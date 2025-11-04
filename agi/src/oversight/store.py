from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..core.manifest import RunManifest
from .models import ApprovalDecision, ApprovalRequest


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _json_loads(text: Optional[str]) -> Any:
    if text in (None, ""):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass
class ApprovalTicket:
    request: ApprovalRequest
    _event: Event
    _decision: Optional[ApprovalDecision] = None

    def resolve(self, decision: ApprovalDecision) -> None:
        self._decision = decision
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> Optional[ApprovalDecision]:
        if self._event.wait(timeout):
            return self._decision
        return None


class OversightStore:
    """State store for oversight data with optional SQLite persistence."""

    def __init__(
        self,
        *,
        base_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        telemetry_limit: int = 2000,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self._db_path = Path(db_path) if db_path is not None else None
        if self._db_path is None and self.base_dir is not None:
            self._db_path = self.base_dir / "oversight.db"
        self._telemetry_limit = telemetry_limit

        self._runs: Dict[str, Dict[str, Any]] = {}
        self._telemetry: List[Dict[str, Any]] = []
        self._memory_records: Dict[str, List[Dict[str, Any]]] = {}
        self._pending: Dict[str, ApprovalTicket] = {}
        self._history: Dict[str, ApprovalDecision] = {}

        self._lock = Lock()
        self._db_lock = Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._init_db()
            self._load_from_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        if self._conn is None:
            raise RuntimeError("SQLite connection not initialised")
        with self._db_lock:
            cursor = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cursor

    def _query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        if self._conn is None:
            raise RuntimeError("SQLite connection not initialised")
        with self._db_lock:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
            self._conn.row_factory = None
            return rows

    def _init_db(self) -> None:
        assert self._conn is not None
        with self._db_lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    manifest TEXT NOT NULL,
                    working_memory TEXT,
                    reflection_summary TEXT,
                    manifest_schema TEXT,
                    run_dir TEXT,
                    inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    name TEXT NOT NULL,
                    record_index INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (name, record_index)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    tool TEXT,
                    tier TEXT,
                    requested_by TEXT,
                    context TEXT,
                    status TEXT NOT NULL,
                    decided_at TEXT,
                    reviewer TEXT,
                    message TEXT,
                    approved INTEGER
                );
                """
            )
            self._conn.commit()

    def _load_from_db(self) -> None:
        if self._conn is None:
            return
        # Runs
        for row in self._query("SELECT * FROM runs"):
            run_id = row["run_id"]
            record = {
                "manifest": _json_loads(row["manifest"]),
                "working_memory": _json_loads(row["working_memory"]),
                "reflection_summary": _json_loads(row["reflection_summary"]),
                "manifest_schema": _json_loads(row["manifest_schema"]),
                "run_dir": row["run_dir"],
            }
            self._runs[run_id] = record

        # Telemetry (keep latest within limit)
        telemetry_rows = self._query(
            "SELECT event FROM telemetry ORDER BY id DESC LIMIT ?",
            (self._telemetry_limit,),
        )
        events = [_json_loads(row["event"]) for row in reversed(telemetry_rows)]
        self._telemetry = [event for event in events if isinstance(event, dict)]

        # Memory records
        rows = self._query("SELECT name, record_index, payload FROM memory_records ORDER BY name, record_index")
        memory: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            payload = _json_loads(row["payload"])
            if not isinstance(payload, dict):
                continue
            memory.setdefault(row["name"], []).append(payload)
        self._memory_records = memory

        # Approvals
        approval_rows = self._query("SELECT * FROM approvals")
        for row in approval_rows:
            request = ApprovalRequest(
                id=row["id"],
                created_at=row["created_at"],
                tool=row["tool"],
                tier=row["tier"],
                requested_by=row["requested_by"],
                context=_json_loads(row["context"]) or {},
            )
            status = row["status"].lower()
            if status == "pending":
                self._pending[request.id] = ApprovalTicket(request=request, _event=Event())
            else:
                decision = ApprovalDecision(
                    approval_id=request.id,
                    approved=bool(row["approved"]),
                    reviewer=row["reviewer"] or "",
                    decided_at=row["decided_at"],
                    message=row["message"],
                )
                self._history[request.id] = decision

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    def ingest_run_dir(self, run_dir: Path) -> str:
        run_dir = run_dir.expanduser().resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

        working_memory_path = run_dir / "working_memory.json"
        reflection_summary_path = run_dir / "reflection_summary.json"
        manifest_schema_path = run_dir / "manifest.schema.json"

        working_memory = _json_loads(working_memory_path.read_text(encoding="utf-8")) if working_memory_path.exists() else None
        reflection_summary = _json_loads(reflection_summary_path.read_text(encoding="utf-8")) if reflection_summary_path.exists() else None
        manifest_schema = _json_loads(manifest_schema_path.read_text(encoding="utf-8")) if manifest_schema_path.exists() else None

        record = {
            "manifest": manifest.model_dump(),
            "working_memory": working_memory,
            "reflection_summary": reflection_summary,
            "manifest_schema": manifest_schema,
            "run_dir": str(run_dir),
        }

        if self._conn is not None:
            self._execute(
                """
                INSERT INTO runs (run_id, manifest, working_memory, reflection_summary, manifest_schema, run_dir)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    manifest=excluded.manifest,
                    working_memory=excluded.working_memory,
                    reflection_summary=excluded.reflection_summary,
                    manifest_schema=excluded.manifest_schema,
                    run_dir=excluded.run_dir
                """,
                (
                    manifest.run_id,
                    _json_dumps(record["manifest"]),
                    _json_dumps(record["working_memory"]),
                    _json_dumps(record["reflection_summary"]),
                    _json_dumps(record["manifest_schema"]),
                    record["run_dir"],
                ),
            )

        with self._lock:
            self._runs[manifest.run_id] = record
        return manifest.run_id

    def ingest_memory(self, name: str, path: Path, *, limit: int = 200) -> None:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Memory log not found: {path}")
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
                if len(records) >= limit:
                    break

        if self._conn is not None:
            self._execute("DELETE FROM memory_records WHERE name = ?", (name,))
            for index, record in enumerate(records):
                self._execute(
                    "INSERT INTO memory_records (name, record_index, payload) VALUES (?, ?, ?)",
                    (name, index, _json_dumps(record)),
                )

        with self._lock:
            self._memory_records[name] = list(records)

    def record_telemetry(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        if self._conn is not None:
            self._execute(
                "INSERT INTO telemetry (event) VALUES (?)",
                (_json_dumps(payload),),
            )
        with self._lock:
            self._telemetry.append(payload)
            if len(self._telemetry) > self._telemetry_limit:
                excess = len(self._telemetry) - self._telemetry_limit
                if excess > 0:
                    del self._telemetry[:excess]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "run_id": run_id,
                    "goal": entry["manifest"].get("goal"),
                    "created_at": entry["manifest"].get("created_at"),
                    "tool_invocations": len(entry["manifest"].get("tool_results") or []),
                }
                for run_id, entry in sorted(self._runs.items())
            ]

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            entry = self._runs[run_id]
            return {
                "manifest": entry["manifest"],
                "working_memory": entry.get("working_memory"),
                "reflection_summary": entry.get("reflection_summary"),
                "manifest_schema": entry.get("manifest_schema"),
                "run_dir": entry.get("run_dir"),
            }

    def telemetry(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._telemetry)
        if limit is not None and limit > 0:
            records = records[-limit:]
        return records

    def memory_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {name: list(records) for name, records in self._memory_records.items()}

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------
    def create_approval_request(self, request: ApprovalRequest) -> ApprovalTicket:
        ticket = ApprovalTicket(request=request, _event=Event())
        with self._lock:
            if request.id in self._pending or request.id in self._history:
                raise ValueError(f"Approval id already tracked: {request.id}")
            self._pending[request.id] = ticket
        if self._conn is not None:
            self._execute(
                """
                INSERT INTO approvals (id, created_at, tool, tier, requested_by, context, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    request.id,
                    request.created_at,
                    request.tool,
                    request.tier,
                    request.requested_by,
                    _json_dumps(request.context),
                ),
            )
        return ticket

    def list_pending_approvals(self) -> List[ApprovalRequest]:
        with self._lock:
            return [ticket.request for ticket in self._pending.values()]

    def list_decisions(self) -> List[ApprovalDecision]:
        with self._lock:
            return list(self._history.values())

    def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> None:
        with self._lock:
            ticket = self._pending.pop(approval_id, None)
            if ticket is None:
                raise KeyError(approval_id)
            self._history[approval_id] = decision
        if self._conn is not None:
            self._execute(
                """
                UPDATE approvals
                SET status = 'resolved',
                    decided_at = ?,
                    reviewer = ?,
                    message = ?,
                    approved = ?
                WHERE id = ?
                """,
                (
                    decision.decided_at,
                    decision.reviewer,
                    decision.message,
                    1 if decision.approved else 0,
                    approval_id,
                ),
            )
        ticket.resolve(decision)


__all__ = ["OversightStore", "ApprovalTicket"]
