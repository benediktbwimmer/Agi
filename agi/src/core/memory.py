from __future__ import annotations

import json
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List

from .telemetry import Telemetry


def _normalise_time(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _hash_source(source: Dict[str, Any]) -> str:
    payload = json.dumps(source, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class MemoryStore:
    path: Path
    telemetry: Telemetry | None = None
    _lock: Lock = field(default_factory=Lock, init=False)
    _claim_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _time_keys: List[datetime] = field(default_factory=list, init=False)
    _time_records: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _tool_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _source_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._index_record(record)

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        self._index_record(record)
        if self.telemetry is not None:
            payload = {
                "record_type": record.get("type"),
                "tool": record.get("tool"),
                "plan_id": record.get("plan_id"),
                "call_id": record.get("call_id"),
                "path": str(self.path),
            }
            self.telemetry.emit("memory.append", **{k: v for k, v in payload.items() if v is not None})

    def query_by_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(r)) for r in self._claim_index.get(claim_id, [])]

    def query_by_time(self, start: str, end: str) -> List[Dict[str, Any]]:
        start_ts = _normalise_time(start)
        end_ts = _normalise_time(end)
        start_idx = bisect_left(self._time_keys, start_ts)
        end_idx = bisect_right(self._time_keys, end_ts)
        return [
            json.loads(json.dumps(record))
            for record in self._time_records[start_idx:end_idx]
        ]

    def query_by_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(r)) for r in self._tool_index.get(tool_name, [])]

    def query_by_source_hash(self, digest: str) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(r)) for r in self._source_index.get(digest, [])]

    def _index_record(self, record: Dict[str, Any]) -> None:
        if "time" in record:
            try:
                ts = _normalise_time(record["time"])
            except ValueError:  # pragma: no cover - invalid timestamp
                pass
            else:
                insert_at = bisect_right(self._time_keys, ts)
                self._time_keys.insert(insert_at, ts)
                self._time_records.insert(insert_at, record)
        claim = record.get("claim")
        if record.get("type") == "semantic" and not isinstance(claim, dict):
            claim = record.get("claim", {})
        if isinstance(claim, dict):
            claim_id = claim.get("id")
            if claim_id:
                self._claim_index.setdefault(claim_id, []).append(record)
        trace = record.get("trace")
        if isinstance(trace, list):
            for step in trace:
                tool = step.get("tool") if isinstance(step, dict) else None
                if tool:
                    bucket = self._tool_index.setdefault(tool, [])
                    if record not in bucket:
                        bucket.append(record)
        tool_name = record.get("tool")
        if tool_name:
            bucket = self._tool_index.setdefault(tool_name, [])
            if record not in bucket:
                bucket.append(record)
        sources = record.get("sources") or record.get("provenance")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    digest = _hash_source(source)
                    self._source_index.setdefault(digest, []).append(record)
