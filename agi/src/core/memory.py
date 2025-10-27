from __future__ import annotations

import json
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .telemetry import Telemetry


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _normalise_time(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _hash_source(source: Dict[str, Any]) -> str:
    payload = json.dumps(source, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _tokenise(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _extract_text(record: Mapping[str, Any], *, max_depth: int = 3) -> str:
    parts: List[str] = []

    def _gather(value: Any, depth: int) -> None:
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(stripped)
            return
        if depth >= max_depth:
            return
        if isinstance(value, Mapping):
            for item in value.values():
                _gather(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _gather(item, depth + 1)

    _gather(record, 0)
    return " ".join(parts)


@dataclass
class MemoryStore:
    path: Path
    telemetry: Telemetry | None = None
    _lock: Lock = field(default_factory=Lock, init=False)
    _records: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _record_times: List[Optional[datetime]] = field(default_factory=list, init=False)
    _claim_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _time_keys: List[datetime] = field(default_factory=list, init=False)
    _time_records: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _tool_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _source_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _token_index: Dict[str, List[int]] = field(default_factory=dict, init=False)
    _plan_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)

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

    def query_by_plan(self, plan_id: str) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(r)) for r in self._plan_index.get(plan_id, [])]

    def recent(
        self, *, limit: int = 20, types: Iterable[str] | None = None
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        type_filter = {t for t in types} if types is not None else None
        results: List[Dict[str, Any]] = []
        for record in reversed(self._time_records):
            if type_filter is not None and record.get("type") not in type_filter:
                continue
            results.append(json.loads(json.dumps(record)))
            if len(results) >= limit:
                break
        return list(reversed(results))

    def semantic_search(
        self,
        query: str,
        *,
        limit: int = 5,
        types: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return memory records that best match ``query`` lexically."""

        if limit <= 0:
            return []

        tokens = _tokenise(query)
        if not tokens:
            return []

        type_filter = {t for t in types} if types is not None else None
        scores: Counter[int] = Counter()
        for token in tokens:
            for record_idx in self._token_index.get(token, []):
                if type_filter is not None:
                    record_type = self._records[record_idx].get("type")
                    if record_type not in type_filter:
                        continue
                scores[record_idx] += 1

        if not scores:
            return []

        def _sort_key(index: int) -> tuple[int, float, int]:
            timestamp = self._record_times[index]
            ts_key = timestamp.timestamp() if timestamp is not None else float("-inf")
            return (-scores[index], -ts_key, -index)

        ranked_indices = sorted(scores, key=_sort_key)
        limited = ranked_indices[:limit]
        return [json.loads(json.dumps(self._records[idx])) for idx in limited]

    def _index_record(self, record: Dict[str, Any]) -> None:
        stored_record = json.loads(json.dumps(record))
        self._records.append(stored_record)
        record_index = len(self._records) - 1

        record_time: Optional[datetime] = None
        if "time" in stored_record:
            try:
                record_time = _normalise_time(stored_record["time"])
            except ValueError:  # pragma: no cover - invalid timestamp
                record_time = None
            else:
                insert_at = bisect_right(self._time_keys, record_time)
                self._time_keys.insert(insert_at, record_time)
                self._time_records.insert(insert_at, stored_record)
        self._record_times.append(record_time)

        claim = stored_record.get("claim")
        if stored_record.get("type") == "semantic" and not isinstance(claim, dict):
            claim = stored_record.get("claim", {})
        if isinstance(claim, dict):
            claim_id = claim.get("id")
            if claim_id:
                self._claim_index.setdefault(claim_id, []).append(stored_record)
        trace = stored_record.get("trace")
        if isinstance(trace, list):
            for step in trace:
                tool = step.get("tool") if isinstance(step, dict) else None
                if tool:
                    bucket = self._tool_index.setdefault(tool, [])
                    if stored_record not in bucket:
                        bucket.append(stored_record)
        tool_name = stored_record.get("tool")
        if tool_name:
            bucket = self._tool_index.setdefault(tool_name, [])
            if stored_record not in bucket:
                bucket.append(stored_record)
        plan_id = stored_record.get("plan_id")
        if isinstance(plan_id, str) and plan_id:
            bucket = self._plan_index.setdefault(plan_id, [])
            if stored_record not in bucket:
                bucket.append(stored_record)
        sources = stored_record.get("sources") or stored_record.get("provenance")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    digest = _hash_source(source)
                    self._source_index.setdefault(digest, []).append(stored_record)

        text_blob = _extract_text(stored_record)
        for token in set(_tokenise(text_blob)):
            self._token_index.setdefault(token, []).append(record_index)


__all__ = [
    "MemoryStore",
    "_hash_source",
]
