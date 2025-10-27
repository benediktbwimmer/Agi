from __future__ import annotations

import json
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import copy


def _normalise_time(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _hash_source(source: Dict[str, Any]) -> str:
    payload = json.dumps(source, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class MemoryStore:
    path: Path
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

    def recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return [
            json.loads(json.dumps(record))
            for record in self._time_records[-limit:]
        ]

    def all(self) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(record)) for record in self._time_records]

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


@dataclass
class WorkingMemory:
    """A lightweight, per-run cache of recent episodes."""

    capacity_per_tool: int = 5
    capacity_global: int = 20
    _episodes_by_tool: Dict[Optional[str], List[Dict[str, Any]]] = field(
        default_factory=dict, init=False
    )
    _seen_call_ids: set[str] = field(default_factory=set, init=False)

    def reset(self) -> None:
        self._episodes_by_tool.clear()
        self._seen_call_ids.clear()

    def hydrate(self, episodes: Iterable[Dict[str, Any]]) -> None:
        for episode in episodes:
            self.add_episode(episode)

    def add_episode(self, episode: Dict[str, Any]) -> None:
        tool = episode.get("tool")
        call_id = episode.get("call_id")
        if call_id and call_id in self._seen_call_ids:
            return
        if call_id:
            self._seen_call_ids.add(call_id)

        bucket = self._episodes_by_tool.setdefault(tool, [])
        bucket.append(copy.deepcopy(episode))
        if len(bucket) > self.capacity_per_tool:
            del bucket[0 : len(bucket) - self.capacity_per_tool]

        global_bucket = self._episodes_by_tool.setdefault(None, [])
        global_bucket.append(copy.deepcopy(episode))
        if len(global_bucket) > self.capacity_global:
            del global_bucket[0 : len(global_bucket) - self.capacity_global]

    def recall(self, tool: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        bucket = self._episodes_by_tool.get(tool)
        if not bucket:
            bucket = self._episodes_by_tool.get(None, [])
        if limit is not None and limit >= 0:
            bucket = bucket[-limit:]
        return [copy.deepcopy(ep) for ep in bucket]

    def snapshot(self) -> Dict[Optional[str], List[Dict[str, Any]]]:
        return {key: [copy.deepcopy(ep) for ep in episodes] for key, episodes in self._episodes_by_tool.items()}
