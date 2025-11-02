from __future__ import annotations

import json
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - numpy not installed
    np = None  # type: ignore[assignment]
try:  # pragma: no cover - optional dependency
    from .vector_index import MemoryVectorIndex
except ImportError:  # pragma: no cover - faiss or numpy not installed
    MemoryVectorIndex = None  # type: ignore[assignment]

from .telemetry import Telemetry


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _normalise_time(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _coerce_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        return _normalise_time(value)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _coerce_timedelta(value: timedelta | int | float | None) -> timedelta | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    raise TypeError(f"Unsupported timedelta value: {value!r}")


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


def _embedding_to_vector(embedding: Mapping[str, Any]) -> Optional[Any]:
    if np is None:
        return None
    try:
        dim = int(embedding.get("dim", 0))
    except (TypeError, ValueError):
        return None
    if dim <= 0 or dim > 4096:
        return None
    values = embedding.get("values")
    if not isinstance(values, Mapping):
        return None
    vector = np.zeros(dim, dtype=np.float32)
    nonzero = False
    for key, weight in values.items():
        try:
            index = int(key)
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if 0 <= index < dim and value != 0.0:
            vector[index] = value
            nonzero = True
    if not nonzero:
        return None
    return vector


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
    _keyword_index: Dict[str, List[int]] = field(default_factory=dict, init=False)
    _plan_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _goal_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    _vector_index: "MemoryVectorIndex | None" = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if MemoryVectorIndex is not None:
            try:
                self._vector_index = MemoryVectorIndex()
            except RuntimeError:  # pragma: no cover - faiss initialisation failure
                self._vector_index = None
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

    def query_reflection_insights(self, goal: str, *, limit: int = 3) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        key = goal.strip().lower()
        if not key:
            return []
        records = self._goal_index.get(key, [])
        if not records:
            return []
        selected = [record for record in records if record.get("type") == "reflection_insight"][-limit:]
        return [json.loads(json.dumps(record)) for record in selected]

    def iter_reflection_insights(
        self,
        *,
        goal: str | None = None,
        limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Return reflection insight records ordered by timestamp."""

        records: List[Dict[str, Any]] = []
        if goal is not None:
            key = goal.strip().lower()
            if key:
                bucket = self._goal_index.get(key, [])
                for record in bucket:
                    if record.get("type") == "reflection_insight":
                        records.append(record)
        else:
            seen: set[int] = set()
            for bucket in self._goal_index.values():
                for record in bucket:
                    if record.get("type") != "reflection_insight":
                        continue
                    record_id = id(record)
                    if record_id in seen:
                        continue
                    seen.add(record_id)
                    records.append(record)

        def _key(item: Mapping[str, Any]) -> str:
            return str(item.get("time") or "")

        records.sort(key=_key)
        if limit is not None and limit > 0:
            records = records[-limit:]
        return [json.loads(json.dumps(record)) for record in records]

    def temporal_window(
        self,
        *,
        anchor: datetime | str | None = None,
        before: timedelta | int | float | None = None,
        after: timedelta | int | float | None = None,
        limit: int | None = 50,
        types: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return records around ``anchor`` constrained by ``before``/``after`` windows.

        ``before`` and ``after`` accept :class:`datetime.timedelta` instances or seconds.
        When ``anchor`` is omitted the most recent timestamp is used as the reference
        point.  Results are returned in chronological order and can be filtered by
        ``types``.
        """

        if before is None and after is None:
            raise ValueError("temporal_window requires at least one of 'before' or 'after'")
        if not self._time_records:
            return []

        before_delta = _coerce_timedelta(before)
        after_delta = _coerce_timedelta(after)

        if anchor is None:
            anchor_dt = self._time_keys[-1]
        else:
            anchor_dt = _coerce_datetime(anchor)

        start_dt = anchor_dt - before_delta if before_delta is not None else anchor_dt
        end_dt = anchor_dt + after_delta if after_delta is not None else anchor_dt

        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        start_idx = bisect_left(self._time_keys, start_dt)
        end_idx = bisect_right(self._time_keys, end_dt)

        type_filter = {t for t in types} if types is not None else None
        results: List[Dict[str, Any]] = []

        for record in self._time_records[start_idx:end_idx]:
            if type_filter is not None and record.get("type") not in type_filter:
                continue
            results.append(json.loads(json.dumps(record)))
            if limit is not None and limit > 0 and len(results) >= limit:
                break

        return results

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
        type_filter = {t for t in types} if types is not None else None
        lexical_scores: Counter[int] = Counter()
        for token in tokens:
            for record_idx in self._token_index.get(token, []):
                if type_filter is not None:
                    record_type = self._records[record_idx].get("type")
                    if record_type not in type_filter:
                        continue
                lexical_scores[record_idx] += 1
                if record_idx in self._keyword_index.get(token, []):
                    lexical_scores[record_idx] += 2

        vector_scores: Dict[int, float] = {}
        vector_weight = 0.75
        if self._vector_index is not None:
            vector_weight = getattr(self._vector_index, "search_weight", vector_weight)
            try:
                vector_results = self._vector_index.search_text(query, limit=max(limit * 3, limit))
            except RuntimeError:  # pragma: no cover - faiss failure
                vector_results = []
                self._vector_index = None
            else:
                for record_idx, similarity in vector_results:
                    record = self._records[record_idx]
                    if type_filter is not None and record.get("type") not in type_filter:
                        continue
                    # retain the strongest similarity per record
                    if similarity > vector_scores.get(record_idx, 0.0):
                        vector_scores[record_idx] = similarity

        combined_scores: Dict[int, float] = {idx: float(score) for idx, score in lexical_scores.items()}
        for idx, similarity in vector_scores.items():
            combined_scores[idx] = combined_scores.get(idx, 0.0) + similarity * vector_weight

        if not combined_scores:
            return []

        def _sort_key(index: int) -> Tuple[float, float, int]:
            timestamp = self._record_times[index]
            ts_key = timestamp.timestamp() if timestamp is not None else float("-inf")
            return (-combined_scores[index], -ts_key, -index)

        ranked_indices = sorted(combined_scores, key=_sort_key)
        limited = ranked_indices[:limit]
        results: List[Dict[str, Any]] = []
        for idx in limited:
            record = json.loads(json.dumps(self._records[idx]))
            record["semantic_score"] = round(combined_scores[idx], 4)
            lexical_hits = lexical_scores.get(idx)
            if lexical_hits:
                record["lexical_hits"] = int(lexical_hits)
            vector_similarity = vector_scores.get(idx)
            if vector_similarity:
                record["vector_similarity"] = round(vector_similarity, 4)
            results.append(record)
        return results

    def search(
        self,
        *,
        query: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        types: Iterable[str] | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve records using lexical and temporal constraints.

        ``query`` performs lexical matching similar to :meth:`semantic_search` but
        allows additional ``start``/``end`` time filters and ``types`` filtering.
        When ``query`` is omitted, the method falls back to a purely temporal
        search returning records in chronological order.
        """

        if limit <= 0:
            return []

        type_filter = {t for t in types} if types is not None else None

        start_dt = _coerce_datetime(start) if start is not None else None
        end_dt = _coerce_datetime(end) if end is not None else None
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        def _time_filter(record_idx: int) -> bool:
            record_time = self._record_times[record_idx]
            if start_dt is not None and (
                record_time is None or record_time < start_dt
            ):
                return False
            if end_dt is not None and (
                record_time is None or record_time > end_dt
            ):
                return False
            return True

        def _type_filter(record: Mapping[str, Any]) -> bool:
            if type_filter is None:
                return True
            return record.get("type") in type_filter

        if query is None or not query.strip():
            if not self._time_records:
                return []

            if start_dt is None:
                start_idx = 0
            else:
                start_idx = bisect_left(self._time_keys, start_dt)
            if end_dt is None:
                end_idx = len(self._time_records)
            else:
                end_idx = bisect_right(self._time_keys, end_dt)

            results: List[Dict[str, Any]] = []
            for record in self._time_records[start_idx:end_idx]:
                if not _type_filter(record):
                    continue
                results.append(json.loads(json.dumps(record)))
                if len(results) >= limit:
                    break
            return results

        tokens = _tokenise(query)
        lexical_scores: Counter[int] = Counter()
        for token in tokens:
            for record_idx in self._token_index.get(token, []):
                record = self._records[record_idx]
                if not _type_filter(record):
                    continue
                if not _time_filter(record_idx):
                    continue
                lexical_scores[record_idx] += 1
                if record_idx in self._keyword_index.get(token, []):
                    lexical_scores[record_idx] += 2

        vector_scores: Dict[int, float] = {}
        vector_weight = 0.75
        if self._vector_index is not None:
            vector_weight = getattr(self._vector_index, "search_weight", vector_weight)
            try:
                vector_results = self._vector_index.search_text(query, limit=max(limit * 3, limit))
            except RuntimeError:  # pragma: no cover - faiss failure
                vector_results = []
                self._vector_index = None
            else:
                for record_idx, similarity in vector_results:
                    record = self._records[record_idx]
                    if not _type_filter(record):
                        continue
                    if not _time_filter(record_idx):
                        continue
                    if similarity > vector_scores.get(record_idx, 0.0):
                        vector_scores[record_idx] = similarity

        combined_scores: Dict[int, float] = {idx: float(score) for idx, score in lexical_scores.items()}
        for idx, similarity in vector_scores.items():
            combined_scores[idx] = combined_scores.get(idx, 0.0) + similarity * vector_weight

        if not combined_scores:
            return []

        def _sort_key(index: int) -> Tuple[float, float, int]:
            timestamp = self._record_times[index]
            ts_key = timestamp.timestamp() if timestamp is not None else float("-inf")
            return (-combined_scores[index], -ts_key, -index)

        ranked_indices = sorted(combined_scores, key=_sort_key)
        limited = ranked_indices[:limit]
        results: List[Dict[str, Any]] = []
        for idx in limited:
            record = json.loads(json.dumps(self._records[idx]))
            record["semantic_score"] = round(combined_scores[idx], 4)
            lexical_hits = lexical_scores.get(idx)
            if lexical_hits:
                record["lexical_hits"] = int(lexical_hits)
            vector_similarity = vector_scores.get(idx)
            if vector_similarity:
                record["vector_similarity"] = round(vector_similarity, 4)
            results.append(record)
        return results

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
        vector_added = False
        if self._vector_index is not None:
            embedding_payload = stored_record.get("embedding")
            if isinstance(embedding_payload, Mapping):
                vector = _embedding_to_vector(embedding_payload)
                if vector is not None:
                    try:
                        self._vector_index.add_vector(vector, record_index)
                    except RuntimeError:  # pragma: no cover - faiss operational failure
                        self._vector_index = None
                    else:
                        vector_added = True
        if self._vector_index is not None and text_blob and not vector_added:
            try:
                self._vector_index.add_text(text_blob, record_index)
            except RuntimeError:  # pragma: no cover - faiss operational failure
                self._vector_index = None
        for token in set(_tokenise(text_blob)):
            self._token_index.setdefault(token, []).append(record_index)
        keywords = stored_record.get("keywords")
        if isinstance(keywords, list):
            for keyword in keywords:
                if not isinstance(keyword, str):
                    continue
                token = keyword.strip().lower()
                if token:
                    self._keyword_index.setdefault(token, []).append(record_index)

        goal = stored_record.get("goal")
        if isinstance(goal, str) and goal.strip():
            key = goal.strip().lower()
            bucket = self._goal_index.setdefault(key, [])
            bucket.append(stored_record)


__all__ = [
    "MemoryStore",
    "_hash_source",
]
