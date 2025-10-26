"""Working memory implementation with context frames and eviction policies."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence


OverflowHandler = Callable[[List[Dict[str, Any]], str], None]


@dataclass
class MemoryRecord:
    """Represents a single entry in working memory."""

    key: str
    value: Any
    priority: float
    created_at: float
    expires_at: Optional[float]
    last_accessed: float
    context_id: int

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def touch(self, now: float) -> None:
        self.last_accessed = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_accessed": self.last_accessed,
            "context_id": self.context_id,
        }


@dataclass
class Frame:
    """A frame is an isolated scope of working memory."""

    id: int
    name: str
    items: Dict[str, MemoryRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "items": [record.to_dict() for record in self.items.values()],
        }


class WorkingMemory:
    """Working memory with scoped storage and configurable eviction policies."""

    def __init__(
        self,
        *,
        capacity: int = 128,
        default_ttl: Optional[float] = None,
        overflow_handler: Optional[OverflowHandler] = None,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._overflow_handler = overflow_handler
        self._time_provider = time_provider

        self._context_counter = 0
        self._frames: List[Frame] = []
        self._queue_boundary = 1
        # Root frame exists by default and cannot be removed.
        self._frames.append(self._create_frame(name="root"))

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def push_context(self, name: Optional[str] = None) -> Frame:
        """Push a new context on the stack (LIFO)."""

        frame = self._create_frame(name)
        self._frames.append(frame)
        return frame

    def pop_context(self) -> Frame:
        """Pop the most recent context frame."""

        if len(self._frames) == 1:
            raise IndexError("cannot pop the root context")

        frame = self._remove_frame(len(self._frames) - 1, reason="context_removed")
        return frame

    def enqueue_context(self, name: Optional[str] = None) -> Frame:
        """Add a context that participates in FIFO ordering."""

        frame = self._create_frame(name)
        self._frames.insert(self._queue_boundary, frame)
        self._queue_boundary += 1
        return frame

    def dequeue_context(self, name: Optional[str] = None) -> Frame:
        """Remove the oldest (non-root) context frame."""

        if len(self._frames) == 1:
            raise IndexError("no contexts to dequeue")

        if name is None:
            if self._queue_boundary <= 1:
                raise IndexError("no queued contexts to dequeue")
            idx = 1
        else:
            idx = self._find_frame_index(name=name, first=True)
            if idx == 0:
                raise ValueError("cannot dequeue the root context")

        frame = self._remove_frame(idx, reason="context_removed")
        return frame

    @contextmanager
    def context(self, name: Optional[str] = None, mode: str = "stack") -> Iterator[Frame]:
        """Context manager for temporary frames."""

        if mode not in {"stack", "queue"}:
            raise ValueError("mode must be 'stack' or 'queue'")

        if mode == "stack":
            frame = self.push_context(name)
            try:
                yield frame
            finally:
                self.pop_context()
        else:
            frame = self.enqueue_context(name)
            try:
                yield frame
            finally:
                self._remove_frame_by_id(frame.id, reason="context_removed")

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def set(
        self,
        key: str,
        value: Any,
        *,
        context: Optional[str] = None,
        ttl: Optional[float] = None,
        priority: float = 1.0,
    ) -> None:
        """Store a value in working memory within the specified context."""

        frame = self._resolve_frame(context)
        now = self._now()
        ttl = self.default_ttl if ttl is None else ttl
        expires_at = None if ttl is None else now + ttl

        record = MemoryRecord(
            key=key,
            value=value,
            priority=priority,
            created_at=now,
            expires_at=expires_at,
            last_accessed=now,
            context_id=frame.id,
        )

        frame.items[key] = record
        self._prune_expired(now)
        evicted = self._evict_if_needed()
        if evicted:
            self._handle_overflow(evicted, reason="capacity")

    def get(
        self,
        key: str,
        *,
        context: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """Retrieve a value from working memory respecting context scoping."""

        now = self._now()
        self._prune_expired(now)

        if context is not None:
            frame = self._resolve_frame(context)
            record = frame.items.get(key)
            if record is None:
                return default
            if record.is_expired(now):
                del frame.items[key]
                self._handle_overflow([record], reason="expired")
                return default
            record.touch(now)
            return record.value

        # Search from top of stack (most specific) to root (least specific).
        for frame in reversed(self._frames):
            record = frame.items.get(key)
            if record is not None:
                if record.is_expired(now):
                    del frame.items[key]
                    self._handle_overflow([record], reason="expired")
                    continue
                record.touch(now)
                return record.value

        return default

    def delete(self, key: str, *, context: Optional[str] = None) -> bool:
        """Delete a value from the specified context."""

        frame = self._resolve_frame(context)
        record = frame.items.pop(key, None)
        if record is None:
            return False
        self._handle_overflow([record], reason="deleted")
        return True

    def clear(self, *, context: Optional[str] = None) -> None:
        """Clear either the full working memory or a specific context."""

        if context is None:
            for idx in range(len(self._frames) - 1, 0, -1):
                self._remove_frame(idx, reason="context_removed")
            root_items = list(self._frames[0].items.values())
            self._frames[0].items.clear()
            self._handle_overflow(root_items, reason="context_removed")
            self._queue_boundary = 1
        else:
            frame = self._resolve_frame(context)
            removed = list(frame.items.values())
            frame.items.clear()
            self._handle_overflow(removed, reason="cleared")

    # ------------------------------------------------------------------
    # Inspection and serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the working memory to a dictionary."""

        self._prune_expired(self._now())
        return {
            "capacity": self.capacity,
            "default_ttl": self.default_ttl,
            "queue_boundary": self._queue_boundary,
            "contexts": [frame.to_dict() for frame in self._frames],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        overflow_handler: Optional[OverflowHandler] = None,
        time_provider: Callable[[], float] = time.time,
    ) -> "WorkingMemory":
        """Restore a :class:`WorkingMemory` instance from serialized data."""

        memory = cls(
            capacity=payload["capacity"],
            default_ttl=payload.get("default_ttl"),
            overflow_handler=overflow_handler,
            time_provider=time_provider,
        )

        memory._frames.clear()
        memory._context_counter = 0
        memory._queue_boundary = payload.get("queue_boundary", 1)

        max_id = -1
        for frame_payload in payload.get("contexts", []):
            frame = memory._create_frame(name=frame_payload.get("name"))
            frame.id = frame_payload.get("id", frame.id)
            max_id = max(max_id, frame.id)
            memory._frames.append(frame)
            for item in frame_payload.get("items", []):
                record = MemoryRecord(
                    key=item["key"],
                    value=item["value"],
                    priority=item.get("priority", 1.0),
                    created_at=item.get("created_at", memory._now()),
                    expires_at=item.get("expires_at"),
                    last_accessed=item.get("last_accessed", item.get("created_at", memory._now())),
                    context_id=frame.id,
                )
                frame.items[record.key] = record

        if not memory._frames:
            memory._frames.append(memory._create_frame(name="root"))

        memory._queue_boundary = min(max(len(memory._frames), 1), max(memory._queue_boundary, 1))
        memory._context_counter = max(memory._context_counter, max_id + 1)

        return memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return float(self._time_provider())

    def _resolve_frame(self, context: Optional[str]) -> Frame:
        if context is None:
            return self._frames[-1]

        if isinstance(context, str):
            idx = self._find_frame_index(name=context, first=False)
            return self._frames[idx]

        raise TypeError("context must be a frame name or None")

    def _find_frame_index(self, name: str, *, first: bool) -> int:
        matches = [idx for idx, frame in enumerate(self._frames) if frame.name == name]
        if not matches:
            raise KeyError(f"context '{name}' not found")
        return matches[0] if first else matches[-1]

    def _create_frame(self, name: Optional[str]) -> Frame:
        frame_id = self._context_counter
        self._context_counter += 1
        frame_name = name or f"context-{frame_id}"
        frame = Frame(id=frame_id, name=frame_name)
        return frame

    def _remove_frame(self, idx: int, *, reason: str) -> Frame:
        if idx <= 0:
            raise ValueError("cannot remove the root context")
        frame = self._frames.pop(idx)
        if idx < self._queue_boundary:
            self._queue_boundary -= 1
        self._handle_overflow(list(frame.items.values()), reason=reason)
        return frame

    def _remove_frame_by_id(self, frame_id: int, *, reason: str) -> Frame:
        for idx, frame in enumerate(self._frames):
            if frame.id == frame_id:
                if idx == 0:
                    raise ValueError("cannot remove the root context")
                return self._remove_frame(idx, reason=reason)
        raise KeyError(f"context with id {frame_id} not found")

    def _prune_expired(self, now: Optional[float] = None) -> None:
        now = self._now() if now is None else now
        expired: List[MemoryRecord] = []
        for frame in self._frames:
            keys_to_delete = [key for key, record in frame.items.items() if record.is_expired(now)]
            for key in keys_to_delete:
                expired.append(frame.items.pop(key))
        if expired:
            self._handle_overflow(expired, reason="expired")

    def _evict_if_needed(self) -> List[MemoryRecord]:
        evicted: List[MemoryRecord] = []
        while self._total_items() > self.capacity:
            victim = self._select_victim()
            if victim is None:
                break
            frame = self._frame_by_id(victim.context_id)
            if frame is None:
                break
            removed = frame.items.pop(victim.key, None)
            if removed is not None:
                evicted.append(removed)
        return evicted

    def _select_victim(self) -> Optional[MemoryRecord]:
        candidates: List[MemoryRecord] = []
        for frame in self._frames:
            candidates.extend(frame.items.values())
        if not candidates:
            return None
        candidates.sort(key=lambda record: (record.priority, record.created_at))
        return candidates[0]

    def _frame_by_id(self, frame_id: int) -> Optional[Frame]:
        for frame in self._frames:
            if frame.id == frame_id:
                return frame
        return None

    def _total_items(self) -> int:
        return sum(len(frame.items) for frame in self._frames)

    def _handle_overflow(self, records: List[MemoryRecord], *, reason: str) -> None:
        if not records:
            return
        if self._overflow_handler is not None:
            summaries = self._summarize(records)
            self._overflow_handler(summaries, reason)

    def _summarize(self, records: Sequence[MemoryRecord]) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        now = self._now()
        for record in records:
            payload = record.to_dict()
            frame = self._frame_by_id(record.context_id)
            payload["context"] = frame.name if frame is not None else None
            if record.expires_at is not None:
                payload["ttl_remaining"] = max(record.expires_at - now, 0.0)
            else:
                payload["ttl_remaining"] = None
            summaries.append(payload)
        return summaries


__all__ = ["WorkingMemory", "MemoryRecord"]
