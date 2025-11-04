"""Simple structured telemetry utilities used across the AGI stack."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, MutableMapping, Protocol

try:  # pragma: no cover - optional oversight integration
    from ..oversight.store import OversightStore
except ImportError:  # pragma: no cover - avoid hard dependency at import time
    OversightStore = None  # type: ignore[assignment]


class TelemetrySink(Protocol):
    """A destination for telemetry events."""

    def write(self, event: Dict[str, Any]) -> None:
        """Persist or forward a telemetry event."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Telemetry:
    """Dispatcher that fan-outs events to the configured sinks."""

    sinks: Iterable[TelemetrySink] = field(default_factory=tuple)
    context: MutableMapping[str, Any] = field(default_factory=dict)

    def emit(self, event: str, **payload: Any) -> None:
        if not self.sinks:
            return
        base: Dict[str, Any] = {"event": event, "time": _now_iso()}
        if self.context:
            base.update(self.context)
        base.update(payload)
        for sink in self.sinks:
            try:
                sink.write(dict(base))
            except Exception:  # pragma: no cover - telemetry failures must not break runs
                continue


@dataclass
class InMemorySink:
    """Sink that keeps telemetry in-memory for inspection in tests."""

    events: List[Dict[str, Any]] = field(default_factory=list)

    def write(self, event: Dict[str, Any]) -> None:
        self.events.append(dict(event))


@dataclass
class JsonLinesSink:
    """Append-only JSONL sink for telemetry events."""

    path: Path
    _lock: Lock = field(default_factory=Lock, init=False)

    def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


@dataclass
class OversightSink:
    """Telemetry sink that mirrors events into an :class:`OversightStore`."""

    store: "OversightStore"

    def __post_init__(self) -> None:  # pragma: no cover - defensive input validation
        if OversightStore is None:
            raise RuntimeError("OversightStore is unavailable; install oversight extras")
        if not isinstance(self.store, OversightStore):
            raise TypeError("store must be an OversightStore instance")

    def write(self, event: Dict[str, Any]) -> None:
        self.store.record_telemetry(event)
