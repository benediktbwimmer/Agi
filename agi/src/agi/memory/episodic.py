"""Episodic memory data models and storage backends."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
import uuid

from pydantic import BaseModel, Field, field_validator


StateSerializer = Callable[[Any], str]
StateDeserializer = Callable[[str], Any]


class EpisodeEvent(BaseModel):
    """A discrete event captured during an episode."""

    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    event_type: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """A collection of related events with accompanying metadata."""

    id: str
    events: List[EpisodeEvent]
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    state_snapshot: Optional[Any] = None

    @field_validator("events", mode="before")
    @classmethod
    def _ensure_events(cls, value: Optional[Iterable[EpisodeEvent]]) -> List[EpisodeEvent]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return list(value)


@dataclass
class EpisodeQuery:
    """Query parameters for retrieving episodes."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: Optional[int] = None
    ascending: bool = True


class EpisodicMemoryStore:
    """Abstract interface for episodic memory storage."""

    def initialize(self) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def append_episode(
        self,
        events: Sequence[EpisodeEvent],
        *,
        tags: Optional[Sequence[str]] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state_snapshot: Optional[Any] = None,
        episode_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Episode:  # pragma: no cover - interface definition
        raise NotImplementedError

    def fetch_episode(self, episode_id: str) -> Optional[Episode]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def fetch_episodes_by_time(self, query: EpisodeQuery) -> List[Episode]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def fetch_episodes_by_tags(
        self,
        tags: Sequence[str],
        *,
        match_all: bool = False,
        limit: Optional[int] = None,
    ) -> List[Episode]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def compress_episode(
        self,
        episode_id: str,
        *,
        summary: str,
        keep_events: int = 0,
    ) -> Optional[Episode]:  # pragma: no cover - interface definition
        raise NotImplementedError


class SQLiteEpisodicMemoryStore(EpisodicMemoryStore):
    """SQLite-backed episodic memory store."""

    def __init__(
        self,
        path: Path | str,
        *,
        timeout: float = 30.0,
        state_serializer: Optional[StateSerializer] = None,
        state_deserializer: Optional[StateDeserializer] = None,
    ) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._state_serializer = state_serializer or (lambda state: json.dumps(state) if state is not None else "null")
        self._state_deserializer = state_deserializer or (
            lambda payload: None if payload in (None, "null") else json.loads(payload)
        )
        self._init_lock = RLock()

    def initialize(self) -> None:
        with self._init_lock:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodes (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        summary TEXT,
                        metadata TEXT NOT NULL,
                        state_snapshot TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE
                    );
                    """
                )

    # -- CRUD operations -------------------------------------------------
    def append_episode(
        self,
        events: Sequence[EpisodeEvent],
        *,
        tags: Optional[Sequence[str]] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state_snapshot: Optional[Any] = None,
        episode_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Episode:
        if not events:
            raise ValueError("Episodes must contain at least one event")

        now = datetime.utcnow()
        created = created_at or now
        episode_uuid = episode_id or str(uuid.uuid4())
        tags = list(tags or [])
        metadata = metadata or {}

        serialized_state = self._serialize_state(state_snapshot)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (id, created_at, updated_at, tags, summary, metadata, state_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    episode_uuid,
                    created.isoformat(),
                    now.isoformat(),
                    json.dumps(tags),
                    summary,
                    json.dumps(metadata),
                    serialized_state,
                ),
            )
            event_rows = [
                (
                    episode_uuid,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.content,
                    json.dumps(event.metadata),
                )
                for event in events
            ]
            conn.executemany(
                """
                INSERT INTO events (episode_id, timestamp, event_type, content, metadata)
                VALUES (?, ?, ?, ?, ?);
                """,
                event_rows,
            )

        return Episode(
            id=episode_uuid,
            events=list(events),
            tags=tags,
            summary=summary,
            created_at=created,
            updated_at=now,
            metadata=metadata,
            state_snapshot=state_snapshot,
        )

    def fetch_episode(self, episode_id: str) -> Optional[Episode]:
        with self._connect() as conn:
            episode_row = conn.execute(
                "SELECT id, created_at, updated_at, tags, summary, metadata, state_snapshot FROM episodes WHERE id = ?;",
                (episode_id,),
            ).fetchone()
            if episode_row is None:
                return None

            events_rows = conn.execute(
                "SELECT timestamp, event_type, content, metadata FROM events WHERE episode_id = ? ORDER BY timestamp ASC;",
                (episode_id,),
            ).fetchall()

        return self._rows_to_episode(episode_row, events_rows)

    def fetch_episodes_by_time(self, query: EpisodeQuery) -> List[Episode]:
        conditions = []
        params: List[Any] = []
        if query.start is not None:
            conditions.append("created_at >= ?")
            params.append(query.start.isoformat())
        if query.end is not None:
            conditions.append("created_at <= ?")
            params.append(query.end.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order = "ASC" if query.ascending else "DESC"
        limit_clause = " LIMIT ?" if query.limit is not None else ""
        if query.limit is not None:
            params.append(query.limit)

        sql = (
            "SELECT id, created_at, updated_at, tags, summary, metadata, state_snapshot FROM episodes "
            f"{where_clause} ORDER BY created_at {order}{limit_clause};"
        )

        with self._connect() as conn:
            episode_rows = conn.execute(sql, params).fetchall()
            episodes = [
                self._rows_to_episode(
                    row,
                    conn.execute(
                        "SELECT timestamp, event_type, content, metadata FROM events WHERE episode_id = ? ORDER BY timestamp ASC;",
                        (row[0],),
                    ).fetchall(),
                )
                for row in episode_rows
            ]

        return episodes

    def fetch_episodes_by_tags(
        self,
        tags: Sequence[str],
        *,
        match_all: bool = False,
        limit: Optional[int] = None,
    ) -> List[Episode]:
        if not tags:
            return []

        with self._connect() as conn:
            episode_rows = conn.execute(
                "SELECT id, created_at, updated_at, tags, summary, metadata, state_snapshot FROM episodes ORDER BY created_at DESC;"
            ).fetchall()

        tag_set = set(tags)
        matched_rows = []
        for row in episode_rows:
            stored_tags = set(json.loads(row[3]))
            if match_all and not tag_set.issubset(stored_tags):
                continue
            if not match_all and tag_set.isdisjoint(stored_tags):
                continue
            matched_rows.append(row)
            if limit is not None and len(matched_rows) >= limit:
                break

        episodes = []
        with self._connect() as conn:
            for row in matched_rows:
                events_rows = conn.execute(
                    "SELECT timestamp, event_type, content, metadata FROM events WHERE episode_id = ? ORDER BY timestamp ASC;",
                    (row[0],),
                ).fetchall()
                episodes.append(self._rows_to_episode(row, events_rows))
        return episodes

    def compress_episode(
        self,
        episode_id: str,
        *,
        summary: str,
        keep_events: int = 0,
    ) -> Optional[Episode]:
        if keep_events < 0:
            raise ValueError("keep_events must be non-negative")

        with self._connect() as conn:
            episode_row = conn.execute(
                "SELECT id, created_at, updated_at, tags, summary, metadata, state_snapshot FROM episodes WHERE id = ?;",
                (episode_id,),
            ).fetchone()
            if episode_row is None:
                return None

            new_updated_at = datetime.utcnow()
            conn.execute(
                "UPDATE episodes SET summary = ?, updated_at = ? WHERE id = ?;",
                (summary, new_updated_at.isoformat(), episode_id),
            )

            if keep_events == 0:
                conn.execute("DELETE FROM events WHERE episode_id = ?;", (episode_id,))
            else:
                event_ids = [row[0] for row in conn.execute(
                    "SELECT id FROM events WHERE episode_id = ? ORDER BY timestamp DESC;", (episode_id,)
                ).fetchall()]
                to_remove = event_ids[keep_events:]
                if to_remove:
                    conn.executemany("DELETE FROM events WHERE id = ?;", [(eid,) for eid in to_remove])

            events_rows = conn.execute(
                "SELECT timestamp, event_type, content, metadata FROM events WHERE episode_id = ? ORDER BY timestamp ASC;",
                (episode_id,),
            ).fetchall()

            episode_row = conn.execute(
                "SELECT id, created_at, updated_at, tags, summary, metadata, state_snapshot FROM episodes WHERE id = ?;",
                (episode_id,),
            ).fetchone()

        return self._rows_to_episode(episode_row, events_rows, summary_override=summary)

    # -- Internal helpers -----------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.timeout, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _rows_to_episode(
        self,
        episode_row: sqlite3.Row | Sequence[Any],
        events_rows: Sequence[sqlite3.Row | Sequence[Any]],
        *,
        summary_override: Optional[str] = None,
    ) -> Episode:
        created_at = datetime.fromisoformat(episode_row[1])
        updated_at = datetime.fromisoformat(episode_row[2])
        tags = json.loads(episode_row[3])
        metadata = json.loads(episode_row[5])
        state_snapshot = self._deserialize_state(episode_row[6])
        events = [
            EpisodeEvent(
                timestamp=datetime.fromisoformat(row[0]),
                event_type=row[1],
                content=row[2],
                metadata=json.loads(row[3]),
            )
            for row in events_rows
        ]
        return Episode(
            id=episode_row[0],
            created_at=created_at,
            updated_at=updated_at,
            tags=tags,
            summary=summary_override if summary_override is not None else episode_row[4],
            metadata=metadata,
            state_snapshot=state_snapshot,
            events=events,
        )

    def _serialize_state(self, state: Optional[Any]) -> Optional[str]:
        if state is None:
            return None
        return self._state_serializer(state)

    def _deserialize_state(self, payload: Optional[str]) -> Optional[Any]:
        if payload in (None, ""):
            return None
        return self._state_deserializer(payload)


__all__ = [
    "Episode",
    "EpisodeEvent",
    "EpisodeQuery",
    "EpisodicMemoryStore",
    "SQLiteEpisodicMemoryStore",
]
