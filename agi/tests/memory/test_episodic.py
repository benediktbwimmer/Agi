from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agi.memory.episodic import (
    EpisodeEvent,
    EpisodeQuery,
    SQLiteEpisodicMemoryStore,
)


@pytest.fixture()
def temp_store(tmp_path: Path) -> SQLiteEpisodicMemoryStore:
    db_path = tmp_path / "episodes.db"
    store = SQLiteEpisodicMemoryStore(db_path)
    store.initialize()
    return store


def test_initialize_idempotent(temp_store: SQLiteEpisodicMemoryStore) -> None:
    temp_store.initialize()  # should not raise


def test_append_and_fetch_episode(temp_store: SQLiteEpisodicMemoryStore) -> None:
    event = EpisodeEvent(event_type="observation", content="Saw a tree")
    episode = temp_store.append_episode([event], tags=["outdoors"], metadata={"weather": "sunny"})

    loaded = temp_store.fetch_episode(episode.id)
    assert loaded is not None
    assert loaded.id == episode.id
    assert loaded.events[0].content == "Saw a tree"
    assert loaded.tags == ["outdoors"]
    assert loaded.metadata["weather"] == "sunny"


def test_fetch_by_time_and_tags(temp_store: SQLiteEpisodicMemoryStore) -> None:
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    first = temp_store.append_episode(
        [EpisodeEvent(event_type="thought", content="First", timestamp=base)],
        tags=["alpha", "beta"],
        created_at=base,
    )
    second_time = base + timedelta(minutes=30)
    temp_store.append_episode(
        [EpisodeEvent(event_type="action", content="Second", timestamp=second_time)],
        tags=["beta"],
        created_at=second_time,
    )

    recent_query = EpisodeQuery(start=base + timedelta(minutes=10))
    recent = temp_store.fetch_episodes_by_time(recent_query)
    assert len(recent) == 1
    assert recent[0].events[0].content == "Second"

    beta_any = temp_store.fetch_episodes_by_tags(["beta"], match_all=False)
    assert {episode.id for episode in beta_any} == {first.id, recent[0].id}

    alpha_only = temp_store.fetch_episodes_by_tags(["alpha"], match_all=True)
    assert len(alpha_only) == 1
    assert alpha_only[0].id == first.id


def test_state_serialization_hooks(tmp_path: Path) -> None:
    db_path = tmp_path / "stateful.db"

    def serializer(state: dict[str, str]) -> str:
        return json.dumps({"payload": state})

    def deserializer(payload: str) -> dict[str, str]:
        return json.loads(payload)["payload"]

    store = SQLiteEpisodicMemoryStore(db_path, state_serializer=serializer, state_deserializer=deserializer)
    store.initialize()

    episode = store.append_episode(
        [EpisodeEvent(event_type="state", content="stored")],
        state_snapshot={"value": "retained"},
    )

    loaded = store.fetch_episode(episode.id)
    assert loaded is not None
    assert loaded.state_snapshot == {"value": "retained"}


def test_compress_episode_removes_events(temp_store: SQLiteEpisodicMemoryStore) -> None:
    events = [
        EpisodeEvent(event_type="log", content=f"event {idx}")
        for idx in range(4)
    ]
    episode = temp_store.append_episode(events)

    compressed = temp_store.compress_episode(episode.id, summary="condensed", keep_events=2)
    assert compressed is not None
    assert compressed.summary == "condensed"
    assert len(compressed.events) == 2
    assert [event.content for event in compressed.events] == ["event 2", "event 3"]

    empty = temp_store.compress_episode(episode.id, summary="final", keep_events=0)
    assert empty is not None
    assert empty.events == []


def test_concurrent_appends(temp_store: SQLiteEpisodicMemoryStore) -> None:
    total_threads = 5
    episodes_per_thread = 4

    def worker(thread_id: int) -> None:
        for idx in range(episodes_per_thread):
            temp_store.append_episode(
                [EpisodeEvent(event_type="thread", content=f"t{thread_id}-{idx}")],
                tags=[f"t{thread_id}"],
            )
            # Ensure sqlite has a chance to schedule between writers
            time.sleep(0.01)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(total_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_episodes = temp_store.fetch_episodes_by_time(EpisodeQuery())
    assert len(all_episodes) == total_threads * episodes_per_thread


def test_durability_across_restarts(tmp_path: Path) -> None:
    db_path = tmp_path / "durable.db"
    store = SQLiteEpisodicMemoryStore(db_path)
    store.initialize()
    episode = store.append_episode([EpisodeEvent(event_type="durable", content="persist")])

    # Simulate restart by creating a new store instance
    store = SQLiteEpisodicMemoryStore(db_path)
    store.initialize()
    loaded = store.fetch_episode(episode.id)
    assert loaded is not None
    assert loaded.events[0].content == "persist"
