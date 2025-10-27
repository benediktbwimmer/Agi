from agi.src.agi.memory.working import WorkingMemory


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def advance(self, delta: float) -> None:
        self._now += delta

    def __call__(self) -> float:
        return self._now


def test_capacity_eviction_uses_priority_and_calls_handler():
    clock = FakeClock()
    events: list[tuple[list[str], str]] = []

    def handler(records: list[dict[str, object]], reason: str) -> None:
        events.append(([record["key"] for record in records], reason))

    memory = WorkingMemory(capacity=2, overflow_handler=handler, time_provider=clock)

    memory.set("high", "keep", priority=0.9)
    memory.set("low", "evict", priority=0.1)
    memory.set("mid", "stay", priority=0.5)

    assert memory.get("high") == "keep"
    assert memory.get("mid") == "stay"
    assert memory.get("low") is None

    assert events[-1][1] == "capacity"
    assert events[-1][0] == ["low"]


def test_context_stack_and_queue_operations():
    memory = WorkingMemory(capacity=10)

    memory.set("global", "root")
    memory.push_context("task")
    memory.set("task_key", "task_value")

    assert memory.get("task_key") == "task_value"
    assert memory.get("global") == "root"

    memory.push_context("inner")
    memory.set("shadow", "inner")
    memory.pop_context()

    assert memory.get("shadow") is None

    queued = memory.enqueue_context("queued")
    memory.set("queued_key", "queued_value", context=queued.name)
    memory.dequeue_context()

    assert memory.get("queued_key") is None

    with memory.context("scoped") as frame:
        memory.set("scoped_key", frame.name, context=frame.name)
        assert memory.get("scoped_key") == frame.name

    assert memory.get("scoped_key") is None


def test_delete_without_context_removes_nearest_scope():
    memory = WorkingMemory(capacity=5)

    memory.set("shared", "root")
    task = memory.push_context("task")
    memory.set("shared", "task", context=task.name)

    assert memory.delete("shared") is True
    # The task-scoped entry is gone but the root value remains accessible.
    assert memory.get("shared") == "root"

    memory.pop_context()

    # Fall back to root once task context is gone.
    memory.set("shared", "root")
    memory.push_context("inner")
    assert memory.delete("shared") is True
    assert memory.get("shared") is None


def test_serialization_round_trip_preserves_state():
    clock = FakeClock(start=100.0)

    memory = WorkingMemory(capacity=4, default_ttl=5.0, time_provider=clock)
    memory.set("root_key", "root_value")

    episode_frame = memory.push_context("episode")
    memory.set("episode_key", "episode_value", context=episode_frame.name, ttl=2.0, priority=0.4)

    snapshot = memory.to_dict()

    restored = WorkingMemory.from_dict(snapshot, time_provider=clock)

    assert restored.capacity == 4
    assert restored.default_ttl == 5.0
    assert restored.get("root_key", context="root") == "root_value"
    assert restored.get("episode_key", context="episode") == "episode_value"

    clock.advance(10)
    assert restored.get("episode_key", context="episode") is None


def test_equal_priority_prefers_recently_accessed_records():
    clock = FakeClock()
    memory = WorkingMemory(capacity=2, time_provider=clock)

    memory.set("a", "A", priority=0.5)
    clock.advance(1)
    memory.set("b", "B", priority=0.5)

    # Refresh "a" so that "b" becomes the least recently touched candidate.
    assert memory.get("a") == "A"

    clock.advance(1)
    memory.set("c", "C", priority=0.5)

    assert memory.get("a") == "A"
    assert memory.get("b") is None
    assert memory.get("c") == "C"


def test_updates_refresh_ttl_and_priority():
    clock = FakeClock()
    events: list[tuple[str, list[str]]] = []

    def handler(records: list[dict[str, object]], reason: str) -> None:
        events.append((reason, [record["key"] for record in records]))

    memory = WorkingMemory(capacity=3, default_ttl=5.0, overflow_handler=handler, time_provider=clock)

    memory.set("item", "v1", priority=0.1)
    clock.advance(1)
    memory.set("item", "v2", ttl=2.0, priority=0.9)

    assert memory.get("item") == "v2"
    clock.advance(1.5)
    assert memory.get("item") == "v2"

    clock.advance(1.0)
    assert memory.get("item") is None
    assert events[-1][0] == "expired"
    assert events[-1][1] == ["item"]
