from __future__ import annotations

import json
from pathlib import Path

from agi.src.core.memory import MemoryStore
from agi.src.core.retrieval import MemoryRetriever


def test_memory_retriever_blends_claim_and_semantic(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    store.append(
        {
            "type": "episode",
            "tool": "sensor_logger",
            "call_id": "call-1",
            "ok": True,
            "stdout": "Energy audit baseline established",
            "time": "2024-01-01T00:00:00+00:00",
            "claim": {"id": "claim-1", "text": "Baseline energy usage"},
        }
    )
    store.append(
        {
            "type": "reflection",
            "tool": "executive_reflection",
            "summary": "Energy audit highlights ventilation losses",
            "time": "2024-01-02T00:00:00+00:00",
            "key_findings": ["audit report"],
        }
    )

    retriever = MemoryRetriever(store, per_claim_limit=2, semantic_limit=3)

    snippets = retriever.retrieve(
        goal="Energy audit",
        context={"focus": "baseline"},
        claim_ids=["claim-1"],
    )

    assert any(snippet["source"] == "claim" for snippet in snippets)
    assert any(snippet["source"] == "semantic" for snippet in snippets)

    fingerprints = {
        json.dumps({k: v for k, v in snippet.items() if k != "source"}, sort_keys=True)
        for snippet in snippets
    }
    assert len(fingerprints) == len(snippets)

    limited = retriever.retrieve(goal="Energy audit", claim_ids=["claim-1"], limit=1)
    assert len(limited) == 1
