from __future__ import annotations

import json
import math
from pathlib import Path

from agi.src.core.world_model import WorldModel


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_world_model_persists_and_recovers_state(tmp_path: Path) -> None:
    storage = tmp_path / "state" / "beliefs.json"
    model = WorldModel(storage_path=storage)

    first_updates = model.update(
        [
            {
                "claim_id": "alpha",
                "passed": True,
                "provenance": [{"kind": "tool", "ref": "artifact.txt"}],
            }
        ]
    )
    assert first_updates
    assert storage.exists()

    data = _read_json(storage)
    assert data["revision"] == 1
    assert data["beliefs"][0]["claim_id"] == "alpha"
    evidence_entry = data["beliefs"][0]["evidence"][0]
    assert evidence_entry["source"]["ref"] == "artifact.txt"
    assert evidence_entry["outcome"] == "support"
    assert math.isclose(data["beliefs"][0]["support"], 1.5)
    assert math.isclose(data["beliefs"][0]["conflict"], 0.0)
    assert "variance" in data["beliefs"][0]

    history_path = storage.with_suffix(storage.suffix + ".history.jsonl")
    assert history_path.exists()
    with history_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) == 1
    assert lines[0]["revision"] == 1

    # Perform another update to test revision increments and append-only history
    second_updates = model.update([
        {"claim_id": "alpha", "passed": False}
    ])
    assert second_updates

    data = _read_json(storage)
    assert data["revision"] == 2
    assert data["beliefs"][0]["claim_id"] == "alpha"
    assert math.isclose(data["beliefs"][0]["support"], 1.5)
    assert math.isclose(data["beliefs"][0]["conflict"], 1.5)

    with history_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) == 2
    assert lines[-1]["revision"] == 2

    recovered = WorldModel(storage_path=storage)
    beliefs = recovered.beliefs
    assert "alpha" in beliefs
    assert beliefs["alpha"].credence == second_updates[0].credence
    # Evidence collected across runs should persist
    evidence = beliefs["alpha"].evidence[0]
    assert evidence.source.ref == "artifact.txt"
    assert math.isclose(beliefs["alpha"].support, 1.5)
    assert math.isclose(beliefs["alpha"].conflict, 1.5)
    assert beliefs["alpha"].variance <= 1.0
