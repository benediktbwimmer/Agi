from __future__ import annotations

from agi.src.core.world_model import WorldModel


def test_world_model_updates_and_checks_units():
    wm = WorldModel()
    updates = wm.update(
        [
            {
                "claim_id": "c1",
                "passed": True,
                "expected_unit": "m/s",
                "observed_unit": "m/s",
                "provenance": [{"kind": "file", "ref": "artifact.txt"}],
            }
        ]
    )
    assert updates[0].credence > 0.5

    try:
        wm.update(
            [
                {
                    "claim_id": "c1",
                    "passed": False,
                    "expected_unit": "m/s",
                    "observed_unit": "kg",
                }
            ]
        )
    except ValueError:
        pass
    else:  # pragma: no cover - safety
        raise AssertionError("Unit mismatch should raise")
