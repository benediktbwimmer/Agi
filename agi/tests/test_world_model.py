from __future__ import annotations

import math

import pytest

from agi.src.core.world_model import WorldModel, _logistic_update


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
    assert math.isclose(updates[0].support, 1.5)
    assert math.isclose(updates[0].conflict, 0.0)
    assert 0.0 < updates[0].uncertainty < 1.0

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


def test_world_model_weight_adjusts_credence_delta():
    wm = WorldModel()
    result = wm.update([
        {"claim_id": "c-weight", "passed": True, "weight": 0.5}
    ])[0]
    expected = _logistic_update(0.5, True, weight=0.5)
    assert math.isclose(result.credence, expected)


def test_world_model_confidence_scales_weight():
    wm = WorldModel()
    result = wm.update([
        {"claim_id": "c-confidence", "passed": True, "weight": 1.5, "confidence": 0.25}
    ])[0]
    expected = _logistic_update(0.5, True, weight=1.5 * 0.25)
    assert math.isclose(result.credence, expected)


def test_world_model_rejects_negative_weight():
    wm = WorldModel()
    with pytest.raises(ValueError):
        wm.update([
            {"claim_id": "bad", "passed": True, "weight": -0.1}
        ])


def test_world_model_accepts_external_timestamp():
    wm = WorldModel()
    ts = "2024-01-02T03:04:05Z"
    result = wm.update([
        {"claim_id": "time", "passed": True, "timestamp": ts}
    ])[0]
    assert result.last_updated == "2024-01-02T03:04:05+00:00"


def test_world_model_tracks_evidence_strength():
    wm = WorldModel()
    first = wm.update([
        {"claim_id": "strength", "passed": True, "weight": 1.0}
    ])[0]
    assert math.isclose(first.support, 1.0)
    assert math.isclose(first.conflict, 0.0)
    assert first.uncertainty < 1.0

    second = wm.update([
        {"claim_id": "strength", "passed": False, "weight": 0.5}
    ])[0]
    assert math.isclose(second.support, 1.0)
    assert math.isclose(second.conflict, 0.5)
    assert second.uncertainty < first.uncertainty
