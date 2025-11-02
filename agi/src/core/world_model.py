from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .types import Belief, Source


@dataclass
class WorldModel:
    """Belief tracker with optional durable storage."""

    storage_path: Path | None = None
    history_path: Path | None = None
    _beliefs: Dict[str, Belief] = field(default_factory=dict, init=False)
    _revision: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.storage_path is not None:
            self._load_state()
            if self.history_path is None:
                self.history_path = self.storage_path.with_suffix(
                    self.storage_path.suffix + ".history.jsonl"
                )

    def update(self, results: Iterable[Dict[str, Any]]) -> List[Belief]:
        updated: List[Belief] = []
        update_time = datetime.now(timezone.utc)
        update_time_iso = update_time.isoformat()
        for result in results:
            claim_id = result["claim_id"]
            if result.get("observed_unit") and result.get("expected_unit"):
                if result["observed_unit"] != result["expected_unit"]:
                    raise ValueError("Unit mismatch for claim %s" % claim_id)
            passed = _coerce_bool(result.get("passed"))
            prior = self._beliefs.get(
                claim_id,
                Belief(
                    claim_id=claim_id,
                    credence=0.5,
                    evidence=[],
                    last_updated=update_time_iso,
                    support=0.0,
                    conflict=0.0,
                ),
            )
            weight = _resolve_weight(
                result.get("weight"),
                confidence=result.get("confidence"),
            )
            posterior = _logistic_update(prior.credence, passed, weight=weight)
            evidence = list(prior.evidence)
            provenance = result.get("provenance") or []
            for source in provenance:
                if isinstance(source, Source):
                    evidence.append(source)
                elif isinstance(source, dict):
                    evidence.append(Source(**source))
            timestamp = _resolve_update_timestamp(
                result.get("timestamp"), default=update_time_iso
            )
            support = prior.support + (weight if passed else 0.0)
            conflict = prior.conflict + (weight if not passed else 0.0)
            belief = Belief(
                claim_id=claim_id,
                credence=posterior,
                evidence=evidence,
                last_updated=timestamp,
                support=support,
                conflict=conflict,
            )
            self._beliefs[claim_id] = belief
            updated.append(belief)

        if updated:
            self._revision += 1
            self._persist_state(update_time_iso)
            self._append_history(updated, update_time_iso)

        return updated

    @property
    def beliefs(self) -> Dict[str, Belief]:
        return dict(self._beliefs)

    def _load_state(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        with self.storage_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        beliefs = {}
        for item in data.get("beliefs", []):
            evidence = [
                src if isinstance(src, Source) else Source(**src)
                for src in item.get("evidence", [])
            ]
            belief = Belief(
                claim_id=item["claim_id"],
                credence=float(item.get("credence", 0.5)),
                evidence=evidence,
                last_updated=item.get("last_updated", datetime.now(timezone.utc).isoformat()),
                support=float(item.get("support", 0.0)),
                conflict=float(item.get("conflict", 0.0)),
            )
            beliefs[belief.claim_id] = belief
        self._beliefs = beliefs
        self._revision = int(data.get("revision", 0))

    def _persist_state(self, timestamp: str) -> None:
        if self.storage_path is None:
            return
        payload = {
            "version": 1,
            "revision": self._revision,
            "updated_at": timestamp,
            "beliefs": [self._serialize_belief(b) for b in self._beliefs.values()],
        }
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(self.storage_path)

    def _append_history(self, updates: List[Belief], timestamp: str) -> None:
        if not updates:
            return
        history_path = self.history_path
        if history_path is None:
            return
        history_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "revision": self._revision,
            "time": timestamp,
            "updates": [self._serialize_belief(b) for b in updates],
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    @staticmethod
    def _serialize_belief(belief: Belief) -> Dict[str, Any]:
        return {
            "claim_id": belief.claim_id,
            "credence": belief.credence,
            "last_updated": belief.last_updated,
            "evidence": [asdict(src) for src in belief.evidence],
            "support": belief.support,
            "conflict": belief.conflict,
        }


def _logistic_update(prior: float, outcome: bool, weight: float = 1.5) -> float:
    prior = min(max(prior, 1e-3), 1 - 1e-3)
    log_odds = math.log(prior / (1 - prior))
    log_odds += weight if outcome else -weight
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    raise ValueError("passed must be a boolean or boolean-like value")


def _resolve_weight(weight: Any, *, confidence: Any | None) -> float:
    base = 1.5 if weight is None else float(weight)
    if math.isnan(base):  # pragma: no cover - defensive
        raise ValueError("weight cannot be NaN")
    if base < 0:
        raise ValueError("weight must be non-negative")
    if confidence is not None:
        conf = float(confidence)
        if math.isnan(conf):  # pragma: no cover - defensive
            raise ValueError("confidence cannot be NaN")
        conf = max(0.0, min(1.0, conf))
        base *= conf
    return base


def _resolve_update_timestamp(
    timestamp: Any, *, default: str, tz: timezone = timezone.utc
) -> str:
    if timestamp is None:
        return default
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=tz)
        return timestamp.isoformat()
    if isinstance(timestamp, str):
        try:
            normalised = timestamp.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalised)
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601 formatted") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.isoformat()
    raise TypeError("timestamp must be a string or datetime")
