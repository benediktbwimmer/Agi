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
        now = datetime.now(timezone.utc).isoformat()
        for result in results:
            claim_id = result["claim_id"]
            if result.get("observed_unit") and result.get("expected_unit"):
                if result["observed_unit"] != result["expected_unit"]:
                    raise ValueError("Unit mismatch for claim %s" % claim_id)
            passed = bool(result.get("passed"))
            prior = self._beliefs.get(
                claim_id,
                Belief(claim_id=claim_id, credence=0.5, evidence=[], last_updated=now),
            )
            posterior = _logistic_update(prior.credence, passed)
            evidence = list(prior.evidence)
            provenance = result.get("provenance") or []
            for source in provenance:
                if isinstance(source, Source):
                    evidence.append(source)
                elif isinstance(source, dict):
                    evidence.append(Source(**source))
            belief = Belief(
                claim_id=claim_id,
                credence=posterior,
                evidence=evidence,
                last_updated=now,
            )
            self._beliefs[claim_id] = belief
            updated.append(belief)

        if updated:
            self._revision += 1
            self._persist_state(now)
            self._append_history(updated, now)

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
        }


def _logistic_update(prior: float, outcome: bool, weight: float = 1.5) -> float:
    prior = min(max(prior, 1e-3), 1 - 1e-3)
    log_odds = math.log(prior / (1 - prior))
    log_odds += weight if outcome else -weight
    odds = math.exp(log_odds)
    return odds / (1 + odds)
