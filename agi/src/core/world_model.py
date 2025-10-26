from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from .types import Belief, Source


@dataclass
class WorldModel:
    _beliefs: Dict[str, Belief] = field(default_factory=dict)

    def update(self, results: Iterable[Dict[str, Any]]) -> List[Belief]:
        updated: List[Belief] = []
        now = datetime.now(timezone.utc).isoformat()
        for result in results:
            claim_id = result["claim_id"]
            if result.get("observed_unit") and result.get("expected_unit"):
                if result["observed_unit"] != result["expected_unit"]:
                    raise ValueError("Unit mismatch for claim %s" % claim_id)
            passed = bool(result.get("passed"))
            prior = self._beliefs.get(claim_id, Belief(claim_id=claim_id, credence=0.5, evidence=[], last_updated=now))
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
        return updated

    @property
    def beliefs(self) -> Dict[str, Belief]:
        return dict(self._beliefs)


def _logistic_update(prior: float, outcome: bool, weight: float = 1.5) -> float:
    prior = min(max(prior, 1e-3), 1 - 1e-3)
    log_odds = math.log(prior / (1 - prior))
    log_odds += weight if outcome else -weight
    odds = math.exp(log_odds)
    return odds / (1 + odds)
