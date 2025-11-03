from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .types import Belief, Evidence, Source


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
            claim_id = str(result["claim_id"])
            expected_unit = result.get("expected_unit")
            observed_unit = result.get("observed_unit") or expected_unit
            if expected_unit and observed_unit and observed_unit != expected_unit:
                raise ValueError("Unit mismatch for claim %s" % claim_id)
            passed = _coerce_bool(result.get("passed"))
            confidence_value = _normalise_confidence(result.get("confidence"))
            weight = _resolve_weight(
                result.get("weight"),
                confidence=confidence_value,
            )
            raw_weight = _normalise_raw_weight(result.get("weight"))
            prior = self._beliefs.get(
                claim_id,
                Belief(
                    claim_id=claim_id,
                    credence=0.5,
                    last_updated=update_time_iso,
                    evidence=[],
                    support=0.0,
                    conflict=0.0,
                    variance=1.0,
                ),
            )
            posterior = _logistic_update(prior.credence, passed, weight=weight)
            timestamp = _resolve_update_timestamp(
                result.get("timestamp"), default=update_time_iso
            )
            provided_evidence = _normalise_evidence_list(
                result.get("evidence"),
                default_outcome="support" if passed else "conflict",
                default_weight=weight,
                default_confidence=confidence_value,
                default_unit=observed_unit,
                default_note=_normalise_note(result),
                timestamp=timestamp,
            )
            if provided_evidence:
                evidence_entries = provided_evidence
            else:
                provenance = _normalise_sources(result.get("provenance"))
                evidence_entries = _build_evidence_entries(
                    provenance,
                    outcome="support" if passed else "conflict",
                    weight=weight,
                    raw_weight=raw_weight,
                    confidence=confidence_value,
                    unit=observed_unit,
                    value=_normalise_value(result.get("observed_value", result.get("value"))),
                    note=_normalise_note(result),
                    timestamp=timestamp,
                )
            evidence = list(prior.evidence)
            evidence.extend(evidence_entries)
            support = prior.support
            conflict = prior.conflict
            for entry in evidence_entries:
                if entry.is_support():
                    support += entry.weight
                else:
                    conflict += entry.weight
            total_weight = support + conflict
            variance = 1.0 if total_weight <= 0 else 1.0 / (1.0 + total_weight)
            belief = Belief(
                claim_id=claim_id,
                credence=posterior,
                evidence=evidence,
                last_updated=timestamp,
                support=support,
                conflict=conflict,
                variance=variance,
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
            last_updated = item.get("last_updated", datetime.now(timezone.utc).isoformat())
            evidence_payload = item.get("evidence", [])
            evidence = [
                _deserialize_evidence(payload, default_time=last_updated)
                for payload in evidence_payload
            ]
            evidence = [entry for entry in evidence if entry is not None]
            belief = Belief(
                claim_id=item["claim_id"],
                credence=float(item.get("credence", 0.5)),
                evidence=evidence,
                last_updated=last_updated,
                support=float(item.get("support", 0.0)),
                conflict=float(item.get("conflict", 0.0)),
                variance=float(item.get("variance", 1.0)),
            )
            beliefs[belief.claim_id] = belief
        self._beliefs = beliefs
        self._revision = int(data.get("revision", 0))

    def _persist_state(self, timestamp: str) -> None:
        if self.storage_path is None:
            return
        payload = {
            "version": 2,
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
            "evidence": [asdict(entry) for entry in belief.evidence],
            "support": belief.support,
            "conflict": belief.conflict,
            "variance": belief.variance,
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


def _resolve_weight(weight: Any, *, confidence: Optional[float]) -> float:
    base = 1.5 if weight is None else float(weight)
    if math.isnan(base):  # pragma: no cover - defensive
        raise ValueError("weight cannot be NaN")
    if base < 0:
        raise ValueError("weight must be non-negative")
    if confidence is not None:
        base *= confidence
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


def _normalise_sources(payload: Any) -> List[Source]:
    sources: List[Source] = []
    if payload is None:
        return sources
    if isinstance(payload, Source):
        return [payload]
    if isinstance(payload, Mapping):
        try:
            return [Source(**{k: payload[k] for k in ("kind", "ref", "note") if k in payload})]
        except TypeError:
            return [Source(kind=str(payload.get("kind", "unknown")), ref=str(payload.get("ref", "")), note=payload.get("note"))]
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        for item in payload:
            if isinstance(item, Source):
                sources.append(item)
            elif isinstance(item, Mapping):
                kind = str(item.get("kind", "unknown"))
                ref = str(item.get("ref", ""))
                note = item.get("note")
                sources.append(Source(kind=kind, ref=ref, note=note))
    return sources


def _build_evidence_entries(
    sources: List[Source],
    *,
    outcome: str,
    weight: float,
    raw_weight: Optional[float],
    confidence: Optional[float],
    unit: Optional[str],
    value: Any,
    note: Optional[str],
    timestamp: str,
) -> List[Evidence]:
    outcome_normalised = outcome.lower()
    if outcome_normalised not in {"support", "conflict"}:
        outcome_normalised = "support"
    effective_sources = sources or [
        Source(kind="system", ref="world_model", note="auto-generated")
    ]
    count = len(effective_sources)
    per_weight = weight / count if count and weight else weight
    per_raw = raw_weight / count if raw_weight is not None and count else raw_weight
    entries: List[Evidence] = []
    for source in effective_sources:
        entries.append(
            Evidence(
                source=source,
                outcome=outcome_normalised,
                weight=per_weight,
                raw_weight=per_raw,
                confidence=confidence,
                unit=unit,
                value=value,
                note=note,
                observed_at=timestamp,
            )
        )
    return entries


def _normalise_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if math.isnan(confidence):  # pragma: no cover - defensive
        raise ValueError("confidence cannot be NaN")
    return max(0.0, min(1.0, confidence))


def _normalise_raw_weight(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight must be numeric") from exc
    if math.isnan(raw):  # pragma: no cover - defensive
        raise ValueError("weight cannot be NaN")
    if raw < 0:
        raise ValueError("weight must be non-negative")
    return raw


def _normalise_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


def _normalise_note(result: Mapping[str, Any]) -> Optional[str]:
    for key in ("note", "summary", "comment"):
        value = result.get(key)
        if value:
            return str(value)
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):  # pragma: no cover - defensive
        return default
    return number


def _deserialize_evidence(payload: Any, *, default_time: str) -> Optional[Evidence]:
    if isinstance(payload, Evidence):
        return payload
    if isinstance(payload, Source):
        return Evidence(
            source=payload,
            outcome="support",
            weight=0.0,
            raw_weight=None,
            confidence=None,
            unit=None,
            value=None,
            note=None,
            observed_at=default_time,
        )
    if isinstance(payload, Mapping):
        if "source" in payload:
            source = payload["source"]
            if not isinstance(source, Source):
                source = Source(
                    kind=str(source.get("kind", "unknown")),
                    ref=str(source.get("ref", "")),
                    note=source.get("note"),
                )
            outcome = str(payload.get("outcome", "support"))
            weight = _to_float(payload.get("weight"), 0.0)
            raw_weight = payload.get("raw_weight")
            raw_weight = None if raw_weight is None else _to_float(raw_weight, 0.0)
            confidence = _normalise_confidence(payload.get("confidence"))
            unit = payload.get("unit")
            value = payload.get("value")
            note = payload.get("note")
            observed_at = payload.get("observed_at", default_time)
            return Evidence(
                source=source,
                outcome=outcome,
                weight=weight,
                raw_weight=raw_weight,
                confidence=confidence,
                unit=unit,
                value=value,
                note=note,
                observed_at=observed_at,
            )
        # Legacy format: plain Source payload
        source = Source(
            kind=str(payload.get("kind", "unknown")),
            ref=str(payload.get("ref", "")),
            note=payload.get("note"),
        )
        return Evidence(
            source=source,
            outcome=str(payload.get("outcome", "support")),
            weight=_to_float(payload.get("weight"), 0.0) if "weight" in payload else 0.0,
            raw_weight=None,
            confidence=None,
            unit=payload.get("unit"),
            value=payload.get("value"),
            note=payload.get("note"),
            observed_at=payload.get("observed_at", default_time),
        )
    return None


def _normalise_evidence_list(
    payload: Any,
    *,
    default_outcome: str,
    default_weight: float,
    default_confidence: Optional[float],
    default_unit: Optional[str],
    default_note: Optional[str],
    timestamp: str,
) -> List[Evidence]:
    if payload is None:
        return []
    if isinstance(payload, Mapping) and "source" in payload:
        payload = [payload]
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        entries: List[Evidence] = []
        for item in payload:
            evidence = _deserialize_evidence(item, default_time=timestamp)
            if evidence is None:
                continue
            if not evidence.outcome:
                evidence.outcome = default_outcome
            if not evidence.weight and default_weight:
                evidence.weight = default_weight
            if evidence.confidence is None and default_confidence is not None:
                evidence.confidence = default_confidence
            if evidence.unit is None:
                evidence.unit = default_unit
            if evidence.note is None and default_note:
                evidence.note = default_note
            if evidence.observed_at is None:
                evidence.observed_at = timestamp
            entries.append(evidence)
        return entries
    return []
