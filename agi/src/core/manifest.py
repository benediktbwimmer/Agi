"""Run manifest schema and helpers.

The orchestrator writes a manifest for every execution so downstream systems
can replay and analyse the run.  Historically this structure was an ad-hoc
dictionary which made validation brittle.  This module formalises the schema
using Pydantic so manifests are self-describing, versioned, and easy to
validate in tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .safety import RiskAssessment, SafetyDecision
from .tools import ToolSpec
from .types import Belief, ToolResult


MANIFEST_SCHEMA_VERSION = "0.2.0"


class ManifestSource(BaseModel):
    """Normalised provenance metadata."""

    kind: str
    ref: str
    note: str | None = None


class ManifestToolResult(BaseModel):
    """Serialised representation of :class:`ToolResult`."""

    call_id: str
    ok: bool
    cost_tokens: int | None = None
    wall_time_ms: int | None = None
    stdout: str | None = None
    data: Any = None
    figures: Sequence[str] | None = None
    provenance: Sequence[ManifestSource] = Field(default_factory=list)


class ManifestEvidenceEntry(BaseModel):
    """Structured evidence backing a belief update."""

    source: ManifestSource
    outcome: str
    weight: float | None = None
    raw_weight: float | None = None
    confidence: float | None = None
    unit: str | None = None
    value: Any = None
    note: str | None = None
    observed_at: str | None = None


class ManifestBeliefUpdate(BaseModel):
    """Serialised representation of :class:`Belief`."""

    claim_id: str
    credence: float
    last_updated: str
    support: float | None = None
    conflict: float | None = None
    variance: float | None = None
    uncertainty: float | None = None
    confidence_interval: Sequence[float] | None = None
    evidence: Sequence[ManifestEvidenceEntry] = Field(default_factory=list)


class ManifestSafetyDecision(BaseModel):
    """Serialised representation of :class:`SafetyDecision`."""

    plan_id: str
    step_id: str
    tool_name: str
    requested_level: str
    tool_level: str
    effective_level: str
    approved: bool
    reason: str | None = None


class ManifestRiskAssessment(BaseModel):
    """Serialised representation of :class:`RiskAssessment`."""

    plan_id: str
    step_id: str
    tool_name: str
    requested_level: str
    tool_level: str
    effective_level: str
    approved: bool
    reason: str | None = None


class ManifestCritique(BaseModel):
    """Critique emitted by the critic during planning."""

    plan_id: str
    status: str
    reviewer: str | None = "critic"
    notes: str | None = None
    summary: str | None = None
    amendments: Sequence[str] | None = None
    issues: Sequence[str] | None = None


class ManifestToolParameter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    required: bool = True
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")


class ManifestToolCapability(BaseModel):
    name: str
    description: str
    safety_tier: str
    parameters: Sequence[ManifestToolParameter] = Field(default_factory=list)
    outputs: Sequence[str] = Field(default_factory=list)


class ManifestToolSpec(BaseModel):
    name: str
    description: str
    safety_tier: str
    version: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: Sequence[ManifestToolCapability] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sensor_profile: dict[str, Any] | None = None


class RunManifest(BaseModel):
    """Versioned manifest describing an orchestrator execution."""

    schema_version: str = Field(default=MANIFEST_SCHEMA_VERSION)
    run_id: str
    created_at: str
    goal: dict[str, Any]
    constraints: dict[str, Any]
    tool_results: Sequence[ManifestToolResult]
    belief_updates: Sequence[ManifestBeliefUpdate]
    safety_audit: Sequence[ManifestSafetyDecision] = Field(default_factory=list)
    risk_assessments: Sequence[ManifestRiskAssessment] = Field(default_factory=list)
    critiques: Sequence[ManifestCritique] = Field(default_factory=list)
    tool_catalog: Sequence[ManifestToolSpec] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("created_at must be ISO-8601 formatted") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        goal: Mapping[str, Any],
        constraints: Mapping[str, Any],
        tool_results: Iterable[ToolResult],
        belief_updates: Iterable[Belief],
        safety_audit: Iterable[SafetyDecision],
        risk_assessments: Iterable[RiskAssessment] | None = None,
        critiques: Iterable[Mapping[str, Any]] | None = None,
        tool_catalog: Iterable[ToolSpec] | Iterable[Mapping[str, Any]] | None = None,
    ) -> "RunManifest":
        """Construct a manifest from runtime dataclasses."""

        created_at = datetime.now(timezone.utc).isoformat()

        def _normalise(items: Iterable[Any]) -> list[Any]:
            normalised: list[Any] = []
            for item in items:
                if isinstance(item, BaseModel):
                    normalised.append(item.model_dump())
                elif is_dataclass(item):
                    normalised.append(asdict(item))
                else:
                    normalised.append(item)
            return normalised

        def _normalise_beliefs(items: Iterable[Any]) -> list[Any]:
            normalised: list[Any] = []
            for item in items:
                if isinstance(item, Belief):
                    interval = [float(value) for value in item.confidence_interval]
                    evidence = [
                        {
                            "source": asdict(ev.source),
                            "outcome": ev.outcome,
                            "weight": ev.weight,
                            "raw_weight": ev.raw_weight,
                            "confidence": ev.confidence,
                            "unit": ev.unit,
                            "value": ev.value,
                            "note": ev.note,
                            "observed_at": ev.observed_at,
                        }
                        for ev in item.evidence
                    ]
                    normalised.append(
                        {
                            "claim_id": item.claim_id,
                            "credence": item.credence,
                            "last_updated": item.last_updated,
                            "support": item.support,
                            "conflict": item.conflict,
                            "variance": item.variance,
                            "uncertainty": item.uncertainty,
                            "confidence_interval": interval,
                            "evidence": evidence,
                        }
                    )
                elif isinstance(item, BaseModel):
                    normalised.append(item.model_dump())
                elif is_dataclass(item):
                    normalised.append(asdict(item))
                else:
                    normalised.append(item)
            return normalised

        manifest = cls(
            run_id=run_id,
            created_at=created_at,
            goal=dict(goal),
            constraints=dict(constraints),
            tool_results=_normalise(tool_results),
            belief_updates=_normalise_beliefs(belief_updates),
            safety_audit=_normalise(safety_audit),
            risk_assessments=_normalise(risk_assessments or []),
            critiques=_normalise(critiques or []),
            tool_catalog=_normalise(tool_catalog or []),
        )
        return manifest

    def write(self, path: Path) -> None:
        """Persist the manifest to disk in canonical JSON form."""

        path.write_text(
            self.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    @classmethod
    def write_schema(cls, path: Path) -> None:
        """Write the JSON schema for the manifest to disk."""

        schema = cls.model_json_schema()
        path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManifestBeliefUpdate",
    "ManifestEvidenceEntry",
    "ManifestCritique",
    "ManifestRiskAssessment",
    "ManifestToolCapability",
    "ManifestToolParameter",
    "ManifestToolSpec",
    "ManifestSafetyDecision",
    "ManifestSource",
    "ManifestToolResult",
    "RunManifest",
]
