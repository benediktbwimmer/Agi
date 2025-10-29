from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


def _normalise_tier(level: str | None) -> str:
    if not level:
        return "T0"
    level = str(level).upper()
    if level not in {"T0", "T1", "T2", "T3"}:
        raise ValueError(f"Unknown safety tier: {level}")
    return level


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(mapping or {})


@dataclass(frozen=True)
class ToolParameter:
    """Description of a structured parameter accepted by a tool capability."""

    name: str
    description: str
    required: bool = True
    schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorProfile:
    """Describes the sensory characteristics of a tool interface."""

    modality: str
    latency_ms: int | None = None
    trust: str = "provisional"
    description: str | None = None


@dataclass(frozen=True)
class ToolCapability:
    """A unit of functionality exposed by a tool."""

    name: str
    description: str
    safety_tier: str = "T0"
    parameters: Sequence[ToolParameter] = field(default_factory=tuple)
    outputs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolSpec:
    """Structured description of a tool for planning and governance."""

    name: str
    description: str
    safety_tier: str = "T0"
    version: str | None = None
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Sequence[ToolCapability] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sensor_profile: SensorProfile | None = None


def _normalise_sensor_profile(profile: SensorProfile | Mapping[str, Any] | None) -> SensorProfile | None:
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        profile = SensorProfile(**profile)
    if not isinstance(profile, SensorProfile):  # pragma: no cover - defensive
        raise TypeError("sensor_profile must be a SensorProfile or mapping")
    latency = profile.latency_ms
    if latency is not None:
        latency = int(latency)
        if latency < 0:
            raise ValueError("sensor_profile.latency_ms must be non-negative")
    trust = str(profile.trust).strip().lower() or "provisional"
    modality = str(profile.modality).strip() or "unknown"
    description = profile.description
    if description is not None:
        description = str(description).strip() or None
    return SensorProfile(
        modality=modality,
        latency_ms=latency,
        trust=trust,
        description=description,
    )


def _ensure_capabilities(spec: ToolSpec) -> Sequence[ToolCapability]:
    if spec.capabilities:
        capabilities = []
        for capability in spec.capabilities:
            params = tuple(
                replace(param, schema=_copy_mapping(param.schema))
                for param in capability.parameters
            )
            capabilities.append(
                replace(
                    capability,
                    safety_tier=_normalise_tier(capability.safety_tier),
                    parameters=params,
                )
            )
        return tuple(capabilities)
    default_capability = ToolCapability(
        name="default",
        description="Default execution path",
        safety_tier=spec.safety_tier,
        parameters=tuple(),
        outputs=tuple(),
    )
    return (default_capability,)


def _normalise_spec(spec: ToolSpec) -> ToolSpec:
    return replace(
        spec,
        name=str(spec.name),
        description=str(spec.description).strip() or str(spec.name),
        safety_tier=_normalise_tier(spec.safety_tier),
        input_schema=_copy_mapping(spec.input_schema),
        output_schema=_copy_mapping(spec.output_schema),
        capabilities=_ensure_capabilities(spec),
        metadata=_copy_mapping(spec.metadata),
        sensor_profile=_normalise_sensor_profile(spec.sensor_profile),
    )


def describe_tool(tool: Any, *, override_name: str | None = None) -> ToolSpec:
    """Return a :class:`ToolSpec` for ``tool``.

    Tools may implement a ``describe`` method that returns a :class:`ToolSpec`.
    If absent, a conservative default specification is synthesised based on the
    tool's attributes.  The ``override_name`` parameter allows callers to
    enforce a specific catalog name while preserving the tool's internal
    metadata.
    """

    if hasattr(tool, "describe"):
        spec = tool.describe()
        if not isinstance(spec, ToolSpec):  # pragma: no cover - defensive
            raise TypeError("tool.describe() must return a ToolSpec instance")
    else:
        name = getattr(tool, "name", tool.__class__.__name__)
        description = (getattr(tool, "__doc__", "") or str(name)).strip() or str(name)
        safety = _normalise_tier(getattr(tool, "safety", "T0"))
        sensor_profile = SensorProfile(
            modality=str(getattr(tool, "modality", "unknown")),
            latency_ms=_getattr_int(tool, "latency_ms"),
            trust=str(getattr(tool, "trust", "provisional")),
        )
        spec = ToolSpec(
            name=name,
            description=description,
            safety_tier=safety,
            capabilities=(
                ToolCapability(
                    name="default",
                    description="Default execution path",
                    safety_tier=safety,
                ),
            ),
            sensor_profile=sensor_profile,
        )

    spec = _normalise_spec(spec)
    if override_name:
        spec = replace(spec, name=str(override_name))
    return spec


def _getattr_int(obj: Any, attr: str) -> int | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


__all__ = [
    "SensorProfile",
    "ToolCapability",
    "ToolParameter",
    "ToolSpec",
    "describe_tool",
]
