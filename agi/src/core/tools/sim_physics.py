from __future__ import annotations

from typing import Any, Dict

from . import SensorProfile, ToolCapability, ToolParameter, ToolSpec
from ..types import RunContext, ToolResult


def simulate_projectile(angle_deg: float, speed: float, gravity: float = 9.81, steps: int = 100):
    import math

    angle = math.radians(angle_deg)
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    dt = 2 * vy / gravity / steps if steps else 0.01
    x = y = 0.0
    path = []
    for _ in range(steps + 1):
        path.append({"x": x, "y": y})
        x += vx * dt
        vy -= gravity * dt
        y += vy * dt
        if y < 0:
            break
    return path


class PhysicsSimulator:
    name = "sim_physics"
    safety = "T0"

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:  # pragma: no cover - simple
        trajectory = simulate_projectile(
            angle_deg=float(args.get("angle_deg", 45.0)),
            speed=float(args.get("speed", 10.0)),
            gravity=float(args.get("gravity", 9.81)),
            steps=int(args.get("steps", 100)),
        )
        return ToolResult(
            call_id=args.get("id", "sim_physics"),
            ok=True,
            stdout=str(trajectory[-1] if trajectory else {}),
            data={"trajectory": trajectory},
            provenance=[],
        )

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Simulate projectile motion in a simple physics model.",
            safety_tier=self.safety,
            sensor_profile=SensorProfile(
                modality="simulation",
                latency_ms=80,
                trust="medium",
                description="Numerical physics integrator for simple projectile motion",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "angle_deg": {"type": "number", "description": "Launch angle in degrees."},
                    "speed": {"type": "number", "description": "Initial speed in metres per second."},
                    "gravity": {"type": "number", "description": "Gravity constant used in m/s^2."},
                    "steps": {"type": "integer", "minimum": 1, "description": "Integration steps."},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "trajectory": {
                                "type": "array",
                                "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
                            }
                        },
                    },
                    "stdout": {"type": "string"},
                },
            },
            capabilities=(
                ToolCapability(
                    name="simulate_projectile",
                    description="Generate a trajectory for a projectile launched under uniform gravity.",
                    safety_tier=self.safety,
                    parameters=(
                        ToolParameter(
                            name="angle_deg",
                            description="Launch angle measured in degrees.",
                            required=False,
                            schema={"type": "number"},
                        ),
                        ToolParameter(
                            name="speed",
                            description="Initial launch speed in m/s.",
                            required=False,
                            schema={"type": "number"},
                        ),
                        ToolParameter(
                            name="gravity",
                            description="Gravity constant applied to the simulation.",
                            required=False,
                            schema={"type": "number"},
                        ),
                        ToolParameter(
                            name="steps",
                            description="Number of integration steps to compute the trajectory.",
                            required=False,
                            schema={"type": "integer", "minimum": 1},
                        ),
                    ),
                    outputs=("data.trajectory",),
                ),
            ),
        )
