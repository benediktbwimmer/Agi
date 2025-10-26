from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
