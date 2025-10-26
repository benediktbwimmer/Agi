from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Gatekeeper:
    policy: Dict[str, object]

    def review(self, tier: str) -> bool:  # pragma: no cover - trivial
        if tier == "T2":
            return False
        return True
