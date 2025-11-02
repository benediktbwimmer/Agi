from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict


class CriticError(RuntimeError):
    pass


@dataclass
class Critic:
    llm: Callable[[Dict[str, Any]], str]

    async def check(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        response = self.llm(plan)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise CriticError("Critic did not return JSON") from exc
        status = str(data.get("status", "")).upper()
        if status not in {"PASS", "FAIL", "REVISION"}:
            raise CriticError("Critic response missing status")
        data["status"] = status
        return data
