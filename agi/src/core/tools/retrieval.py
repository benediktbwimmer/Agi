from __future__ import annotations

from typing import Any, Dict, List

from ..types import RunContext, Source, ToolResult


class RetrievalTool:
    name = "retrieval"
    safety = "T1"

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:  # pragma: no cover - demo stub
        query = args.get("query", "")
        return ToolResult(
            call_id=args.get("id", "retrieval"),
            ok=True,
            stdout=f"no-op retrieval for query: {query}",
            provenance=[Source(kind="tool", ref="retrieval", note="stub result")],
        )
