from __future__ import annotations

from typing import Any, Dict, List

from ..types import RunContext, Source, ToolResult


class RetrievalTool:
    name = "retrieval"
    safety = "T1"

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:
        query = args.get("query", "")
        tool_hint = args.get("tool_hint")

        episodes = ctx.recall_from_episodic(tool=tool_hint, limit=5)
        if not episodes and ctx.working_memory:
            episodes = list(ctx.working_memory)

        if episodes:
            summary_lines: List[str] = []
            provenance: List[Source] = []
            for episode in episodes:
                call_id = episode.get("call_id", "")
                tool_name = episode.get("tool", "memory")
                stdout = episode.get("stdout") or episode.get("summary") or "(no output)"
                timestamp = episode.get("time", "unknown time")
                summary_lines.append(f"[{timestamp}] {tool_name}: {stdout}")
                if call_id:
                    provenance.append(
                        Source(
                            kind="memory",
                            ref=str(call_id),
                            note=f"retrieved from episodic store ({tool_name})",
                        )
                    )
            stdout = " | ".join(summary_lines)
        else:
            provenance = []
            stdout = f"no episodic memory matched query: {query}"

        return ToolResult(
            call_id=args.get("id", "retrieval"),
            ok=True,
            stdout=stdout,
            provenance=provenance or [Source(kind="tool", ref="retrieval", note="no matches")],
        )
