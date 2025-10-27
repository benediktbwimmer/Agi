from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set, Tuple

from ..types import RunContext, Source, ToolResult


class RetrievalTool:
    name = "retrieval"
    safety = "T1"

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:
        query = (args.get("query") or "").strip()
        tool_hint = args.get("tool_hint")
        limit = int(args.get("limit", 5))

        episodes = ctx.recall_from_episodic(
            tool=tool_hint,
            limit=limit,
            text_query=query or None,
        )

        def _matches_query(episode: Dict[str, Any]) -> bool:
            if not query:
                return True
            needle = query.lower()
            haystacks: List[str] = []
            for key in ("stdout", "summary", "goal"):
                value = episode.get(key)
                if isinstance(value, str):
                    haystacks.append(value.lower())
            claim_ids = episode.get("claim_ids")
            if isinstance(claim_ids, list):
                haystacks.extend(str(cid).lower() for cid in claim_ids)
            return any(needle in field for field in haystacks)

        def _dedupe(
            records: Iterable[Dict[str, Any]]
        ) -> Tuple[List[Dict[str, Any]], Set[str]]:
            seen: Set[str] = set()
            unique: List[Dict[str, Any]] = []
            for episode in records:
                call_id = episode.get("call_id")
                key = str(call_id) if call_id is not None else None
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                unique.append(episode)
            return unique, seen

        unique_episodes, seen_ids = _dedupe(episodes)

        if ctx.working_memory:
            wm_matches = [
                episode
                for episode in ctx.working_memory
                if (
                    (episode.get("call_id") is None)
                    or (str(episode.get("call_id")) not in seen_ids)
                )
                and _matches_query(episode)
            ]
            unique_episodes.extend(wm_matches)

        if limit == 0:
            unique_episodes = []
        elif limit > 0:
            unique_episodes = unique_episodes[-limit:]

        if unique_episodes:
            summary_lines: List[str] = []
            provenance: List[Source] = []
            for episode in unique_episodes:
                call_id = episode.get("call_id", "")
                tool_name = episode.get("tool", tool_hint or "memory")
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
            stdout = f"no episodic memory matched query: {query}" if query else "no episodic memory available"

        return ToolResult(
            call_id=args.get("id", "retrieval"),
            ok=True,
            stdout=stdout,
            provenance=provenance
            or [Source(kind="tool", ref="retrieval", note="no matches")],
        )
