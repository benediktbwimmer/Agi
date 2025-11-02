from __future__ import annotations

"""Utilities for encoding tool outputs into persistent memory records."""

import json
import math
import re
from collections import Counter
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agi.src.core.types import ToolResult


_SUMMARY_LIMIT = 240
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{4,}")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_EMBED_DIM = 512
_METRIC_PATTERN = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_\- ]{1,24})[:=]\s*(?P<value>-?\d+(?:\.\d+)?)"
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def _extract_keywords(text: str, *, limit: int = 5) -> List[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return []
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def _split_sentences(text: str) -> List[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text) if sentence.strip()]
    return sentences


def _extract_claims(text: str, *, limit: int = 3) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    scored: List[tuple[float, str]] = []
    for sentence in sentences:
        lower = sentence.lower()
        score = 0.0
        if any(keyword in lower for keyword in ("should", "must", "needs", "ensure", "warning")):
            score += 2.0
        if any(keyword in lower for keyword in ("success", "failure", "risk", "anomaly", "plan")):
            score += 1.5
        score += min(len(sentence.split()) / 12.0, 1.0)
        scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    return [sentence for _, sentence in scored[:limit]]


def _stable_bucket(token: str) -> int:
    digest = sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % _EMBED_DIM


def _encode_embedding(text: str) -> Optional[Dict[str, Any]]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return None
    buckets: Dict[int, float] = {}
    for token in tokens:
        bucket = _stable_bucket(token)
        buckets[bucket] = buckets.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in buckets.values()))
    if not norm:
        return None
    values = {str(bucket): round(value / norm, 6) for bucket, value in buckets.items()}
    return {"dim": _EMBED_DIM, "values": values}


def _collect_numeric_metrics(payload: Mapping[str, Any], *, prefix: str = "", limit: int = 8) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    queue: List[tuple[str, Any]] = []
    for key, value in payload.items():
        queue.append((f"{prefix}{key}" if not prefix else f"{prefix}.{key}", value))
    while queue and len(metrics) < limit:
        key, value = queue.pop(0)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                queue.append((f"{key}.{sub_key}", sub_value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for idx, item in enumerate(value[:limit]):
                queue.append((f"{key}[{idx}]", item))
    return metrics


def _extract_text_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for match in _METRIC_PATTERN.finditer(text):
        key = match.group("key").strip().lower().replace(" ", "_")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        metrics.setdefault(key, value)
        if len(metrics) >= 5:
            break
    return metrics


def summarise_tool_result(result: ToolResult) -> Dict[str, Any]:
    """Return derived metadata describing ``result`` for storage in memory."""

    summary: Dict[str, Any] = {}
    structured: Dict[str, Any] = {}
    stdout = result.stdout
    if isinstance(stdout, str):
        trimmed = stdout.strip()
        if trimmed:
            summary["summary"] = _truncate(trimmed, _SUMMARY_LIMIT)
            summary["tokens"] = len(trimmed.split())
            keywords = _extract_keywords(trimmed)
            if keywords:
                summary["keywords"] = keywords
            if trimmed.startswith("{") or trimmed.startswith("["):
                try:
                    parsed = json.loads(trimmed)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    structured.setdefault("keys", sorted(parsed.keys())[:5])
                    metrics = _collect_numeric_metrics(parsed)
                    if metrics:
                        structured.setdefault("metrics", metrics)
                elif isinstance(parsed, list):
                    structured["list_length"] = len(parsed)
            claims = _extract_claims(trimmed)
            if claims:
                summary["claims"] = claims
            embedding = _encode_embedding(trimmed)
            if embedding:
                summary["embedding"] = embedding
            text_metrics = _extract_text_metrics(trimmed)
            if text_metrics:
                structured.setdefault("metrics", {}).update(text_metrics)

    data = result.data
    if isinstance(data, Mapping):
        keys = list(data.keys())
        if keys:
            structured.setdefault("data_keys", keys[:5])
        numeric_metrics = _collect_numeric_metrics(data)
        if numeric_metrics:
            structured.setdefault("metrics", {}).update(numeric_metrics)
    figures = result.figures or []
    if figures:
        structured["figures"] = len(figures)
    if structured:
        summary["structured"] = structured
    return summary


__all__ = [
    "summarise_tool_result",
]
