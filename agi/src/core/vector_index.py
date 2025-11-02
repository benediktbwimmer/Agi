from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    faiss = None
    _import_error = exc
else:
    _import_error = None


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _stable_bucket(token: str, dim: int) -> int:
    digest = sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def _encode_tokens(tokens: Sequence[str], dim: int) -> np.ndarray | None:
    if not tokens:
        return None
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        bucket = _stable_bucket(token, dim)
        vector[bucket] += 1.0
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return None
    vector /= norm
    return vector


@dataclass
class MemoryVectorIndex:
    """Faiss-backed vector index tuned for memory record retrieval."""

    dim: int = 512
    search_weight: float = 0.75

    def __post_init__(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available") from _import_error
        self._index = faiss.IndexFlatIP(self.dim)
        self._lock = Lock()
        self._ids: List[int] = []

    @staticmethod
    def is_available() -> bool:
        return faiss is not None

    def _encode(self, text: str) -> np.ndarray | None:
        tokens = _tokenise(text)
        return _encode_tokens(tokens, self.dim)

    def add_text(self, text: str, record_idx: int) -> None:
        vector = self._encode(text)
        if vector is None:
            return
        self.add_vector(vector, record_idx)

    def add_vector(self, vector: np.ndarray, record_idx: int) -> None:
        if vector.shape != (self.dim,):
            raise RuntimeError(f"Expected vector of shape {(self.dim,)}, received {vector.shape}")
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        with self._lock:
            batch = vector.reshape(1, -1).copy()
            faiss.normalize_L2(batch)
            self._index.add(batch)
            self._ids.append(record_idx)

    def search_text(self, text: str, *, limit: int) -> List[Tuple[int, float]]:
        vector = self._encode(text)
        if vector is None:
            return []
        return self.search_vector(vector, limit=limit)

    def search_vector(self, vector: np.ndarray, *, limit: int) -> List[Tuple[int, float]]:
        if limit <= 0 or not self._ids:
            return []
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        with self._lock:
            query = vector.reshape(1, -1).copy()
            faiss.normalize_L2(query)
            top_k = min(limit, len(self._ids))
            scores, indices = self._index.search(query, top_k)
        results: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            record_idx = self._ids[idx]
            results.append((record_idx, float(score)))
        return results

    def describe(self) -> Dict[str, int | float]:
        return {
            "dim": self.dim,
            "size": len(self._ids),
            "weight": self.search_weight,
        }


__all__ = ["MemoryVectorIndex"]
