"""Convenience exports for the AGI package.

To keep import-time side effects minimal we lazily proxy attributes from
``agi.src.agi``.  This mirrors the legacy public surface while avoiding
dependence on the import order within the ``agi.src`` package.
"""

from __future__ import annotations

import importlib
from typing import Any

# Preload the ``agi.src`` package so that submodule imports work even if callers
# import :mod:`agi` first.  This mirrors the legacy namespace packaging behaviour
# from when the project lived under ``src/`` only.
importlib.import_module("agi.src")

__all__ = (
    "ExecutiveAgent",
    "MemoryRecord",
    "MemoryStore",
    "Orchestrator",
    "Planner",
    "PlannerError",
    "WorkingMemory",
    "WorldModel",
)


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agi.src import agi as _api

        return getattr(_api, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
