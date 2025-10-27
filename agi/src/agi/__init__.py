"""Lazy public interface for ``agi.src`` with backwards-compatible aliases."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Dict

__all__ = (
    "ExecutiveAgent",
    "MemoryRecord",
    "MemoryStore",
    "Orchestrator",
    "Planner",
    "PlannerError",
    "WorkingMemory",
    "WorldModel",
    "memory",
)

_IMPORT_MAP: Dict[str, str] = {
    "ExecutiveAgent": "agi.src.core.executive",
    "MemoryStore": "agi.src.core.memory",
    "Orchestrator": "agi.src.core.orchestrator",
    "Planner": "agi.src.core.planner",
    "PlannerError": "agi.src.core.planner",
    "WorldModel": "agi.src.core.world_model",
    "MemoryRecord": "agi.src.memory",
    "WorkingMemory": "agi.src.memory",
}

_memory_alias_ready = False
_memory_module_name = f"{__name__}.memory"
_working_module_name = f"{_memory_module_name}.working"


def _ensure_memory_alias() -> ModuleType:
    global _memory_alias_ready
    if _memory_alias_ready:
        return sys.modules[_memory_module_name]

    memory_pkg = importlib.import_module("agi.src.memory")
    working_module = importlib.import_module("agi.src.memory.working")
    core_memory = importlib.import_module("agi.src.core.memory")

    memory_pkg.MemoryStore = getattr(core_memory, "MemoryStore")
    memory_pkg.MemoryRecord = getattr(memory_pkg, "MemoryRecord")
    memory_pkg.WorkingMemory = getattr(memory_pkg, "WorkingMemory")
    memory_pkg.working = working_module

    sys.modules[_memory_module_name] = memory_pkg
    sys.modules[_working_module_name] = working_module

    if not hasattr(working_module, "MemoryRecord"):
        working_module.MemoryRecord = memory_pkg.MemoryRecord  # type: ignore[attr-defined]
    if not hasattr(working_module, "WorkingMemory"):
        working_module.WorkingMemory = memory_pkg.WorkingMemory  # type: ignore[attr-defined]
    working_module.__all__ = ("MemoryRecord", "WorkingMemory")

    _memory_alias_ready = True
    return memory_pkg


def __getattr__(name: str) -> Any:
    if name == "memory":
        return _ensure_memory_alias()
    module_name = _IMPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))

