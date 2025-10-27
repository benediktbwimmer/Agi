"""High level entrypoints that compose the AGI subsystems."""
from __future__ import annotations

import sys
from types import ModuleType

from agi.src.core.executive import ExecutiveAgent
from agi.src.core.memory import MemoryStore
from agi.src.core.orchestrator import Orchestrator
from agi.src.core.planner import Planner, PlannerError
from agi.src.core.world_model import WorldModel
from agi.src.memory import MemoryRecord, WorkingMemory

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

_memory_module_name = f"{__name__}.memory"
_memory_module = sys.modules.get(_memory_module_name)
if not isinstance(_memory_module, ModuleType):
    _memory_module = ModuleType(_memory_module_name)

_working_module_name = f"{_memory_module_name}.working"
_working_module = sys.modules.get(_working_module_name)
if not isinstance(_working_module, ModuleType):
    _working_module = ModuleType(_working_module_name)

_memory_module.MemoryRecord = MemoryRecord
_memory_module.MemoryStore = MemoryStore
_memory_module.WorkingMemory = WorkingMemory
_memory_module.__all__ = ("MemoryRecord", "MemoryStore", "WorkingMemory")
if getattr(_memory_module, "__path__", None) is None:
    _memory_module.__path__ = []  # type: ignore[attr-defined]

_working_module.MemoryRecord = MemoryRecord
_working_module.WorkingMemory = WorkingMemory
_working_module.__all__ = ("MemoryRecord", "WorkingMemory")

sys.modules[_memory_module_name] = _memory_module
sys.modules[_working_module_name] = _working_module

_memory_module.working = _working_module
memory = _memory_module
