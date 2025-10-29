"""Shared type definitions for the AGI runtime."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, TypedDict


UID = str


class Prediction(TypedDict, total=False):
    """Prediction metadata attached to a :class:`Claim`."""

    id: UID
    metric: str
    expectation: str
    eval_procedure: str


@dataclass
class Source:
    """Provenance information for beliefs and tool outputs."""

    kind: str
    ref: str
    note: Optional[str] = None


@dataclass
class Claim:
    """A declarative claim the system is attempting to verify."""

    id: UID
    text: str
    predictions: List[Prediction]
    variables: Dict[str, Any]
    provenance: Optional[List[Source]] = None


@dataclass
class ToolCall:
    """Instruction to invoke a tool during a plan."""

    id: UID
    tool: str
    args: Dict[str, Any]
    safety_level: str = "T0"


@dataclass
class ToolResult:
    """Outcome from executing a tool call."""

    call_id: UID
    ok: bool
    cost_tokens: Optional[int] = None
    wall_time_ms: Optional[int] = None
    stdout: Optional[str] = None
    data: Any = None
    figures: Optional[List[str]] = None
    provenance: List[Source] = field(default_factory=list)


@dataclass
class BranchCondition:
    """Condition controlling execution of a plan branch."""

    kind: str = "always"
    step_id: Optional[UID] = None
    value: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "BranchCondition":
        """Normalise a raw condition into a :class:`BranchCondition`."""

        if raw is None:
            return cls()
        if isinstance(raw, BranchCondition):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text or text.lower() in {"always", "true"}:
                return cls()
            if text.lower() in {"never", "false"}:
                return cls(kind="never")
            if text.startswith("on_success(") and text.endswith(")"):
                return cls(kind="success", step_id=text[len("on_success(") : -1].strip())
            if text.startswith("on_failure(") and text.endswith(")"):
                return cls(kind="failure", step_id=text[len("on_failure(") : -1].strip())
            return cls(kind="expression", value=text)
        if isinstance(raw, dict):
            kind = str(raw.get("when", raw.get("kind", "always"))).lower()
            step = raw.get("step") or raw.get("step_id")
            value = raw.get("value") or raw.get("contains") or raw.get("expression")
            return cls(kind=kind, step_id=step, value=value)
        raise TypeError(f"Unsupported branch condition format: {raw!r}")

    def evaluate(self, results: Mapping[str, "ToolResult"]) -> bool:
        """Evaluate the condition against known tool results."""

        if self.kind in {"always", ""}:
            return True
        if self.kind == "never":
            return False
        if self.kind == "success":
            if not self.step_id:
                return False
            result = results.get(self.step_id)
            return bool(result and result.ok)
        if self.kind == "failure":
            if not self.step_id:
                return False
            result = results.get(self.step_id)
            return bool(result and not result.ok)
        if self.kind == "stdout_contains":
            if not self.step_id or self.value is None:
                return False
            result = results.get(self.step_id)
            if result is None or result.stdout is None:
                return False
            return str(self.value) in result.stdout
        if self.kind == "expression":
            return _evaluate_expression(str(self.value or ""), results)
        return False

    def to_payload(self) -> Any:
        """Serialise the condition for external consumers."""

        if self.kind in {"always", ""} and not self.step_id and self.value is None:
            return None
        payload: Dict[str, Any] = {"when": self.kind}
        if self.step_id:
            payload["step"] = self.step_id
        if self.value is not None:
            payload["value"] = self.value
        return payload


class _ResultView:
    """Lightweight proxy exposing safe attributes of a :class:`ToolResult`."""

    __slots__ = ("_result",)

    def __init__(self, result: ToolResult | None):
        self._result = result

    @property
    def ok(self) -> bool:
        return bool(self._result and self._result.ok)

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def stdout(self) -> Optional[str]:
        return None if self._result is None else self._result.stdout

    @property
    def data(self) -> Any:
        return None if self._result is None else self._result.data

    @property
    def call_id(self) -> Optional[str]:
        return None if self._result is None else self._result.call_id

    def __bool__(self) -> bool:  # pragma: no cover - indirect through expression evaluation
        return self.ok

    def __getitem__(self, key: Any) -> Any:
        if key in {"ok", "failed", "stdout", "data", "call_id"}:
            return getattr(self, key)
        raise KeyError(key)


def _evaluate_expression(expression: str, results: Mapping[str, "ToolResult"]) -> bool:
    expression = expression.strip()
    if not expression:
        return False

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    def result_lookup(step_id: Any) -> _ResultView:
        return _ResultView(results.get(str(step_id)))

    def contains(value: Any, needle: Any) -> bool:
        if value is None:
            return False
        try:
            haystack = "".join(value) if isinstance(value, (list, tuple, set)) else str(value)
        except Exception:  # pragma: no cover - defensive
            haystack = str(value)
        return str(needle) in haystack

    env_functions: Dict[str, Callable[..., Any]] = {
        "result": result_lookup,
        "contains": contains,
    }

    constants: Dict[str, Any] = {"True": True, "False": False, "None": None}

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.BoolOp):
            values = [_eval(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(_eval(node.operand))
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.In):
                    ok = left in right
                elif isinstance(op, ast.NotIn):
                    ok = left not in right
                elif isinstance(op, ast.Is):
                    ok = left is right
                elif isinstance(op, ast.IsNot):
                    ok = left is not right
                else:
                    raise ValueError
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError
            name = node.func.id
            func = env_functions.get(name)
            if func is None:
                raise ValueError
            args = [_eval(arg) for arg in node.args]
            kwargs = {kw.arg: _eval(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        if isinstance(node, ast.Attribute):
            value = _eval(node.value)
            if not hasattr(value, node.attr):
                raise ValueError
            return getattr(value, node.attr)
        if isinstance(node, ast.Subscript):
            value = _eval(node.value)
            index = _eval(node.slice)
            return value[index]
        if isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]
            raise ValueError
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
        raise ValueError

    try:
        return bool(_eval(tree.body))
    except (ValueError, KeyError, TypeError):
        return False


@dataclass
class PlanStep:
    """Node within a hierarchical plan."""

    id: UID
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    safety_level: str = "T0"
    description: Optional[str] = None
    goal: Optional[str] = None
    sub_steps: List["PlanStep"] = field(default_factory=list)
    branches: List["PlanBranch"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.args = dict(self.args)
        self.sub_steps = [_coerce_plan_step(step) for step in self.sub_steps]
        self.branches = [_coerce_plan_branch(branch) for branch in self.branches]

    def iter_tool_calls(self) -> Iterator["PlanStep"]:
        if self.tool:
            yield self
        for step in self.sub_steps:
            yield from step.iter_tool_calls()
        for branch in self.branches:
            for step in branch.steps:
                yield from step.iter_tool_calls()


@dataclass
class PlanBranch:
    """Conditional branch within a plan step."""

    condition: BranchCondition = field(default_factory=BranchCondition)
    steps: List[PlanStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.condition = BranchCondition.from_raw(self.condition)
        self.steps = [_coerce_plan_step(step) for step in self.steps]


@dataclass
class Plan:
    """Structured plan produced by the planner."""

    id: UID
    claim_ids: List[UID]
    steps: List[PlanStep]
    expected_cost: Dict[str, Optional[float]]
    risks: List[str]
    ablations: List[str]

    def __post_init__(self) -> None:
        self.steps = [_coerce_plan_step(step) for step in self.steps]

    def iter_tool_calls(self) -> Iterator[PlanStep]:
        for step in self.steps:
            yield from step.iter_tool_calls()


def _coerce_plan_step(step: Any) -> PlanStep:
    if isinstance(step, PlanStep):
        return step
    if isinstance(step, ToolCall):
        return PlanStep(
            id=step.id,
            tool=step.tool,
            args=dict(step.args),
            safety_level=step.safety_level,
        )
    if isinstance(step, dict):
        data = dict(step)
        return PlanStep(
            id=data["id"],
            tool=data.get("tool"),
            args=dict(data.get("args", {})),
            safety_level=data.get("safety_level", "T0"),
            description=data.get("description"),
            goal=data.get("goal"),
            sub_steps=[_coerce_plan_step(s) for s in data.get("sub_steps", [])],
            branches=[_coerce_plan_branch(b) for b in data.get("branches", [])],
        )
    raise TypeError(f"Unsupported plan step type: {step!r}")


def _coerce_plan_branch(branch: Any) -> PlanBranch:
    if isinstance(branch, PlanBranch):
        return branch
    if isinstance(branch, dict):
        data = dict(branch)
        return PlanBranch(
            condition=BranchCondition.from_raw(data.get("condition")),
            steps=[_coerce_plan_step(s) for s in data.get("steps", [])],
        )
    return PlanBranch(condition=BranchCondition.from_raw(branch))


@dataclass
class Belief:
    """Belief state for a specific claim."""

    claim_id: UID
    credence: float
    evidence: List[Source]
    last_updated: str


@dataclass
class Report:
    """Structured summary returned to callers after orchestration."""

    goal: str
    summary: str
    key_findings: List[str]
    belief_deltas: List[Belief]
    artifacts: List[str]


class Tool(Protocol):
    """Runtime contract for orchestrator tools."""

    name: str
    safety: str

    async def run(self, args: Dict[str, Any], ctx: "RunContext") -> ToolResult:  # pragma: no cover - interface
        ...


@dataclass
class RunContext:
    """Execution context provided to tools during a run."""

    working_dir: str
    timeout_s: int
    env_whitelist: List[str]
    network: str
    record_provenance: bool


__all__ = [
    "BranchCondition",
    "Belief",
    "Claim",
    "Plan",
    "PlanBranch",
    "PlanStep",
    "Prediction",
    "Report",
    "RunContext",
    "Source",
    "Tool",
    "ToolCall",
    "ToolResult",
    "UID",
]

