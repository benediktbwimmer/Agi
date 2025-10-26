from __future__ import annotations

import ast
import operator
from typing import Any, Dict

from ..types import RunContext, ToolResult


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class Calculator:
    name = "calculator"
    safety = "T0"

    async def run(self, args: Dict[str, Any], ctx: RunContext) -> ToolResult:  # pragma: no cover - trivial
        expr = args.get("expression")
        if not isinstance(expr, str):
            raise ValueError("'expression' must be provided")
        value = _eval_expr(ast.parse(expr, mode="eval").body)
        return ToolResult(call_id=args.get("id", "calculator"), ok=True, stdout=str(value), provenance=[])


def _eval_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("Unsupported expression")
