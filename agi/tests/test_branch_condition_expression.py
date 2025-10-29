from agi.src.core.types import BranchCondition, ToolResult


def _result(step_id: str, *, ok: bool, stdout: str | None = None, data=None) -> ToolResult:
    return ToolResult(call_id=step_id, ok=ok, stdout=stdout, data=data)


def test_expression_combines_success_and_stdout():
    results = {
        "step-1": _result("step-1", ok=True, stdout="value=42"),
        "step-2": _result("step-2", ok=False, stdout="error"),
    }
    condition = BranchCondition(kind="expression", value="result('step-1').ok and contains(result('step-1').stdout, 'value')")
    assert condition.evaluate(results) is True


def test_expression_accesses_structured_data():
    results = {
        "lookup": _result("lookup", ok=True, data={"score": 0.87, "label": "cat"}),
    }
    condition = BranchCondition(
        kind="expression",
        value="result('lookup').data['score'] > 0.5 and result('lookup').data['label'] == 'cat'",
    )
    assert condition.evaluate(results) is True


def test_expression_handles_unknown_step_safely():
    results = {}
    condition = BranchCondition(kind="expression", value="result('missing').ok")
    assert condition.evaluate(results) is False
