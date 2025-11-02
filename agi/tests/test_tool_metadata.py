from agi.src.core.tools import ToolCapability, ToolSpec, describe_tool
from agi.src.core.tools.calculator import Calculator


class NoDescribeTool:
    safety = "t1"


def test_calculator_provides_structured_spec():
    spec = describe_tool(Calculator())
    assert isinstance(spec, ToolSpec)
    assert spec.name == "calculator"
    assert spec.safety_tier == "T0"
    assert spec.capabilities
    capability = spec.capabilities[0]
    assert isinstance(capability, ToolCapability)
    assert any(param.name == "expression" for param in capability.parameters)


def test_describe_tool_synthesises_metadata_from_attributes():
    spec = describe_tool(NoDescribeTool(), override_name="alias")
    assert spec.name == "alias"
    assert spec.safety_tier == "T1"
    assert spec.description
    assert spec.capabilities[0].safety_tier == "T1"
