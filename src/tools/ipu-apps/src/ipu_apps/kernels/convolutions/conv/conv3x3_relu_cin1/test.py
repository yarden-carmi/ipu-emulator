"""Execute each conv3x3_relu_cin1 case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("conv3x3_relu_cin1"))
def test_case(name):
    state, _ = run_case("conv3x3_relu_cin1", load_cases("conv3x3_relu_cin1")[name])
    assert state.is_halted
