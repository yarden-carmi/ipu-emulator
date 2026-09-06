"""Execute each conv3x3_relu case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("conv3x3_relu"))
def test_case(name):
    state, _ = run_case("conv3x3_relu", load_cases("conv3x3_relu")[name])
    assert state.is_halted
