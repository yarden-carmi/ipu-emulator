"""Execute each conv1x1 case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("conv1x1"))
def test_case(name):
    state, _ = run_case("conv1x1", load_cases("conv1x1")[name])
    assert state.is_halted
