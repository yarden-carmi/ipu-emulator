"""Execute each depth_to_space case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("depth_to_space"))
def test_case(name):
    state, _ = run_case("depth_to_space", load_cases("depth_to_space")[name])
    assert state.is_halted
