"""Execute each maxpool2d_window case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("maxpool2d_window"))
def test_case(name):
    state, _ = run_case("maxpool2d_window", load_cases("maxpool2d_window")[name])
    assert state.is_halted
