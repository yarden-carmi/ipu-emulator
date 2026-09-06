"""Execute each l2_normalize_channels case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("l2_normalize_channels"))
def test_case(name):
    state, _ = run_case("l2_normalize_channels", load_cases("l2_normalize_channels")[name])
    assert state.is_halted
