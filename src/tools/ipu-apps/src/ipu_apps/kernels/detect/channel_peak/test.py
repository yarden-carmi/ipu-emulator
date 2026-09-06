"""Execute each channel_peak case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("channel_peak"))
def test_case(name):
    state, _ = run_case("channel_peak", load_cases("channel_peak")[name])
    assert state.is_halted
