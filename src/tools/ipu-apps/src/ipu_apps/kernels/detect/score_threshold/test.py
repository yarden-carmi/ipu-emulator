"""Execute each score_threshold case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("score_threshold"))
def test_case(name):
    state, _ = run_case("score_threshold", load_cases("score_threshold")[name])
    assert state.is_halted
