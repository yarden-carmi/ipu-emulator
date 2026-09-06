"""Execute each maxpool2d_nms7 case and verify its output."""
import pytest
from ipu_apps.kernel_registry.cases import load_cases, run_case


@pytest.mark.parametrize("name", load_cases("maxpool2d_nms7"))
def test_case(name):
    state, _ = run_case("maxpool2d_nms7", load_cases("maxpool2d_nms7")[name])
    assert state.is_halted
