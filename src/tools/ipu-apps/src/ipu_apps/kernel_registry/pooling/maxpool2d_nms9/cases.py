"""Runnable cases for maxpool2d_nms9."""
from . import App
from ipu_apps.kernel_registry.pooling.window_cases import make_cases

CASES = make_cases(App, 9)
