"""Runnable cases for conv3x3_relu."""
from . import App
from ipu_apps.kernels.convolutions.conv.case_support import make_cases

CASES = make_cases(App)
