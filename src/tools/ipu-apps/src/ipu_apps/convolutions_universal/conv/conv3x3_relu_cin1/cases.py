"""Runnable cases for conv3x3_relu_cin1."""
from . import App
from ipu_apps.convolutions_universal.conv.case_support import make_cases

CASES = make_cases(App)
