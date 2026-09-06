"""maxpool2d_window memory-only harness and registry declaration."""
from ipu_apps.kernel_registry.pooling._window import WindowPoolApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(WindowPoolApp):
    fixed_kernel = None


SPEC = memory_spec("maxpool2d_window", "maxpool2d", App,
                   ("shape", "kernel_size", "stride", "padding"), cost=1)
