"""conv1x1 memory-only harness and registry declaration."""
from ipu_apps.convolutions_universal.conv._memory import ConvApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(ConvApp):
    kernel_size = 1
    single_channel = False


SPEC = memory_spec("conv1x1", "conv2d", App,
                   ("shape", "out_channels", "kernel_size", "stride", "padding", "activation"), cost=1)
