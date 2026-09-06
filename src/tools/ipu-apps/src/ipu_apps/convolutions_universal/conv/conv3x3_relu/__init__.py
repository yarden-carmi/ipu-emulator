"""conv3x3_relu memory-only harness and registry declaration."""
from ipu_apps.convolutions_universal.conv._memory import ConvApp
from ipu_apps.kernel_registry.memory import memory_spec


class App(ConvApp):
    kernel_size = 3
    single_channel = False


SPEC = memory_spec("conv3x3_relu", "conv2d", App,
                   ("shape", "out_channels", "kernel_size", "stride", "padding", "activation"), cost=1)
