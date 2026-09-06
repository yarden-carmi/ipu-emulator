"""L2 normalization of columns in a preformatted (channels, columns) matrix."""
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape, memory_spec


class App(MemoryApp):
    @staticmethod
    def memory_layout(*, shape):
        channels, columns = positive_shape(shape, 2)
        tiles = ceildiv(columns, 128)
        rows = channels * tiles
        return MemoryLayout(rows + 1, rows, {2: 0, 3: rows + 1, 4: rows, 5: tiles, 6: channels})


SPEC = memory_spec("l2_normalize_channels", "l2_normalize", App, ("shape",))
