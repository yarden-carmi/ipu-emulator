"""Memory geometry shared by centred max-pool kernels."""
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape


class WindowPoolApp(MemoryApp):
    fixed_kernel = None

    @classmethod
    def memory_layout(cls, *, shape, kernel_size, stride, padding):
        c, h, w = positive_shape(shape, 3)
        k = kernel_size
        if type(k) is not int or k < 1 or k > 127 or k % 2 != 1:
            raise ValueError("kernel_size must be odd and between 1 and 127")
        if stride != 1 or padding != k // 2:
            raise ValueError("requires stride=1 and padding=kernel_size//2")
        if cls.fixed_kernel is not None and k != cls.fixed_kernel:
            raise ValueError(f"requires kernel_size={cls.fixed_kernel}")
        tiles = ceildiv(w, 129 - k)
        plane = (h + k - 1) * tiles
        rows = c * plane
        general = cls.fixed_kernel is None
        output = rows + int(general)
        crs = {2: 0, 3: output, 4: rows if general else 128,
               5: tiles, 6: plane, 7: h, 8: c}
        if general:
            crs.update({9: k, 10: 384, 11: k - 1})
        return MemoryLayout(output, c * h * tiles, crs)
