import bpy
import numpy as np
from mathutils import Matrix
from bpy_extras.io_utils import axis_conversion


def global_matrix(convert_axis: bool) -> Matrix:
    if convert_axis:
        return axis_conversion(
            from_forward="-Z", from_up="Y", to_forward="-Y", to_up="Z"
        ).to_4x4()

    return Matrix.Identity(4)


def is_nds_dim(dim: int) -> bool:
    return 8 <= dim <= 1024 and dim & (dim - 1) == 0


def image_to_rgba(img: bpy.types.Image) -> bytes:
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)

    img.pixels.foreach_get(buf)
    px = (np.clip(buf, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    
    return px.reshape(h, w * 4)[::-1].tobytes()
