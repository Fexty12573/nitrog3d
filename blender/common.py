
from mathutils import Matrix
from bpy_extras.io_utils import axis_conversion


def global_matrix(convert_axis: bool) -> Matrix:
    if convert_axis:
        return axis_conversion(
            from_forward="-Z",
            from_up="Y",
            to_forward="-Y",
            to_up="Z"
        ).to_4x4()

    return Matrix.Identity(4)
