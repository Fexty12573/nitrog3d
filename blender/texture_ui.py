import bpy
import bpy.utils.previews
import numpy as np
from bpy.props import BoolProperty, EnumProperty
from bpy.types import Context, Operator

from .common import image_to_rgba, is_nds_dim
from .texture_props import COLOR0_ITEMS, FORMAT_ITEMS
from ..nitro import tex_encoder as texenc
from ..nitro.tex0 import TexFmt


_previews = None

_BPP = {
    TexFmt.PLTT4: 2,
    TexFmt.PLTT16: 4,
    TexFmt.PLTT256: 8,
    TexFmt.A3I5: 8,
    TexFmt.A5I3: 8,
    TexFmt.DIRECT: 16,
}


def _resolve_format(rgba: list[int], w: int, h: int, fmt: str) -> TexFmt:
    if fmt != "AUTO":
        return TexFmt[fmt]
    return texenc.analyze_texture(rgba, w, h).suggested


def _quantize_rgba(rgba: list[int], w: int, h: int, fmt: str, color0: str) -> list[int]:
    format = _resolve_format(rgba, w, h, fmt)
    transparent = format.is_pltt_n() and any(a < 128 for a in rgba[3::4])
    return texenc.quantize_rgba(
        rgba,
        w,
        h,
        format,
        color0_transparent=(
            color0 == "TRANSPARENT" or (color0 == "AUTO" and transparent)
        ),
    )


def _make_preview(quantized: list[int], w: int, h: int) -> int:
    # First pad the image to a square because icons can only be square :(
    side = max(w, h)
    buf = np.zeros((side, side, 4), dtype=np.float32)
    q = np.asarray(quantized, dtype=np.float32).reshape(h, w, 4) / 255.0

    y0, x0 = (side - h) // 2, (side - w) // 2
    buf[y0 : y0 + h, x0 : x0 + w] = q

    _previews.clear()
    entry = _previews.new("nitro")
    entry.image_size = (side, side)
    entry.image_pixels_float.foreach_set(buf[::-1].ravel())
    return entry.icon_id


class NITROG3D_OT_convert(Operator):
    bl_idname = "nitrog3d.convert"
    bl_label = "Convert to NITRO"
    bl_options = {"REGISTER", "INTERNAL"}

    format: EnumProperty(
        name="Format",
        description="The pixel format of the texture",
        items=FORMAT_ITEMS,
        default="AUTO",
    )

    color0: EnumProperty(
        name="Color 0 Mode",
        description="The way in which palette color 0 behaves",
        items=COLOR0_ITEMS,
        default="AUTO",
    )

    @classmethod
    def poll(cls, context: Context):
        return getattr(context.space_data, "image", None) is not None

    def invoke(self, context: Context, event):
        img = context.space_data.image
        w, h = img.size
        if not is_nds_dim(w) or not is_nds_dim(h):
            self.report({"ERROR"}, f"Size {w}x{h} is not a valid DS texture size")
            return {"CANCELLED"}

        self.format = img.nitro.format
        self.color0 = img.nitro.color0

        self.ensure_state(img)
        self.rebuild_preview()

        return context.window_manager.invoke_popup(self, width=340)

    def ensure_state(self, img: bpy.types.Image):
        if getattr(self, "_image", None) == img.name:
            return

        w, h = img.size
        self._image: str = img.name
        self._size: tuple[int, int] = (w, h)
        self._rgba: list[int] = list(image_to_rgba(img))
        self._stats = texenc.analyze_texture(self._rgba, w, h)
        self._sig: tuple[str, str] | None = None
        self._icon = 0
        self._error = ""

    def rebuild_preview(self):
        w, h = self._size
        self._sig = (self.format, self.color0)
        try:
            self._icon = _make_preview(
                _quantize_rgba(self._rgba, w, h, self.format, self.color0), w, h
            )
            self._error = ""
        except NotImplementedError as e:
            self._icon = 0
            self._error = str(e)

    def draw(self, context: Context):
        layout = self.layout
        # invoke_popup() has no title parameter, so the header is ours to draw.
        layout.label(text="Convert to NITRO")
        layout.separator()
        layout.prop(self, "format")
        layout.prop(self, "color0")

        img = getattr(context.space_data, "image", None)
        if img is None:
            layout.label(text="No image in this editor", icon="ERROR")
            return

        self.ensure_state(img)
        if self._sig != (self.format, self.color0):
            self.rebuild_preview()

        layout.separator()
        if self._error:
            box = layout.box()
            box.alert = True
            box.label(text=self._error, icon="ERROR")
        else:
            layout.template_icon(icon_value=self._icon, scale=12)

        layout.label(text=self._stats_text())
        layout.separator()

        row = layout.row()
        row.operator_context = "EXEC_DEFAULT"
        for label, bake in (("Apply on Export", False), ("Apply Now", True)):
            op = row.operator("nitrog3d.convert_apply", text=label)
            op.bake = bake
            op.format = self.format
            op.color0 = self.color0

    def _stats_text(self) -> str:
        w, h = self._size
        fmt = _resolve_format(self._rgba, w, h, self.format)
        label = fmt.name if self.format != "AUTO" else f"Auto → {fmt.name}"
        parts = [
            label,
            f"{self._stats.ncolors555} colors",
            f"{self._stats.alpha_kind} alpha",
        ]
        bpp = _BPP.get(fmt)
        if bpp is not None:
            parts.append(f"{w * h * bpp / 8 / 1024:.1f} KB")
        return " · ".join(parts)

    def execute(self, context: Context):
        return {"FINISHED"}


class NITROG3D_OT_convert_apply(Operator):
    bl_idname = "nitrog3d.convert_apply"
    bl_label = "Apply"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    format: EnumProperty(items=FORMAT_ITEMS, default="AUTO", options={"HIDDEN"})
    color0: EnumProperty(items=COLOR0_ITEMS, default="AUTO", options={"HIDDEN"})
    bake: BoolProperty(default=False, options={"HIDDEN"})

    @classmethod
    def poll(cls, context: Context):
        return getattr(context.space_data, "image", None) is not None

    def execute(self, context: Context):
        img = context.space_data.image
        w, h = img.size

        img.nitro.format = self.format
        img.nitro.color0 = self.color0

        if not self.bake:
            self.report({"INFO"}, f"{img.name}: will export as {self.format}")
            return {"FINISHED"}

        rgba = list(image_to_rgba(img))
        try:
            quantized = _quantize_rgba(rgba, w, h, self.format, self.color0)
        except NotImplementedError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        px = np.asarray(quantized, dtype=np.float32).reshape(h, w * 4)[::-1]
        img.pixels.foreach_set((px / 255.0).ravel())
        img.update()
        img.pack()

        self.report({"INFO"}, f"{img.name}: baked as {self.format}")
        return {"FINISHED"}


classes = [
    NITROG3D_OT_convert,
    NITROG3D_OT_convert_apply,
]


def _menu(self, context):
    self.layout.operator("nitrog3d.convert", text="Convert to NITRO")


def register():
    global _previews
    _previews = bpy.utils.previews.new()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.IMAGE_MT_image.append(_menu)


def unregister():
    global _previews
    bpy.types.IMAGE_MT_image.remove(_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.utils.previews.remove(_previews)
    _previews = None
