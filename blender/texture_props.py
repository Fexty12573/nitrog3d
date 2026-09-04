import bpy
from bpy.types import PropertyGroup
from bpy.props import EnumProperty, IntProperty, PointerProperty, StringProperty


FORMAT_ITEMS = [
    ("AUTO", "Auto", "Automatically determine a suitable format", 0),
    None,
    ("PLTT4", "2bpp", "2 bits per pixel, indexed (4 colors)", 2),
    ("PLTT16", "4bpp", "4 bits per pixel, indexed (16 colors)", 3),
    ("PLTT256", "8bpp", "8 bits per pixel, indexed (256 colors)", 4),
    ("A3I5", "A3I5", "3 bits alpha, 5 bits color, indexed", 1),
    ("A5I3", "A5I3", "5 bits alpha, 3 bits color, indexed", 6),
    ("DIRECT", "RGB5551", "15 bits RGB, 1 bit alpha", 7),
    ("COMP4X4", "Texeled", "4x4 Compressed Texels (Not writeable yet)", 5),
]


COLOR0_ITEMS = [
    ("AUTO", "Auto", "Reserve color 0 only if there are transparent pixels", -1),
    ("NORMAL", "Opaque", "Never reserve color 0 as transparent", 0),
    ("TRANSPARENT", "Transparent", "Reserve color 0 as transparent", 1),
]


class TextureProps(PropertyGroup):
    name: StringProperty(name="Name", description="The name of the texture", default="")

    width: IntProperty(name="Width", description="The width of the texture", default=0)

    height: IntProperty(
        name="Height", description="The height of the texture", default=0
    )

    format: EnumProperty(
        name="Format",
        description="The pixel format of the texture",
        items=FORMAT_ITEMS,
        default="AUTO",
    )

    color0: EnumProperty(
        name="Color 0 Behavior",
        description="Whether palette color 0 is transparent or not",
        items=COLOR0_ITEMS,
        default="AUTO",
    )


def register():
    bpy.utils.register_class(TextureProps)
    bpy.types.Image.nitro = PointerProperty(type=TextureProps)


def unregister():
    del bpy.types.Image.nitro
    bpy.utils.unregister_class(TextureProps)
