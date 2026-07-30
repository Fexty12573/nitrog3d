import bpy
import traceback
from bpy.types import Operator, OperatorFileListElement
from bpy.props import StringProperty, CollectionProperty, BoolProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper
from pathlib import Path
from . import importer


class IMPORT_SCENE_OT_nsbmd(Operator, ImportHelper):
    """
    Import a Nintendo DS NSBMD model.
    """

    bl_idname = "import_scene.nsbmd"
    bl_label = "Import NSBMD"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".nsbmd"
    filter_glob: StringProperty(default="*.nsbmd", options={"HIDDEN"})

    files: CollectionProperty(name="File Path", type=OperatorFileListElement)
    directory: StringProperty(subtype="DIR_PATH")

    convert_axis: BoolProperty(
        name="Y-up to Z-up",
        description="Rotate the model from Y-up to Z-up",
        default=True
    )

    flip_uv: BoolProperty(
        name="Flip UV V",
        description="Flip texture coordinates vertically",
        default=True
    )

    import_textures: BoolProperty(
        name="Import Textures",
        description="Import textures embedded in the NSBMD",
        default=True
    )

    create_armature: BoolProperty(
        name="Create Armature",
        description="Build an armature from the joint tree and bind meshes to it",
        default=True
    )

    bone_length: FloatProperty(
        name="Bone Length",
        description="Display length for generated bones",
        default=0.1, min=0.01, soft_max=10.0
    )

    def execute(self, context):
        if not self.directory:
            files = [Path(self.filepath)]
        else:
            dir = Path(self.directory)
            files = [dir / str(file.name) for file in self.files]

        nsbmds = list(filter(lambda f: f.suffix == ".nsbmd", files))

        objs = tris = 0
        for file in nsbmds:
            try:
                created, model = importer.import_nsbmd(
                    context,
                    str(file),
                    importer.ImportOptions(
                        self.convert_axis,
                        self.flip_uv,
                        self.import_textures,
                        self.create_armature,
                        self.bone_length,
                    )
                )
            except Exception as e:
                traceback.print_exc()
                self.report(
                    {"ERROR"},
                    f"NSBMD import failed ({file.name}): {e}"
                )

                return {"CANCELLED"}

            objs += len(created)
            tris += sum(len(mesh.faces)
                        for sub in model.models for mesh in sub.meshes)

        self.report(
            {"INFO"},
            f"Imported {len(nsbmds)} model(s): {objs} object(s), {tris} triangle(s)"
        )
        return {"FINISHED"}


def _menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_nsbmd.bl_idname,
                         text="Nitro Model (.nsbmd)")


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_nsbmd)
    bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_nsbmd)
