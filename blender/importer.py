
import bpy
import numpy as np
from bpy_extras.io_utils import axis_conversion
from dataclasses import dataclass
from mathutils import Matrix, Vector
from ..nitro import model as mdl
from ..nitro.mdl0 import CullMode
from .common import global_matrix


@dataclass
class ImportOptions:
    convert_axis: bool = True
    flip_uv: bool = True
    import_textures: bool = True
    create_armature: bool = True
    bone_length: float = 0.1


def import_nsbmd(context: bpy.types.Context, filepath: str, opts: ImportOptions = ImportOptions()):
    with open(filepath, "rb") as f:
        data = f.read()

    model = mdl.load(data)
    global_mtx = global_matrix(opts.convert_axis)
    image_cache: dict[str, bpy.types.Image] = {}
    created: list[bpy.types.Object] = []

    if context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set("OBJECT")
        except RuntimeError:
            pass

    collection = context.collection
    for smi, sub in enumerate(model.models):
        bl_materials: list[bpy.types.Material] = []
        name_to_material: dict[str, bpy.types.Material] = {}

        for imat in sub.materials:
            if not opts.import_textures:
                imat.texture = None

            bmat = _make_material(imat, image_cache)
            bmat["nsbmd_model"] = smi
            bmat["nsbmd_name"] = imat.name

            bl_materials.append(bmat)
            name_to_material[imat.name] = bmat

        arm_obj: bpy.types.Armature | None = None
        if opts.create_armature and sub.bones:
            arm_obj = _build_armature(
                sub, global_mtx, opts.bone_length, collection)
            created.append(arm_obj)

        anchor = arm_obj
        for mi, mesh in enumerate(sub.meshes):
            obj = _build_mesh_object(
                sub,
                mesh,
                mi,
                smi,
                global_mtx,
                bl_materials,
                opts.flip_uv,
                collection
            )
            if arm_obj is not None:
                _bind_to_armature(obj, mesh, sub, arm_obj)

            created.append(obj)
            if anchor is None:
                anchor = obj

        if anchor is not None:
            anchor["nsbmd_marker"] = "NSBMD"

    return created, model


def _make_material(imat: mdl.ImportedMaterial, image_cache: dict[str, bpy.types.Image]) -> bpy.types.Material:
    mat = bpy.data.materials.new(imat.name)
    mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    r, g, b = imat.diffuse
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0

    if imat.texture is not None:
        img = _make_image(imat.texture, image_cache)
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (-300, 0)
        tex_node.image = img
        tex_node.interpolation = "Closest"
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        if imat.texture.has_alpha:
            nt.links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.blend_method = "HASHED"
    elif imat.alpha < 1.0:
        mat.blend_method = "HASHED"

    if imat.alpha < 1.0 and "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = imat.alpha

    mat.use_backface_culling = (
        imat.polygon_attr is not None
        and imat.polygon_attr.cull_mode == CullMode.BACK
    )
    return mat


def _make_image(tex: mdl.DecodedTexture, image_cache: dict[str, bpy.types.Image]) -> bpy.types.Image:
    if tex.name in image_cache:
        return image_cache[tex.name]

    img = bpy.data.images.new(tex.name, tex.width, tex.height, alpha=True)
    w, h = tex.width, tex.height
    src = tex.rgba

    pixels = [0.0] * (w * h * 4)
    inv = 1.0 / 255.0
    for y in range(h):
        src_row = (h - 1 - y) * w * 4
        dst_row = y * w * 4
        for x in range(w * 4):
            pixels[dst_row + x] = src[src_row + x] * inv

    img.pixels[:] = pixels
    img.pack()
    img.use_fake_user = True

    image_cache[tex.name] = img
    return img


def _build_armature(sub: mdl.ImportedSubModel, global_mtx: Matrix, bone_length: float, collection: bpy.types.Collection) -> bpy.types.Armature:
    arm_data = bpy.data.armatures.new(sub.name + "_Armature")
    arm_obj = bpy.data.objects.new(sub.name + "_Armature", arm_data)
    collection.objects.link(arm_obj)

    view_layer = bpy.context.view_layer
    view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones: list[bpy.types.EditBone] = []
    for bone in sub.bones:
        eb = arm_data.edit_bones.new(bone.name)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, bone_length, 0.0)
        eb.inherit_scale = 'NONE' if bone.scale_compensate else 'FULL'
        full = global_mtx @ _ds_matrix_to_blender(bone.world_mtx)
        loc, rot, _ = full.decompose()
        eb.matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        edit_bones.append(eb)

    for i, bone in enumerate(sub.bones):
        if 0 <= bone.parent < len(edit_bones) and bone.parent != i:
            edit_bones[i].parent = edit_bones[bone.parent]

    bpy.ops.object.mode_set(mode="OBJECT")

    for db in arm_data.bones:
        db["nsbmd_rest"] = [db.matrix_local[i][j]
                            for i in range(4) for j in range(4)]

    return arm_obj


def _build_mesh_object(sub: mdl.ImportedSubModel,
                       mesh: mdl.ImportedMesh,
                       mesh_idx: int,
                       model_idx: int,
                       global_mtx: Matrix,
                       bl_materials: list,
                       flip_uv: bool,
                       collection: bpy.types.Collection) -> bpy.types.Object:
    me = bpy.data.meshes.new(f"{sub.name}_{mesh.name}_{mesh_idx}")
    verts = [tuple(global_mtx @ Vector(v)) for v in mesh.vertices]
    me.from_pydata(verts, [], mesh.faces)

    if mesh.has_uv and mesh.loop_uvs:
        uvl = me.uv_layers.new(name="UVMap")
        for i, uv in enumerate(mesh.loop_uvs):
            u, v = uv
            uvl.data[i].uv = (u, 1.0 - v if flip_uv else v)

    if mesh.has_colors and mesh.loop_colors:
        ca = me.color_attributes.new(
            name="Color", type="BYTE_COLOR", domain="CORNER")
        for i, col in enumerate(mesh.loop_colors):
            ca.data[i].color = (*col, 1.0)

    me.update()
    me.validate(clean_customdata=False)

    if mesh.has_normals and mesh.loop_normals and len(mesh.loop_normals) == len(me.loops):
        gl3 = global_mtx.to_3x3()
        normals = []
        for n in mesh.loop_normals:
            nv = (gl3 @ Vector(n))
            if nv.length > 1e-8:
                nv.normalize()
            normals.append(tuple(nv))

        try:
            me.normals_split_custom_set(normals)
        except (RuntimeError, ValueError):
            pass

    if 0 <= mesh.material < len(bl_materials):
        me.materials.append(bl_materials[mesh.material])

    obj = bpy.data.objects.new(me.name, me)
    obj["nsbmd_model"] = model_idx
    obj["nsbmd_shape"] = mesh_idx
    obj["nsbmd_name"] = mesh.name

    collection.objects.link(obj)

    return obj


def _bind_to_armature(obj: bpy.types.Object, mesh: mdl.ImportedMesh, sub: mdl.ImportedSubModel, arm_obj: bpy.types.Object):
    groups: dict[str, bpy.types.VertexGroup] = {}
    for vi, bone_idx in enumerate(mesh.vertex_bone):
        if not (0 <= bone_idx < len(sub.bones)):
            continue

        name = sub.bones[bone_idx].name
        vg = groups.get(name)
        if vg is None:
            vg = obj.vertex_groups.new(name=name)
            groups[name] = vg
        vg.add([vi], 1.0, "REPLACE")

    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    obj.parent = arm_obj


def _ds_matrix_to_blender(mat: np.ndarray) -> Matrix:
    return Matrix((mat[0, :], mat[1, :], mat[2, :], mat[3, :])).transposed()
