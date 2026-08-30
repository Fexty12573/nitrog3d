from __future__ import annotations

import bpy
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, field
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix

from ..nitro import model as mdl
from ..nitro.mdl0 import MDL0, CullMode, PolygonAttr
from ..nitro.model_builder import ModelBuilder
from ..nitro.nsbmd import NSBMD
from ..nitro.tex0 import TexImageParam
from .common import global_matrix

MAX_NAME_LEN = 16


@dataclass
class ExportOptions:
    convert_axis: bool = True
    flip_uv: bool = True
    use_selection: bool = False
    use_active_collection: bool = False
    visible_only: bool = False
    apply_modifiers: bool = True
    model_name: str = ""


@dataclass
class ExportResult:
    submodels: int = 0
    meshes: int = 0
    triangles: int = 0
    materials: int = 0
    warnings: list[str] = field(default_factory=list)


def export_nsbmd(context: bpy.types.Context,
                 filepath: str,
                 opts: ExportOptions = ExportOptions()) -> ExportResult:
    if context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass

    warnings: list[str] = []
    ir = scene_to_ir(context, opts, warnings)
    if not ir.models:
        raise ValueError("Nothing to export")

    data = build_nsbmd(ir)
    with open(filepath, "wb") as f:
        f.write(data)

    return ExportResult(
        submodels=len(ir.models),
        meshes=sum(len(s.meshes) for s in ir.models),
        triangles=sum(len(m.faces) for s in ir.models for m in s.meshes),
        materials=sum(len(s.materials) for s in ir.models),
        warnings=warnings,
    )


def scene_to_ir(context: bpy.types.Context,
                opts: ExportOptions,
                warnings: list[str]) -> mdl.ImportedModel:
    to_ds = global_matrix(opts.convert_axis).inverted()

    groups = _group_by_armature(_collect_objects(context, opts))
    result = mdl.ImportedModel()

    with _rest_pose(context, [arm for arm, _ in groups if arm is not None]):
        depsgraph = context.evaluated_depsgraph_get()
        for arm, objects in groups:
            sub = _build_submodel(
                arm,
                objects,
                to_ds,
                depsgraph,
                opts,
                warnings
            )

            if sub.meshes:
                result.models.append(sub)
            else:
                warnings.append(f"{sub.name}: no geometry, skipped")

    return result


def build_nsbmd(model: mdl.ImportedModel, stash: bytes | None = None) -> bytes:
    if stash is not None:
        # TODO: Re-parse stash and re-emit the parts whose Blender counterparts are unedited
        raise NotImplementedError(
            "stash-and-regenerate is not implemented yet")

    names = _NameAllocator()
    models = {
        names.take(sub.name): ModelBuilder(sub).build() for sub in model.models
    }

    # TODO: Textures
    return NSBMD.build(MDL0.build(models)).write()


def _collect_objects(context: bpy.types.Context,
                     opts: ExportOptions) -> list[bpy.types.Object]:
    if opts.use_selection:
        pool = context.selected_objects
    elif opts.use_active_collection:
        pool = context.collection.all_objects
    else:
        pool = context.scene.objects

    return [
        obj for obj in pool
        if obj.type == "MESH" and (not opts.visible_only or obj.visible_get())
    ]


def _group_by_armature(
    objects: list[bpy.types.Object],
) -> list[tuple[bpy.types.Object | None, list[bpy.types.Object]]]:
    groups: dict[str | None,
                 tuple[bpy.types.Object | None, list[bpy.types.Object]]] = {}
    for obj in objects:
        arm = _armature_of(obj)
        key = arm.name if arm is not None else None
        groups.setdefault(key, (arm, []))[1].append(obj)

    ordered = [groups[k] for k in sorted(k for k in groups if k is not None)]
    if None in groups:
        ordered.append(groups[None])
    return ordered


def _armature_of(obj: bpy.types.Object) -> bpy.types.Object | None:
    for m in obj.modifiers:
        if m.type == "ARMATURE" and m.object is not None:
            return m.object
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    return None


@contextmanager
def _rest_pose(context: bpy.types.Context, armatures: list[bpy.types.Object]):
    """
    Avoid baking poses into the exported geometry by temporarily
    switching all armatures to REST pose.
    """

    saved = [(a, a.data.pose_position) for a in armatures]
    for arm, _ in saved:
        arm.data.pose_position = "REST"
    if saved:
        context.view_layer.update()

    try:
        yield
    finally:
        for arm, previous in saved:
            arm.data.pose_position = previous
        if saved:
            context.view_layer.update()


def _build_submodel(arm: bpy.types.Object | None,
                    objects: list[bpy.types.Object],
                    to_ds: Matrix,
                    depsgraph: bpy.types.Depsgraph,
                    opts: ExportOptions,
                    warn: list[str]) -> mdl.ImportedSubModel:
    sub = mdl.ImportedSubModel(_submodel_name(arm, objects, opts))

    node_names = _NameAllocator()
    if arm is not None:
        sub.bones, bone_index = _build_bones(arm, to_ds, node_names)
    else:
        sub.bones = [mdl.Bone(node_names.take(sub.name),
                              -1,
                              np.identity(4),
                              np.identity(4))]
        bone_index = {}

    materials = _MaterialTable()
    shape_names = _NameAllocator()
    for obj in objects:
        sub.meshes.extend(_extract_meshes(
            obj, to_ds, depsgraph, opts, bone_index,
            materials, shape_names, warn
        ))

    sub.materials = materials.materials
    return sub


def _submodel_name(arm: bpy.types.Object | None,
                   objects: list[bpy.types.Object],
                   opts: ExportOptions) -> str:
    if opts.model_name:
        return opts.model_name
    if arm is not None:
        suffix = "_Armature"
        name = arm.name
        return name[:-len(suffix)] if name.endswith(suffix) else name
    return objects[0].name if objects else "model"


def _build_bones(arm: bpy.types.Object,
                 to_ds: Matrix,
                 names: _NameAllocator) -> tuple[list[mdl.Bone], dict[str, int]]:
    roots = [b for b in arm.data.bones if b.parent is None]
    if len(roots) != 1:
        # TODO: Support multiple root bones
        raise ValueError(
            f"{arm.name}: an NSBMD node tree needs exactly one root bone, "
            f"found {len(roots)} ({', '.join(b.name for b in roots) or 'none'})"
        )

    ordered = _preorder(roots[0])
    index = {b.name: i for i, b in enumerate(ordered)}

    bones: list[mdl.Bone] = []
    for b in ordered:
        world = _blender_matrix_to_ds(
            to_ds @ arm.matrix_world @ b.matrix_local)
        direction = world.copy()
        direction[3, :3] = 0.0
        bones.append(mdl.Bone(
            name=names.take(b.name),
            parent=index[b.parent.name] if b.parent is not None else -1,
            world_mtx=world,
            world_dir_mtx=direction,
            scale_compensate=(b.inherit_scale == "NONE"),
        ))
    return bones, index


def _preorder(root: bpy.types.Bone) -> list[bpy.types.Bone]:
    out: list[bpy.types.Bone] = []

    def walk(bone: bpy.types.Bone):
        out.append(bone)
        for child in bone.children:
            walk(child)

    walk(root)
    return out


class _Bucket:
    def __init__(self):
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []
        self.vertex_bone: list[int] = []
        self.uvs: list[tuple[float, float]] = []
        self.normals: list[tuple[float, float, float]] = []
        self.colors: list[tuple[float, float, float]] = []
        self.vmap: dict[int, int] = {}


def _extract_meshes(obj: bpy.types.Object,
                    to_ds: Matrix,
                    depsgraph: bpy.types.Depsgraph,
                    opts: ExportOptions,
                    bone_index: dict[str, int],
                    materials: _MaterialTable,
                    shape_names: _NameAllocator,
                    warn: list[str]) -> list[mdl.ImportedMesh]:
    source = obj.evaluated_get(depsgraph) if opts.apply_modifiers else obj
    try:
        me = source.to_mesh()
    except RuntimeError:
        warn.append(f"{obj.name}: could not be evaluated to a mesh, skipped")
        return []

    try:
        return _extract(obj, me, to_ds, opts, bone_index,
                        materials, shape_names, warn)
    finally:
        source.to_mesh_clear()


def _extract(obj: bpy.types.Object,
             me: bpy.types.Mesh,
             to_ds: Matrix,
             opts: ExportOptions,
             bone_index: dict[str, int],
             materials: _MaterialTable,
             shape_names: _NameAllocator,
             warn: list[str]) -> list[mdl.ImportedMesh]:
    me.calc_loop_triangles()
    if not me.loop_triangles:
        return []

    mtx = to_ds @ obj.matrix_world
    normal_mtx = mtx.to_3x3().inverted_safe().transposed()
    # A mirrored object flips winding
    flip_winding = mtx.determinant() < 0.0

    uv_layer = me.uv_layers.active
    color_layer = me.color_attributes.active_color
    has_uv = uv_layer is not None
    has_colors = color_layer is not None
    color_per_corner = has_colors and color_layer.domain == "CORNER"

    vertex_bone = _vertex_bone_map(obj, me, bone_index, warn)
    buckets: dict[int, _Bucket] = {}

    for tri in me.loop_triangles:
        if len(set(tri.vertices)) < 3:
            continue

        bucket = buckets.setdefault(tri.material_index, _Bucket())
        corners = list(zip(tri.vertices, tri.loops))
        if flip_winding:
            corners.reverse()

        face: list[int] = []
        for vi, _ in corners:
            idx = bucket.vmap.get(vi)
            if idx is None:
                idx = len(bucket.vertices)
                bucket.vmap[vi] = idx
                co = mtx @ me.vertices[vi].co
                bucket.vertices.append((co.x, co.y, co.z))
                bucket.vertex_bone.append(vertex_bone[vi])
            face.append(idx)
        bucket.faces.append(tuple(face))

        for vi, li in corners:
            n = (normal_mtx @ _corner_normal(me, li)).normalized()
            bucket.normals.append((n.x, n.y, n.z))
            if has_uv:
                u, v = uv_layer.data[li].uv
                bucket.uvs.append((u, 1.0 - v if opts.flip_uv else v))
            if has_colors:
                c = color_layer.data[li if color_per_corner else vi].color
                bucket.colors.append((c[0], c[1], c[2]))

    return [
        _bucket_to_mesh(obj, me, slot, bucket, len(buckets),
                        has_uv, has_colors, materials, shape_names)
        for slot, bucket in sorted(buckets.items())
    ]


def _bucket_to_mesh(obj: bpy.types.Object,
                    me: bpy.types.Mesh,
                    slot: int,
                    bucket: _Bucket,
                    slot_count: int,
                    has_uv: bool,
                    has_colors: bool,
                    materials: _MaterialTable,
                    shape_names: _NameAllocator) -> mdl.ImportedMesh:
    bl_mat = me.materials[slot] if slot < len(me.materials) else None
    return mdl.ImportedMesh(
        name=shape_names.take(_shape_name(obj, slot, slot_count)),
        vertices=bucket.vertices,
        faces=bucket.faces,
        loop_uvs=bucket.uvs,
        loop_normals=bucket.normals,
        loop_colors=bucket.colors,
        vertex_bone=bucket.vertex_bone,
        material=materials.index_of(bl_mat),
        has_uv=has_uv,
        has_normals=True,
        has_colors=has_colors,
    )


def _shape_name(obj: bpy.types.Object, slot: int, slot_count: int) -> str:
    base = obj.get("nsbmd_name", obj.name)
    return base if slot_count == 1 else f"{base}_{slot}"


def _corner_normal(me: bpy.types.Mesh, loop_index: int):
    try:
        return me.corner_normals[loop_index].vector.copy()
    except (AttributeError, IndexError):
        return me.loops[loop_index].normal.copy()


def _vertex_bone_map(obj: bpy.types.Object,
                     me: bpy.types.Mesh,
                     bone_index: dict[str, int],
                     warn: list[str]) -> list[int]:
    """Rigid binding: one bone per vertex, the highest-weighted group that
    names a bone. Weighted (NODEMIX) skinning is not wired up yet."""
    if not bone_index:
        return [0] * len(me.vertices)

    group_bone = {
        gi: bone_index[g.name]
        for gi, g in enumerate(obj.vertex_groups) if g.name in bone_index
    }

    out: list[int] = []
    weighted = 0
    for v in me.vertices:
        best, best_weight, bound = 0, -1.0, 0
        for element in v.groups:
            bone = group_bone.get(element.group)
            if bone is None or element.weight <= 0.0:
                continue
            bound += 1
            if element.weight > best_weight:
                best, best_weight = bone, element.weight
        weighted += bound > 1
        out.append(best)

    if weighted:
        warn.append(
            f"{obj.name}: {weighted} vertices are weighted to several bones; "
            f"exported rigid to the heaviest one"
        )
    return out


class _MaterialTable:
    def __init__(self):
        self.materials: list[mdl.ImportedMaterial] = []
        self._index: dict[str, int] = {}
        self._names = _NameAllocator()

    def index_of(self, bl_mat: bpy.types.Material | None) -> int:
        key = bl_mat.name if bl_mat is not None else ""
        if key not in self._index:
            self._index[key] = len(self.materials)
            self.materials.append(_build_material(bl_mat, self._names))
        return self._index[key]


def _build_material(bl_mat: bpy.types.Material | None, names: _NameAllocator) -> mdl.ImportedMaterial:
    if bl_mat is None:
        return mdl.ImportedMaterial(
            names.take("default"), polygon_attr=PolygonAttr.build()
        )

    imat = mdl.ImportedMaterial(names.take(
        bl_mat.get("nsbmd_name", bl_mat.name)
    ))
    imat.diffuse, imat.alpha = _base_color(bl_mat)

    # If we stored polygon attr from import, use that.
    stored = bl_mat.get("nsbmd_poly_attr")
    if stored is not None:
        imat.polygon_attr = PolygonAttr(int(stored))
    else:
        imat.polygon_attr = PolygonAttr.build(
            cull=CullMode.BACK if bl_mat.use_backface_culling else CullMode.NONE,
            alpha=int(round(imat.alpha * 31)),
        )

    tex_image_param = bl_mat.get("nsbmd_tex_img_param")
    if tex_image_param is not None:
        imat.tex_img_param = TexImageParam(int(tex_image_param))

    imat.tex_name = bl_mat.get("nsbmd_tex_name")
    imat.pltt_name = bl_mat.get("nsbmd_pltt_name")

    size = bl_mat.get("nsbmd_orig_size")
    if size is not None:
        imat.tex_size = (int(size[0]), int(size[1]))

    return imat


def _base_color(bl_mat: bpy.types.Material) -> tuple[tuple[float, float, float], float]:
    bsdf = None
    if bl_mat.use_nodes and bl_mat.node_tree is not None:
        bsdf = next(
            (n for n in bl_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"),
            None
        )

    if bsdf is not None:
        c = bsdf.inputs["Base Color"].default_value
        alpha = (bsdf.inputs["Alpha"].default_value
                 if "Alpha" in bsdf.inputs else 1.0)
        return (c[0], c[1], c[2]), float(alpha)

    c = bl_mat.diffuse_color
    return (c[0], c[1], c[2]), float(c[3])


class _NameAllocator:
    def __init__(self):
        self._used: set[str] = set()

    def take(self, name: str) -> str:
        base = _sanitize_name(name)
        if base not in self._used:
            self._used.add(base)
            return base

        for i in range(1, 10000):
            suffix = f"_{i}"
            candidate = base[:MAX_NAME_LEN - len(suffix)] + suffix
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate

        raise ValueError(
            f"could not derive a unique 16-byte name from {name!r}")


def _sanitize_name(name: str) -> str:
    cleaned = (name or "").strip() or "unnamed"
    return cleaned.encode("ascii", "replace").decode("ascii")[:MAX_NAME_LEN]


def _blender_matrix_to_ds(m: Matrix) -> np.ndarray:
    return np.array([[m[r][c] for c in range(4)] for r in range(4)],
                    dtype=float).T
