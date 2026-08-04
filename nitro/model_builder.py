
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import matrix as mat
from . import binary as bin
from . import mdl0
from .dl_encoder import DlEncoder
from .model import ImportedSubModel, ImportedMesh, ImportedMaterial, Bone
from .quantize import box_exponent_for, pos_scale_for
from .sbc_encoder import SbcEncoder, BoneMapping

MAX_NODES = 255
MAX_SHAPES = 255
MAX_MATERIALS = 255
EPSILON = 1e-6


class ModelBuilder:
    def __init__(self, sub: ImportedSubModel):
        self.sub = sub
        self.pos_scale = 1.0
        self.shapes: list[EmittedShape] = []
        self.dls: list[bytes] = []
        self.bones: list[tuple[int, Bone]] = []
        self.id_map: dict[int, int] = {}
        self.sbc = bytes()
        self.first_unused_mtx_stack_id = 0
        self.total_vertices = 0

    def build(self) -> mdl0.Model:
        self._plan_nodes()
        self._plan_shapes()
        self._compute_pos_scale()
        self._encode_dls()
        self._encode_sbc()
        nodeset = self._build_nodes()
        matset = self._build_materials()
        shapeset = self._build_shapes()
        info = self._build_info()

        return mdl0.Model.build(
            info,
            nodeset,
            self.sbc,
            matset,
            shapeset
        )

    def _plan_nodes(self):
        self.bones = _preorder_bones(self.sub.bones)
        self.id_map = _remap_bone_ids(self.bones)
        if len(self.bones) > MAX_NODES:
            raise ValueError(
                f"{len(self.bones)} nodes exceeds the maximum of {MAX_NODES}"
            )

    def _plan_shapes(self):
        # TODO: We do one shape per mesh, grouped by bound node.
        # Grouping is a temporary solution until encoding multi-mtx DLs is supported.
        by_node: dict[int, list[ImportedMesh]] = {}
        for mesh in self.sub.meshes:
            bound = set(mesh.vertex_bone)
            if len(bound) != 1:
                raise NotImplementedError(
                    f"{mesh.name}: multi-mtx shape ({len(bound)} nodes) not yet supported"
                )

            by_node.setdefault(
                self.id_map[bound.pop()],
                []
            ).append(mesh)

        for node, meshes in by_node.items():
            for mesh in meshes:
                i = len(self.shapes)
                self.shapes.append(EmittedShape(
                    i, mesh.name or f"shape{i}", mesh, node, mesh.material
                ))

        if len(self.shapes) > MAX_SHAPES:
            raise ValueError(
                f"{len(self.shapes)} shapes exceeds the maximum of {MAX_SHAPES}"
            )

    def _compute_pos_scale(self):
        self.pos_scale = pos_scale_for(self.sub)

    def _encode_dls(self):
        for shape in self.shapes:
            bone = self.sub.bones[shape.mesh.vertex_bone[0]]
            enc = DlEncoder(
                shape.mesh, bone.world_mtx,
                bone.world_dir_mtx, self.pos_scale
            )
            self.dls.append(enc.encode())
            self.total_vertices += enc.total_vertices

    def _encode_sbc(self):
        enc = SbcEncoder(
            self.sub,
            BoneMapping(self.bones, self.id_map),
            {s.name: s.index for s in self.shapes}
        )
        self.sbc = enc.encode()
        self.first_unused_mtx_stack_id = enc.next_mtx_stack_id

    def _build_nodes(self) -> mdl0.NodeSet:
        nodes: dict[str, mdl0.NodeData] = {}
        for old_id, bone in self.bones:
            parent = bone.parent
            if parent < 0 or parent == old_id:  # root
                local = bone.world_mtx
            else:
                local = bone.world_mtx @ mat.inverse(
                    self.sub.bones[parent].world_mtx
                )

            t, r, s = _decompose_srt(local)
            nodes[bone.name] = (
                mdl0.NodeData.builder()
                    .translate(t)
                    .rotate(r)
                    .scale(s)
                    .build()
            )

        return mdl0.NodeSet.build(nodes)

    def _build_materials(self) -> mdl0.MaterialSet:
        # TODO: Make this not lossy
        if len(self.sub.materials) > MAX_MATERIALS:
            raise ValueError(
                f"{len(self.sub.materials)} materials exceeds the maximum of {MAX_MATERIALS}"
            )

        names: list[str] = []
        mats: list[mdl0.Material] = []
        tex_names: list[str | None] = []
        pal_names: list[str | None] = []
        for imat in self.sub.materials:
            names.append(imat.name)
            mats.append(_build_material(imat))
            tex = imat.texture
            tex_names.append(tex.tex_name if tex else None)
            pal_names.append(tex.pltt_name if tex else None)

        return mdl0.MaterialSet.build(names, mats, tex_names, pal_names)

    def _build_shapes(self) -> mdl0.ShapeSet:
        shapes: dict[str, mdl0.Shape] = {}
        for i, (shape, dl) in enumerate(zip(self.shapes, self.dls)):
            shapes[shape.name] = mdl0.Shape.build(tag=i, flag=0, dl=dl)

        return mdl0.ShapeSet.build(shapes)

    def _build_info(self) -> mdl0.ModelInfo:
        lo, hi = _world_bounds(self.sub)
        extent = tuple(h - l for l, h in zip(lo, hi))

        box_scale = float(1 << box_exponent_for(lo, extent))
        tris = sum(len(p.mesh.faces) for p in self.shapes)

        return (
            mdl0.ModelInfo.builder()
                .node_count(len(self.bones))
                .mat_count(len(self.sub.materials))
                .shape_count(len(self.shapes))
                .triangle_count(tris)
                .vertex_count(self.total_vertices)
                .first_unused_mtx_stack_id(self.first_unused_mtx_stack_id)
                .pos_scale(self.pos_scale)
                .polygon_count(tris)
                .bounding_box(*(c / box_scale for c in lo),
                              *(e / box_scale for e in extent))
                .box_pos_scale(self.pos_scale)
                .build()
        )


@dataclass(slots=True)
class EmittedShape:
    index: int
    name: str
    mesh: ImportedMesh
    node: int  # remapped
    material: int


def _preorder_bones(bones: list[Bone]) -> list[tuple[int, Bone]]:
    root_id, root = next(
        filter(lambda b: b[1].parent == -1, enumerate(bones)), (0, bones[0])
    )
    return _get_children(root_id, root, bones)


def _get_children(id: int, bone: Bone, bones: list[Bone]) -> list[tuple[int, Bone]]:
    children: list[Bone] = []
    for (cid, child) in filter(lambda bone: bone[1].parent == id and bone[0] != id, enumerate(bones)):
        children.extend(_get_children(cid, child, bones))
    return [(id, bone), *children]


def _remap_bone_ids(bones: list[tuple[int, Bone]]) -> dict[int, int]:
    return {old_id: new_id for new_id, (old_id, _) in enumerate(bones)}


def _decompose_srt(m: np.ndarray):
    t = tuple(_snap(v) for v in m[3, :3])
    s: list[float] = []
    r: list[float] = []
    for i in range(3):
        row = m[i, :3]
        n = float(np.linalg.norm(row))
        s.append(_snap(n))
        r.extend((row / n if n > EPSILON else row).tolist())
    return t, [_snap(v) for v in r], tuple(s)


def _snap(v: float) -> float:
    """Snaps values close to 0/1/-1 so we can make proper use of SRT flags"""
    for target in (0.0, 1.0, -1.0):
        if abs(v - target) < EPSILON:
            return target
    return v


def _build_material(mat: ImportedMaterial) -> mdl0.Material:
    # TODO: Clean this up
    alpha = bin.clamp(int(round(mat.alpha * 31)), 0, 31)
    poly_attr = (alpha << 16) | ((mat.cull_mode & 0x3) << 6)
    return (
        mdl0.Material.builder()
            .diffuse(mat.diffuse)
            .poly_attr(poly_attr, 0x3F1FFFFF)
            .build()
    )


def _world_bounds(sub: ImportedSubModel):
    pts = [v for mesh in sub.meshes for v in mesh.vertices]
    if not pts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    a = np.array(pts)
    return tuple(a.min(axis=0).tolist()), tuple(a.max(axis=0).tolist())
