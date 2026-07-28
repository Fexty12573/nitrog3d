from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .nsbmd import NSBMD
from .mdl0 import Model
from .tex0 import TEX0
from .dl import GeometryBuilder, Triangle, Vertex
from .sbc import SbcInterpreter, DrawCall


@dataclass(slots=True)
class ImportedModel:
    models: list[ImportedSubModel] = []
    textures: list[DecodedTexture] = []
    anims: list[ImportedAnim] = []


@dataclass(slots=True)
class ImportedSubModel:
    name: str
    bones: list[Bone] = []
    meshes: list[ImportedMesh] = []
    materials: list[ImportedMaterial] = []


@dataclass(slots=True)
class ImportedMesh:
    name: str
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    loop_uvs: list[tuple[float, float]] = []
    loop_normals: list[tuple[float, float, float]] = []
    loop_colors: list[tuple[float, float, float]] = []
    vertex_bone: list[int] = []
    material = -1
    has_uv = False
    has_normals = False
    has_colors = False


@dataclass(slots=True)
class ImportedMaterial:
    name: str
    texture: DecodedTexture | None = None
    diffuse = (1.0, 1.0, 1.0)
    alpha = 1.0
    cull_mode = 0
    wireframe = False


@dataclass(slots=True)
class DecodedTexture:
    name: str
    width: int
    height: int
    rgba: bytes
    has_alpha: bool


@dataclass(slots=True)
class ImportedAnim:
    name: str
    num_frames: int
    tracks: dict[str,]


@dataclass(slots=True)
class Bone:
    name: str
    parent: int
    world_mtx: np.ndarray


def load(data: bytes | bytearray) -> ImportedModel:
    nsbmd = NSBMD(data)
    result = ImportedModel()

    if nsbmd.model_set is None:
        return result

    for name, model in nsbmd.model_set:
        sub = ImportedSubModel(name)

        builder = GeometryBuilder()
        interpreter = SbcInterpreter(
            model,
            builder,
            material_texture_dims(model, nsbmd.tex_pltt_set)
        )

        interpreter.run()

        node_names = model.nodes.dict.keys()
        for i in range(len(model.nodes)):
            bname = node_names[i] if i < len(node_names) else f"bone{i}"
            sub.bones.append(Bone(
                bname,
                interpreter.node_parent[i],
                interpreter.node_world[i]
            ))

            sub.meshes = _build_meshes(model, builder, interpreter.draw_calls)
            # TODO: Build materials


def material_texture_dims(model: Model, tex_set: TEX0 | None):
    if not tex_set:
        return {}

    dims = {}
    for i in range(len(model.materials)):
        tn = model.materials.texture_name(i)
        if tn is None:
            continue
        idx = tex_set.tex_dict.index_of(tn)
        if idx >= 0:
            e = tex_set.tex_dict.data[idx]
            dims[i] = (e.s, e.t)

    return dims


def _build_meshes(model: Model, builder: GeometryBuilder, draw_calls: list[DrawCall]) -> list[ImportedMesh]:
    tris = builder.triangles
    meshes: list[ImportedMesh] = []
    node_names = model.nodes.dict.keys()
    shape_use = _shape_use_counts(draw_calls)

    def any_tri(tris: list[Triangle], f: Callable[[Vertex], bool]) -> bool:
        return any(f(v) for tri in tris for v in tri)

    def key(pos: tuple[float, float, float]) -> tuple[float, float, float]:
        return (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5))

    for dc in draw_calls:
        if dc.tri_end <= dc.tri_start:
            continue

        call_tris = tris[dc.tri_start:dc.tri_end]
        has_uv = any_tri(call_tris, lambda v: v.uv is not None)
        has_nrm = any_tri(call_tris, lambda v: v.normal is not None)
        has_col = any_tri(call_tris, lambda v: v.color is not None)

        mesh = ImportedMesh(f"shape{dc.shape}" if dc.shape >= 0 else "calldl")
        mesh.material = dc.material
        mesh.has_uv = has_uv
        mesh.has_normals = has_nrm
        mesh.has_colors = has_col

        index_of = {}
        for tri in call_tris:
            face: list[int] = []
            for v in tri:
                k = key(v.pos)
                idx = index_of.get(k)
                if idx is None:
                    idx = len(mesh.vertices)
                    index_of[k] = idx
                    mesh.vertices.append(v.pos)
                    mesh.vertex_bone.append(v.node)
                face.append(idx)

            if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
                continue

            mesh.faces.append((face[0], face[1], face[2]))
            for v in tri:
                if has_uv:
                    mesh.loop_uvs.append(
                        v.uv if v.uv is not None else (0.0, 0.0))
                if has_nrm:
                    mesh.loop_normals.append(
                        v.normal if v.normal is not None else (0.0, 0.0, 1.0))
                if has_col:
                    mesh.loop_colors.append(
                        v.color if v.color is not None else (1.0, 1.0, 1.0))
        if not mesh.faces:
            continue
        meshes.append(mesh)

    return meshes


def _shape_use_counts(draw_calls: list[DrawCall]) -> dict[int, int]:
    counts = {}
    for dc in draw_calls:
        if dc.shape >= 0:
            counts[dc.shape] = counts.get(dc.shape) + 1
    return counts
