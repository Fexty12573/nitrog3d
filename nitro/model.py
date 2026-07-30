from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from . import texture
from .binary import bgr555_to_float
from .nsbmd import NSBMD
from .mdl0 import Model, MatFlag
from .tex0 import TEX0, TexFmt
from .dl import GeometryBuilder, Triangle, Vertex
from .sbc import SbcInterpreter, DrawCall


@dataclass(slots=True)
class ImportedModel:
    models: list[ImportedSubModel] = field(default_factory=list)
    textures: dict[str, DecodedTexture] = field(default_factory=dict)
    anims: list[ImportedAnim] = field(default_factory=list)


@dataclass(slots=True)
class ImportedSubModel:
    name: str
    bones: list[Bone] = field(default_factory=list)
    meshes: list[ImportedMesh] = field(default_factory=list)
    materials: list[ImportedMaterial] = field(default_factory=list)


@dataclass(slots=True)
class ImportedMesh:
    name: str
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    loop_uvs: list[tuple[float, float]] = field(default_factory=list)
    loop_normals: list[tuple[float, float, float]
                       ] = field(default_factory=list)
    loop_colors: list[tuple[float, float, float]] = field(default_factory=list)
    vertex_bone: list[int] = field(default_factory=list)
    material: int = -1
    has_uv: bool = False
    has_normals: bool = False
    has_colors: bool = False


@dataclass(slots=True)
class ImportedMaterial:
    name: str
    texture: DecodedTexture | None = None
    diffuse: tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 1.0
    cull_mode: int = 0
    wireframe: bool = False


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


TextureCache = dict[tuple[str, str], DecodedTexture | None]


def load(data: bytes | bytearray) -> ImportedModel:
    nsbmd = NSBMD(data)
    result = ImportedModel()

    if nsbmd.model_set is None:
        return result

    tex_cache: TextureCache = {}

    for name, model in nsbmd.model_set:
        sub = ImportedSubModel(name)

        builder = GeometryBuilder()
        interpreter = SbcInterpreter(
            model,
            builder,
            _material_texture_dims(model, nsbmd.tex_pltt_set)
        )

        interpreter.run()

        sub.bones = _build_bones(model, interpreter)
        sub.meshes = _build_meshes(model, builder, interpreter.draw_calls)

        mb = MaterialBuilder(model, nsbmd.tex_pltt_set,
                             tex_cache, result.textures)
        sub.materials = mb.build()
        result.models.append(sub)

    return result


def _material_texture_dims(model: Model, tex_set: TEX0 | None):
    if not tex_set:
        return {}

    dims = {}
    for i in range(len(model.materials)):
        tn = model.materials.texture_name(i)
        if tn is None:
            continue
        e = tex_set.tex_dict[tn]
        if e is not None:
            dims[i] = (e.s, e.t)

    return dims


def _build_bones(model: Model, interp: SbcInterpreter) -> list[Bone]:
    names = model.nodes.dict.keys()
    return [
        Bone(
            names[i] if i < len(names) else f"bone{i}",
            interp.node_parent[i],
            interp.node_world[i]
        ) for i in range(len(model.nodes))
    ]


def _build_meshes(model: Model, builder: GeometryBuilder, draw_calls: list[DrawCall]) -> list[ImportedMesh]:
    tris = builder.triangles
    meshes: list[ImportedMesh] = []
    shape_use = _shape_use_counts(draw_calls)

    for dc in draw_calls:
        if dc.tri_end <= dc.tri_start:
            continue

        call_tris = tris[dc.tri_start:dc.tri_end]
        name = f"shape{dc.shape}" if dc.shape >= 0 else "calldl"
        mesh = _build_mesh(name, dc.material, call_tris)
        if mesh is not None:
            meshes.append(mesh)

    return meshes


def _build_mesh(name: str, material: int, tris: list[Triangle]) -> ImportedMesh | None:
    def any_tri(ts: list[Triangle], f: Callable[[Vertex], bool]) -> bool:
        return any(f(v) for tri in ts for v in tri)

    def key(pos: tuple[float, float, float]) -> tuple[float, float, float]:
        return (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5))

    has_uv = any_tri(tris, lambda v: v.uv is not None)
    has_nrm = any_tri(tris, lambda v: v.normal is not None)
    has_col = any_tri(tris, lambda v: v.color is not None)

    mesh = ImportedMesh(name, material=material, has_uv=has_uv,
                        has_normals=has_nrm, has_colors=has_col)

    index_of: dict[tuple, int] = {}
    for tri in tris:
        keys = [key(v.pos) for v in tri]
        if len(set(keys)) < 3:
            continue  # Degenerate face

        for v, k in zip(tri, keys):
            if k not in index_of:
                index_of[k] = len(mesh.vertices)
                mesh.vertices.append(v.pos)
                mesh.vertex_bone.append(v.node)
        mesh.faces.append(tuple(index_of[k] for k in keys))

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

    return mesh if mesh.faces else None


def _shape_use_counts(draw_calls: list[DrawCall]) -> dict[int, int]:
    counts = {}
    for dc in draw_calls:
        if dc.shape >= 0:
            counts[dc.shape] = counts.get(dc.shape, 0) + 1
    return counts


class MaterialBuilder:
    def __init__(self, model: Model, tex_set: TEX0 | None, tex_cache: TextureCache, out_textures: dict[str, DecodedTexture]):
        self.model = model
        self.tex_set = tex_set
        self.tex_cache = tex_cache
        self.textures = out_textures

    def build(self) -> list[ImportedMaterial]:
        mats: list[ImportedMaterial] = []
        names = self.model.materials.dict.keys()
        for i, m in enumerate(self.model.materials):
            name = names[i] if i < len(names) else f"material{i}"
            im = ImportedMaterial(name)
            im.diffuse = bgr555_to_float(m.diff_amb)
            im.alpha = m.alpha / 31.0
            im.cull_mode = m.cull_mode
            im.wireframe = m.hasflag(MatFlag.WIREFRAME)

            tex_name = self.model.materials.texture_name(i)
            pltt_name = self.model.materials.palette_name(i)
            if self.tex_set and tex_name is not None:
                im.texture = self._resolve_texture(tex_name, pltt_name)
            mats.append(im)
        return mats

    def _resolve_texture(self, tex_name: str, pal_name: str | None) -> DecodedTexture | None:
        ckey = (tex_name, pal_name)
        if ckey in self.tex_cache:
            return self.tex_cache[ckey]

        entry = self.tex_set.tex_dict[tex_name]
        if entry is None:
            self.tex_cache[ckey] = None
            return None

        pal_bytes = b""
        if entry.fmt != TexFmt.DIRECT:
            pal_bytes = self._find_palette_bytes(pal_name)
        w, h, rgba = texture.decode_texture(entry, pal_bytes)
        has_alpha = entry.fmt.has_alpha() or entry.transparent_color

        disp_name = self._display_name(tex_name, pal_name)
        dt = DecodedTexture(disp_name, w, h, rgba, has_alpha)

        self.tex_cache[ckey] = dt
        self.textures[disp_name] = dt
        return dt

    def _display_name(self, tex: str, pal: str | None) -> str:
        return tex if pal in (None, tex) else f"{tex}.{pal}"

    def _find_palette_bytes(self, pal_name: str | None) -> bytes:
        if pal_name is None:
            return b""
        entry = self.tex_set.pltt_dict[pal_name]
        return entry.data if entry is not None else b""
