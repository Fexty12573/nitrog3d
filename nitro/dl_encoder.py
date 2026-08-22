from __future__ import annotations
from dataclasses import dataclass, field
from itertools import chain
from . import binary as bin
from .dl import DlCmd, PrimType, NORMAL_SCALE, TEXCOORD_SCALE
from .model import ImportedMesh
from .sbc_encoder import SbcEncoder
from .quantize import decide_vertex_form, to_fx
from . import matrix as mat
import struct
import numpy as np


class DlEncoder:
    def __init__(self,
                 mesh: ImportedMesh,
                 mtx_map: dict[int, tuple[np.ndarray, np.ndarray]],
                 bone_map: dict[int, int],
                 entry_matrix: int | None,
                 matrix_slots: dict[int, int],
                 pos_scale: float):
        assert len(set(mesh.vertex_bone)) <= 29, \
            "Meshes with more than 29 bones are not supported"

        self.mesh = mesh
        self.dl = bin.BinaryWriter()
        self.cmds: list[_Command] = []

        S = mat.scale(pos_scale, pos_scale, pos_scale)
        self.mtx_map = {
            bone: _BoneMatrices(
                pos=mat.inverse(S @ pos),
                dir=mat.inverse(dir)
            ) for bone, (pos, dir) in mtx_map.items()
        }

        self.bone_map = bone_map
        self.entry_matrix = entry_matrix
        self.matrix_slots = matrix_slots
        self.pos_scale = pos_scale

        self.prev_normal: int | None = None
        self.prev_color: int | None = None
        self.prev_uv: int | None = None
        self.prev_vtx: tuple[int, int, int] | None = None

        self.has_restore = False
        self.has_normal = mesh.has_normals
        self.has_color = mesh.has_colors
        self.has_uv = mesh.has_uv
        self.total_vertices = 0

    def encode(self) -> bytes:
        self._begin(PrimType.TRIANGLES)

        current = self.entry_matrix

        for i, face in _sorted_faces(self.mesh):
            base = i * 3
            for j, vi in enumerate(face):
                bone = self.bone_map[self.mesh.vertex_bone[vi]]
                if bone != current:
                    self._restore_mtx(self.matrix_slots[bone])
                    if self.pos_scale != 1.0:
                        self._scale(
                            self.pos_scale,
                            self.pos_scale,
                            self.pos_scale
                        )
                    current = bone

                v = self.mesh.vertices[vi]
                if self.mesh.has_uv:
                    uv = self.mesh.loop_uvs[base+j]
                    self._texcoord(uv)
                if self.mesh.has_normals:
                    n = self.mesh.loop_normals[base+j]
                    self._normal(bone, n)
                if self.mesh.has_colors:
                    c = self.mesh.loop_colors[base+j]
                    self._color(c)

                self._vertex(bone, v)

        self._end()
        self._flush()

        return self.dl.get_bytes()

    def _begin(self, prim_type: PrimType):
        self._emit(DlCmd.BEGIN, "<I", int(prim_type))

    def _end(self):
        self._emit(DlCmd.END)

    def _normal(self, bone: int, n: tuple[float, float, float]):
        local = mat.mul_no_translate(n, self.mtx_map[bone].dir)
        encoded = _encode_normal(local)
        if encoded != self.prev_normal:
            self._emit(DlCmd.NORMAL, "<I", encoded)
            self.prev_normal = encoded

    def _texcoord(self, uv: tuple[float, float]):
        encoded = _encode_uv(uv)
        if encoded != self.prev_uv:
            self._emit(DlCmd.TEXCOORD, "<I", encoded)
            self.prev_uv = encoded

    def _color(self, c: tuple[float, float, float]):
        encoded = _encode_color(c)
        if encoded != self.prev_color:
            self._emit(DlCmd.COLOR, "<I", encoded)
            self.prev_color = encoded

    def _restore_mtx(self, slot: int):
        self._emit(DlCmd.RESTORE_MTX, "<I", slot)
        self.prev_normal = None  # Matrix restore clears normal
        self.has_restore = True

    def _scale(self, sx: float, sy: float, sz: float):
        self._emit(DlCmd.SCALE, "<iii", to_fx(sx), to_fx(sy), to_fx(sz))

    def _vertex(self, bone: int, v: tuple[float, float, float]):
        local = mat.mul(v, self.mtx_map[bone].pos)
        vtx = tuple(map(bin.to_fx, local))
        self._emit_vtx(vtx)
        self.prev_vtx = vtx
        self.total_vertices += 1

    def _emit_vtx(self, vtx: tuple[int, int, int]):
        match decide_vertex_form(vtx, self.prev_vtx):
            case DlCmd.VERTEX:
                self._vertex_xyz(vtx)
            case DlCmd.VERTEX_XY:
                self._vertex_xy(vtx)
            case DlCmd.VERTEX_XZ:
                self._vertex_xz(vtx)
            case DlCmd.VERTEX_YZ:
                self._vertex_yz(vtx)
            case DlCmd.VERTEX_10:
                self._vertex10(vtx)
            case DlCmd.VERTEX_DIFF:
                self._vertex_diff(vtx)
            case _:
                raise ValueError(f"Unexpected vertex form: {vtx}")

    def _vertex_xyz(self, v: tuple[int, int, int]):
        self._emit(DlCmd.VERTEX, "<II", *_encode_vertex3x16(v))

    def _vertex_xy(self, v: tuple[int, int, int]):
        self._emit(DlCmd.VERTEX_XY, "<I", _encode_vertex2x16(v[0], v[1]))

    def _vertex_xz(self, v: tuple[int, int, int]):
        self._emit(DlCmd.VERTEX_XZ, "<I", _encode_vertex2x16(v[0], v[2]))

    def _vertex_yz(self, v: tuple[int, int, int]):
        self._emit(DlCmd.VERTEX_YZ, "<I", _encode_vertex2x16(v[1], v[2]))

    def _vertex10(self, v: tuple[int, int, int]):
        shifted = tuple(map(lambda c: c >> 6, v))
        self._emit(DlCmd.VERTEX_10, "<I", _encode_vertex3x10(shifted))

    def _vertex_diff(self, v: tuple[int, int, int]):
        assert self.prev_vtx is not None
        delta = tuple((c - p for c, p in zip(v, self.prev_vtx)))
        self._emit(DlCmd.VERTEX_DIFF, "<I", _encode_vertex3x10(delta))

    def _emit(self, cmd: DlCmd, fmt: str | None = None, *args):
        packed = struct.pack(fmt, *args) if fmt is not None else b""
        self.cmds.append(_Command(cmd=cmd, args=packed))
        if len(self.cmds) == 4:
            self._flush()

    def _flush(self):
        if not self.cmds:
            return

        # Pad to 4 bytes
        while len(self.cmds) < 4:
            self.cmds.append(_Command(cmd=DlCmd.NOP))

        self.dl.write_u8s(list(map(lambda c: int(c.cmd), self.cmds)))
        concatenated = b"".join(map(lambda c: c.args, self.cmds))
        self.dl.write_bytes(concatenated)

        if len(concatenated) % 4 != 0:
            self.dl.write_bytes(b"\x00" * (4 - len(concatenated) % 4))

        self.cmds.clear()


def _encode_vertex3x16(v: tuple[int, int, int]) -> tuple[int, int]:
    return ((v[0] & 0xFFFF) | ((v[1] & 0xFFFF) << 16), (v[2] & 0xFFFF))


def _encode_vertex3x10(v: tuple[int, int, int]) -> int:
    return bin.pack3x10(v[0] & 0x3FF, v[1] & 0x3FF, v[2] & 0x3FF)


def _encode_vertex2x16(v0: int, v1: int) -> int:
    return (v0 & 0xFFFF) | ((v1 & 0xFFFF) << 16)


def _encode_normal(n: tuple[float, float, float]) -> int:
    quantized = tuple(
        bin.clamp(int(round(c / NORMAL_SCALE)), -512, 511) for c in n)
    return bin.pack3x10(*quantized)


def _encode_color(c: tuple[float, float, float]) -> int:
    return bin.float_to_bgr555(*c)


def _sorted_faces(mesh: ImportedMesh) -> list[tuple[int, tuple[int, int, int]]]:
    return list(enumerate(mesh.faces))  # TODO: Sort by bone


def _encode_uv(uv: tuple[float, float]) -> int:
    s = int(round(uv[0] / TEXCOORD_SCALE)) & 0xFFFF
    t = int(round(uv[1] / TEXCOORD_SCALE)) & 0xFFFF
    return s | (t << 16)


@dataclass
class _Command:
    cmd: DlCmd
    args: bytes = field(default_factory=bytes)


@dataclass(slots=True)
class _BoneMatrices:
    pos: np.ndarray
    dir: np.ndarray
