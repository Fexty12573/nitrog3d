
from __future__ import annotations
from .binary import FX16_SCALE, FX32_SCALE, sign_extend, read_u32_le
from . import matrix as mat
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from itertools import batched


class MtxMode(IntEnum):
    PROJECTION = 0
    POSITION = 1
    POSITION_VECTOR = 2
    TEXTURE = 3


class TexGen(IntEnum):
    NONE = 0
    TEXCOORD = 1
    NORMAL = 2
    VERTEX = 3


class PrimType(IntEnum):
    TRIANGLES = 0
    QUADS = 1
    TRI_STRIP = 2
    QUAD_STRIP = 3


@dataclass(slots=True)
class Vertex:
    pos: tuple[float, float, float]
    normal: tuple[float, float]
    uv: tuple[float, float]
    color: tuple[float, float, float]
    node: int


class GeometryBuilder:
    def __init__(self):
        self.pos_stack = [np.identity(4) for _ in range(31)]
        self.dir_stack = [np.identity(4) for _ in range(31)]
        self.cur_pos = np.identity(4)
        self.cur_dir = np.identity(4)
        self.cur_tex = np.identity(4)
        self.stack_ptr = 0
        self.mtx_mode = MtxMode.POSITION_VECTOR

        self.last_tex = (0.0, 0.0)
        self.last_vtx = (0.0, 0.0, 0.0)
        self.last_col = (1.0, 1.0, 1.0)
        self.alpha = 31
        self.tex_width = 1
        self.tex_height = 1
        self.tex_gen = TexGen.NONE
        self.next_poly_attr = 0

        self.cur_nrm: tuple[float, float, float] | None = None
        self.cur_uv: tuple[float, float] | None = None

        self.current_bound_node = 0
        self.triangles: list[tuple[Vertex, Vertex, Vertex]] = []

        self._prim_type: PrimType | None = None
        self._prim: list[Vertex] = []

        self.num_vertices_emitted = 0
        self.num_triangles = 0
        self.num_quads = 0

        self._first_pos = None
        self._first_dir = None
        self._first_pos_ref = None
        self._single_matrix = True

    def _emit_vertex(self, v: tuple[float, float, float]):
        if self._first_pos is None:
            self._first_pos = np.copy(self.cur_pos)
            self._first_dir = np.copy(self.cur_dir)
            self._first_pos_ref = self.cur_pos
        elif self.cur_pos is not self._first_pos_ref:
            # Current matrix changed since first vertex -> multi-matrix DL
            self._single_matrix = False

        self.num_vertices_emitted += 1
        self._prim.append(Vertex(v, self.cur_nrm,
                          self.cur_uv, self.last_col, self.current_bound_node))

    def _flush_primitive(self):
        verts = self._prim
        self._prim = []

        if self._prim_type is None or len(verts) < 3:
            return

        match self._prim_type:
            case PrimType.TRIANGLES:
                self.num_triangles += len(verts) // 3
                # An incomplete trailing primitive is discarded, as on hardware
                for group in batched(verts, 3):
                    if len(group) == 3:
                        self.triangles.append(group)

            case PrimType.QUADS:
                self.num_quads += len(verts) // 4
                for group in batched(verts, 4):
                    if len(group) == 4:
                        a, b, c, d = group
                        self.triangles.append((a, b, c))
                        self.triangles.append((a, c, d))
            case PrimType.TRI_STRIP:
                self.num_triangles += len(verts) - 2
                for i in range(2, len(verts)):
                    if i % 2 == 0:
                        self.triangles.append(
                            (verts[i - 2], verts[i - 1], verts[i]))
                    else:
                        self.triangles.append(
                            (verts[i - 1], verts[i - 2], verts[i]))
            case PrimType.QUAD_STRIP:
                self.num_quads += (len(verts) - 2) // 2
                i = 3
                while i < len(verts):
                    a, b = verts[i - 3], verts[i - 2]
                    c, d = verts[i], verts[i - 1]
                    self.triangles.append((a, b, c))
                    self.triangles.append((a, c, d))
                    i += 2

    def run_dl(self, dl: bytes | bytearray):
        self._first_pos = None
        self._first_dir = None
        self._first_pos_ref = None
        self._single_matrix = True
        off = 0
        n = len(dl)
        while off < n:
            if off + 4 > n:
                break
            cmds = tuple(dl[off:off + 4])
            off += 4
            for cmd in cmds:
                off = self._exec(cmd, dl, off)

    def _exec(self, cmd: DlCmd, dl: bytes | bytearray, off: int) -> int:
        match cmd:
            case DlCmd.NOP:
                return off
            case DlCmd.MTX_MODE:
                self.mtx_mode = MtxMode(read_u32_le(dl, off))
                return off + 4
            case DlCmd.PUSH_MTX:
                self.push_mtx()
                return off
            case DlCmd.POP_MTX:
                self.pop_mtx(read_u32_le(dl, off))
                return off + 4
            case DlCmd.STORE_MTX:
                self.store_mtx(read_u32_le(dl, off))
                return off + 4
            case DlCmd.RESTORE_MTX:
                self.restore_mtx(read_u32_le(dl, off))
                return off + 4
            case DlCmd.IDENTITY:
                self.identity()
                return off
            case DlCmd.LOAD_MTX44:
                self.load_mtx44([read_u32_le(dl, off + i * 4)
                                for i in range(16)])
                return off + 64
            case DlCmd.LOAD_MTX43:
                self.load_mtx43([read_u32_le(dl, off + i * 4)
                                for i in range(12)])
                return off + 48
            case DlCmd.MUL_MTX44:
                self.mul_mtx44([read_u32_le(dl, off + i * 4)
                               for i in range(16)])
                return off + 64
            case DlCmd.MUL_MTX43:
                self.mul_mtx43([read_u32_le(dl, off + i * 4)
                               for i in range(12)])
                return off + 48
            case DlCmd.MUL_MTX33:
                self.mul_mtx33([read_u32_le(dl, off + i * 4)
                               for i in range(9)])
                return off + 36
            case DlCmd.SCALE:
                self.scale(read_u32_le(dl, off), read_u32_le(
                    dl, off + 4), read_u32_le(dl, off + 8))
                return off + 12
            case DlCmd.TRANSLATE:
                self.translate(read_u32_le(dl, off), read_u32_le(
                    dl, off + 4), read_u32_le(dl, off + 8))
                return off + 12
            case DlCmd.COLOR:
                self.color(read_u32_le(dl, off))
                return off + 4
            case DlCmd.NORMAL:
                self.normal(read_u32_le(dl, off))
                return off + 4
            case DlCmd.TEXCOORD:
                self.texcoord(read_u32_le(dl, off))
                return off + 4
            case DlCmd.VERTEX:
                self.vertex(read_u32_le(dl, off), read_u32_le(dl, off + 4))
                return off + 8
            case DlCmd.VERTEX_10:
                self.vertex10(read_u32_le(dl, off))
                return off + 4
            case DlCmd.VERTEX_XY:
                self.vertex_xy(read_u32_le(dl, off))
                return off + 4
            case DlCmd.VERTEX_XZ:
                self.vertex_xz(read_u32_le(dl, off))
                return off + 4
            case DlCmd.VERTEX_YZ:
                self.vertex_yz(read_u32_le(dl, off))
                return off + 4
            case DlCmd.VERTEX_DIFF:
                self.vertex_diff(read_u32_le(dl, off))
                return off + 4
            case DlCmd.POLY_ATTR:
                self.next_poly_attr = read_u32_le(dl, off)
                return off + 4
            case DlCmd.TEX_IMG_PARAM:
                self.tex_image_param(read_u32_le(dl, off))
                return off + 4
            case DlCmd.TEX_PLTT_BASE:
                return off + 4
            case DlCmd.MAT_COL_0:
                return off + 4
            case DlCmd.MAT_COL_1:
                return off + 4
            case DlCmd.LIGHT_VEC:
                return off + 4
            case DlCmd.LIGHT_COL:
                return off + 4
            case DlCmd.SHININESS:
                return off + 128
            case DlCmd.BEGIN:
                self.begin(PrimType(read_u32_le(dl, off)))
                return off + 4
            case DlCmd.END:
                self._flush_primitive()
                return off
            case DlCmd.SWAP_BUFFERS:
                return off + 4
            case DlCmd.VIEWPORT:
                return off + 4
            case DlCmd.BOXTEST:
                return off + 12
            case DlCmd.POSTEST:
                return off + 8
            case DlCmd.VECTEST:
                return off + 4
            case _:
                return off

    def push_mtx(self):
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.pos_stack[self.stack_ptr] = self.cur_pos
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.dir_stack[self.stack_ptr] = self.cur_dir
        self.stack_ptr += 1

    def pop_mtx(self, cmd: int):
        self.stack_ptr -= sign_extend(cmd & 0x3F, 6)
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = self.pos_stack[self.stack_ptr]
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.cur_dir = self.dir_stack[self.stack_ptr]

    def store_mtx(self, index: int):
        index &= 31
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.pos_stack[index] = self.cur_pos
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.dir_stack[index] = self.cur_dir

    def restore_mtx(self, index: int):
        index &= 31
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = self.pos_stack[index]
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.cur_dir = self.dir_stack[index]

    def identity(self):
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = np.identity(4)
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.cur_dir = np.identity(4)
        if self.mtx_mode == MtxMode.TEXTURE:
            self.cur_tex = np.identity(4)

    def _load(self, m: np.ndarray):
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = m
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.cur_dir = m
        if self.mtx_mode == MtxMode.TEXTURE:
            self.cur_tex = m

    def load_mtx44(self, vals: list[int]):
        self._load(mat.from4x4(_fx32list(vals)))

    def load_mtx43(self, vals: list[int]):
        self._load(mat.from4x3(_fx32list(vals)))

    def mul(self, m: np.ndarray):
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = np.dot(self.cur_pos, m)
        if self.mtx_mode == MtxMode.POSITION_VECTOR:
            self.cur_dir = np.dot(self.cur_dir, m)
        if self.mtx_mode == MtxMode.TEXTURE:
            self.cur_tex = np.dot(self.cur_tex, m)

    def mul_mtx44(self, vals: list[int]):
        self.mul(mat.from4x4(_fx32list(vals)))

    def mul_mtx43(self, vals: list[int]):
        self.mul(mat.from4x3(_fx32list(vals)))

    def mul_mtx33(self, vals: list[int]):
        self.mul(mat.from3x3(_fx32list(vals)))

    def scale(self, sx: int, sy: int, sz: int):
        self.scale_vec(fx32(sx), fx32(sy), fx32(sz))

    def scale_vec(self, x: float, y: float, z: float):
        m = mat.scale(x, y, z)
        if self.mtx_mode in (MtxMode.POSITION, MtxMode.POSITION_VECTOR):
            self.cur_pos = np.dot(self.cur_pos, m)
        if self.mtx_mode == MtxMode.TEXTURE:
            self.cur_tex = np.dot(self.cur_tex, m)

    def translate(self, tx: int, ty: int, tz: int):
        self.translate_vec(fx32(tx), fx32(ty), fx32(tz))

    def translate_vec(self, x: float, y: float, z: float):
        self.mul(mat.translate(x, y, z))

    def color(self, rgb: int):
        self.last_col = _bgr555_to_float(rgb)

    def normal(self, packed: int):
        nx = sign_extend(packed & 0x3FF, 10) / 512.0
        ny = sign_extend((packed >> 10) & 0x3FF, 10) / 512.0
        nz = sign_extend((packed >> 20) & 0x3FF, 10) / 512.0
        n = mat.mul_no_translate((nx, ny, nz), self.cur_dir)
        self.cur_nrm = n
        if self.tex_gen == TexGen.NORMAL:
            tm = self.cur_tex
            u = (tm[0, 0] * nx + tm[1, 0] * ny + tm[2, 0]
                 * nz + self.last_tex[0]) / self.tex_width
            v = (tm[0, 1] * nx + tm[1, 1] * ny + tm[2, 1]
                 * nz + self.last_tex[1]) / self.tex_height
            self.cur_uv = (u, v)

    def texcoord(self, packed: int):
        s = _s16(packed) / 16.0
        t = _s16(packed >> 16) / 16.0
        self.last_tex = (s, t)
        if self.tex_gen == TexGen.NONE:
            self.cur_uv = (s / self.tex_width, t / self.tex_height)
        elif self.tex_gen == TexGen.TEXCOORD:
            tm = self.cur_tex
            u = (tm[0, 0] * s + tm[1, 0] * t +
                 tm[2, 0] + tm[3, 0]) / self.tex_width
            v = (tm[0, 1] * s + tm[1, 1] * t +
                 tm[2, 1] + tm[3, 1]) / self.tex_height
            self.cur_uv = (u, v)

    def tex_image_param(self, cmd: int):
        self.tex_width = 8 << ((cmd >> 20) & 7)
        self.tex_height = 8 << ((cmd >> 23) & 7)
        self.tex_gen = TexGen((cmd >> 30) & 3)

    def begin(self, prim_type: PrimType):
        self._flush_primitive()
        self._prim_type = prim_type
        self._prim = []
        self.alpha = (self.next_poly_attr >> 16) & 31

    def _vertex(self, v: tuple[float, float, float]):
        self.last_vtx = v
        self._emit_vertex(mat.mul(v, self.cur_pos))

    def vertex(self, cmd1: int, cmd2: int):
        x = fx16(cmd1)
        y = fx16(cmd1 >> 16)
        z = fx16(cmd2)
        self._vertex((x, y, z))

    def vertex10(self, v: int):
        x = sign_extend(v & 0x3FF, 10) / 64.0
        y = sign_extend((v >> 10) & 0x3FF, 10) / 64.0
        z = sign_extend((v >> 20) & 0x3FF, 10) / 64.0
        self._vertex((x, y, z))

    def vertex_xy(self, v: int):
        self._vertex((fx16(v), fx16(v >> 16), self.last_vtx[2]))

    def vertex_xz(self, v: int):
        self._vertex((fx16(v), self.last_vtx[1], fx16(v >> 16)))

    def vertex_yz(self, v: int):
        self._vertex((self.last_vtx[0], fx16(v), fx16(v >> 16)))

    def vertex_diff(self, v: int):
        dx = sign_extend(v & 0x3FF, 10) / FX16_SCALE
        dy = sign_extend((v >> 10) & 0x3FF, 10) / FX16_SCALE
        dz = sign_extend((v >> 20) & 0x3FF, 10) / FX16_SCALE
        self._vertex(
            (self.last_vtx[0] + dx, self.last_vtx[1] + dy, self.last_vtx[2] + dz))

    def get_pos_mtx(self) -> np.ndarray:
        return self._first_pos if self._first_pos is not None else self.cur_pos

    def get_dir_mtx(self) -> np.ndarray:
        return self._first_dir if self._first_dir is not None else self.cur_dir


def _s16(v: int) -> int:
    return sign_extend(v & 0xFFFF, 16)


def _s32(v: int) -> int:
    return v - 0x100000000 if v >= 0x80000000 else v


def fx16(v: int) -> float:
    return _s16(v) / FX16_SCALE


def fx32(v: int) -> float:
    return _s32(v) / FX32_SCALE


def _fx32list(vs: list[int]) -> list[float]:
    return list(map(fx32, vs))


def _expand5(v: int) -> int:
    return (v * 255 + 15) // 31


def _bgr555_to_float(rgb: int) -> tuple[float, float, float]:
    b = _expand5((rgb >> 10) & 0x1F)
    g = _expand5((rgb >> 5) & 0x1F)
    r = _expand5(rgb & 0x1F)
    return (r / 255.0, g / 255.0, b / 255.0)


class DlCmd(IntEnum):
    NOP = 0x00
    MTX_MODE = 0x10
    PUSH_MTX = 0x11
    POP_MTX = 0x12
    STORE_MTX = 0x13
    RESTORE_MTX = 0x14
    IDENTITY = 0x15
    LOAD_MTX44 = 0x16
    LOAD_MTX43 = 0x17
    MUL_MTX44 = 0x18
    MUL_MTX43 = 0x19
    MUL_MTX33 = 0x1A
    SCALE = 0x1B
    TRANSLATE = 0x1C
    COLOR = 0x20
    NORMAL = 0x21
    TEXCOORD = 0x22
    VERTEX = 0x23
    VERTEX_10 = 0x24
    VERTEX_XY = 0x25
    VERTEX_XZ = 0x26
    VERTEX_YZ = 0x27
    VERTEX_DIFF = 0x28
    POLY_ATTR = 0x29
    TEX_IMG_PARAM = 0x2A
    TEX_PLTT_BASE = 0x2B
    MAT_COL_0 = 0x30
    MAT_COL_1 = 0x31
    LIGHT_VEC = 0x32
    LIGHT_COL = 0x33
    SHININESS = 0x34
    BEGIN = 0x40
    END = 0x41
    SWAP_BUFFERS = 0x50
    VIEWPORT = 0x60
    BOXTEST = 0x70
    POSTEST = 0x71
    VECTEST = 0x72
