
from __future__ import annotations
from .binary import *
from . import matrix as mat
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from itertools import islice
from typing import Callable


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


_NORMAL_SCALE = 1.0 / 512.0
_VERTEX10_SCALE = 1.0 / 64.0
_TEXCOORD_SCALE = 1.0 / 16.0
_MTX_STACK_MASK = 0x1F
_ALPHA_MASK = 0x1F


@dataclass(slots=True)
class Vertex:
    pos: tuple[float, float, float]
    normal: tuple[float, float, float] | None
    uv: tuple[float, float] | None
    color: tuple[float, float, float]
    node: int


Triangle = tuple[Vertex, Vertex, Vertex]


class GeometryBuilder:
    def __init__(self):
        # TODO: Check if this is 31 or 32
        self.pos_stack = [np.identity(4) for _ in range(32)]
        self.dir_stack = [np.identity(4) for _ in range(32)]
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
        self.triangles: list[Triangle] = []

        self._prim_type: PrimType | None = None
        self._prim: list[Vertex] = []

        self.num_vertices_emitted = 0
        self.num_triangles = 0
        self.num_quads = 0

        self._first_pos = None
        self._first_dir = None
        self._first_pos_ref = None
        self._single_matrix = True

    @property
    def single_matrix(self):
        return self._single_matrix

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

    def _flush_primitives(self):
        verts = self._prim
        self._prim = []

        if self._prim_type is None or len(verts) < 3:
            return

        match self._prim_type:
            case PrimType.TRIANGLES:
                self.num_triangles += len(verts) // 3
                # An incomplete trailing primitive is discarded, as on hardware
                for group in _batched(verts, 3):
                    if len(group) == 3:
                        self.triangles.append(group)

            case PrimType.QUADS:
                self.num_quads += len(verts) // 4
                for group in _batched(verts, 4):
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

    def _exec(self, cmd: int, dl: bytes | bytearray, off: int) -> int:
        words, handler = _DL_HANDLERS.get(cmd, (0, None))
        if handler is not None:
            params = [read_u32_le(dl, off + i * 4) for i in range(words)]
            getattr(self, handler)(*params)

        return off + words * 4

    def set_mtx_mode(self, mode: MtxMode | int):
        self.mtx_mode = MtxMode(mode)

    def _targets(self, *, include_dir=True) -> tuple[str, ...]:
        match self.mtx_mode:
            case MtxMode.POSITION: return ("cur_pos",)
            case MtxMode.POSITION_VECTOR: return ("cur_pos", "cur_dir") if include_dir else ("cur_pos",)
            case MtxMode.TEXTURE: return ("cur_tex",)
            case _: return ()

    def _apply(self, fn: Callable[[np.ndarray], np.ndarray], *, include_dir=True):
        for name in self._targets(include_dir=include_dir):
            setattr(self, name, fn(getattr(self, name)))

    def _stack_targets(self) -> tuple[tuple[str, list[np.ndarray]], ...]:
        match self.mtx_mode:
            case MtxMode.POSITION: return (("cur_pos", self.pos_stack),)
            case MtxMode.POSITION_VECTOR:
                return (("cur_pos", self.pos_stack), ("cur_dir", self.dir_stack))
            case _: return ()

    def push_mtx(self):
        for name, stack in self._stack_targets():
            stack[self.stack_ptr] = getattr(self, name)
        self.stack_ptr += 1

    def pop_mtx(self, cmd: int):
        self.stack_ptr -= sign_extend(cmd & 0x3F, 6)
        for name, stack in self._stack_targets():
            setattr(self, name, stack[self.stack_ptr])

    def store_mtx(self, index: int):
        index &= _MTX_STACK_MASK
        for name, stack in self._stack_targets():
            stack[index] = getattr(self, name)

    def restore_mtx(self, index: int):
        index &= _MTX_STACK_MASK
        for name, stack in self._stack_targets():
            setattr(self, name, stack[index])

    def identity(self):
        self._apply(lambda _: np.identity(4))

    def _load(self, m: np.ndarray):
        self._apply(lambda _: m)

    def load_mtx44(self, *vals: int):
        self._load(mat.from4x4(_fx32list(vals)))

    def load_mtx43(self, *vals: int):
        self._load(mat.from4x3(_fx32list(vals)))

    def mul(self, m: np.ndarray):
        self._apply(lambda cur: np.dot(m, cur))

    def mul_mtx44(self, *vals: int):
        self.mul(mat.from4x4(_fx32list(vals)))

    def mul_mtx43(self, *vals: int):
        self.mul(mat.from4x3(_fx32list(vals)))

    def mul_mtx33(self, *vals: int):
        self.mul(mat.from3x3(_fx32list(vals)))

    def scale(self, sx: int, sy: int, sz: int):
        self.scale_vec(fx32(sx), fx32(sy), fx32(sz))

    def scale_vec(self, x: float, y: float, z: float):
        # Skip the direction matrix even in POSITION_VECTOR mode
        m = mat.scale(x, y, z)
        self._apply(lambda cur: np.dot(m, cur), include_dir=False)

    def translate(self, tx: int, ty: int, tz: int):
        self.translate_vec(fx32(tx), fx32(ty), fx32(tz))

    def translate_vec(self, x: float, y: float, z: float):
        self.mul(mat.translate(x, y, z))

    def color(self, rgb: int):
        self.last_col = bgr555_to_float(rgb)

    def normal(self, packed: int):
        (nx, ny, nz) = tuple(map(lambda x: x * _NORMAL_SCALE, unpack3x10(packed)))
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
        s = s16(packed) * _TEXCOORD_SCALE
        t = s16(packed >> 16) * _TEXCOORD_SCALE
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

    def begin(self, prim_type: PrimType | int):
        self._flush_primitives()
        self._prim_type = PrimType(prim_type)
        self._prim = []
        self.alpha = (self.next_poly_attr >> 16) & _ALPHA_MASK

    def end(self):
        self._flush_primitives()

    def _vertex(self, v: tuple[float, float, float]):
        self.last_vtx = v
        self._emit_vertex(mat.mul(v, self.cur_pos))

    def vertex(self, cmd1: int, cmd2: int):
        x = fx16(cmd1)
        y = fx16(cmd1 >> 16)
        z = fx16(cmd2)
        self._vertex((x, y, z))

    def vertex10(self, v: int):
        (x, y, z) = tuple(map(lambda x: x * _VERTEX10_SCALE, unpack3x10(v)))
        self._vertex((x, y, z))

    def vertex_xy(self, v: int):
        self._vertex((fx16(v), fx16(v >> 16), self.last_vtx[2]))

    def vertex_xz(self, v: int):
        self._vertex((fx16(v), self.last_vtx[1], fx16(v >> 16)))

    def vertex_yz(self, v: int):
        self._vertex((self.last_vtx[0], fx16(v), fx16(v >> 16)))

    def vertex_diff(self, v: int):
        (dx, dy, dz) = tuple(map(lambda x: x / FX16_SCALE, unpack3x10(v)))
        self._vertex(
            (self.last_vtx[0] + dx, self.last_vtx[1] + dy, self.last_vtx[2] + dz))

    def poly_attr(self, cmd: int):
        self.next_poly_attr = cmd

    def get_pos_mtx(self) -> np.ndarray:
        return self._first_pos if self._first_pos is not None else self.cur_pos

    def get_dir_mtx(self) -> np.ndarray:
        return self._first_dir if self._first_dir is not None else self.cur_dir


def _fx32list(vs: list[int]) -> list[float]:
    return list(map(fx32, vs))


def _batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


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


# CMD -> (words, handler, pass as list)
_DL_HANDLERS: dict[DlCmd, tuple[int, str | None]] = {
    DlCmd.NOP: (0, None),
    DlCmd.MTX_MODE: (1, "set_mtx_mode"),
    DlCmd.PUSH_MTX: (0, "push_mtx"),
    DlCmd.POP_MTX: (1, "pop_mtx"),
    DlCmd.STORE_MTX: (1, "store_mtx"),
    DlCmd.RESTORE_MTX: (1, "restore_mtx"),
    DlCmd.IDENTITY: (0, "identity"),
    DlCmd.LOAD_MTX44: (16, "load_mtx44"),
    DlCmd.LOAD_MTX43: (12, "load_mtx43"),
    DlCmd.MUL_MTX44: (16, "mul_mtx44"),
    DlCmd.MUL_MTX43: (12, "mul_mtx43"),
    DlCmd.MUL_MTX33: (9, "mul_mtx33"),
    DlCmd.SCALE: (3, "scale"),
    DlCmd.TRANSLATE: (3, "translate"),
    DlCmd.COLOR: (1, "color"),
    DlCmd.NORMAL: (1, "normal"),
    DlCmd.TEXCOORD: (1, "texcoord"),
    DlCmd.VERTEX: (2, "vertex"),
    DlCmd.VERTEX_10: (1, "vertex10"),
    DlCmd.VERTEX_XY: (1, "vertex_xy"),
    DlCmd.VERTEX_XZ: (1, "vertex_xz"),
    DlCmd.VERTEX_YZ: (1, "vertex_yz"),
    DlCmd.VERTEX_DIFF: (1, "vertex_diff"),
    DlCmd.POLY_ATTR: (1, "poly_attr"),
    DlCmd.TEX_IMG_PARAM: (1, "tex_image_param"),
    DlCmd.TEX_PLTT_BASE: (1, None),
    DlCmd.MAT_COL_0: (1, None),
    DlCmd.MAT_COL_1: (1, None),
    DlCmd.LIGHT_VEC: (1, None),
    DlCmd.LIGHT_COL: (1, None),
    DlCmd.SHININESS: (32, None),
    DlCmd.BEGIN: (1, "begin"),
    DlCmd.END: (0, "end"),
    DlCmd.SWAP_BUFFERS: (1, None),
    DlCmd.VIEWPORT: (1, None),
    DlCmd.BOXTEST: (3, None),
    DlCmd.POSTEST: (2, None),
    DlCmd.VECTEST: (1, None),
}
