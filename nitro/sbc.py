
from .mdl0 import Model, SrtFlag, NodeData
from .dl import GeometryBuilder, MtxMode
from .binary import read_u32_le
from . import matrix as mat
import numpy as np
from enum import IntEnum, IntFlag
from dataclasses import dataclass


class SbcCmd(IntEnum):
    NOP = 0
    RET = 1
    NODE = 2
    MTX = 3
    MAT = 4
    SHP = 5
    NODEDESC = 6
    BB = 7
    BBY = 8
    NODEMIX = 9
    CALLDL = 10
    POSSCALE = 11
    ENVMAP = 12
    PRJMAP = 13

    CMD_MASK = 0x1F
    FLAG_MASK = 0xE0


class SbcOpt(IntFlag):
    NONE = 0x00
    STORE = 0x20
    RESTORE = 0x40


_PIVOT = [
    [4, 5, 7, 8],
    [3, 5, 6, 8],
    [3, 4, 6, 7],
    [1, 2, 7, 8],
    [0, 2, 6, 8],
    [0, 1, 6, 7],
    [1, 2, 4, 5],
    [0, 2, 3, 5],
    [0, 1, 3, 4],
]

_PIVOT_TABLE = [
    [
        [(1, 1), (1, 2), (2, 1), (2, 2)],
        [(1, 0), (1, 2), (2, 0), (2, 2)],
        [(1, 0), (1, 1), (2, 0), (2, 1)]
    ],
    [
        [(0, 1), (0, 2), (2, 1), (2, 2)],
        [(0, 0), (0, 2), (2, 0), (2, 2)],
        [(0, 0), (0, 1), (2, 0), (2, 1)]
    ],
    [
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 0), (0, 2), (1, 0), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
]


class SbcInterpreter:
    def __init__(self, model: Model, builder: GeometryBuilder, mat_tex_dims: dict[int, tuple[int, int]]):
        self.model = model
        self.builder = builder
        self.mat_tex_dims = mat_tex_dims
        self.sbc = model.sbc
        self.nodes = model.nodes.nodes
        self.materials = model.materials.materials
        self.shapes = model.shapes.shapes
        self.evp = model.evp_matrices

        n = len(self.nodes)
        self.node_parent = [-1] * n
        self.node_world = [mat.identity() for _ in range(n)]
        self.node_seen = [False] * n

        # Shared with builder now so shape DLs can adjust the bound node
        # when running a RestoreMtx command
        self.stack_to_node: dict[int, int] = builder.stack_to_node
        self.current_node = -1
        self.current_mat = 0
        self.draw_calls: list[DrawCall] = []

    def run(self):
        sbc = self.sbc
        off = 0
        n = len(sbc)
        guard = 0
        while off < n:
            guard += 1
            if guard > 1_000_000:
                break
            opcode = sbc[off]
            cmd = opcode & SbcCmd.CMD_MASK
            opt = opcode & SbcCmd.FLAG_MASK

            match cmd:
                case SbcCmd.NOP:
                    off += 1
                case SbcCmd.RET:
                    break
                case SbcCmd.NODE:
                    off += 3
                case SbcCmd.MTX:
                    idx = sbc[off + 1]
                    self.builder.restore_mtx(idx)
                    self.current_node = self.stack_to_node.get(
                        idx, self.current_node)
                    off += 2
                case SbcCmd.MAT:
                    off = self._do_mat(off)
                case SbcCmd.SHP:
                    off = self._do_shp(off)
                case SbcCmd.NODEDESC:
                    off = self._do_nodedesc(off, opt)
                case SbcCmd.BB | SbcCmd.BBY:
                    off = self._do_billboard(off, opt)
                case SbcCmd.NODEMIX:
                    off = self._do_nodemix(off)
                case SbcCmd.CALLDL:
                    off = self._do_calldl(off)
                case SbcCmd.POSSCALE:
                    off = self._do_posscale(off, opt)
                case SbcCmd.ENVMAP:
                    off += 3
                case SbcCmd.PRJMAP:
                    off += 3
                case _:
                    off += 1

    def _node_rot(self, node: NodeData) -> np.ndarray | None:
        if node.hasflag(SrtFlag.ROTATION_ZERO):
            return None
        if node.hasflag(SrtFlag.HAS_PIVOT):
            a, b = node.a, node.b
            row, col = node.pivot_idx()

            r = np.zeros((3, 3))
            r[row, col] = -1.0 if node.hasflag(SrtFlag.PIVOT_NEGATIVE) else 1.0
            piv = _PIVOT_TABLE[row][col]
            r[piv[0]] = a
            r[piv[1]] = b
            r[piv[2]] = -b if node.hasflag(SrtFlag.SIGN_REVC) else b
            r[piv[3]] = -a if node.hasflag(SrtFlag.SIGN_REVD) else a
            return r
        return np.array(node.rot_mtx()).reshape(3, 3)

    def _apply_joint(self, node_id: int):
        node = self.nodes[node_id]
        b = self.builder
        b.mtx_mode = MtxMode.POSITION_VECTOR
        trans_zero = node.hasflag(SrtFlag.TRANSLATION_ZERO)
        rot_zero = node.hasflag(SrtFlag.ROTATION_ZERO)
        scale_one = node.hasflag(SrtFlag.SCALE_ONE)
        rot3x3 = self._node_rot(node)
        trans = node.translation()
        if not trans_zero:
            if not rot_zero:
                b.mul(mat.from_rot_trans(rot3x3, trans))
            else:
                b.translate_vec(trans[0], trans[1], trans[2])
        elif not rot_zero:
            b.mul(mat.from3x3(rot3x3))
        if not scale_one:
            b.scale_vec(node.sx, node.sy, node.sz)

    def _stack_operands(self, off: int, opt: int, base: int) -> tuple[int | None, int | None, int]:
        n, store, restore = base, None, None
        if opt & SbcOpt.STORE:
            store, n = self.sbc[off + n], n + 1
        if opt & SbcOpt.RESTORE:
            restore, n = self.sbc[off + n], n + 1
        return store, restore, n

    def _draw_call(self, dl: bytes | bytearray, shape: int):
        b = self.builder

        # Using the current node from *before* running the DL as it can now modify the bound node
        node = max(self.current_node, 0)
        b.current_bound_node = node
        start = len(b.triangles)
        b.run_dl(dl)

        self.draw_calls.append(DrawCall(
            shape,
            self.current_mat,
            node,
            start,
            len(b.triangles),
            b.get_pos_mtx(),
            b.get_dir_mtx(),
            b.single_matrix
        ))

    def _do_mat(self, off: int) -> int:
        idx = self.sbc[off + 1]
        self.current_mat = idx
        if idx < len(self.materials):
            self.builder.tex_image_param(self.materials[idx].tex_image_param)
            dims = self.mat_tex_dims.get(idx)
            if dims:
                self.builder.tex_width, self.builder.tex_height = dims
        return off + 2

    def _do_shp(self, off: int) -> int:
        idx = self.sbc[off + 1]
        self._draw_call(self.shapes[idx].dl, idx)
        return off + 2

    def _do_billboard(self, off: int, opt: int) -> int:
        b = self.builder
        store, restore, num = self._stack_operands(off, opt, 2)
        if restore is not None:
            b.restore_mtx(restore)
            self.current_node = self.stack_to_node.get(
                restore, self.current_node)
        if store is not None:
            b.store_mtx(store)
            self.stack_to_node[store] = self.current_node
        return off + num

    def _do_nodedesc(self, off: int, opt: int) -> int:
        b = self.builder
        node_id = self.sbc[off + 1]
        store, restore, num = self._stack_operands(off, opt, 4)
        if restore is not None:
            b.restore_mtx(restore)
            self.current_node = self.stack_to_node.get(
                restore, self.current_node)

        parent = self.current_node
        self._apply_joint(node_id)
        self.node_parent[node_id] = parent
        self.node_world[node_id] = b.cur_pos
        self.node_seen[node_id] = True
        if store is not None:
            b.store_mtx(store)
            self.stack_to_node[store] = node_id

        self.current_node = node_id
        return off + num

    def _do_nodemix(self, off: int) -> int:
        b = self.builder
        store_idx = self.sbc[off + 1]
        num_terms = self.sbc[off + 2]
        sum_m = np.zeros((4, 4))
        sum_n = np.zeros((4, 4))
        first_node = self.current_node

        i = off + 3
        for term in range(num_terms):
            stack_idx = self.sbc[i]
            evp_idx = self.sbc[i + 1]
            weight = self.sbc[i + 2] / 255.0
            if term == 0:
                first_node = evp_idx
            if self.evp is not None and evp_idx < len(self.evp.m):
                inv_m = mat.from4x3(self.evp.m[evp_idx].inv_m)
                inv_n = mat.from3x3(self.evp.m[evp_idx].inv_n)
                m_term = np.dot(inv_m, b.pos_stack[stack_idx])
                n_term = np.dot(inv_n, b.dir_stack[stack_idx])
            else:
                m_term = b.pos_stack[stack_idx]
                n_term = b.dir_stack[stack_idx]

            sum_m[:, :3] += weight * m_term[:, :3]
            sum_n[:3, :3] += weight * n_term[:3, :3]
            i += 3

        sum_m[3, 3] = sum_n[3, 3] = 1.0
        b.cur_pos, b.cur_dir = sum_m, sum_n

        b.store_mtx(store_idx)
        self.stack_to_node[store_idx] = first_node
        self.current_node = first_node
        return off + 3 + num_terms * 3

    def _do_calldl(self, off: int) -> int:
        addr = read_u32_le(self.sbc, off + 1)
        length = read_u32_le(self.sbc, off + 5)
        dl = bytes(self.sbc[off + addr: off + addr + length])
        self._draw_call(dl, -1)
        return off + 9

    def _do_posscale(self, off: int, opt: int) -> int:
        scale = self.model.info.inv_pos_scale if opt else self.model.info.pos_scale
        self.builder.scale_vec(scale, scale, scale)
        return off + 1


@dataclass(slots=True)
class DrawCall:
    shape: int
    material: int
    node: int
    tri_start: int
    tri_end: int

    bind_pos: np.ndarray | None = None
    bind_dir: np.ndarray | None = None
    single_mtx: bool | None = None
