
import numpy as np

from . import matrix as mat
from .binary import to_fx
from .dl import DlCmd
from .model import ImportedSubModel

S16_MIN, S16_MAX = -0x8000, 0x7FFF


def exponent_for(lo: int, hi: int) -> int:
    """Smallest e with (lo >> e) >= -0x8000 and (hi >> e) <= 0x7fff."""
    e = 0
    while lo < S16_MIN or hi > S16_MAX:
        lo, hi, e = lo >> 1, hi >> 1, e + 1
    return e


def local_extent(sub: ImportedSubModel) -> tuple[int, int]:
    """Fixed-point min/max over every vertex of an `ImportedSubModel`."""
    inv_cache: dict[int, np.ndarray] = {}
    lo = hi = None

    for mesh in sub.meshes:
        for v, node in zip(mesh.vertices, mesh.vertex_bone):
            inv_n = inv_cache.get(node)
            if inv_n is None:
                inv_n = inv_cache[node] = mat.inverse(
                    sub.bones[node].world_mtx)
            for c in mat.mul(v, inv_n):
                n = to_fx(c)          # quantise first, then accumulate
                if lo is None or n < lo:
                    lo = n
                if hi is None or n > hi:
                    hi = n

    return (0, 0) if lo is None else (lo, hi)


def pos_scale_for(sub: ImportedSubModel) -> float:
    """Computes Model-wide geometry scale."""
    return float(1 << exponent_for(*local_extent(sub)))


def box_exponent_for(corner: tuple[float, float, float], extent: tuple[float, float, float]) -> int:
    """`box_test` picks its own exponent, over min[xyz], max[xyz], ext[xyz]."""
    lo3 = [to_fx(v) for v in corner]
    ext = [to_fx(v) for v in extent]
    vals = lo3 + [a + b for a, b in zip(lo3, ext)] + ext
    return exponent_for(min(vals), max(vals))


def decide_vertex_form(cur: tuple[int, int, int] | list[int], prev: tuple[int, int, int] | list[int] | None) -> DlCmd:
    """Decide the next vertex form, given the current vertex and optionally a previous vertex."""
    if prev is not None:
        if cur[2] == prev[2]:
            return DlCmd.VERTEX_XY
        if cur[1] == prev[1]:
            return DlCmd.VERTEX_XZ
        if cur[0] == prev[0]:
            return DlCmd.VERTEX_YZ

    if all((c & 0x3F) == 0 for c in cur):
        return DlCmd.VERTEX_10

    if prev is None:
        return DlCmd.VERTEX

    if all(-0x200 <= c - p < 0x200 for c, p in zip(cur, prev)):
        return DlCmd.VERTEX_DIFF

    return DlCmd.VERTEX
