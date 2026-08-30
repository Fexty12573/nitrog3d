import struct

try:
    from itertools import batched
except ImportError:  # Python < 3.12
    from itertools import islice

    def batched(iterable, n):
        it = iter(iterable)
        while group := tuple(islice(it, n)):
            yield group

import numpy as np
import pytest

from nitro import matrix as mat
from nitro.dl import (DlCmd, GeometryBuilder, MtxMode, PrimType, TexGen,
                      fx16, fx32)


def fx16_raw(v: float) -> int:
    return int(round(v * 4096)) & 0xFFFF


def fx32_raw(v: float) -> int:
    return int(round(v * 4096)) & 0xFFFFFFFF


def p_u32(*vals: int) -> bytes:
    return struct.pack(f"<{len(vals)}I", *[v & 0xFFFFFFFF for v in vals])


def p_fx32(*vals: float) -> bytes:
    return p_u32(*[fx32_raw(v) for v in vals])


def p_vertex(x: float, y: float, z: float) -> bytes:
    return p_u32(fx16_raw(x) | (fx16_raw(y) << 16), fx16_raw(z))


def p_vertex10(x: float, y: float, z: float) -> bytes:
    def q(v):
        return int(round(v * 64)) & 0x3FF
    return p_u32(q(x) | (q(y) << 10) | (q(z) << 20))


def p_normal(nx: float, ny: float, nz: float) -> bytes:
    def q(v):
        return int(round(v * 512)) & 0x3FF
    return p_u32(q(nx) | (q(ny) << 10) | (q(nz) << 20))


def p_mtx43(tx=0.0, ty=0.0, tz=0.0, basis=None) -> bytes:
    """A 4x3 matrix in NITRO row-major order: three basis rows then translation."""
    basis = basis if basis is not None else (1, 0, 0, 0, 1, 0, 0, 0, 1)
    return p_fx32(*basis, tx, ty, tz)


# 90 degrees about Z, row-major: (1,0,0) -> (0,1,0)
ROT_Z90 = (0, 1, 0, -1, 0, 0, 0, 0, 1)


def make_dl(*commands: tuple[int, bytes]) -> bytes:
    """Pack (command, params) pairs into NITRO display-list form.

    Commands travel four-per-word with all four parameter blocks following,
    so the tail is padded with NOPs to keep whole groups.
    """
    cmds = list(commands)
    while len(cmds) % 4 != 0:
        cmds.append((DlCmd.NOP, b""))

    out = bytearray()
    for group in batched(cmds, 4):
        out += bytes(int(c) for c, _ in group)
        for _, params in group:
            out += params
    return bytes(out)


def tri_dl(prim=PrimType.TRIANGLES,
           verts=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))) -> bytes:
    cmds = [(DlCmd.BEGIN, p_u32(int(prim)))]
    cmds += [(DlCmd.VERTEX, p_vertex(*v)) for v in verts]
    cmds.append((DlCmd.END, b""))
    return make_dl(*cmds)


def run(*commands: tuple[int, bytes]) -> GeometryBuilder:
    b = GeometryBuilder()
    b.run_dl(make_dl(*commands))
    return b


class TestFixedPoint:
    def test_fx16_converts_signed_1_12(self):
        assert fx16(0x1000) == pytest.approx(1.0)
        assert fx16(0xF000) == pytest.approx(-1.0)
        assert fx16(0) == 0.0

    def test_fx16_ignores_high_half(self):
        # fx16 masks to 16 bits so a packed pair decodes independently
        assert fx16(0xDEAD1000) == pytest.approx(1.0)

    def test_fx32_converts_signed_19_12(self):
        assert fx32(0x1000) == pytest.approx(1.0)
        assert fx32(0xFFFFF000) == pytest.approx(-1.0)


class TestMatrixCommands:
    def test_mtx_mode_sets_mode(self):
        b = run((DlCmd.MTX_MODE, p_u32(int(MtxMode.TEXTURE))))
        assert b.mtx_mode == MtxMode.TEXTURE

    def test_push_then_pop_restores_matrix(self):
        b = run(
            (DlCmd.PUSH_MTX, b""),
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0, ty=2.0, tz=3.0)),
            (DlCmd.POP_MTX, p_u32(1)),
        )
        assert b.stack_ptr == 0
        assert b.cur_pos == pytest.approx(np.identity(4))

    def test_push_advances_stack_pointer(self):
        b = run((DlCmd.PUSH_MTX, b""), (DlCmd.PUSH_MTX, b""))
        assert b.stack_ptr == 2

    def test_pop_mtx_sign_extends_6_bit_count(self):
        # A negative pop count moves the stack pointer back up
        b = GeometryBuilder()
        b.stack_ptr = 4
        b.pop_mtx(0x3F)  # -1
        assert b.stack_ptr == 5

    def test_store_and_restore_mtx_round_trip(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=4.0, ty=5.0, tz=6.0)),
            (DlCmd.STORE_MTX, p_u32(7)),
            (DlCmd.LOAD_MTX43, p_mtx43()),
        ))
        stored = b.pos_stack[7]
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 0.0, 0.0))

        b.run_dl(make_dl((DlCmd.RESTORE_MTX, p_u32(7))))
        assert b.cur_pos is stored
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (4.0, 5.0, 6.0))

    def test_store_mtx_masks_index_to_5_bits(self):
        b = GeometryBuilder()
        b.cur_pos = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0, 0.0, 0.0])
        b.store_mtx(0x27)  # 0x27 & 31 == 7
        assert b.pos_stack[7] is b.cur_pos

    def test_load_mtx43_places_translation_in_last_row(self):
        b = run((DlCmd.LOAD_MTX43, p_mtx43(tx=2.0, ty=3.0, tz=4.0)))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (2.0, 3.0, 4.0))

    def test_load_mtx44_loads_all_16_values(self):
        # Every cell distinct, so a transposed or short read fails here.
        vals = list(range(1, 17))
        b = run((DlCmd.LOAD_MTX44, p_fx32(*vals)))
        assert b.cur_pos == pytest.approx(mat.from4x4(vals))

    def test_load_mtx43_loads_all_12_values(self):
        vals = list(range(1, 13))
        b = run((DlCmd.LOAD_MTX43, p_fx32(*vals)))
        assert b.cur_pos == pytest.approx(mat.from4x3(vals))

    def test_load_replaces_rather_than_composes(self):
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.LOAD_MTX43, p_mtx43(tx=2.0)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (2.0, 0.0, 0.0))

    def test_mul_applies_the_operand_before_the_current_matrix(self):
        # The whole point of the convention: v @ operand @ current. Stated once
        # over the full matrix; the tests below check it through visible effects.
        prior = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0, 2.0, 3.0])
        operand = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0, ty=2.0, tz=3.0)),
            (DlCmd.MUL_MTX44, p_fx32(*operand)),
        )
        assert b.cur_pos == pytest.approx(
            np.matmul(mat.from4x4(operand), prior))

    def test_mul_mtx43_composes_with_current(self):
        # Rotate, then multiply in a translation: the translation happens first
        # and so gets rotated. Reversing the operands would give (1, 0, 0).
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(basis=ROT_Z90)),
            (DlCmd.MUL_MTX43, p_mtx43(tx=1.0)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 1.0, 0.0))

    def test_mul_mtx44_composes_with_current(self):
        # Translate, then multiply in a scale: the scale runs first, so it does
        # not multiply the translation. Reversing would give (2, 0, 0).
        double = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.MUL_MTX44, p_fx32(*double)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 0.0, 0.0))

    def test_mul_mtx33_composes_with_current(self):
        # Scale-by-2 ahead of a translate-by-1: (1,1,1) -> (2,2,2) -> (3,2,2).
        # Reversing the operands would give (4, 2, 2).
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.MUL_MTX33, p_fx32(2, 0, 0, 0, 2, 0, 0, 0, 2)),
        )
        assert mat.mul((1.0, 1.0, 1.0), b.cur_pos) == pytest.approx(
            (3.0, 2.0, 2.0))

    def test_mul_mtx33_contributes_no_translation(self):
        b = run((DlCmd.MUL_MTX33, p_fx32(*ROT_Z90)))
        assert b.cur_pos[3, :].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 1.0, 0.0))

    def test_identity_resets_the_current_matrix(self):
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=5.0, ty=5.0, tz=5.0)),
            (DlCmd.IDENTITY, b""),
        )
        assert b.cur_pos == pytest.approx(np.identity(4))
        assert b.cur_dir == pytest.approx(np.identity(4))

    def test_identity_in_texture_mode_only_resets_texture_matrix(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=5.0)),
            (DlCmd.MTX_MODE, p_u32(int(MtxMode.TEXTURE))),
            (DlCmd.SCALE, p_fx32(2.0, 2.0, 2.0)),
        ))
        # Guard the guard: IDENTITY clearing cur_tex only proves something if
        # cur_tex was dirty to begin with.
        assert b.cur_tex != pytest.approx(np.identity(4))

        b.run_dl(make_dl((DlCmd.IDENTITY, b"")))
        assert b.cur_tex == pytest.approx(np.identity(4))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (5.0, 0.0, 0.0))

    def test_scale_command(self):
        b = run((DlCmd.SCALE, p_fx32(2.0, 3.0, 4.0)))
        assert mat.mul((1.0, 1.0, 1.0), b.cur_pos) == pytest.approx(
            (2.0, 3.0, 4.0))

    def test_scale_composes_before_the_current_matrix(self):
        # Scale runs first, so the already-loaded translation is not scaled.
        # Reversing the operands would give (2, 0, 0).
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.SCALE, p_fx32(2.0, 2.0, 2.0)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 0.0, 0.0))
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (3.0, 0.0, 0.0))

    def test_translate_command_moves_the_origin(self):
        b = run((DlCmd.TRANSLATE, p_fx32(1.0, -2.0, 0.5)))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, -2.0, 0.5))

    def test_translate_accumulates_with_the_current_matrix(self):
        b = run(
            (DlCmd.TRANSLATE, p_fx32(1.0, 0.0, 0.0)),
            (DlCmd.TRANSLATE, p_fx32(0.0, 2.0, 0.0)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 2.0, 0.0))

    def test_translate_composes_before_the_current_matrix(self):
        # Translation runs first and is therefore rotated by the loaded matrix.
        # Reversing the operands would give (1, 0, 0).
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(basis=ROT_Z90)),
            (DlCmd.TRANSLATE, p_fx32(1.0, 0.0, 0.0)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 1.0, 0.0))

    def test_translate_agrees_with_load_mtx43(self):
        # matrix.translate and matrix.from4x3 must use the same convention
        by_cmd = run((DlCmd.TRANSLATE, p_fx32(1.0, 2.0, 3.0)))
        by_load = run((DlCmd.LOAD_MTX43, p_mtx43(tx=1.0, ty=2.0, tz=3.0)))
        assert by_cmd.cur_pos == pytest.approx(by_load.cur_pos)

    def test_scale_does_not_touch_direction_matrix(self):
        # In POSITION_VECTOR mode the DS applies MTX_SCALE to the position
        # matrix only, leaving the vector matrix untouched.
        b = run((DlCmd.SCALE, p_fx32(2.0, 2.0, 2.0)))
        assert b.cur_dir == pytest.approx(np.identity(4))

    def test_mul_applies_to_direction_matrix_in_position_vector_mode(self):
        b = run((DlCmd.MUL_MTX33, p_fx32(2, 0, 0, 0, 2, 0, 0, 0, 2)))
        assert b.cur_dir == pytest.approx(b.cur_pos)
        assert mat.mul((1.0, 1.0, 1.0), b.cur_dir) == pytest.approx(
            (2.0, 2.0, 2.0))

    def test_mul_in_texture_mode_composes_the_texture_matrix(self):
        b = run(
            (DlCmd.MTX_MODE, p_u32(int(MtxMode.TEXTURE))),
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.MUL_MTX44, p_fx32(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1)),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.cur_tex) == pytest.approx(
            (1.0, 0.0, 0.0))
        assert b.cur_pos == pytest.approx(np.identity(4))

    def test_pop_mtx_restores_both_position_and_direction(self):
        b = run(
            (DlCmd.MUL_MTX33, p_fx32(*ROT_Z90)),
            (DlCmd.PUSH_MTX, b""),
            (DlCmd.IDENTITY, b""),
            (DlCmd.POP_MTX, p_u32(1)),
        )
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 1.0, 0.0))
        assert mat.mul((1.0, 0.0, 0.0), b.cur_dir) == pytest.approx(
            (0.0, 1.0, 0.0))

    def test_texture_mode_isolates_texture_matrix(self):
        b = run(
            (DlCmd.MTX_MODE, p_u32(int(MtxMode.TEXTURE))),
            (DlCmd.SCALE, p_fx32(2.0, 2.0, 2.0)),
        )
        assert b.cur_pos == pytest.approx(np.identity(4))
        assert mat.mul((1.0, 1.0, 1.0), b.cur_tex) == pytest.approx(
            (2.0, 2.0, 2.0))

    def test_position_mode_leaves_direction_matrix_alone(self):
        b = run(
            (DlCmd.MTX_MODE, p_u32(int(MtxMode.POSITION))),
            (DlCmd.MUL_MTX33, p_fx32(2, 0, 0, 0, 2, 0, 0, 0, 2)),
        )
        assert b.cur_dir == pytest.approx(np.identity(4))
        assert b.cur_pos != pytest.approx(np.identity(4))


class TestVertexCommands:
    def test_vertex_decodes_three_fx16_components(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 2.0, 3.0)),
        )
        assert b.last_vtx == pytest.approx((1.0, 2.0, 3.0))
        assert b.num_vertices_emitted == 1

    def test_vertex_handles_negative_components(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(-1.0, -0.5, -2.0)),
        )
        assert b.last_vtx == pytest.approx((-1.0, -0.5, -2.0))

    def test_vertex10_decodes_10_bit_components(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX_10, p_vertex10(1.0, -1.0, 0.5)),
        )
        assert b.last_vtx == pytest.approx((1.0, -1.0, 0.5))

    def test_vertex_xy_reuses_previous_z(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 1.0, 2.5)),
            (DlCmd.VERTEX_XY, p_u32(fx16_raw(2.0) | (fx16_raw(3.0) << 16))),
        )
        assert b.last_vtx == pytest.approx((2.0, 3.0, 2.5))

    def test_vertex_xz_reuses_previous_y(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 2.5, 1.0)),
            (DlCmd.VERTEX_XZ, p_u32(fx16_raw(2.0) | (fx16_raw(3.0) << 16))),
        )
        assert b.last_vtx == pytest.approx((2.0, 2.5, 3.0))

    def test_vertex_yz_reuses_previous_x(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(2.5, 1.0, 1.0)),
            (DlCmd.VERTEX_YZ, p_u32(fx16_raw(2.0) | (fx16_raw(3.0) << 16))),
        )
        assert b.last_vtx == pytest.approx((2.5, 2.0, 3.0))

    def test_vertex_diff_accumulates_onto_previous_vertex(self):
        delta = 8  # 8 / 4096 in each axis
        packed = delta | (delta << 10) | (delta << 20)
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 2.0, 3.0)),
            (DlCmd.VERTEX_DIFF, p_u32(packed)),
        )
        d = delta / 4096.0
        assert b.last_vtx == pytest.approx((1.0 + d, 2.0 + d, 3.0 + d))

    def test_vertex_diff_handles_negative_deltas(self):
        delta = -8 & 0x3FF
        packed = delta | (delta << 10) | (delta << 20)
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 1.0, 1.0)),
            (DlCmd.VERTEX_DIFF, p_u32(packed)),
        )
        d = 8 / 4096.0
        assert b.last_vtx == pytest.approx((1.0 - d, 1.0 - d, 1.0 - d))

    def test_vertex_is_transformed_by_current_matrix(self):
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=10.0)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 1.0)),
            (DlCmd.END, b""),
        )
        assert b.triangles[0][0].pos == pytest.approx((11.0, 0.0, 0.0))

    def test_last_vtx_stores_untransformed_position(self):
        # last_vtx feeds the VTX_XY/XZ/YZ/DIFF deltas, so it must stay in
        # model space rather than picking up the current matrix.
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=10.0)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
        )
        assert b.last_vtx == pytest.approx((1.0, 0.0, 0.0))

    def test_vertex_records_current_bound_node(self):
        b = GeometryBuilder()
        b.current_bound_node = 5
        b.run_dl(tri_dl())
        assert [v.node for v in b.triangles[0]] == [5, 5, 5]

    def test_vertex_captures_current_color(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (DlCmd.COLOR, p_u32(0x1F)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        ))
        assert b.triangles[0][0].color == pytest.approx((1.0, 0.0, 0.0))

    def test_vertex_captures_current_uv(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (DlCmd.TEXCOORD, p_u32((16 * 1) | ((16 * 2) << 16))),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        ))
        assert b.triangles[0][0].uv == pytest.approx((1.0, 2.0))


class TestAttributeCommands:
    def test_color_expands_bgr555(self):
        b = run((DlCmd.COLOR, p_u32(0x1F)))  # red = 31
        assert b.last_col == pytest.approx((1.0, 0.0, 0.0))

    def test_color_channel_order_is_bgr(self):
        b = run((DlCmd.COLOR, p_u32(31 << 10)))  # blue field
        assert b.last_col == pytest.approx((0.0, 0.0, 1.0))

    def test_color_green_field(self):
        b = run((DlCmd.COLOR, p_u32(31 << 5)))
        assert b.last_col == pytest.approx((0.0, 1.0, 0.0))

    def test_color_black_and_white_endpoints(self):
        assert run((DlCmd.COLOR, p_u32(0))
                   ).last_col == pytest.approx((0.0, 0.0, 0.0))
        assert run((DlCmd.COLOR, p_u32(0x7FFF))
                   ).last_col == pytest.approx((1.0, 1.0, 1.0))

    def test_color_expands_midpoints_over_the_full_range(self):
        # Endpoints alone can't distinguish v/31 from the rounded 5->8 bit
        # expansion the hardware uses, so pin an interior value too.
        b = run((DlCmd.COLOR, p_u32(15 | (15 << 5) | (15 << 10))))
        assert b.last_col == pytest.approx((123 / 255,) * 3)

    def test_color_ignores_bit_15(self):
        b = run((DlCmd.COLOR, p_u32(0x8000)))
        assert b.last_col == pytest.approx((0.0, 0.0, 0.0))

    def test_texcoord_scales_by_texture_size(self):
        b = GeometryBuilder()
        b.tex_width, b.tex_height = 32, 64
        b.run_dl(make_dl((DlCmd.TEXCOORD, p_u32((16 * 32) | ((16 * 64) << 16)))))
        assert b.last_tex == pytest.approx((32.0, 64.0))
        assert b.cur_uv == pytest.approx((1.0, 1.0))

    def test_texcoord_handles_negative_values(self):
        b = run((DlCmd.TEXCOORD, p_u32((-16 & 0xFFFF) | ((-32 & 0xFFFF) << 16))))
        assert b.last_tex == pytest.approx((-1.0, -2.0))

    def test_texcoord_texgen_texcoord_applies_texture_matrix(self):
        b = GeometryBuilder()
        b.tex_gen = TexGen.TEXCOORD
        b.cur_tex = mat.from4x3([2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0])
        b.run_dl(make_dl((DlCmd.TEXCOORD, p_u32((16 * 1) | ((16 * 1) << 16)))))
        # u = m00*s + m10*t + m20 + m30 = 2*1 + 0*1 + 0 + 0
        assert b.cur_uv == pytest.approx((2.0, 2.0))

    def test_tex_image_param_decodes_s_size_and_texgen(self):
        cmd = (3 << 20) | (int(TexGen.TEXCOORD) << 30)
        b = run((DlCmd.TEX_IMG_PARAM, p_u32(cmd)))
        assert b.tex_width == 8 << 3
        assert b.tex_gen == TexGen.TEXCOORD

    def test_tex_image_param_decodes_t_size_from_bits_23_25(self):
        b = run((DlCmd.TEX_IMG_PARAM, p_u32(2 << 23)))
        assert b.tex_height == 8 << 2

    def test_tex_image_param_s_and_t_sizes_are_independent(self):
        b = run((DlCmd.TEX_IMG_PARAM, p_u32((1 << 20) | (4 << 23))))
        assert (b.tex_width, b.tex_height) == (8 << 1, 8 << 4)

    def test_tex_image_param_ignores_repeat_and_flip_bits(self):
        # Bits 16-19 are repeat/flip S/T and must not leak into the sizes.
        b = run((DlCmd.TEX_IMG_PARAM, p_u32(0xF << 16)))
        assert (b.tex_width, b.tex_height) == (8, 8)

    def test_tex_image_param_minimum_size(self):
        b = run((DlCmd.TEX_IMG_PARAM, p_u32(0)))
        assert b.tex_width == 8
        assert b.tex_gen == TexGen.NONE

    def test_poly_attr_alpha_latches_on_begin(self):
        b = run(
            (DlCmd.POLY_ATTR, p_u32(7 << 16)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
        )
        assert b.alpha == 7

    def test_poly_attr_alone_does_not_change_alpha(self):
        b = run((DlCmd.POLY_ATTR, p_u32(7 << 16)))
        assert b.alpha == 31
        assert b.next_poly_attr == 7 << 16


class TestNormalCommand:
    def test_decodes_three_10_bit_components(self):
        b = run((DlCmd.NORMAL, p_normal(0.5, -0.5, 0.25)))
        assert b.cur_nrm == pytest.approx((0.5, -0.5, 0.25))

    def test_full_negative_scale_is_representable(self):
        # 10-bit signed over 512: -1.0 is exactly the sign bit.
        b = run((DlCmd.NORMAL, p_u32(512)))
        assert b.cur_nrm[0] == pytest.approx(-1.0)

    def test_largest_positive_component_is_511_over_512(self):
        b = run((DlCmd.NORMAL, p_u32(511)))
        assert b.cur_nrm[0] == pytest.approx(511 / 512)

    def test_is_rotated_by_the_direction_matrix(self):
        b = GeometryBuilder()
        b.cur_dir = mat.from3x3([0, 1, 0, 1, 0, 0, 0, 0, 1])  # swap x/y
        b.run_dl(make_dl((DlCmd.NORMAL, p_normal(0.5, -0.25, 0.0))))
        assert b.cur_nrm == pytest.approx((-0.25, 0.5, 0.0))

    def test_translation_is_not_applied_to_normals(self):
        b = GeometryBuilder()
        b.cur_dir = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 9.0, 9.0, 9.0])
        b.run_dl(make_dl((DlCmd.NORMAL, p_normal(0.5, 0.0, 0.0))))
        assert b.cur_nrm == pytest.approx((0.5, 0.0, 0.0))

    def test_texgen_normal_derives_uv_from_the_normal(self):
        b = GeometryBuilder()
        b.tex_gen = TexGen.NORMAL
        b.run_dl(make_dl((DlCmd.NORMAL, p_normal(0.5, -0.25, 0.0))))
        assert b.cur_uv == pytest.approx((0.5, -0.25))

    def test_texgen_none_leaves_uv_untouched(self):
        b = run((DlCmd.NORMAL, p_normal(0.5, -0.25, 0.0)))
        assert b.cur_uv is None


class TestPrimitiveAssembly:
    def test_triangles_emits_one_triangle_per_three_vertices(self):
        verts = [(float(i), 0.0, 0.0) for i in range(6)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRIANGLES, verts))
        assert len(b.triangles) == 2
        assert b.num_triangles == 2

    def test_triangles_preserves_vertex_order(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRIANGLES, verts))
        assert tuple(v.pos[0] for v in b.triangles[0]
                     ) == pytest.approx((0.0, 1.0, 2.0))

    def test_quads_split_into_two_triangles(self):
        verts = [(float(i), 0.0, 0.0) for i in range(4)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.QUADS, verts))

        assert b.num_quads == 1
        xs = [tuple(v.pos[0] for v in t) for t in b.triangles]
        assert xs == [(0.0, 1.0, 2.0), (0.0, 2.0, 3.0)]

    def test_quads_emits_two_quads(self):
        verts = [(float(i), 0.0, 0.0) for i in range(8)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.QUADS, verts))

        assert b.num_quads == 2
        assert len(b.triangles) == 4

    def test_tri_strip_alternates_winding(self):
        verts = [(float(i), 0.0, 0.0) for i in range(4)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRI_STRIP, verts))

        assert len(b.triangles) == 2
        assert b.num_triangles == 2
        xs = [tuple(v.pos[0] for v in t) for t in b.triangles]
        assert xs == [(0.0, 1.0, 2.0), (2.0, 1.0, 3.0)]

    def test_tri_strip_with_five_vertices_emits_three_triangles(self):
        verts = [(float(i), 0.0, 0.0) for i in range(5)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRI_STRIP, verts))

        xs = [tuple(v.pos[0] for v in t) for t in b.triangles]
        assert xs == [(0.0, 1.0, 2.0), (2.0, 1.0, 3.0), (2.0, 3.0, 4.0)]

    def test_tri_strip_minimum_three_vertices(self):
        verts = [(float(i), 0.0, 0.0) for i in range(3)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRI_STRIP, verts))
        assert len(b.triangles) == 1

    def test_quad_strip_pairs_vertices(self):
        verts = [(float(i), 0.0, 0.0) for i in range(4)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.QUAD_STRIP, verts))

        assert b.num_quads == 1
        xs = [tuple(v.pos[0] for v in t) for t in b.triangles]
        assert xs == [(0.0, 1.0, 3.0), (0.0, 3.0, 2.0)]

    def test_quad_strip_with_six_vertices_emits_two_quads(self):
        verts = [(float(i), 0.0, 0.0) for i in range(6)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.QUAD_STRIP, verts))

        assert b.num_quads == 2
        xs = [tuple(v.pos[0] for v in t) for t in b.triangles]
        assert xs == [(0.0, 1.0, 3.0), (0.0, 3.0, 2.0),
                      (2.0, 3.0, 5.0), (2.0, 5.0, 4.0)]

    def test_fewer_than_three_vertices_emits_nothing(self):
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRIANGLES, [
                 (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]))
        assert b.triangles == []
        assert b.num_triangles == 0
        assert b.num_vertices_emitted == 2

    @pytest.mark.parametrize("count,expected", [(3, 1), (4, 1), (5, 1), (6, 2), (7, 2)])
    def test_triangles_discards_incomplete_trailing_primitive(self, count, expected):
        verts = [(float(i), 0.0, 0.0) for i in range(count)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRIANGLES, verts))
        assert len(b.triangles) == expected
        assert b.num_triangles == expected

    @pytest.mark.parametrize("count,expected", [(4, 2), (5, 2), (6, 2), (7, 2), (8, 4)])
    def test_quads_discards_incomplete_trailing_primitive(self, count, expected):
        verts = [(float(i), 0.0, 0.0) for i in range(count)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.QUADS, verts))
        assert len(b.triangles) == expected
        assert b.num_quads == expected // 2

    def test_incomplete_tail_does_not_corrupt_the_kept_primitives(self):
        verts = [(float(i), 0.0, 0.0) for i in range(5)]
        b = GeometryBuilder()
        b.run_dl(tri_dl(PrimType.TRIANGLES, verts))
        assert tuple(v.pos[0] for v in b.triangles[0]
                     ) == pytest.approx((0.0, 1.0, 2.0))

    def test_end_without_begin_emits_nothing(self):
        b = run((DlCmd.END, b""))
        assert b.triangles == []

    def test_begin_flushes_pending_primitive(self):
        # Two BEGINs with no intervening END: the first batch still lands.
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
        )
        assert len(b.triangles) == 1

    def test_unterminated_primitive_is_dropped(self):
        # No END and no following BEGIN: nothing flushes it.
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
        )
        assert b.triangles == []

    def test_end_twice_does_not_duplicate_geometry(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
            (DlCmd.END, b""),
        )
        assert len(b.triangles) == 1

    def test_counters_accumulate_across_display_lists(self):
        b = GeometryBuilder()
        b.run_dl(tri_dl())
        b.run_dl(tri_dl())
        assert b.num_triangles == 2
        assert b.num_vertices_emitted == 6
        assert len(b.triangles) == 2


class TestBindMatrixTracking:
    def test_single_matrix_dl_is_flagged(self):
        b = GeometryBuilder()
        b.run_dl(tri_dl())
        assert b._single_matrix is True

    def test_matrix_change_mid_primitive_clears_single_matrix(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.RESTORE_MTX, p_u32(0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        )
        assert b._single_matrix is False

    def test_bind_matrices_snapshot_state_at_first_vertex(self):
        b = run(
            (DlCmd.LOAD_MTX43, p_mtx43(tx=1.0)),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.LOAD_MTX43, p_mtx43(tx=5.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        )
        assert mat.mul((0.0, 0.0, 0.0), b.get_pos_mtx()
                       ) == pytest.approx((1.0, 0.0, 0.0))

    def test_bind_matrix_is_a_copy_not_a_live_reference(self):
        b = run(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        )
        snapshot = np.copy(b.get_pos_mtx())
        b.cur_pos[0, 0] = 99.0
        assert b.get_pos_mtx() == pytest.approx(snapshot)

    def test_bind_matrices_fall_back_to_current_when_no_vertices(self):
        b = run((DlCmd.LOAD_MTX43, p_mtx43(tx=3.0)))
        assert b.get_pos_mtx() is b.cur_pos
        assert b.get_dir_mtx() is b.cur_dir

    def test_run_dl_resets_tracking_between_calls(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.RESTORE_MTX, p_u32(0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        ))
        assert b._single_matrix is False

        b.run_dl(tri_dl())
        assert b._single_matrix is True
        assert b._first_pos is not None


def assert_trailing_triangle_intact(b: GeometryBuilder, cmd):
    """The triangle after the command under test must be byte-for-byte intact.

    Checking the count alone is too weak: a desynchronised stream can still
    yield one triangle, just assembled from the wrong bytes.
    """
    assert len(b.triangles) == 1, f"{cmd!r} desynchronised the list"
    assert [v.pos for v in b.triangles[0]] == pytest.approx(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ), f"{cmd!r} desynchronised the list"


class TestCommandLengths:
    """Every command must consume exactly its operand bytes.

    A wrong length desynchronises the rest of the list, so each case runs the
    command under test ahead of a triangle and checks the triangle survives.
    """

    @pytest.mark.parametrize("cmd,params", [
        (DlCmd.NOP, b""),
        (DlCmd.MTX_MODE, p_u32(int(MtxMode.POSITION_VECTOR))),
        (DlCmd.PUSH_MTX, b""),
        (DlCmd.POP_MTX, p_u32(0)),
        (DlCmd.STORE_MTX, p_u32(1)),
        (DlCmd.RESTORE_MTX, p_u32(1)),
        (DlCmd.IDENTITY, b""),
        (DlCmd.NORMAL, p_normal(0.0, 0.0, 0.5)),
        (DlCmd.LOAD_MTX44, p_fx32(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)),
        (DlCmd.LOAD_MTX43, p_mtx43()),
        (DlCmd.MUL_MTX44, p_fx32(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)),
        (DlCmd.MUL_MTX43, p_mtx43()),
        (DlCmd.MUL_MTX33, p_fx32(1, 0, 0, 0, 1, 0, 0, 0, 1)),
        (DlCmd.SCALE, p_fx32(1.0, 1.0, 1.0)),
        (DlCmd.TRANSLATE, p_fx32(0.0, 0.0, 0.0)),
        (DlCmd.COLOR, p_u32(0x7FFF)),
        (DlCmd.TEXCOORD, p_u32(0)),
        (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
        (DlCmd.VERTEX_10, p_vertex10(0.0, 0.0, 0.0)),
        (DlCmd.VERTEX_XY, p_u32(0)),
        (DlCmd.VERTEX_XZ, p_u32(0)),
        (DlCmd.VERTEX_YZ, p_u32(0)),
        (DlCmd.VERTEX_DIFF, p_u32(0)),
        (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
        (DlCmd.END, b""),
        (DlCmd.POLY_ATTR, p_u32(0)),
        (DlCmd.TEX_IMG_PARAM, p_u32(0)),
        (DlCmd.TEX_PLTT_BASE, p_u32(0)),
        (DlCmd.MAT_COL_0, p_u32(0)),
        (DlCmd.MAT_COL_1, p_u32(0)),
        (DlCmd.LIGHT_VEC, p_u32(0)),
        (DlCmd.LIGHT_COL, p_u32(0)),
        (DlCmd.SHININESS, bytes(128)),
        (DlCmd.SWAP_BUFFERS, p_u32(0)),
        (DlCmd.VIEWPORT, p_u32(0)),
        (DlCmd.BOXTEST, bytes(12)),
        (DlCmd.POSTEST, bytes(8)),
        (DlCmd.VECTEST, p_u32(0)),
    ])
    def test_command_consumes_exact_operand_length(self, cmd, params):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (cmd, params),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        ))
        assert_trailing_triangle_intact(b, cmd)

    def test_unknown_opcode_is_skipped_without_operands(self):
        b = GeometryBuilder()
        b.run_dl(make_dl(
            (0x7F, b""),
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        ))
        assert_trailing_triangle_intact(b, 0x7F)

    def test_four_commands_share_one_word(self):
        # Exercises the packing rule directly: 4 opcodes, then 4 operand blocks.
        dl = make_dl(
            (DlCmd.BEGIN, p_u32(int(PrimType.TRIANGLES))),
            (DlCmd.VERTEX, p_vertex(0.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(1.0, 0.0, 0.0)),
            (DlCmd.VERTEX, p_vertex(0.0, 1.0, 0.0)),
            (DlCmd.END, b""),
        )
        assert len(dl) == 4 + (4 + 8 + 8 + 8) + 4 + 0
        b = GeometryBuilder()
        b.run_dl(dl)
        assert len(b.triangles) == 1

    def test_truncated_trailing_group_is_ignored(self):
        b = GeometryBuilder()
        b.run_dl(tri_dl() + b"\x23\x00")  # stray bytes, no full command word
        assert len(b.triangles) == 1

    def test_empty_display_list_is_a_noop(self):
        b = GeometryBuilder()
        b.run_dl(b"")
        assert b.triangles == []
        assert b.num_vertices_emitted == 0

    def test_accepts_bytearray(self):
        b = GeometryBuilder()
        b.run_dl(bytearray(tri_dl()))
        assert len(b.triangles) == 1
