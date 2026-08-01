import struct

import numpy as np
import pytest

from nitro import matrix as mat
from nitro.binary import BinaryReader
from nitro.dictionary import Dictionary
from nitro.dl import GeometryBuilder, MtxMode
from nitro.mdl0 import (Envelope, EvpMatrices, MatFlag, Material, MaterialSet,
                        Model, ModelInfo, NodeData, NodeSet, Shape, ShapeSet,
                        SrtFlag)
from nitro.sbc import SbcCmd, SbcOpt, SbcInterpreter
from tests.test_dl import ROT_Z90, tri_dl
from tests.test_mdl0 import (make_material_bytes, make_model_info_bytes,
                             make_node_bytes)

RIGID = SrtFlag.TRANSLATION_ZERO | SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE


def node(flag=RIGID, **kwargs) -> NodeData:
    return NodeData(BinaryReader(make_node_bytes(flag, **kwargs)))


def material(tex_image_param=0) -> Material:
    flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | \
        MatFlag.TEXMTX_TRANSLATION_ZERO
    return Material(BinaryReader(make_material_bytes(flag, tex_image_param=tex_image_param)))


def shape(dl: bytes) -> Shape:
    s = Shape.__new__(Shape)
    s.tag, s.size, s.flag = 1, 16, 0
    s.dl_offset = s.dl_size = 0
    s.dl = dl
    return s


def envelope(inv_m=None, inv_n=None) -> Envelope:
    e = Envelope.__new__(Envelope)
    e.inv_m = list(inv_m) if inv_m is not None else [
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    e.inv_n = list(inv_n) if inv_n is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return e


def build_model(sbc: bytes, *, nodes=None, materials=None, shapes=None,
                envelopes=None, **info_overrides) -> Model:
    nodes = nodes if nodes is not None else [node()]
    materials = materials if materials is not None else [material()]
    shapes = shapes if shapes is not None else [shape(tri_dl())]

    node_set = NodeSet.__new__(NodeSet)
    node_set.nodes = nodes
    node_set.dict = Dictionary(0, [f"node{i}" for i in range(len(nodes))],
                               [0] * len(nodes), 4)

    mat_set = MaterialSet.__new__(MaterialSet)
    mat_set.materials = materials
    mat_set.dict = Dictionary(0, [f"mat{i}" for i in range(len(materials))],
                              [0] * len(materials), 4)
    mat_set.dict_tex_to_mat = Dictionary(0, [], [], 4)
    mat_set.dict_pltt_to_mat = Dictionary(0, [], [], 4)

    shape_set = ShapeSet.__new__(ShapeSet)
    shape_set.shapes = shapes
    shape_set.dict = Dictionary(0, [f"shape{i}" for i in range(len(shapes))],
                                [0] * len(shapes), 4)

    model = Model.__new__(Model)
    model.info = ModelInfo(BinaryReader(
        make_model_info_bytes(**info_overrides)))
    model.nodes = node_set
    model.materials = mat_set
    model.shapes = shape_set
    model.sbc = sbc
    if envelopes is None:
        model.evp_matrices = None
    else:
        model.evp_matrices = EvpMatrices.__new__(EvpMatrices)
        model.evp_matrices.m = list(envelopes)
    return model


def interpret(sbc: bytes, *, mat_tex_dims=None, builder=None, **kwargs) -> SbcInterpreter:
    model = build_model(sbc, **kwargs)
    interp = SbcInterpreter(
        model, builder or GeometryBuilder(), mat_tex_dims or {})
    interp.run()
    return interp


def nodedesc(node_id: int, flag=SbcOpt.NONE, store=0, restore=0, parent=0) -> bytes:
    """SBC_NODEDESC: opcode, node id, parent id, scale-compensate, [store], [restore]."""
    out = bytes([SbcCmd.NODEDESC | flag, node_id, parent, 0])
    if flag & SbcOpt.STORE:
        out += bytes([store])
    if flag & SbcOpt.RESTORE:
        out += bytes([restore])
    return out


class TestControlFlow:
    def test_ret_halts_before_later_commands(self):
        interp = interpret(bytes([SbcCmd.RET]) +
                           nodedesc(0), nodes=[node(), node()])
        assert interp.node_seen == [False, False]

    def test_nop_advances_one_byte(self):
        sbc = bytes([SbcCmd.NOP, SbcCmd.NOP]) + \
            nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]

    def test_running_off_the_end_terminates(self):
        # No RET at all: the loop must stop at the buffer end, not run away.
        interp = interpret(nodedesc(0))
        assert interp.node_seen == [True]

    def test_empty_sbc_does_nothing(self):
        interp = interpret(b"")
        assert interp.draw_calls == []
        assert interp.current_node == -1

    def test_unknown_opcode_advances_one_byte(self):
        sbc = bytes([0x1F]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]

    @pytest.mark.parametrize("cmd", [SbcCmd.NODE, SbcCmd.ENVMAP, SbcCmd.PRJMAP])
    def test_three_byte_commands_consume_their_operands(self, cmd):
        sbc = bytes([cmd, 0xFF, 0xFF]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]

    def test_initial_state(self):
        interp = interpret(b"", nodes=[node(), node()])
        assert interp.current_node == -1
        assert interp.current_mat == 0
        assert interp.current_draw_node == -1
        assert interp.node_parent == [-1, -1]
        assert interp.stack_to_node == {}
        assert interp.node_visible == [True, True]
        assert interp.billboard_nodes == {}
        assert interp.tex_gen_cmds == []


def sbc_node(node_id: int, visible: bool = True) -> bytes:
    """SBC_NODE: opcode, node id, visibility (bit 0)."""
    return bytes([SbcCmd.NODE, node_id, int(visible)])


class TestNode:
    """SBC_NODE names the node that the following draw commands belong to.

    The interpreter used to skip the command wholesale, dropping both operands.
    An exporter has to reproduce them, so they are now recorded.
    """

    def test_records_the_visibility_bit_per_node(self):
        sbc = sbc_node(1, visible=False) + bytes([SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.node_visible == [True, False]

    def test_visible_node_stays_visible(self):
        interp = interpret(sbc_node(0, visible=True) + bytes([SbcCmd.RET]))
        assert interp.node_visible == [True]

    @pytest.mark.parametrize("byte,visible", [
        (0x00, False), (0x01, True), (0xFE, False), (0xFF, True),
    ])
    def test_only_bit_zero_of_the_visibility_operand_counts(self, byte, visible):
        # G3D_BinaryFormat.pdf spells the operand "--- --- --- --- --- --- --- V",
        # and sbc.c reads `*(rs->c + 2) & 1`, so the upper bits are reserved.
        sbc = bytes([SbcCmd.NODE, 0, byte, SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_visible == [visible]
        assert interp.current_visible is visible

    def test_records_the_node_id_separately_from_the_current_matrix_node(self):
        # The runtime keeps NNSG3dRS::currentNode (set by NODE) apart from
        # currentNodeDesc (whose matrix is loaded). Conflating them would bind
        # a skinned mesh's vertices to the mesh node instead of the joints.
        sbc = nodedesc(0) + sbc_node(1) + bytes([SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.current_draw_node == 1
        assert interp.current_node == 0

    def test_out_of_range_node_id_is_ignored_but_still_consumed(self):
        sbc = sbc_node(9, visible=False) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_visible == [True]
        assert interp.current_draw_node == 9
        assert interp.node_seen == [True]

    def test_draw_calls_carry_the_visibility_in_force(self):
        sbc = (sbc_node(0, visible=False) + bytes([SbcCmd.SHP, 0])
               + sbc_node(0, visible=True) + bytes([SbcCmd.SHP, 0, SbcCmd.RET]))
        interp = interpret(sbc)
        assert [c.visible for c in interp.draw_calls] == [False, True]

    def test_geometry_of_a_hidden_node_is_still_imported(self):
        # The hardware would skip it; dropping it here would lose data the
        # exporter has to write back.
        b = GeometryBuilder()
        sbc = sbc_node(0, visible=False) + bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interpret(sbc, builder=b)
        assert len(b.triangles) == 1

    def test_a_later_node_command_overrides_an_earlier_one(self):
        sbc = sbc_node(0, visible=False) + sbc_node(0, True) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_visible == [True]

    def test_consumes_three_bytes(self):
        sbc = sbc_node(0) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]


class TestTexGenCommands:
    """ENVMAP / PRJMAP carry a material ID plus a reserved flag byte."""

    @pytest.mark.parametrize("cmd", [SbcCmd.ENVMAP, SbcCmd.PRJMAP])
    def test_records_material_and_flag(self, cmd):
        sbc = bytes([cmd, 1, 0, SbcCmd.RET])
        interp = interpret(sbc, materials=[material(), material()])
        assert len(interp.tex_gen_cmds) == 1
        rec = interp.tex_gen_cmds[0]
        assert (rec.cmd, rec.material, rec.flag) == (cmd, 1, 0)

    @pytest.mark.parametrize("cmd", [SbcCmd.ENVMAP, SbcCmd.PRJMAP])
    def test_keeps_the_flag_byte_verbatim(self, cmd):
        # The format calls it "flag for expansion (currently always 0)", so a
        # non-zero value is news; don't silently mask it away.
        interp = interpret(bytes([cmd, 0, 0x80, SbcCmd.RET]))
        assert interp.tex_gen_cmds[0].flag == 0x80

    def test_records_both_kinds_in_stream_order(self):
        sbc = (bytes([SbcCmd.MAT, 0, SbcCmd.ENVMAP, 0, 0])
               + bytes([SbcCmd.MAT, 1, SbcCmd.PRJMAP, 1, 0, SbcCmd.RET]))
        interp = interpret(sbc, materials=[material(), material()])
        assert [(r.cmd, r.material) for r in interp.tex_gen_cmds] == [
            (SbcCmd.ENVMAP, 0), (SbcCmd.PRJMAP, 1)]

    def test_does_not_disturb_the_current_material(self):
        sbc = bytes([SbcCmd.MAT, 1, SbcCmd.ENVMAP, 0, 0, SbcCmd.RET])
        interp = interpret(sbc, materials=[material(), material()])
        assert interp.current_mat == 1


class TestNodeDesc:
    def test_marks_node_seen_and_becomes_current(self):
        interp = interpret(nodedesc(0) + bytes([SbcCmd.RET]))
        assert interp.node_seen == [True]
        assert interp.current_node == 0

    def test_root_node_has_no_parent(self):
        interp = interpret(nodedesc(0) + bytes([SbcCmd.RET]))
        assert interp.node_parent[0] == -1

    def test_second_node_parents_to_the_first(self):
        sbc = nodedesc(0) + nodedesc(1) + bytes([SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.node_parent == [-1, 0]
        assert interp.current_node == 1

    def test_records_world_matrix_per_node(self):
        # node_world starts out full of identities, so a rigid node would make
        # this pass without the interpreter recording anything.
        n = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                 translation=(1.0, 2.0, 3.0))
        interp = interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n])
        assert mat.mul((0.0, 0.0, 0.0), interp.node_world[0]) == pytest.approx(
            (1.0, 2.0, 3.0))

    def test_parent_comes_from_traversal_order_not_the_parent_byte(self):
        # The interpreter derives the hierarchy from the order nodes are
        # visited and ignores the parent id in the stream, so a bogus parent
        # byte must not change the result.
        sbc = nodedesc(0, parent=200) + \
            nodedesc(1, parent=200) + bytes([SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.node_parent == [-1, 0]

    def test_store_flag_maps_stack_slot_to_node(self):
        sbc = nodedesc(0, SbcOpt.STORE, store=5) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.stack_to_node == {5: 0}

    def test_store_flag_writes_the_matrix_stack(self):
        b = GeometryBuilder()
        sbc = nodedesc(0, SbcOpt.STORE, store=5) + bytes([SbcCmd.RET])
        interpret(sbc, builder=b)
        assert b.pos_stack[5] is b.cur_pos

    def test_restore_flag_reparents_to_the_stored_node(self):
        sbc = (nodedesc(0, SbcOpt.STORE, store=3)   # stack 3 -> node 0
               + nodedesc(1)                        # current becomes node 1
               + nodedesc(2, SbcOpt.RESTORE, restore=3)
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node(), node()])
        # node 2's parent is whatever stack slot 3 held, i.e. node 0
        assert interp.node_parent == [-1, 0, 0]

    def test_store_and_restore_flag_reads_both_operands(self):
        sbc = (nodedesc(0, SbcOpt.STORE, store=3)
               + nodedesc(1)
               + nodedesc(2, SbcOpt.STORE | SbcOpt.RESTORE, store=9, restore=3)
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node(), node()])
        assert interp.node_parent[2] == 0
        assert interp.stack_to_node == {3: 0, 9: 2}

    @pytest.mark.parametrize("flag,length", [
        (SbcOpt.NONE, 4),
        (SbcOpt.STORE, 5),
        (SbcOpt.RESTORE, 5),
        (SbcOpt.STORE | SbcOpt.RESTORE, 6),
    ])
    def test_operand_length_per_flag(self, flag, length):
        # A wrong length desynchronises the stream, so the trailing NODEDESC
        # only runs if the flagged one consumed exactly `length` bytes.
        head = nodedesc(0, flag, store=1, restore=1)
        assert len(head) == length
        interp = interpret(head + nodedesc(1) + bytes([SbcCmd.RET]),
                           nodes=[node(), node()])
        assert interp.node_seen == [True, True]

    def test_applies_joint_translation_to_the_builder(self):
        n = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                 translation=(1.0, 2.0, 3.0))
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert b.mtx_mode == MtxMode.POSITION_VECTOR
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 2.0, 3.0))

    def test_joint_translations_accumulate_down_the_chain(self):
        n0 = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                  translation=(1.0, 0.0, 0.0))
        n1 = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                  translation=(0.0, 2.0, 0.0))
        b = GeometryBuilder()
        interp = interpret(nodedesc(0) + nodedesc(1) + bytes([SbcCmd.RET]),
                           nodes=[n0, n1], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), interp.node_world[0]) == pytest.approx(
            (1.0, 0.0, 0.0))
        assert mat.mul((0.0, 0.0, 0.0), interp.node_world[1]) == pytest.approx(
            (1.0, 2.0, 0.0))

    def test_joint_with_both_rotation_and_translation(self):
        # Rotation and translation come from one combined matrix, so the
        # translation is *not* rotated: (1,0,0) rotates to (0,1,0), then the
        # translation is added.
        n = node(SrtFlag.SCALE_ONE, translation=(1.0, 2.0, 3.0),
                 rotation=ROT_Z90)
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 2.0, 3.0))
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 3.0, 3.0))

    def test_joint_scale_is_applied_before_translation(self):
        # A joint is SRT: the vertex is scaled first, then translated, so the
        # translation itself is not scaled.
        n = node(SrtFlag.ROTATION_ZERO, translation=(1.0, 0.0, 0.0),
                 scale=[2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 0.0, 0.0))
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (3.0, 0.0, 0.0))

    def test_joint_scale_is_applied_before_rotation(self):
        n = node(0, translation=(0.0, 0.0, 0.0), rotation=ROT_Z90,
                 scale=[2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        # scale x by 2 -> (2,0,0), then rotate 90 about Z -> (0,2,0).
        # Rotating first would instead stretch y and give (0,1,0).
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 2.0, 0.0))

    def test_rotation_only_joint_applies_rotation(self):
        n = node(SrtFlag.TRANSLATION_ZERO | SrtFlag.SCALE_ONE,
                 rotation=ROT_Z90)
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 1.0, 0.0))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 0.0, 0.0))

    def test_child_joint_transform_runs_before_the_parent(self):
        # The parent rotates; the child is offset along x. The child's offset
        # must be rotated by the parent, landing it on +y.
        parent = node(SrtFlag.TRANSLATION_ZERO | SrtFlag.SCALE_ONE,
                      rotation=ROT_Z90)
        child = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                     translation=(1.0, 0.0, 0.0))
        interp = interpret(nodedesc(0) + nodedesc(1) + bytes([SbcCmd.RET]),
                           nodes=[parent, child])
        assert mat.mul((0.0, 0.0, 0.0), interp.node_world[1]) == pytest.approx(
            (0.0, 1.0, 0.0))

    def test_applies_joint_scale(self):
        n = node(SrtFlag.TRANSLATION_ZERO | SrtFlag.ROTATION_ZERO,
                 scale=[2.0, 3.0, 4.0, 0.0, 0.0, 0.0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((1.0, 1.0, 1.0), b.cur_pos) == pytest.approx(
            (2.0, 3.0, 4.0))

    def test_rigid_node_leaves_matrix_untouched(self):
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]),
                  nodes=[node(RIGID)], builder=b)
        assert b.cur_pos == pytest.approx(np.identity(4))


def pivot_flag(index: int, *extra) -> int:
    flag = SrtFlag.HAS_PIVOT | SrtFlag.TRANSLATION_ZERO | SrtFlag.SCALE_ONE
    flag |= index << 4
    for e in extra:
        flag |= e
    return flag


class TestPivotRotation:
    """Compressed rotations: a pivot index plus two values reconstruct a 3x3.

    The node stores the index of the one element that is +/-1 and the two
    remaining magnitudes; _PIVOT says which cells the magnitudes go in and the
    SIGN_REV flags pick their signs.
    """

    def _cur_pos(self, flag, a, b):
        n = node(flag, pivot=(a, b))
        builder = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]),
                  nodes=[n], builder=builder)
        return builder.cur_pos

    def test_pivot_index_4_with_sign_revc_is_a_y_rotation(self):
        # index 4 is the centre cell, so the pivot axis is Y and the layout is
        # [[a, 0, b], [0, 1, 0], [-b, 0, a]] -- a 90 degree turn for a=0, b=1.
        m = self._cur_pos(pivot_flag(4, SrtFlag.SIGN_REVC), 0.0, 1.0)
        assert mat.mul((1.0, 0.0, 0.0), m) == pytest.approx((0.0, 0.0, 1.0))
        assert mat.mul((0.0, 1.0, 0.0), m) == pytest.approx((0.0, 1.0, 0.0))
        assert mat.mul((0.0, 0.0, 1.0), m) == pytest.approx((-1.0, 0.0, 0.0))

    def test_pivot_index_4_is_orthonormal_for_a_unit_pair(self):
        # a/b round to the nearest 1/4096, hence the loose tolerances here and
        # in the sign tests below.
        a, b = 0.6, 0.8
        m = self._cur_pos(pivot_flag(4, SrtFlag.SIGN_REVC), a, b)
        r = m[:3, :3]
        assert np.matmul(r, r.T) == pytest.approx(np.identity(3), abs=1e-3)

    def test_pivot_index_0_with_sign_revc_is_an_x_rotation(self):
        # index 0 pins [0][0] to 1 and fills cells 4, 5, 7, 8.
        m = self._cur_pos(pivot_flag(0, SrtFlag.SIGN_REVC), 0.0, 1.0)
        assert mat.mul((1.0, 0.0, 0.0), m) == pytest.approx((1.0, 0.0, 0.0))
        assert mat.mul((0.0, 1.0, 0.0), m) == pytest.approx((0.0, 0.0, 1.0))
        assert mat.mul((0.0, 0.0, 1.0), m) == pytest.approx((0.0, -1.0, 0.0))

    def test_pivot_negative_flips_the_pinned_element(self):
        plain = self._cur_pos(pivot_flag(4, SrtFlag.SIGN_REVC), 0.6, 0.8)
        negated = self._cur_pos(
            pivot_flag(4, SrtFlag.SIGN_REVC, SrtFlag.PIVOT_NEGATIVE), 0.6, 0.8)
        assert plain[1, 1] == pytest.approx(1.0)
        assert negated[1, 1] == pytest.approx(-1.0)
        # only the pinned cell differs
        assert np.count_nonzero(
            ~np.isclose(plain[:3, :3], negated[:3, :3])) == 1

    def test_sign_rev_flags_select_the_two_derived_signs(self):
        base = self._cur_pos(pivot_flag(4), 0.6, 0.8)
        assert (base[2, 0], base[2, 2]) == pytest.approx((0.8, 0.6), abs=1e-3)

        revc = self._cur_pos(pivot_flag(4, SrtFlag.SIGN_REVC), 0.6, 0.8)
        assert (revc[2, 0], revc[2, 2]) == pytest.approx((-0.8, 0.6), abs=1e-3)

        revd = self._cur_pos(pivot_flag(4, SrtFlag.SIGN_REVD), 0.6, 0.8)
        assert (revd[2, 0], revd[2, 2]) == pytest.approx((0.8, -0.6), abs=1e-3)

    def test_rotation_zero_beats_has_pivot(self):
        # ROTATION_ZERO wins, so no pivot data is read and no rotation applied
        m = self._cur_pos(pivot_flag(4, SrtFlag.ROTATION_ZERO), 0.0, 0.0)
        assert m == pytest.approx(np.identity(4))

    @pytest.mark.parametrize("index", range(9))
    def test_every_pivot_index_fills_all_nine_cells(self, index):
        m = self._cur_pos(pivot_flag(index, SrtFlag.SIGN_REVC), 0.6, 0.8)
        r = m[:3, :3]
        assert np.count_nonzero(r) == 5, f"pivot index {index} left gaps"
        assert np.matmul(r, r.T) == pytest.approx(np.identity(3), abs=1e-3)


class TestMtx:
    def test_restores_matrix_and_current_node(self):
        sbc = (nodedesc(0, SbcOpt.STORE, store=5)
               + nodedesc(1)
               + bytes([SbcCmd.MTX, 5])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.current_node == 0

    def test_restore_from_unmapped_slot_keeps_current_node(self):
        sbc = nodedesc(0) + bytes([SbcCmd.MTX, 12, SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.current_node == 0

    def test_consumes_two_bytes(self):
        sbc = bytes([SbcCmd.MTX, 0]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]

    def test_restores_the_stacked_matrix_into_the_builder(self):
        b = GeometryBuilder()
        marker = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 7.0, 8.0, 9.0])
        b.pos_stack[4] = marker
        interpret(bytes([SbcCmd.MTX, 4, SbcCmd.RET]), builder=b)
        assert b.cur_pos is marker


class TestMat:
    def test_sets_current_material(self):
        sbc = bytes([SbcCmd.MAT, 1, SbcCmd.RET])
        interp = interpret(sbc, materials=[material(), material()])
        assert interp.current_mat == 1

    def test_applies_tex_image_param_to_the_builder(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.MAT, 0, SbcCmd.RET]),
                  materials=[material(tex_image_param=3 << 20)], builder=b)
        assert b.tex_width == 8 << 3

    def test_mat_tex_dims_override_the_decoded_size(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.MAT, 0, SbcCmd.RET]),
                  materials=[material(tex_image_param=3 << 20)],
                  mat_tex_dims={0: (16, 32)}, builder=b)
        assert (b.tex_width, b.tex_height) == (16, 32)

    def test_out_of_range_material_index_is_ignored(self):
        b = GeometryBuilder()
        before = (b.tex_width, b.tex_height, b.tex_gen)
        interp = interpret(bytes([SbcCmd.MAT, 9, SbcCmd.RET]),
                           materials=[material()], builder=b)
        assert interp.current_mat == 9  # recorded, but no builder state touched
        assert (b.tex_width, b.tex_height, b.tex_gen) == before

    def test_consumes_two_bytes(self):
        sbc = bytes([SbcCmd.MAT, 0]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]


class TestShp:
    def test_emits_one_draw_call(self):
        sbc = bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interp = interpret(sbc)
        assert len(interp.draw_calls) == 1

    def test_draw_call_records_shape_material_and_node(self):
        sbc = (nodedesc(0)
               + bytes([SbcCmd.MAT, 1])
               + bytes([SbcCmd.SHP, 1])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc,
                           materials=[material(), material()],
                           shapes=[shape(tri_dl()), shape(tri_dl())])

        call = interp.draw_calls[0]
        assert call.shape == 1
        assert call.material == 1
        assert call.node == 0

    def test_draw_call_triangle_range_brackets_the_geometry(self):
        sbc = bytes([SbcCmd.SHP, 0, SbcCmd.SHP, 0, SbcCmd.RET])
        interp = interpret(sbc)

        first, second = interp.draw_calls
        assert (first.tri_start, first.tri_end) == (0, 1)
        assert (second.tri_start, second.tri_end) == (1, 2)

    def test_draw_call_captures_bind_matrices(self):
        # A translating joint ahead of the shape, so an identity or missing
        # bind matrix is distinguishable from the real one.
        n = node(SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
                 translation=(1.0, 2.0, 3.0))
        sbc = nodedesc(0) + bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interp = interpret(sbc, nodes=[n])

        call = interp.draw_calls[0]
        assert mat.mul((0.0, 0.0, 0.0), call.bind_pos) == pytest.approx(
            (1.0, 2.0, 3.0))
        assert call.bind_dir == pytest.approx(call.bind_pos)
        assert call.single_mtx is True

    def test_shape_geometry_reaches_the_builder(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.SHP, 0, SbcCmd.RET]), builder=b)
        assert len(b.triangles) == 1

    def test_vertices_are_bound_to_the_current_node(self):
        b = GeometryBuilder()
        sbc = nodedesc(0) + nodedesc(1) + bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interpret(sbc, nodes=[node(), node()], builder=b)
        assert [v.node for v in b.triangles[0]] == [1, 1, 1]

    def test_shape_before_any_node_binds_to_node_zero(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.SHP, 0, SbcCmd.RET]), builder=b)
        assert b.current_bound_node == 0
        assert [v.node for v in b.triangles[0]] == [0, 0, 0]

    def test_empty_shape_emits_a_zero_length_draw_call(self):
        sbc = bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interp = interpret(sbc, shapes=[shape(b"")])
        call = interp.draw_calls[0]
        assert call.tri_start == call.tri_end == 0

    def test_consumes_two_bytes(self):
        sbc = bytes([SbcCmd.SHP, 0]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]


class TestPosScale:
    def test_flagless_uses_pos_scale(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.POSSCALE, SbcCmd.RET]), builder=b,
                  pos_scale=2.0, inv_pos_scale=0.5)
        assert mat.mul((1.0, 1.0, 1.0), b.cur_pos) == pytest.approx(
            (2.0, 2.0, 2.0))

    def test_flagged_uses_inverse_pos_scale(self):
        b = GeometryBuilder()
        interpret(bytes([SbcCmd.POSSCALE | SbcOpt.STORE, SbcCmd.RET]), builder=b,
                  pos_scale=2.0, inv_pos_scale=0.5)
        assert mat.mul((1.0, 1.0, 1.0), b.cur_pos) == pytest.approx(
            (0.5, 0.5, 0.5))

    def test_consumes_one_byte(self):
        sbc = bytes([SbcCmd.POSSCALE]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]


class TestBillboard:
    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    @pytest.mark.parametrize("flag,length", [
        (SbcOpt.NONE, 2),
        (SbcOpt.STORE, 3),
        (SbcOpt.RESTORE, 3),
        (SbcOpt.STORE | SbcOpt.RESTORE, 4),
    ])
    def test_operand_length_per_flag(self, cmd, flag, length):
        head = bytes([cmd | flag, 0, 1, 1])[:length]
        interp = interpret(head + nodedesc(0) + bytes([SbcCmd.RET]))
        assert interp.node_seen == [
            True], "billboard consumed the wrong byte count"

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_store_flag_maps_stack_slot_to_the_billboard_node(self, cmd):
        sbc = nodedesc(
            0) + bytes([cmd | SbcOpt.STORE, 0, 6]) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.stack_to_node == {6: 0}

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_restore_flag_updates_current_node(self, cmd):
        sbc = (nodedesc(0, SbcOpt.STORE, store=2)
               + nodedesc(1)
               + bytes([cmd | SbcOpt.RESTORE, 0, 2])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.current_node == 0

    @pytest.mark.parametrize("cmd,kind", [(SbcCmd.BB, SbcCmd.BB),
                                          (SbcCmd.BBY, SbcCmd.BBY)])
    def test_records_which_nodes_are_billboards_and_of_which_kind(self, cmd, kind):
        # The node's own NodeData carries no billboard flag -- screen-aligned vs
        # Y-axis-only is only ever stated by the choice of SBC opcode, so an
        # exporter can't recover it from anywhere else.
        sbc = nodedesc(0) + nodedesc(1) + bytes([cmd, 1, SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.billboard_nodes == {1: kind}

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_node_id_operand_is_read_and_not_confused_with_the_stack_slot(self, cmd):
        # The operand order is NodeID first, then the optional stack indices --
        # reading the store slot as the node id would give {2: ...} here.
        sbc = nodedesc(0) + nodedesc(1) + \
            bytes([cmd | SbcOpt.STORE, 0, 2, SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.billboard_nodes == {0: cmd}
        assert interp.stack_to_node == {2: 0}

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_node_id_wins_over_the_restored_slot_for_the_current_node(self, cmd):
        # Stack slot 2 holds node 0, but the command names node 1: the matrix
        # being billboarded is node 1's, so that is what the slot ends up bound
        # to. Reading the operand is the only way to get this right.
        sbc = (nodedesc(0, SbcOpt.STORE, store=2)
               + nodedesc(1)
               + bytes([cmd | SbcOpt.STORE | SbcOpt.RESTORE, 1, 3, 2])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.current_node == 1
        assert interp.stack_to_node == {2: 0, 3: 1}

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_out_of_range_node_id_falls_back_to_the_restored_slot(self, cmd):
        sbc = (nodedesc(0, SbcOpt.STORE, store=2)
               + nodedesc(1)
               + bytes([cmd | SbcOpt.RESTORE, 9, 2])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.billboard_nodes == {}
        assert interp.current_node == 0


def nodemix(dest: int, *terms: tuple[int, int, int]) -> bytes:
    """SBC_NODEMIX: dest stack slot, term count, then (stack, node, ratio)*.

    Ratio is an 8-bit fraction over 256, so the ratios of a full-weight blend
    sum to 256 and no single term can carry full weight -- hence the SDK's
    NNS_G3D_ASSERT(numMtx >= 2). `terms` here therefore normally has >= 2
    entries, matching what g3dcvtr emits.
    """
    out = bytes([SbcCmd.NODEMIX, dest, len(terms)])
    for stack, node, ratio in terms:
        out += bytes([stack, node, ratio])
    return out


HALF = 128   # 128/256 == exactly 0.5, so two of them are full weight


class TestNodeMix:
    def test_identity_envelope_and_stack_yields_identity(self):
        b = GeometryBuilder()
        interpret(nodemix(0, (0, 0, HALF), (0, 0, HALF)) + bytes([SbcCmd.RET]),
                  envelopes=[envelope()], builder=b)
        assert b.cur_pos == pytest.approx(np.identity(4))
        assert b.cur_dir == pytest.approx(np.identity(4))

    def test_ratio_is_a_fraction_of_256_not_255(self):
        # Ratio_N is 0.8 fixed point: the SDK computes `w = ratio << 4` as an
        # fx32 (FX32_SHIFT == 12), i.e. ratio/256. g3dcvtr confirms it -- the
        # SDK's own evp_wgt.imd sample turns IMD weight pairs like "95 5" into
        # ratios 243/13, which sum to 256 exactly, never to 255.
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 256.0, 0.0, 0.0])
        interpret(nodemix(0, (0, 0, 1), (0, 0, 1)) + bytes([SbcCmd.RET]),
                  envelopes=[envelope()], builder=b)
        # two terms of 1/256 each -> 2.0, not 2 * 256/255 == 2.008
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos)[0] == pytest.approx(2.0)

    def test_ratios_summing_to_256_preserve_the_blended_matrix(self):
        # A real g3dcvtr blend: IMD "95 5" -> 243/13. The two stack matrices
        # here are the same, so any correctly normalised scale returns it intact.
        b = GeometryBuilder()
        m = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 7.0, 0.0, 0.0])
        b.pos_stack[0] = b.pos_stack[1] = m
        interpret(nodemix(2, (0, 0, 243), (1, 1, 13)) + bytes([SbcCmd.RET]),
                  envelopes=[envelope(), envelope()], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (7.0, 0.0, 0.0))

    def test_blends_two_stack_matrices_by_weight(self):
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0.0, 0.0, 0.0])
        b.pos_stack[1] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 4.0, 0.0, 0.0])
        interpret(nodemix(5, (0, 0, HALF), (1, 1, HALF)) + bytes([SbcCmd.RET]),
                  envelopes=[envelope(), envelope()], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (2.0, 0.0, 0.0))

    def test_envelope_inverse_is_applied_to_the_stack_matrix(self):
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 5.0, 0.0, 0.0])
        evp = envelope(inv_m=[1, 0, 0, 0, 1, 0, 0, 0, 1, -5.0, 0.0, 0.0])
        sbc = bytes([SbcCmd.NODEMIX, 0, 1, 0, 0, 255, SbcCmd.RET])
        interpret(sbc, envelopes=[evp], builder=b)

        # stack(+5) composed with envelope inverse(-5) cancels out
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 0.0, 0.0))

    def test_envelope_inverse_is_applied_before_the_stack_matrix(self):
        # Skinning is v @ inv_bind @ node_world: the inverse bind pose takes
        # the vertex out of model space first. With pure translations either
        # order gives the same answer, so use a rotation to pin it down.
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 5.0, 0.0, 0.0])
        evp = envelope(inv_m=[*ROT_Z90, 0.0, 0.0, 0.0])
        interpret(nodemix(0, (0, 0, HALF), (0, 0, HALF)) + bytes([SbcCmd.RET]),
                  envelopes=[evp], builder=b)

        # rotate (1,0,0) to (0,1,0), then translate: (5,1,0).
        # The other order would give (0,6,0).
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (5.0, 1.0, 0.0))

    def test_result_is_stored_to_the_named_stack_slot(self):
        b = GeometryBuilder()
        sbc = bytes([SbcCmd.NODEMIX, 9, 1, 0, 0, 255, SbcCmd.RET])
        interpret(sbc, envelopes=[envelope()], builder=b)
        assert b.pos_stack[9] is b.cur_pos

    def test_missing_envelopes_fall_back_to_the_stack_matrix(self):
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 2.0, 0.0, 0.0])
        interpret(nodemix(0, (0, 0, HALF), (0, 0, HALF)) + bytes([SbcCmd.RET]),
                  envelopes=None, builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (2.0, 0.0, 0.0))

    def test_bottom_row_is_normalised_to_one(self):
        b = GeometryBuilder()
        sbc = bytes([SbcCmd.NODEMIX, 0, 1, 0, 0, 128, SbcCmd.RET])
        interpret(sbc, envelopes=[envelope()], builder=b)
        assert b.cur_pos[3, 3] == pytest.approx(1.0)
        assert b.cur_dir[3, 3] == pytest.approx(1.0)

    def test_consumes_three_bytes_per_term(self):
        sbc = (bytes([SbcCmd.NODEMIX, 0, 2, 0, 0, 128, 1, 1, 127])
               + nodedesc(0) + bytes([SbcCmd.RET]))
        interp = interpret(sbc, envelopes=[envelope(), envelope()])
        assert interp.node_seen == [
            True], "NODEMIX consumed the wrong byte count"

    def test_first_term_becomes_the_current_node(self):
        sbc = bytes([SbcCmd.NODEMIX, 0, 1, 0, 1, 255, SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()],
                           envelopes=[envelope(), envelope()])
        assert interp.current_node == 1
        assert interp.stack_to_node == {0: 1}

    def test_zero_terms_leaves_an_identity_bottom_row(self):
        sbc = bytes([SbcCmd.NODEMIX, 0, 0, SbcCmd.RET])
        b = GeometryBuilder()
        interp = interpret(sbc, envelopes=[envelope()], builder=b)
        assert b.cur_pos[3, 3] == pytest.approx(1.0)
        assert interp.current_node == -1


class TestCallDl:
    def _sbc(self, dl: bytes, head: bytes = b"", tail: bytes | None = None) -> bytes:
        """CALLDL: opcode, u32 offset (relative to the opcode), u32 length.

        The display list is parked *after* ``tail``, which ends the stream.
        Execution resumes at the byte after the operands, so an inline display
        list would be decoded as SBC opcodes -- real files keep their display
        lists out of the instruction path and so does this helper.
        """
        tail = bytes([SbcCmd.RET]) if tail is None else tail
        return head + bytes([SbcCmd.CALLDL]) + \
            struct.pack("<II", 9 + len(tail), len(dl)) + tail + dl

    def test_emits_a_draw_call_with_no_shape_index(self):
        interp = interpret(self._sbc(tri_dl()))
        assert len(interp.draw_calls) == 1
        assert interp.draw_calls[0].shape == -1

    def test_draw_call_fields_line_up(self):
        interp = interpret(self._sbc(tri_dl()))
        call = interp.draw_calls[0]
        assert (call.tri_start, call.tri_end) == (0, 1)
        assert call.node == 0
        assert call.material == 0
        assert call.bind_pos is not None
        assert call.bind_dir is not None
        assert call.single_mtx is True

    def test_inline_geometry_reaches_the_builder(self):
        b = GeometryBuilder()
        interpret(self._sbc(tri_dl()), builder=b)
        assert len(b.triangles) == 1

    def test_reads_the_display_list_from_the_given_offset_and_length(self):
        # A short length would truncate the list and drop the triangle; a wrong
        # offset would decode neighbouring bytes instead.
        dl = tri_dl()
        interp = interpret(self._sbc(dl))
        assert (interp.draw_calls[0].tri_start,
                interp.draw_calls[0].tri_end) == (0, 1)

        # dropping the final command word cuts off the END, so nothing flushes
        short = bytes([SbcCmd.CALLDL]) + \
            struct.pack("<II", 10, len(dl) - 4) + bytes([SbcCmd.RET]) + dl
        assert interpret(short).draw_calls[0].tri_end == 0

    def test_binds_to_the_current_node(self):
        b = GeometryBuilder()
        sbc = self._sbc(tri_dl(), head=nodedesc(0) + nodedesc(1))
        interp = interpret(sbc, nodes=[node(), node()], builder=b)
        assert interp.draw_calls[0].node == 1
        assert [v.node for v in b.triangles[0]] == [1, 1, 1]

    def test_records_current_material(self):
        interp = interpret(self._sbc(tri_dl(), head=bytes([SbcCmd.MAT, 1])),
                           materials=[material(), material()])
        assert interp.draw_calls[0].material == 1

    def test_consumes_nine_bytes(self):
        # operands point past the trailing NODEDESC, so the offset skips it
        sbc = self._sbc(tri_dl(),
                        tail=nodedesc(0) + bytes([SbcCmd.RET]))
        interp = interpret(sbc)
        assert interp.node_seen == [
            True], "CALLDL consumed the wrong byte count"


class TestSbcFlags:
    @pytest.mark.parametrize("flag,bits", [
        (SbcOpt.NONE, 0b000),
        (SbcOpt.STORE, 0b001),
        (SbcOpt.RESTORE, 0b010),
        (SbcOpt.STORE | SbcOpt.RESTORE, 0b011),
    ])
    def test_flag_values_are_the_bit_pattern_shifted_into_the_mask(self, flag, bits):
        assert flag == bits << 5
        assert flag & SbcCmd.FLAG_MASK == flag
