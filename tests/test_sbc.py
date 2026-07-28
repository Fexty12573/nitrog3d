import struct

import numpy as np
import pytest

from nitro import matrix as mat
from nitro.binary import BinaryReader
from nitro.dictionary import Dictionary
from nitro.dl import GeometryBuilder, MtxMode, PrimType
from nitro.mdl0 import (Envelope, EvpMatrices, MatFlag, Material, MaterialSet,
                        Model, ModelInfo, NodeData, NodeSet, Shape, ShapeSet,
                        SrtFlag)
from nitro.sbc import DrawCall, SbcCmd, SbcFlag, SbcInterpreter
from tests.test_dl import p_mtx43, tri_dl
from tests.test_mdl0 import (fx32, make_material_bytes, make_model_info_bytes,
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


def nodedesc(node_id: int, flag=SbcFlag.F000, store=0, restore=0) -> bytes:
    """SBC_NODEDESC: opcode, node id, parent id, scale-compensate, [store], [restore]."""
    out = bytes([SbcCmd.NODEDESC | flag, node_id, 0, 0])
    if flag == SbcFlag.F001:
        out += bytes([store])
    elif flag == SbcFlag.F010:
        out += bytes([restore])
    elif flag == SbcFlag.F011:
        out += bytes([store, restore])
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
    def test_three_byte_commands_skip_their_operands(self, cmd):
        sbc = bytes([cmd, 0xFF, 0xFF]) + nodedesc(0) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.node_seen == [True]

    def test_initial_state(self):
        interp = interpret(b"", nodes=[node(), node()])
        assert interp.current_node == -1
        assert interp.current_mat == 0
        assert interp.node_parent == [-1, -1]
        assert interp.stack_to_node == {}


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
        interp = interpret(nodedesc(0) + bytes([SbcCmd.RET]))
        assert interp.node_world[0] is not None
        assert interp.node_world[0] == pytest.approx(np.identity(4))

    def test_store_flag_maps_stack_slot_to_node(self):
        sbc = nodedesc(0, SbcFlag.F001, store=5) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.stack_to_node == {5: 0}

    def test_store_flag_writes_the_matrix_stack(self):
        b = GeometryBuilder()
        sbc = nodedesc(0, SbcFlag.F001, store=5) + bytes([SbcCmd.RET])
        interpret(sbc, builder=b)
        assert b.pos_stack[5] is b.cur_pos

    def test_restore_flag_reparents_to_the_stored_node(self):
        sbc = (nodedesc(0, SbcFlag.F001, store=3)   # stack 3 -> node 0
               + nodedesc(1)                        # current becomes node 1
               + nodedesc(2, SbcFlag.F010, restore=3)
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node(), node()])
        # node 2's parent is whatever stack slot 3 held, i.e. node 0
        assert interp.node_parent == [-1, 0, 0]

    def test_store_and_restore_flag_reads_both_operands(self):
        sbc = (nodedesc(0, SbcFlag.F001, store=3)
               + nodedesc(1)
               + nodedesc(2, SbcFlag.F011, store=9, restore=3)
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node(), node()])
        assert interp.node_parent[2] == 0
        assert interp.stack_to_node == {3: 0, 9: 2}

    @pytest.mark.parametrize("flag,length", [
        (SbcFlag.F000, 4),
        (SbcFlag.F001, 5),
        (SbcFlag.F010, 5),
        (SbcFlag.F011, 6),
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
        n = node(SrtFlag.SCALE_ONE, translation=(1.0, 2.0, 3.0),
                 rotation=[1, 0, 0, 0, 1, 0, 0, 0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (1.0, 2.0, 3.0))

    def test_joint_scale_is_applied_after_translation(self):
        # _apply_joint muls the translation first, then the scale, so the scale
        # ends up multiplying the translation too.
        n = node(SrtFlag.ROTATION_ZERO, translation=(1.0, 0.0, 0.0),
                 scale=[2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert mat.mul((1.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (4.0, 0.0, 0.0))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (2.0, 0.0, 0.0))

    def test_rotation_only_joint_applies_rotation(self):
        n = node(SrtFlag.TRANSLATION_ZERO | SrtFlag.SCALE_ONE,
                 rotation=[0, 1, 0, -1, 0, 0, 0, 0])
        b = GeometryBuilder()
        interpret(nodedesc(0) + bytes([SbcCmd.RET]), nodes=[n], builder=b)
        assert b.cur_pos != pytest.approx(np.identity(4))
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (0.0, 0.0, 0.0))

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


class TestMtx:
    def test_restores_matrix_and_current_node(self):
        sbc = (nodedesc(0, SbcFlag.F001, store=5)
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
        sbc = bytes([SbcCmd.MAT, 9, SbcCmd.RET])
        interp = interpret(sbc, materials=[material()])
        assert interp.current_mat == 9  # recorded, but no builder state touched

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
        sbc = bytes([SbcCmd.SHP, 0, SbcCmd.RET])
        interp = interpret(sbc)

        call = interp.draw_calls[0]
        assert call.bind_pos is not None
        assert call.bind_dir is not None
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
        interpret(bytes([SbcCmd.POSSCALE | SbcFlag.F001, SbcCmd.RET]), builder=b,
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
        (SbcFlag.F000, 2),
        (SbcFlag.F001, 3),
        (SbcFlag.F010, 3),
        (SbcFlag.F011, 4),
    ])
    def test_operand_length_per_flag(self, cmd, flag, length):
        head = bytes([cmd | flag, 0, 1, 1])[:length]
        interp = interpret(head + nodedesc(0) + bytes([SbcCmd.RET]))
        assert interp.node_seen == [
            True], "billboard consumed the wrong byte count"

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_store_flag_maps_stack_slot_to_current_node(self, cmd):
        sbc = nodedesc(
            0) + bytes([cmd | SbcFlag.F001, 0, 6]) + bytes([SbcCmd.RET])
        interp = interpret(sbc)
        assert interp.stack_to_node == {6: 0}

    @pytest.mark.parametrize("cmd", [SbcCmd.BB, SbcCmd.BBY])
    def test_restore_flag_updates_current_node(self, cmd):
        sbc = (nodedesc(0, SbcFlag.F001, store=2)
               + nodedesc(1)
               + bytes([cmd | SbcFlag.F010, 0, 2])
               + bytes([SbcCmd.RET]))
        interp = interpret(sbc, nodes=[node(), node()])
        assert interp.current_node == 0


class TestNodeMix:
    def test_identity_envelope_and_stack_yields_identity(self):
        sbc = bytes([SbcCmd.NODEMIX, 0, 1, 0, 0, 255, SbcCmd.RET])
        b = GeometryBuilder()
        interpret(sbc, envelopes=[envelope()], builder=b)
        assert b.cur_pos == pytest.approx(np.identity(4))
        assert b.cur_dir == pytest.approx(np.identity(4))

    def test_blends_two_stack_matrices_by_weight(self):
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0.0, 0.0, 0.0])
        b.pos_stack[1] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 4.0, 0.0, 0.0])
        # two terms, half weight each (128/255 ~ 0.502)
        sbc = bytes([SbcCmd.NODEMIX, 5, 2, 0, 0, 128, 1, 1, 128, SbcCmd.RET])
        interpret(sbc, envelopes=[envelope(), envelope()], builder=b)

        w = 128 / 255.0
        assert mat.mul((0.0, 0.0, 0.0), b.cur_pos) == pytest.approx(
            (4.0 * w, 0.0, 0.0))

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

    def test_result_is_stored_to_the_named_stack_slot(self):
        b = GeometryBuilder()
        sbc = bytes([SbcCmd.NODEMIX, 9, 1, 0, 0, 255, SbcCmd.RET])
        interpret(sbc, envelopes=[envelope()], builder=b)
        assert b.pos_stack[9] is b.cur_pos

    def test_missing_envelopes_fall_back_to_the_stack_matrix(self):
        b = GeometryBuilder()
        b.pos_stack[0] = mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 2.0, 0.0, 0.0])
        sbc = bytes([SbcCmd.NODEMIX, 0, 1, 0, 0, 255, SbcCmd.RET])
        interpret(sbc, envelopes=None, builder=b)
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
    def _sbc(self, dl: bytes) -> bytes:
        # CALLDL: opcode, u32 offset (relative to the opcode), u32 length.
        return bytes([SbcCmd.CALLDL]) + struct.pack("<II", 9, len(dl)) + dl + \
            bytes([SbcCmd.RET])

    def test_emits_a_draw_call_with_no_shape_index(self):
        interp = interpret(self._sbc(tri_dl()))
        call = interp.draw_calls[0]
        assert call.shape == -1

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

    def test_binds_to_the_current_node(self):
        b = GeometryBuilder()
        dl = tri_dl()
        sbc = nodedesc(0) + nodedesc(1) + self._sbc(dl)
        # the CALLDL offset is relative to its own opcode, so recompute it
        head = nodedesc(0) + nodedesc(1)
        sbc = head + bytes([SbcCmd.CALLDL]) + struct.pack("<II", 9, len(dl)) + \
            dl + bytes([SbcCmd.RET])
        interp = interpret(sbc, nodes=[node(), node()], builder=b)
        assert interp.draw_calls[0].node == 1
        assert [v.node for v in b.triangles[0]] == [1, 1, 1]

    def test_records_current_material(self):
        dl = tri_dl()
        head = bytes([SbcCmd.MAT, 1])
        sbc = head + bytes([SbcCmd.CALLDL]) + struct.pack("<II", 9, len(dl)) + \
            dl + bytes([SbcCmd.RET])
        interp = interpret(sbc, materials=[material(), material()])
        assert interp.draw_calls[0].material == 1

    def test_consumes_nine_bytes(self):
        dl = tri_dl()
        # operands point past the trailing NODEDESC, so the offset skips it
        tail = nodedesc(0) + bytes([SbcCmd.RET])
        sbc = bytes([SbcCmd.CALLDL]) + struct.pack("<II", 9 + len(tail), len(dl)) + \
            tail + dl
        interp = interpret(sbc)
        assert interp.node_seen == [
            True], "CALLDL consumed the wrong byte count"


class TestSbcFlags:
    @pytest.mark.parametrize("flag,bits", [
        (SbcFlag.F000, 0b000),
        (SbcFlag.F001, 0b001),
        (SbcFlag.F010, 0b010),
        (SbcFlag.F011, 0b011),
        (SbcFlag.F100, 0b100),
        (SbcFlag.F101, 0b101),
        (SbcFlag.F110, 0b110),
        (SbcFlag.F111, 0b111),
    ])
    def test_flag_values_are_the_bit_pattern_shifted_into_the_mask(self, flag, bits):
        assert flag == bits << 5
        assert flag & SbcCmd.FLAG_MASK == flag
