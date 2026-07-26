import struct

import pytest

from nitro.binary import BinaryReader, BinaryWriter, FX16_SCALE, FX32_SCALE
from nitro.dictionary import Dictionary
from nitro.mdl0 import (Envelope, EvpMatrices, MatFlag, Material, MDL0,
                        Model, ModelInfo, NodeData, NodeSet, MaterialSet,
                        Shape, ShapeSet, SrtFlag, TexToMatData)


def fx32(v: float) -> int:
    return int(round(v * FX32_SCALE))


def fx16(v: float) -> int:
    return int(round(v * FX16_SCALE))


def make_node_bytes(flag: int, translation=(0, 0, 0), rotation=None, pivot=None, scale=None) -> bytes:
    buf = struct.pack("<Hh", flag, 0)
    if not (flag & SrtFlag.TRANSLATION_ZERO):
        buf += struct.pack("<3i", *[fx32(v) for v in translation])
    if not (flag & (SrtFlag.ROTATION_ZERO | SrtFlag.HAS_PIVOT)):
        buf += struct.pack("<8h", *[fx16(v) for v in (rotation or [0] * 8)])
    if (flag & SrtFlag.HAS_PIVOT) and not (flag & SrtFlag.ROTATION_ZERO):
        buf += struct.pack("<2h", *[fx16(v) for v in (pivot or [0, 0])])
    if not (flag & SrtFlag.SCALE_ONE):
        buf += struct.pack("<6i", *[fx32(v) for v in (scale or [0] * 6)])
    return buf


class TestNodeData:
    def test_all_optional_fields_present_when_no_flags_set(self):
        raw = make_node_bytes(0, translation=(1.0, 2.0, 3.0))
        r = BinaryReader(raw)
        node = NodeData(r)
        assert r.tell() == len(raw)
        assert (node.tx, node.ty, node.tz) == pytest.approx((1.0, 2.0, 3.0))

    def test_translation_zero_skips_translation_fields(self):
        flag = SrtFlag.TRANSLATION_ZERO
        raw = make_node_bytes(flag)
        r = BinaryReader(raw)
        node = NodeData(r)
        assert r.tell() == len(raw)
        assert (node.tx, node.ty, node.tz) == (0, 0, 0)

    def test_rotation_zero_and_scale_one_and_translation_zero_is_minimal(self):
        flag = SrtFlag.TRANSLATION_ZERO | SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE
        raw = make_node_bytes(flag)
        assert len(raw) == 4
        r = BinaryReader(raw)
        NodeData(r)
        assert r.tell() == len(raw)

    def test_has_pivot_reads_pivot_instead_of_full_rotation(self):
        flag = SrtFlag.HAS_PIVOT
        raw = make_node_bytes(flag, pivot=(0.5, -0.5))
        r = BinaryReader(raw)
        node = NodeData(r)
        assert r.tell() == len(raw)
        assert (node.a, node.b) == pytest.approx((0.5, -0.5))
        # full 3x3 rotation fields shouldn't be touched
        assert node._01 == 0.0 and node._22 == 0.0

    def test_has_pivot_with_rotation_zero_skips_pivot_too(self):
        # ROTATION_ZERO takes precedence. No pivot rotation data is read even though HAS_PIVOT is also set.
        flag = SrtFlag.HAS_PIVOT | SrtFlag.ROTATION_ZERO
        raw = make_node_bytes(flag)
        r = BinaryReader(raw)
        node = NodeData(r)
        assert r.tell() == len(raw)
        assert (node.a, node.b) == (0.0, 0.0)

    def test_scale_one_skips_scale_fields(self):
        flag = SrtFlag.SCALE_ONE
        raw = make_node_bytes(flag)
        r = BinaryReader(raw)
        node = NodeData(r)
        assert r.tell() == len(raw)
        assert (node.sx, node.sy, node.sz) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("flag", [
        0,
        SrtFlag.TRANSLATION_ZERO,
        SrtFlag.ROTATION_ZERO,
        SrtFlag.SCALE_ONE,
        SrtFlag.HAS_PIVOT,
        SrtFlag.TRANSLATION_ZERO | SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
        SrtFlag.HAS_PIVOT | SrtFlag.ROTATION_ZERO,
        SrtFlag.HAS_PIVOT | SrtFlag.SCALE_ONE,
    ])
    def test_consumes_exactly_the_bytes_it_produced(self, flag):
        raw = make_node_bytes(flag)
        r = BinaryReader(raw)
        NodeData(r)
        assert r.tell() == len(raw)

    @pytest.mark.parametrize("flag", [
        0,
        SrtFlag.TRANSLATION_ZERO,
        SrtFlag.ROTATION_ZERO,
        SrtFlag.SCALE_ONE,
        SrtFlag.HAS_PIVOT,
        SrtFlag.TRANSLATION_ZERO | SrtFlag.ROTATION_ZERO | SrtFlag.SCALE_ONE,
        SrtFlag.HAS_PIVOT | SrtFlag.ROTATION_ZERO,
        SrtFlag.HAS_PIVOT | SrtFlag.SCALE_ONE,
    ])
    def test_write_reads_back_identically(self, flag):
        original = NodeData(BinaryReader(make_node_bytes(
            flag, translation=(1.0, -2.0, 3.5), rotation=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            pivot=(0.25, -0.25), scale=[1.5, 2.5, 3.5, 0.5, 0.5, 0.5])))

        w = BinaryWriter()
        original.write(w)
        roundtripped = NodeData(BinaryReader(w.get_bytes()))

        assert w.length == len(make_node_bytes(flag))
        assert roundtripped.flag == original.flag
        assert (roundtripped.tx, roundtripped.ty, roundtripped.tz) == \
            pytest.approx((original.tx, original.ty, original.tz))
        assert (roundtripped.a, roundtripped.b) == pytest.approx((original.a, original.b))
        assert (roundtripped.sx, roundtripped.sy, roundtripped.sz) == \
            pytest.approx((original.sx, original.sy, original.sz))


def make_material_bytes(flag: int, tex_image_param=0, poly_attr=0) -> bytes:
    buf = struct.pack("<HH", 0, 0)  # tag, size
    buf += struct.pack("<II", 0, 0)  # diff_amb, spec_emi
    buf += struct.pack("<II", poly_attr, 0)  # poly_attr, poly_attr_mask
    # tex_image_param, tex_image_param_mask
    buf += struct.pack("<II", tex_image_param, 0)
    buf += struct.pack("<HH", 0, flag)  # tex_pltt_base, flag
    buf += struct.pack("<HH", 0, 0)  # orig_width, orig_height
    buf += struct.pack("<ii", 0, 0)  # mag_w, mag_h
    if not (flag & MatFlag.TEXMTX_SCALE_ONE):
        buf += struct.pack("<2i", fx32(1.0), fx32(1.0))
    if not (flag & MatFlag.TEXMTX_ROTATION_ZERO):
        buf += struct.pack("<2h", fx16(0.0), fx16(1.0))
    if not (flag & MatFlag.TEXMTX_TRANSLATION_ZERO):
        buf += struct.pack("<2i", fx32(0.0), fx32(0.0))
    if flag & MatFlag.EFFECTMTX:
        buf += struct.pack("<16i", *[fx32(0.0)] * 16)
    return buf


class TestMaterial:
    def test_all_texmtx_fields_present_when_no_flags_set(self):
        raw = make_material_bytes(0)
        r = BinaryReader(raw)
        mat = Material(r)
        assert r.tell() == len(raw)
        assert mat.effect_mtx is None

    def test_texmtx_scale_one_skips_scale_fields(self):
        flag = MatFlag.TEXMTX_SCALE_ONE
        raw = make_material_bytes(flag)
        r = BinaryReader(raw)
        mat = Material(r)
        assert r.tell() == len(raw)
        assert (mat.scale_s, mat.scale_t) == (1.0, 1.0)

    def test_texmtx_rotation_zero_skips_rotation_fields(self):
        flag = MatFlag.TEXMTX_ROTATION_ZERO
        raw = make_material_bytes(flag)
        r = BinaryReader(raw)
        mat = Material(r)
        assert r.tell() == len(raw)
        assert (mat.rot_sin, mat.rot_cos) == (0.0, 1.0)

    def test_effectmtx_reads_16_entry_matrix(self):
        flag = MatFlag.EFFECTMTX
        raw = make_material_bytes(flag)
        r = BinaryReader(raw)
        mat = Material(r)
        assert r.tell() == len(raw)
        assert mat.effect_mtx == pytest.approx([0.0] * 16)

    @pytest.mark.parametrize("flag", [
        0,
        MatFlag.TEXMTX_SCALE_ONE,
        MatFlag.TEXMTX_ROTATION_ZERO,
        MatFlag.TEXMTX_TRANSLATION_ZERO,
        MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO,
        MatFlag.EFFECTMTX,
        MatFlag.TEXMTX_SCALE_ONE | MatFlag.EFFECTMTX,
    ])
    def test_consumes_exactly_the_bytes_it_produced(self, flag):
        raw = make_material_bytes(flag)
        r = BinaryReader(raw)
        Material(r)
        assert r.tell() == len(raw)

    @pytest.mark.parametrize("flag", [
        0,
        MatFlag.TEXMTX_SCALE_ONE,
        MatFlag.TEXMTX_ROTATION_ZERO,
        MatFlag.TEXMTX_TRANSLATION_ZERO,
        MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO,
        MatFlag.EFFECTMTX,
        MatFlag.TEXMTX_SCALE_ONE | MatFlag.EFFECTMTX,
    ])
    def test_write_reads_back_identically(self, flag):
        original = Material(BinaryReader(make_material_bytes(
            flag, tex_image_param=0x12345678, poly_attr=0x87654321)))

        w = BinaryWriter()
        original.write(w)
        roundtripped = Material(BinaryReader(w.get_bytes()))

        assert w.length == len(make_material_bytes(flag))
        assert roundtripped.flag == original.flag
        assert roundtripped.tex_image_param == original.tex_image_param
        assert roundtripped.poly_attr == original.poly_attr
        assert (roundtripped.scale_s, roundtripped.scale_t) == \
            pytest.approx((original.scale_s, original.scale_t))
        assert (roundtripped.rot_sin, roundtripped.rot_cos) == \
            pytest.approx((original.rot_sin, original.rot_cos))
        assert roundtripped.effect_mtx == original.effect_mtx

    def test_tex_image_param_bitfields(self):
        tex_image_param = (3 << 20) | (2 << 23) | (5 << 26) | (1 << 29)
        flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO
        raw = make_material_bytes(flag, tex_image_param=tex_image_param)
        mat = Material(BinaryReader(raw))
        assert mat.tex_width == 8 << 3
        assert mat.tex_height == 8 << 2
        assert mat.tex_format == 5

    def test_poly_attr_bitfields(self):
        poly_attr = (1 << 6) | (5 << 16)
        flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO
        raw = make_material_bytes(flag, poly_attr=poly_attr)
        mat = Material(BinaryReader(raw))
        assert mat.cull_mode == 1
        assert mat.alpha == 5


def make_model_info_bytes(**overrides):
    base = dict(
        sbc_type=1, scaling_rule=2, tex_mtx_mode=0, node_count=3, mat_count=2,
        shape_count=4, first_unused_mtx_stack_id=5, pos_scale=1.5, inv_pos_scale=0.5,
        vertex_count=100, polygon_count=50, triangle_count=20, quad_count=10,
        box_x=1.0, box_y=-1.0, box_z=2.0, box_w=3.0, box_h=4.0, box_d=5.0,
        box_pos_scale=1.0, box_inv_pos_scale=1.0,
    )
    base.update(overrides)
    buf = struct.pack("<BBBBBBBB", base["sbc_type"], base["scaling_rule"], base["tex_mtx_mode"],
                       base["node_count"], base["mat_count"], base["shape_count"],
                       base["first_unused_mtx_stack_id"], 0)
    buf += struct.pack("<ii", fx32(base["pos_scale"]), fx32(base["inv_pos_scale"]))
    buf += struct.pack("<HHHH", base["vertex_count"], base["polygon_count"],
                        base["triangle_count"], base["quad_count"])
    buf += struct.pack("<6h", *[fx16(v) for v in (
        base["box_x"], base["box_y"], base["box_z"], base["box_w"], base["box_h"], base["box_d"])])
    buf += struct.pack("<ii", fx32(base["box_pos_scale"]), fx32(base["box_inv_pos_scale"]))
    return buf


class TestModelInfo:
    def test_write_reads_back_identically(self):
        raw = make_model_info_bytes()
        original = ModelInfo(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)  # regression: used to raise TypeError (patch_u16 misuse)

        assert w.get_bytes() == raw
        roundtripped = ModelInfo(BinaryReader(w.get_bytes()))
        assert roundtripped.vertex_count == original.vertex_count
        assert roundtripped.polygon_count == original.polygon_count
        assert roundtripped.triangle_count == original.triangle_count
        assert roundtripped.quad_count == original.quad_count
        assert roundtripped.node_count == original.node_count


class TestTexToMatData:
    def test_write_reads_back_identically(self):
        raw = struct.pack("<HBB", 40, 5, 0x80)
        original = TexToMatData(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw
        roundtripped = TexToMatData(BinaryReader(w.get_bytes()))
        assert roundtripped.offset == original.offset
        assert roundtripped.mat_count == original.mat_count
        assert roundtripped.flags == original.flags


class TestEnvelope:
    def test_write_reads_back_identically(self):
        raw = struct.pack("<12i", *[fx32(v) for v in range(12)]) + \
            struct.pack("<9i", *[fx32(v) for v in range(9)])
        original = Envelope(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw


class TestEvpMatrices:
    def test_write_reads_back_identically(self):
        single = struct.pack("<12i", *[0] * 12) + struct.pack("<9i", *[fx32(1.0)] * 9)
        raw = single * 3
        original = EvpMatrices(BinaryReader(raw), node_count=3)

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw
        assert len(original) == 3


class TestNodeSet:
    def test_write_reads_back_identically(self):
        n0 = NodeData(BinaryReader(make_node_bytes(0)))
        n1 = NodeData(BinaryReader(make_node_bytes(SrtFlag.SCALE_ONE)))
        node_set = NodeSet.__new__(NodeSet)
        node_set.dict = Dictionary(0, ["node0", "node1"], [0, 0], 4)
        node_set.nodes = [n0, n1]

        w = BinaryWriter()
        node_set.write(w)
        out = NodeSet(BinaryReader(w.get_bytes()))

        assert out.dict.names == ["node0", "node1"]
        assert len(out.nodes) == 2
        assert out.nodes[0].flag == n0.flag
        assert out.nodes[1].flag == n1.flag


class TestMaterialSet:
    def test_write_reads_back_identically(self):
        flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO
        mat = Material(BinaryReader(make_material_bytes(flag)))
        tex_to_mat = TexToMatData(BinaryReader(struct.pack("<HBB", 0, 1, 0)))
        tex_to_mat.materials = [0]

        mat_set = MaterialSet.__new__(MaterialSet)
        mat_set.dict = Dictionary(0, ["mat0"], [0], 4)
        mat_set.dict_tex_to_mat = Dictionary(0, ["tex0"], [tex_to_mat], 4)
        mat_set.dict_pltt_to_mat = Dictionary(0, [], [], 4)
        mat_set.materials = [mat]

        w = BinaryWriter()
        mat_set.write(w)
        out = MaterialSet(BinaryReader(w.get_bytes()))

        assert out.dict.names == ["mat0"]
        assert len(out.materials) == 1
        assert out.texture_name(0) == "tex0"


class TestShapeSet:
    def test_write_reads_back_identically(self):
        shp0 = Shape.__new__(Shape)
        shp0.tag, shp0.size, shp0.flag = 1, 16, 0
        shp0.dl_offset = shp0.dl_size = 0
        shp0.dl = b"\x01\x02\x03\x04"
        shp1 = Shape.__new__(Shape)
        shp1.tag, shp1.size, shp1.flag = 1, 16, 0
        shp1.dl_offset = shp1.dl_size = 0
        shp1.dl = b"\xaa\xbb"

        shape_set = ShapeSet.__new__(ShapeSet)
        shape_set.dict = Dictionary(0, ["shape0", "shape1"], [0, 0], 4)
        shape_set.shapes = [shp0, shp1]

        w = BinaryWriter()
        shape_set.write(w)  # regression: used to append garbage instead of patching in place
        out = ShapeSet(BinaryReader(w.get_bytes()))

        assert out.dict.names == ["shape0", "shape1"]
        assert out.shapes[0].dl == b"\x01\x02\x03\x04"
        assert out.shapes[1].dl == b"\xaa\xbb"


def build_model(sbc=b"\x01\x02\x03"):
    flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO
    info = ModelInfo(BinaryReader(make_model_info_bytes(node_count=1, mat_count=1, shape_count=1)))

    node_set = NodeSet.__new__(NodeSet)
    node_set.dict = Dictionary(0, ["node0"], [0], 4)
    node_set.nodes = [NodeData(BinaryReader(make_node_bytes(0)))]

    mat_set = MaterialSet.__new__(MaterialSet)
    mat_set.dict = Dictionary(0, ["mat0"], [0], 4)
    mat_set.dict_tex_to_mat = Dictionary(0, [], [], 4)
    mat_set.dict_pltt_to_mat = Dictionary(0, [], [], 4)
    mat_set.materials = [Material(BinaryReader(make_material_bytes(flag)))]

    shp = Shape.__new__(Shape)
    shp.tag, shp.size, shp.flag = 1, 16, 0
    shp.dl_offset = shp.dl_size = 0
    shp.dl = b"\xde\xad\xbe\xef"
    shape_set = ShapeSet.__new__(ShapeSet)
    shape_set.dict = Dictionary(0, ["shape0"], [0], 4)
    shape_set.shapes = [shp]

    model = Model.__new__(Model)
    model.info = info
    model.nodes = node_set
    model.sbc = sbc
    model.materials = mat_set
    model.shapes = shape_set
    model.evp_matrices = EvpMatrices.__new__(EvpMatrices)
    model.evp_matrices.m = [Envelope(BinaryReader(
        struct.pack("<12i", *[0] * 12) + struct.pack("<9i", *[fx32(1.0)] * 9)))]
    return model


class TestModel:
    def test_write_reads_back_identically(self):
        model = build_model()

        w = BinaryWriter()
        model.write(w)
        out = Model(BinaryReader(w.get_bytes()))

        assert out.sbc == model.sbc
        assert out.nodes.dict.names == ["node0"]
        assert out.materials.dict.names == ["mat0"]
        assert out.shapes.shapes[0].dl == b"\xde\xad\xbe\xef"
        assert out.evp_matrices is not None
        assert len(out.evp_matrices) == 1

    def test_write_without_evp_matrices(self):
        model = build_model()
        model.evp_matrices = None

        w = BinaryWriter()
        model.write(w)
        out = Model(BinaryReader(w.get_bytes()))

        assert out.evp_matrices is None


class TestMDL0:
    def test_write_reads_back_identically(self):
        mdl0 = MDL0.__new__(MDL0)
        mdl0.dict = Dictionary(0, ["model0"], [0], 4)
        mdl0.models = [build_model()]

        w = BinaryWriter()
        mdl0.write(w)
        out = MDL0(BinaryReader(w.get_bytes()))

        assert out.dict.names == ["model0"]
        assert len(out.models) == 1
        assert out.models[0].sbc == mdl0.models[0].sbc

    def test_write_multiple_models_at_nonzero_base(self):
        # Regression: MDL0 dictionary offsets are block-relative, like every
        # sibling dictionary (NodeSet, MaterialSet, ShapeSet), but __init__
        # used to seek() them as absolute file offsets. That only happened
        # to work when MDL0 sat at position 0, which is never true once it's
        # embedded after an NSBMD header.
        mdl0 = MDL0.__new__(MDL0)
        mdl0.dict = Dictionary(0, ["model0", "model1"], [0, 0], 4)
        mdl0.models = [build_model(b"\x01\x02\x03"), build_model(b"\xff\xff")]

        w = BinaryWriter()
        w.write_bytes(b"\x00" * 24)  # simulate a preceding NSBMD header
        mdl0.write(w)

        r = BinaryReader(w.get_bytes())
        r.seek(24)
        out = MDL0(r)

        assert out.dict.names == ["model0", "model1"]
        assert out.models[0].sbc == b"\x01\x02\x03"
        assert out.models[1].sbc == b"\xff\xff"
