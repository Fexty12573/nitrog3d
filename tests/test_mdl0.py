import struct

import pytest

from nitro.binary import BinaryReader, FX16_SCALE, FX32_SCALE
from nitro.mdl0 import Material, MatFlag, NodeData, SrtFlag


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
