import struct

from nitro.binary import BinaryReader, BinaryWriter
from nitro.dictionary import Dictionary, read_dictionary
from nitro.tex0 import (PlttDictData, PlttInfo, TEX0, Tex4x4Info, TexDictData,
                        TexFmt, TexInfo)


class TestTexInfo:
    def test_write_reads_back_identically(self):
        raw = struct.pack("<IHHHHI", 0xCAFEBABE, 5, 10, 0x1234, 0, 100)
        original = TexInfo(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw


class TestTex4x4Info:
    def test_init_calls_base_class_constructor(self):
        # Regression: __init__ used to call super(r) instead of
        # super().__init__(r), which raised TypeError on every TEX0 parse.
        raw = struct.pack("<IHHHHI", 0, 5, 10, 0, 0, 100) + struct.pack("<I", 200)
        info = Tex4x4Info(BinaryReader(raw))
        assert info.tex_size == 5 << 3
        assert info.tex_offset == 100
        assert info.tex_pltt_idx_offset == 200

    def test_write_reads_back_identically(self):
        # Regression: write() used to emit only the 4-byte extra field and
        # drop the inherited 16-byte TexInfo header entirely.
        raw = struct.pack("<IHHHHI", 0xCAFEBABE, 5, 10, 0x1234, 0, 100) + struct.pack("<I", 200)
        original = Tex4x4Info(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw


class TestPlttInfo:
    def test_write_reads_back_identically(self):
        raw = struct.pack("<IHHHHI", 0xCAFEBABE, 3, 20, 0x5678, 0, 300)
        original = PlttInfo(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw

    def test_flags_precede_the_dictionary_offset(self):
        # Unlike TexInfo, PlttInfo stores flags first. Reading them the other
        # way round meant the dictionary offset was mistaken for flags, and
        # writing zeroed a real flag bit (0x8000 in some retail files).
        raw = struct.pack("<IHHHHI", 0, 3, 0x8000, 132, 0, 300)
        info = PlttInfo(BinaryReader(raw))
        assert info.flags == 0x8000
        assert info.dict_offset == 132

    def test_write_preserves_a_flag_bit(self):
        raw = struct.pack("<IHHHHI", 0, 3, 0x8000, 132, 0, 300)
        original = PlttInfo(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw
        assert PlttInfo(BinaryReader(w.get_bytes())).flags == 0x8000


class TestTexDictData:
    def test_offset_is_scaled_by_8_on_read(self):
        # Regression: __init__ used to store the raw field value without the
        # <<3 scaling that PlttDictData (and write()) already assume.
        raw = struct.pack("<HHI", 40, 0, 0)
        entry = TexDictData(BinaryReader(raw))
        assert entry.offset == 40 << 3

    def test_write_reads_back_identically(self):
        raw = struct.pack("<HHI", 40, 0x1234, 0xDEADBEEF)
        original = TexDictData(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw
        roundtripped = TexDictData(BinaryReader(w.get_bytes()))
        assert roundtripped.offset == original.offset

    def test_tex_image_param_bitfields(self):
        tex_image_param = (3 << 4) | (2 << 7) | (TexFmt.A5I3 << 10) | (1 << 13)
        entry = TexDictData(BinaryReader(struct.pack("<HHI", 0, tex_image_param, 0)))
        assert entry.s == 8 << 3
        assert entry.t == 8 << 2
        assert entry.fmt == TexFmt.A5I3
        assert entry.transparent_color is True


class TestPlttDictData:
    def test_write_reads_back_identically(self):
        raw = struct.pack("<HH", 40, 0x5678)
        original = PlttDictData(BinaryReader(raw))

        w = BinaryWriter()
        original.write(w)

        assert w.get_bytes() == raw


def build_tex0(tex_data=None, pltt_data=None):
    t = TEX0.__new__(TEX0)
    t.tex_info = TexInfo.__new__(TexInfo)
    t.tex_info.vram_key = 0
    t.tex_info.flags = 0
    t.tex4x4_info = Tex4x4Info.__new__(Tex4x4Info)
    t.tex4x4_info.vram_key = 0
    t.tex4x4_info.flags = 0
    t.pltt_info = PlttInfo.__new__(PlttInfo)
    t.pltt_info.vram_key = 0
    t.pltt_info.flags = 0

    tex_image_param = TexFmt.A5I3 << 10  # 8x8 A5I3 => 64 bytes
    tex_entry = TexDictData(BinaryReader(struct.pack("<HHI", 0, tex_image_param, 0)))
    tex_entry.data = tex_data if tex_data is not None else bytes(range(64))
    t.tex_dict = Dictionary(0, ["tex0"], [tex_entry], 8)

    pltt_entry = PlttDictData(BinaryReader(struct.pack("<HH", 0, 0)))
    pltt_entry.data = pltt_data if pltt_data is not None else bytes(range(32))
    t.pltt_dict = Dictionary(0, ["pltt0"], [pltt_entry], 4)
    return t


class TestTEX0:
    def test_write_reads_back_identically(self):
        original = build_tex0()

        w = BinaryWriter()
        original.write(w)
        out = TEX0(BinaryReader(w.get_bytes()))

        assert out.tex_dict.names == ["tex0"]
        assert out.tex_dict.values()[0].data == original.tex_dict.values()[0].data
        assert out.pltt_dict.names == ["pltt0"]
        assert out.pltt_dict.values()[0].data == original.pltt_dict.values()[0].data

    def test_write_points_pltt_dict_offset_at_the_palette_dictionary(self):
        # The offset must be where the palette dictionary actually lands, not 0.
        original = build_tex0()

        w = BinaryWriter()
        original.write(w)
        out = TEX0(BinaryReader(w.get_bytes()))

        tex_count = len(original.tex_dict)
        expected = 60 + 8 + (tex_count + 1) * 4 + 4 + tex_count * 8 + tex_count * 16
        assert out.pltt_info.dict_offset == expected

        # and the dictionary really is there
        r = BinaryReader(w.get_bytes())
        r.seek(out.pltt_info.dict_offset)
        assert read_dictionary(r, lambda rd: PlttDictData(rd)).names == ["pltt0"]

    def test_write_preserves_pltt_flags(self):
        original = build_tex0()
        original.pltt_info.flags = 0x8000

        w = BinaryWriter()
        original.write(w)

        assert TEX0(BinaryReader(w.get_bytes())).pltt_info.flags == 0x8000

    def test_write_deduplicates_shared_texture_data(self):
        data = bytes(range(64))
        t = build_tex0(tex_data=data)
        second = TexDictData(BinaryReader(struct.pack("<HHI", 0, TexFmt.A5I3 << 10, 0)))
        second.data = data
        t.tex_dict = Dictionary(0, ["tex0", "tex1"], [t.tex_dict.values()[0], second], 8)

        w = BinaryWriter()
        t.write(w)
        out = TEX0(BinaryReader(w.get_bytes()))

        assert out.tex_dict.values()[0].data == data
        assert out.tex_dict.values()[1].data == data
        assert out.tex_dict.values()[0].offset == out.tex_dict.values()[1].offset
