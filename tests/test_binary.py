import struct

import pytest

from nitro.binary import BinaryReader, BinaryWriter, FX16_SCALE, FX32_SCALE


class TestBinaryReaderIntegers:
    def test_read_u8(self):
        r = BinaryReader(bytes([0x00, 0x7F, 0xFF]))
        assert r.read_u8() == 0
        assert r.read_u8() == 127
        assert r.read_u8() == 255

    def test_read_s8_boundaries(self):
        r = BinaryReader(bytes([0x00, 0x7F, 0x80, 0xFF]))
        assert r.read_s8() == 0
        assert r.read_s8() == 127
        assert r.read_s8() == -128
        assert r.read_s8() == -1

    def test_read_u16_little_endian(self):
        r = BinaryReader(struct.pack("<H", 0xBEEF))
        assert r.read_u16() == 0xBEEF

    def test_read_s16_negative(self):
        r = BinaryReader(struct.pack("<h", -1234))
        assert r.read_s16() == -1234

    def test_read_u32_little_endian(self):
        r = BinaryReader(struct.pack("<I", 0xDEADBEEF))
        assert r.read_u32() == 0xDEADBEEF

    def test_read_s32_negative(self):
        r = BinaryReader(struct.pack("<i", -100000))
        assert r.read_s32() == -100000

    def test_read_f32(self):
        r = BinaryReader(struct.pack("<f", 3.5))
        assert r.read_f32() == pytest.approx(3.5)


class TestBinaryReaderFixedPoint:
    def test_read_fx16_positive(self):
        r = BinaryReader(struct.pack("<h", 1 * int(FX16_SCALE)))
        assert r.read_fx16() == pytest.approx(1.0)

    def test_read_fx16_negative(self):
        r = BinaryReader(struct.pack("<h", int(-0.5 * FX16_SCALE)))
        assert r.read_fx16() == pytest.approx(-0.5)

    def test_read_fx32_fraction(self):
        r = BinaryReader(struct.pack("<i", int(1.25 * FX32_SCALE)))
        assert r.read_fx32() == pytest.approx(1.25)

    def test_read_fx16s(self):
        vals = [1.0, -1.0, 0.5]
        raw = struct.pack("<3h", *[int(v * FX16_SCALE) for v in vals])
        r = BinaryReader(raw)
        assert r.read_fx16s(3) == pytest.approx(vals)

    def test_read_fx32s(self):
        vals = [1.0, -2.5, 0.25]
        raw = struct.pack("<3i", *[int(v * FX32_SCALE) for v in vals])
        r = BinaryReader(raw)
        assert r.read_fx32s(3) == pytest.approx(vals)


class TestBinaryReaderBytesAndArrays:
    def test_read_bytes(self):
        r = BinaryReader(b"\x01\x02\x03\x04")
        assert r.read_bytes(2) == b"\x01\x02"
        assert r.read_bytes(2) == b"\x03\x04"

    def test_read_u16s(self):
        raw = struct.pack("<3H", 1, 2, 3)
        r = BinaryReader(raw)
        assert r.read_u16s(3) == [1, 2, 3]

    def test_read_u32s(self):
        raw = struct.pack("<3I", 10, 20, 30)
        r = BinaryReader(raw)
        assert r.read_u32s(3) == [10, 20, 30]


class TestBinaryReaderStrings:
    def test_read_str(self):
        r = BinaryReader(b"BMD0")
        assert r.read_str(4) == "BMD0"

    def test_read_str_invalid_ascii_replaces(self):
        r = BinaryReader(b"\xff\xfe")
        # errors="replace" must not raise on invalid ascii
        assert r.read_str(2) == "��"

    def test_read_name_strips_padding(self):
        r = BinaryReader(b"joint1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        assert r.read_key(16) == "joint1"

    def test_read_name_full_length_no_padding(self):
        name = "0123456789abcdef"  # exactly 16 chars, no null terminator
        r = BinaryReader(name.encode("ascii"))
        assert r.read_key(16) == name

    def test_read_name_only_strips_null_bytes_not_trailing_garbage(self):
        # Documents current behavior: read_name removes ALL \x00 bytes, so
        # bytes after an embedded null are kept rather than truncated there.
        r = BinaryReader(b"ab\x00cd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        assert r.read_key(16) == "abcd"


class TestBinaryReaderPositioning:
    def test_tell_and_seek(self):
        r = BinaryReader(b"\x00\x01\x02\x03\x04")
        r.read_u16()
        assert r.tell() == 2
        r.seek(0)
        assert r.tell() == 0

    def test_skip(self):
        r = BinaryReader(b"\x00\x01\x02\x03")
        r.skip(2)
        assert r.read_u8() == 2

    def test_length(self):
        r = BinaryReader(b"\x00\x01\x02")
        assert r.length == 3

    def test_eof(self):
        r = BinaryReader(b"\x00")
        assert not r.eof()
        r.read_u8()
        assert r.eof()


class TestBinaryWriterIntegers:
    def test_write_u8(self):
        w = BinaryWriter()
        w.write_u8(200)
        assert w.get_bytes() == bytes([200])

    def test_write_u8_masks_overflow(self):
        w = BinaryWriter()
        w.write_u8(0x1FF)  # only the low byte should be kept
        assert w.get_bytes() == bytes([0xFF])

    def test_write_s8_clamps_to_range(self):
        w = BinaryWriter()
        w.write_s8(500)
        w.write_s8(-500)
        assert struct.unpack("<bb", w.get_bytes()) == (127, -128)

    def test_write_u16(self):
        w = BinaryWriter()
        w.write_u16(0xBEEF)
        assert struct.unpack("<H", w.get_bytes())[0] == 0xBEEF

    def test_write_s16_clamps_to_range(self):
        w = BinaryWriter()
        w.write_s16(100000)
        assert struct.unpack("<h", w.get_bytes())[0] == 0x7FFF

    def test_write_u32(self):
        w = BinaryWriter()
        w.write_u32(0xDEADBEEF)
        assert struct.unpack("<I", w.get_bytes())[0] == 0xDEADBEEF

    def test_write_s32_clamps_to_range(self):
        w = BinaryWriter()
        w.write_s32(1 << 40)
        assert struct.unpack("<i", w.get_bytes())[0] == 0x7FFFFFFF


class TestBinaryWriterFixedPoint:
    def test_write_fx16_roundtrip(self):
        w = BinaryWriter()
        w.write_fx16(1.5)
        r = BinaryReader(w.get_bytes())
        assert r.read_fx16() == pytest.approx(1.5)

    def test_write_fx32_roundtrip(self):
        w = BinaryWriter()
        w.write_fx32(-3.25)
        r = BinaryReader(w.get_bytes())
        assert r.read_fx32() == pytest.approx(-3.25)


class TestBinaryWriterBuffer:
    def test_write_bytes_appends(self):
        w = BinaryWriter()
        w.write_bytes(b"\x01\x02")
        w.write_bytes(b"\x03\x04")
        assert w.get_bytes() == b"\x01\x02\x03\x04"

    def test_write_bytes_overwrites_in_place(self):
        w = BinaryWriter()
        w.write_bytes(b"\xff\xff\xff\xff")
        w.seek(1)
        w.write_bytes(b"\x00\x00")
        assert w.get_bytes() == b"\xff\x00\x00\xff"

    def test_write_bytes_zero_fills_gap_on_forward_seek(self):
        w = BinaryWriter()
        w.seek(4)
        w.write_bytes(b"\xaa\xbb")
        assert w.get_bytes() == b"\x00\x00\x00\x00\xaa\xbb"

    def test_length_reflects_buffer_size(self):
        # Regression test: `length` used to call len() on the int position
        # (self.pos) instead of the byte buffer, raising a TypeError.
        w = BinaryWriter()
        assert w.length == 0
        w.write_u32(1)
        assert w.length == 4
        w.seek(0)
        assert w.length == 4  # seeking back must not shrink the buffer

    def test_write_str(self):
        w = BinaryWriter()
        w.write_str("BMD0")
        assert w.get_bytes() == b"BMD0"

    def test_align_pads_to_boundary(self):
        w = BinaryWriter()
        w.write_u8(1)
        w.align(4)
        assert w.get_bytes() == b"\x01\x00\x00\x00"
        assert w.tell() == 4

    def test_align_noop_when_already_aligned(self):
        w = BinaryWriter()
        w.write_u32(1)
        w.align(4)
        assert w.tell() == 4


class TestBinaryWriterKeys:
    def test_write_key_pads_with_nulls(self):
        w = BinaryWriter()
        w.write_key("joint1", 16)
        assert w.get_bytes() == b"joint1" + b"\x00" * 10

    def test_write_key_truncates_when_too_long(self):
        w = BinaryWriter()
        w.write_key("0123456789abcdefXYZ", 16)
        assert w.get_bytes() == b"0123456789abcdef"

    def test_write_key_roundtrips_through_read_key(self):
        w = BinaryWriter()
        w.write_key("joint1", 16)
        r = BinaryReader(w.get_bytes())
        assert r.read_key(16) == "joint1"

    def test_write_key_default_length_is_16(self):
        w = BinaryWriter()
        w.write_key("abc")
        assert w.length == 16

    def test_write_key_replaces_non_ascii_instead_of_raising(self):
        w = BinaryWriter()
        w.write_key("café", 8)
        r = BinaryReader(w.get_bytes())
        assert r.read_key(8).startswith("caf")


class TestBinaryWriterPatch:
    def test_patch_u16_writes_at_offset(self):
        w = BinaryWriter()
        w.write_u16(0)
        w.write_u16(0xAAAA)
        w.patch_u16(0, 0x1234)
        assert struct.unpack("<HH", w.get_bytes()) == (0x1234, 0xAAAA)

    def test_patch_u16_restores_position(self):
        w = BinaryWriter()
        w.write_u16(0)
        w.write_u16(0)
        w.patch_u16(0, 1)
        assert w.tell() == 4

    def test_patch_u32_writes_at_offset(self):
        w = BinaryWriter()
        w.write_u32(0)
        w.write_u32(0xAAAAAAAA)
        w.patch_u32(0, 0xDEADBEEF)
        assert struct.unpack("<II", w.get_bytes()) == (0xDEADBEEF, 0xAAAAAAAA)

    def test_patch_u32_restores_position(self):
        w = BinaryWriter()
        w.write_u32(0)
        w.write_u32(0)
        w.patch_u32(0, 1)
        assert w.tell() == 8
