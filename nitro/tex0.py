
from .binary import BinaryReader, BinaryWriter
from .dictionary import read_dictionary
from enum import IntEnum


class TexFmt(IntEnum):
    NONE = 0
    A3I5 = 1
    PLTT4 = 2
    PLTT16 = 3
    PLTT256 = 4
    COMP4X4 = 5
    A5I3 = 6
    DIRECT = 7


_DATA_BITS = [
    0,  # NONE
    8,  # A3I5
    2,  # PLTT4
    4,  # PLTT16
    8,  # PLTT256
    2,  # COMP4X4
    8,  # A5I3
    16  # DIRECT
]


class TEX0:
    SIGNATURE = "TEX0"

    def __init__(self, r: BinaryReader):
        base = r.tell()
        sig = r.read_str(4)
        if sig != self.SIGNATURE:
            raise ValueError(f"Expected TEX0 signature, got {sig}")

        self.section_size = r.read_u32()
        self.tex_info = TexInfo(r)
        self.tex4x4_info = Tex4x4Info(r)
        self.pltt_info = PlttInfo(r)

        self.tex_dict = read_dictionary(r, lambda rd: TexDictData(rd))
        for entry in self.tex_dict.values():
            entry.read_data(
                r,
                self.tex_info.tex_offset,
                self.tex4x4_info.tex_offset,
                self.tex4x4_info.tex_pltt_idx_offset,
                base)

        self.pltt_dict = read_dictionary(r, lambda rd: PlttDictData(rd))
        offsets = sorted({e.offset for e in self.pltt_dict.values()})
        stream_len = r.length
        pltt_offset = self.pltt_info.pltt_offset
        for entry in self.pltt_dict.values():
            idx = offsets.index(entry.offset)
            offset = offsets[idx]
            if idx == len(offsets) - 1:
                length = stream_len - (offset + pltt_offset + base)
            else:
                length = offsets[idx + 1] - offset
            entry.read_data(r, pltt_offset, length, base)


class TexInfo:
    def __init__(self, r: BinaryReader):
        self.vram_key = r.read_u32()
        self.tex_size = r.read_u16() << 3
        self.dict_offset = r.read_u16()
        self.flags = r.read_u16()
        r.skip(2)
        self.tex_offset = r.read_u32()


class Tex4x4Info(TexInfo):
    def __init__(self, r: BinaryReader):
        super(r)
        self.tex_pltt_idx_offset = r.read_u32()


class PlttInfo:
    def __init__(self, r: BinaryReader):
        self.vram_key = r.read_u32()
        self.pltt_size = r.read_u16() << 3
        self.dict_offset = r.read_u16()
        self.flags = r.read_u16()
        r.skip(2)
        self.pltt_offset = r.read_u32()


class TexDictData:
    def __init__(self, r: BinaryReader):
        self.tex_image_param = r.read_u32()
        self.offset = (self.tex_image_param & 0xFFFF) << 3
        self.s = 8 << ((self.tex_image_param >> 20) & 0x7)
        self.t = 8 << ((self.tex_image_param >> 23) & 0x7)
        self.fmt = (self.tex_image_param >> 26) & 0x7
        self.transparent_color = ((self.tex_image_param >> 29) & 1) == 1
        self.extra_param = r.read_u32()
        self.data = b""
        self.data4x4 = b""

    def read_data(self, r: BinaryReader, base_tex: int, base_tex4x4: int, base_text4x4_info: int, tex_set_offset: int):
        pos = r.tell()
        n_bytes = self.s * self.t * _DATA_BITS[self.fmt] // 8
        if self.fmt == TexFmt.COMP4X4:
            r.seek(self.offset + base_tex4x4 + tex_set_offset)
            self.data = r.read_bytes(n_bytes)
            r.seek(self.offset // 2 + base_text4x4_info + tex_set_offset)
            self.data4x4 = r.read_bytes(n_bytes // 2)
        else:
            r.seek(self.offset + base_tex + tex_set_offset)
            self.data = r.read_bytes(n_bytes)
        r.seek(pos)


class PlttDictData:
    def __init__(self, r: BinaryReader):
        self.offset = r.read_u16() << 3
        self.flag = r.read_u16()
        self.data = b""

    def read_data(self, r: BinaryReader, base_pltt: int, length: int, tex_set_offset: int):
        pos = r.tell()
        r.seek(self.offset + base_pltt + tex_set_offset)
        self.data = r.read_bytes(max(0, length))
        r.seek(pos)
