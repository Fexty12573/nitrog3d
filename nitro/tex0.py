
from __future__ import annotations
from .binary import BinaryReader, BinaryWriter
from .dictionary import read_dictionary, write_dictionary, make_dictionary
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

    def has_alpha(self) -> bool:
        return self in (self.A3I5, self.A5I3, self.COMP4X4, self.DIRECT)

    def is_pltt_n(self) -> bool:
        return self in (self.PLTT4, self.PLTT16, self.PLTT256)


class TexGen(IntEnum):
    NONE = 0
    TEXCOORD = 1
    NORMAL = 2
    VERTEX = 3


class TexRepeat(IntEnum):
    NONE = 0
    S = 1
    T = 2
    ST = 3


class TexFlip(IntEnum):
    NONE = 0
    S = 1
    T = 2
    ST = 3


class TexColor0Mode(IntEnum):
    NORMAL = 0
    TRANSPARENT = 1


class TexImageParam:
    def __init__(self, value: int):
        self.v = value & 0xFFFFFFFF

    def __eq__(self, value):
        if isinstance(value, TexImageParam):
            return self.v == value.v
        elif isinstance(value, int):
            return self.v == value
        return False

    @property
    def texgen(self) -> TexGen:
        return TexGen((self.v >> 30) & 0x3)

    @property
    def color0_mode(self) -> TexColor0Mode:
        return TexColor0Mode(((self.v >> 29) & 1))

    @property
    def format(self) -> TexFmt:
        return TexFmt((self.v >> 26) & 0x7)

    @property
    def width(self) -> int:
        return 8 << ((self.v >> 20) & 0x7)

    @property
    def height(self) -> int:
        return 8 << ((self.v >> 23) & 0x7)

    @property
    def flip(self) -> TexFlip:
        return TexFlip((self.v >> 18) & 0x3)

    @property
    def repeat(self) -> TexRepeat:
        return TexRepeat((self.v >> 16) & 0x3)

    @property
    def addr(self) -> int:
        return self.v & 0xFFFF

    @classmethod
    def build(cls,
              texgen: TexGen,
              color0_mode: TexColor0Mode,
              fmt: TexFmt,
              width: int,
              height: int,
              flip: TexFlip,
              repeat: TexRepeat,
              addr: int) -> TexImageParam:
        w = 0
        w |= (texgen & 0x3) << 30
        w |= (color0_mode & 0x1) << 29
        w |= (fmt & 0x7) << 26
        w |= ((width.bit_length() - 4) & 0x7) << 20
        w |= ((height.bit_length() - 4) & 0x7) << 23
        w |= (flip & 0x3) << 18
        w |= (repeat & 0x3) << 16
        w |= addr & 0xFFFF
        return cls(w)


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
                base
            )

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

    def write(self, w: BinaryWriter):
        base = w.tell()
        w.write_str(self.SIGNATURE)
        pos_size = w.tell()
        w.write_u32(0)

        # Nasty manual offset calculations because I want byte-exact load-save behavior.
        # Existing files have deduplicated data if 2 texture/palettes have the same data
        # so we need to track offsets as well.
        tex_pool = bytearray()
        tex4x4_pool = bytearray()
        tex4x4_idx_pool = bytearray()
        reg_map = {}
        comp_map = {}
        tex_new_off = []
        for entry in self.tex_dict.values():
            orig = entry.offset
            if entry.fmt != TexFmt.COMP4X4:
                if orig not in reg_map:
                    reg_map[orig] = len(tex_pool)
                    tex_pool += entry.data
                tex_new_off.append(reg_map[orig])
            else:
                if orig not in comp_map:
                    comp_map[orig] = len(tex4x4_pool)
                    tex4x4_pool += entry.data
                    tex4x4_idx_pool += entry.data4x4
                tex_new_off.append(comp_map[orig])

        uniq_offs = []
        seen_pltt = {}
        for entry in self.pltt_dict.values():
            if entry.offset not in seen_pltt:
                seen_pltt[entry.offset] = len(uniq_offs)
                uniq_offs.append((entry.offset, entry.data))

        pltt_pool = bytearray()
        pos_of = {}
        for i, (orig, data) in enumerate(uniq_offs):
            pos_of[orig] = len(pltt_pool)
            pltt_pool += data
            if i != len(uniq_offs) - 1 and len(pltt_pool) % 8 != 0:
                pltt_pool += b"\x00" * (8 - len(pltt_pool) % 8)

        pltt_new_off = [pos_of[e.offset] for e in self.pltt_dict.values()]

        tex_count = len(self.tex_dict)
        pltt_count = len(self.pltt_dict)

        # Dictionary size: header + tree + entry-header + payloads + names
        tex_dict_bytes = 8 + (tex_count + 1) * 4 + 4 + \
            tex_count * 8 + tex_count * 16
        pltt_dict_bytes = 8 + (pltt_count + 1) * 4 + 4 + \
            pltt_count * 4 + pltt_count * 16

        pos_tex_dict = 60  # 8 header + 52 info structs
        pos_pltt_dict = pos_tex_dict + tex_dict_bytes
        pos_data = pos_pltt_dict + pltt_dict_bytes

        self.tex_info.dict_offset = pos_tex_dict
        self.tex_info.tex_size = len(tex_pool)
        self.tex_info.tex_offset = pos_data

        self.tex4x4_info.dict_offset = pos_tex_dict
        self.tex4x4_info.tex_size = len(tex4x4_pool)
        self.tex4x4_info.tex_offset = pos_data + len(tex_pool)
        self.tex4x4_info.tex_pltt_idx_offset = pos_data + \
            len(tex_pool) + len(tex4x4_pool)

        self.pltt_info.dict_offset = pos_pltt_dict
        self.pltt_info.pltt_size = len(pltt_pool)
        self.pltt_info.pltt_offset = pos_data + \
            len(tex_pool) + len(tex4x4_pool) + len(tex4x4_idx_pool)

        self.tex_info.write(w)
        self.tex4x4_info.write(w)
        self.pltt_info.write(w)

        for entry, offset in zip(self.tex_dict.values(), tex_new_off):
            entry.offset = offset
        write_dictionary(w, self.tex_dict, lambda wr, v: v.write(wr))

        for entry, offset in zip(self.pltt_dict.values(), pltt_new_off):
            entry.offset = offset
        write_dictionary(w, self.pltt_dict, lambda wr, v: v.write(wr))

        w.write_bytes(tex_pool)
        w.write_bytes(tex4x4_pool)
        w.write_bytes(tex4x4_idx_pool)
        w.write_bytes(pltt_pool)

        w.patch_u32(pos_size, w.tell() - base)

    @classmethod
    def build(cls,
              tex_info: TexInfo,
              tex4x4_info: Tex4x4Info,
              pltt_info: PlttInfo,
              textures: dict[str, TexDictData],
              palettes: dict[str, PlttDictData]) -> TEX0:
        t = cls.__new__(cls)
        t.section_size = 0
        t.tex_info = tex_info
        t.tex4x4_info = tex4x4_info
        t.pltt_info = pltt_info
        t.tex_dict = make_dictionary(textures, TexDictData.SIZE)
        t.pltt_dict = make_dictionary(palettes, PlttDictData.SIZE)
        return t


class TexInfo:
    def __init__(self, r: BinaryReader):
        self.vram_key = r.read_u32()
        self.tex_size = r.read_u16() << 3
        self.dict_offset = r.read_u16()
        self.flags = r.read_u16()
        r.skip(2)
        self.tex_offset = r.read_u32()

    def write(self, w: BinaryWriter):
        w.write_u32(self.vram_key)
        w.write_u16(self.tex_size >> 3)
        w.write_u16(self.dict_offset)
        w.write_u16(self.flags)
        w.write_u16(0)
        w.write_u32(self.tex_offset)

    @classmethod
    def build(cls, vram_key: int, flags: int) -> TexInfo:
        t = cls.__new__(cls)
        t.vram_key = vram_key
        t.flags = flags
        t.tex_size = 0
        t.dict_offset = 0
        t.tex_offset = 0
        return t


class Tex4x4Info(TexInfo):
    def __init__(self, r: BinaryReader):
        super().__init__(r)
        self.tex_pltt_idx_offset = r.read_u32()

    def write(self, w: BinaryWriter):
        super().write(w)
        w.write_u32(self.tex_pltt_idx_offset)

    @classmethod
    def build(cls, vram_key: int, flags: int) -> Tex4x4Info:
        t = cls.__new__(cls)
        t.vram_key = vram_key
        t.flags = flags
        t.tex_size = 0
        t.dict_offset = 0
        t.tex_offset = 0
        t.tex_pltt_idx_offset = 0
        return t


class PlttInfo:
    """Palette set header.

    Note the field order differs from TexInfo: the flags come *before* the
    dictionary offset here, where TexInfo has them the other way round.
    """

    def __init__(self, r: BinaryReader):
        self.vram_key = r.read_u32()
        self.pltt_size = r.read_u16() << 3
        self.flags = r.read_u16()
        self.dict_offset = r.read_u16()
        r.skip(2)
        self.pltt_offset = r.read_u32()

    def write(self, w: BinaryWriter):
        w.write_u32(self.vram_key)
        w.write_u16(self.pltt_size >> 3)
        w.write_u16(self.flags)
        w.write_u16(self.dict_offset)
        w.write_u16(0)
        w.write_u32(self.pltt_offset)

    @classmethod
    def build(cls, vram_key: int, flags: int) -> PlttInfo:
        p = cls.__new__(cls)
        p.vram_key = vram_key
        p.flags = flags
        p.pltt_size = 0
        p.dict_offset = 0
        p.pltt_offset = 0
        return p


class TexDictData:
    SIZE = 8

    def __init__(self, r: BinaryReader):
        self.tex_image_param = TexImageParam(r.read_u32())
        self.offset = self.tex_image_param.addr << 3
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

    def write(self, w: BinaryWriter):
        w.write_u32((self.tex_image_param.v & 0xFFFF0000)
                    | ((self.offset >> 3) & 0x0000FFFF))
        w.write_u32(self.extra_param)

    @classmethod
    def build(cls, param: TexImageParam, extra_param: int, data: bytes, data4x4: bytes = b"") -> TexDictData:
        d = cls.__new__(cls)
        d.offset = 0
        d.tex_image_param = param
        d.extra_param = extra_param
        d.data = data
        d.data4x4 = data4x4
        return d

    @property
    def s(self) -> int:
        return self.tex_image_param.width

    @property
    def t(self) -> int:
        return self.tex_image_param.height

    @property
    def fmt(self) -> TexFmt:
        return self.tex_image_param.format

    @property
    def transparent_color(self) -> bool:
        return self.tex_image_param.color0_mode == TexColor0Mode.TRANSPARENT


class PlttDictData:
    SIZE = 4

    def __init__(self, r: BinaryReader):
        self.offset = r.read_u16() << 3
        self.flag = r.read_u16()
        self.data = b""

    def read_data(self, r: BinaryReader, base_pltt: int, length: int, tex_set_offset: int):
        pos = r.tell()
        r.seek(self.offset + base_pltt + tex_set_offset)
        self.data = r.read_bytes(max(0, length))
        r.seek(pos)

    def write(self, w: BinaryWriter):
        w.write_u16((self.offset >> 3) & 0xFFFF)
        w.write_u16(self.flag)

    @classmethod
    def build(cls, data: bytes, flag: int = 0) -> PlttDictData:
        d = cls.__new__(cls)
        d.offset = 0
        d.flag = flag
        d.data = data
        return d
