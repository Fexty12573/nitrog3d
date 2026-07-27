
from .binary import BinaryReader
from .dictionary import read_dictionary
from enum import IntEnum


class SrtAnimFlag(IntEnum):
    VALUES_FX16 = 0x10000000
    VALUE_CONST = 0x20000000
    FRAME_STEP_2 = 0x40000000
    FRAME_STEP_4 = 0x80000000
    LAST_INTERP_MASK = 0xFFFF


class SRT0:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        sig = r.read_str(4)
        if sig != "SRT0":
            raise ValueError(f"Expected SRT0 signature, got {sig}")

        self.section_size = r.read_u32()
        self.dict = read_dictionary(r, lambda rd: rd.read_u32())

        offsets = self.dict.values()
        self.anims: list[SrtAnim] = []
        for i, offset in enumerate(offsets):
            end = offsets[i + 1] if i + 1 < len(offsets) else self.section_size
            r.seek(base + offset)
            self.anims.append(SrtAnim(r, base + end))


class SrtAnim:
    def __init__(self, r: BinaryReader, end: int):
        self.cat0 = r.read_str(1)
        self.revision = r.read_u8()
        self.cat1 = r.read_str(2)[::-1]
        if self.cat0 != "M" or self.cat1 != "TA":
            raise ValueError(f"Expected SRT Anim signature, got {self.cat0} {self.cat1}")

        self.frames = r.read_u16()
        self.flags = r.read_u8()
        self.tex_mtx_mode = r.read_u8()
        self.dict = read_dictionary(r, lambda rd: SrtAnimDictData(rd))
        self.data = r.read_bytes(end - r.tell())


class SrtAnimDictData:
    def __init__(self, r: BinaryReader):
        self.scale_s_flags = r.read_u32()
        self.scale_s_value = r.read_u32()
        self.scale_t_flags = r.read_u32()
        self.scale_t_value = r.read_u32()
        self.rot_flags = r.read_u32()
        self.rot_value = r.read_u32()
        self.trans_s_flags = r.read_u32()
        self.trans_s_value = r.read_u32()
        self.trans_t_flags = r.read_u32()
        self.trans_t_value = r.read_u32()
