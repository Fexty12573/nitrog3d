
from .binary import BinaryReader, BinaryWriter
from .mdl0 import MDL0
from .tex0 import TEX0


class NSBMDHeader:
    def __init__(self, r: BinaryReader):
        self.signature = r.read_str(4)
        if self.signature not in ("BMD0", "BTX0"):
            raise ValueError(
                f"Not and NSBMD/NSBTX file (signature {self.signature})")

        self.endianness = r.read_u16()
        self.version = r.read_u16()
        self.file_size = r.read_u32()
        self.header_size = r.read_u16()
        self.num_blocks = r.read_u16()
        self.block_offsets = r.read_u32s(self.num_blocks)


class NSBMD:
    def __init__(self, data: bytes | bytearray):
        r = BinaryReader(data)
        self.header = NSBMDHeader(r)
        self.model_set = None
        self.tex_pltt_set = None

        if self.header.signature == "BTX0":
            if self.header.num_blocks > 0:
                r.seek(self.header.block_offsets[0])
                self.tex_pltt_set = TEX0(r)
            return

        if self.header.num_blocks > 0:  # MDL0 Block is always the first block
            r.seek(self.header.block_offsets[0])
            self.model_set = MDL0(r)
        if self.header.num_blocks > 1:  # TEX0 Block is always the second block, if there is one
            r.seek(self.header.block_offsets[1])
            self.tex_pltt_set = TEX0(r)

    def write(self) -> bytes:
        w = BinaryWriter()
        h = self.header
        sig = h.signature
        if sig == "BTX0":
            num_blocks = 1
        else:
            num_blocks = 2 if self.tex_pltt_set is not None else 1

        w.write_str(sig)
        w.write_u16(h.endianness)
        w.write_u16(h.version)
        pos_file_size = w.tell()
        w.write_u32(0)
        w.write_u16(h.header_size)
        w.write_u16(num_blocks)
        pos_block_table = w.tell()
        for _ in range(num_blocks):
            w.write_u32(0)

        if sig == "BTX0":
            w.patch_u32(pos_block_table, w.tell())
            self.tex_pltt_set.write(w)
        else:
            w.patch_u32(pos_block_table, w.tell())
            self.model_set.write(w)
            if self.tex_pltt_set is not None:
                w.patch_u32(pos_block_table + 4, w.tell())
                self.tex_pltt_set.write(w)

        w.patch_u32(pos_file_size, w.tell())
        return w.get_bytes()
