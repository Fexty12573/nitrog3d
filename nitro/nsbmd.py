
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
