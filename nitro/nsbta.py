
from .container import Container
from .binary import BinaryReader, BinaryWriter
from .srt0 import SRT0


class NSBTA:
    def __init__(self, data: bytes | bytearray):
        r = BinaryReader(data)
        self.container = Container(r, ["BTA0"])

        if self.container.num_blocks > 0:
            r.seek(self.container.block_offsets[0])
            self.srt_anim_set = SRT0(r)
