from __future__ import annotations
from .binary import BinaryReader, BinaryWriter
from .tex0 import TEX0
from .container import Container


class NSBTX:
    def __init__(self, data: bytes | bytearray):
        r = BinaryReader(data)
        self.container = Container(r, ["BTX0"])

        if self.container.num_blocks < 1:
            raise RuntimeError("Missing TEX0 Block")

        r.seek(self.container.block_offsets[0])
        self.tex0 = TEX0(r)

    def write(self) -> bytes:
        w = BinaryWriter()
        cont = self.container
        cont.num_blocks = 1
        cont.block_offsets = [0] * cont.num_blocks
        cont.write(w)

        cont.block_offsets[0] = w.tell()
        self.tex0.write(w)

        cont.file_size = w.length

        w.seek(0)
        cont.write(w)

        return w.get_bytes()

    @classmethod
    def build(cls, tex: TEX0) -> NSBTX:
        nsbtx = cls.__new__(cls)
        nsbtx.container = Container.build(
            sig="BTX0",
            num_blocks=1
        )

        nsbtx.tex0 = tex
        return nsbtx
