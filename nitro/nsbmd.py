
from __future__ import annotations
from .binary import BinaryReader, BinaryWriter
from .mdl0 import MDL0
from .tex0 import TEX0
from .container import Container


class NSBMD:
    def __init__(self, data: bytes | bytearray):
        r = BinaryReader(data)
        self.container = Container(r, ["BMD0", "BTX0"])
        self.model_set = None
        self.tex_pltt_set = None

        if self.container.signature == "BTX0":
            if self.container.num_blocks > 0:
                r.seek(self.container.block_offsets[0])
                self.tex_pltt_set = TEX0(r)
            return

        if self.container.num_blocks > 0:  # MDL0 Block is always the first block
            r.seek(self.container.block_offsets[0])
            self.model_set = MDL0(r)
        if self.container.num_blocks > 1:  # TEX0 Block is always the second block, if there is one
            r.seek(self.container.block_offsets[1])
            self.tex_pltt_set = TEX0(r)

    def write(self) -> bytes:
        w = BinaryWriter()
        cont = self.container
        if cont.signature == "BTX0":
            cont.num_blocks = 1
        else:
            cont.num_blocks = 2 if self.tex_pltt_set is not None else 1

        cont.block_offsets = [0] * cont.num_blocks
        cont.write(w)

        if cont.signature == "BTX0":
            cont.block_offsets[0] = w.tell()
            self.tex_pltt_set.write(w)
        else:
            cont.block_offsets[0] = w.tell()
            self.model_set.write(w)
            if self.tex_pltt_set is not None:
                cont.block_offsets[1] = w.tell()
                self.tex_pltt_set.write(w)

        cont.file_size = w.length

        w.seek(0)
        cont.write(w)

        return w.get_bytes()

    @classmethod
    def build(cls, mdl: MDL0, tex: TEX0 | None = None) -> NSBMD:
        nsbmd = cls.__new__(cls)
        nsbmd.container = Container.build(
            sig="BMD0",
            num_blocks=1 if tex is None else 2
        )

        nsbmd.model_set = mdl
        nsbmd.tex_pltt_set = tex
        return nsbmd
