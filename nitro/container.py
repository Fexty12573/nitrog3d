
from __future__ import annotations
from .binary import BinaryReader, BinaryWriter
from typing import Literal


Endian = Literal["little", "big"]


class Container:
    def __init__(self, r: BinaryReader, expected_sigs: list[str]):
        self.signature = r.read_str(4)
        if self.signature not in expected_sigs:
            raise ValueError(
                f"Expected one of {expected_sigs}, got {self.signature}")

        self.endianness = r.read_u16()
        self.version = r.read_u16()
        self.file_size = r.read_u32()
        self.header_size = r.read_u16()
        self.num_blocks = r.read_u16()
        self.block_offsets = r.read_u32s(self.num_blocks)

    def write(self, w: BinaryWriter):
        w.write_str(self.signature)
        w.write_u16(self.endianness)
        w.write_u16(self.version)
        w.write_u32(self.file_size)
        w.write_u16(self.header_size)
        w.write_u16(self.num_blocks)
        w.write_u32s(self.block_offsets)

    @classmethod
    def build(cls,
              sig: str,
              num_blocks: int = 0,
              endian: Endian = 'little',
              version: int = 2) -> Container:
        c = cls.__new__(cls)
        c.signature = sig
        c.endianness = 0xFEFF if endian == "little" else 0xFFFE
        c.version = version
        c.file_size = 0
        c.header_size = 16
        c.num_blocks = num_blocks
        c.block_offsets = [0] * num_blocks
        return c
