
import struct

FX32_SCALE = float(1 << 12)
FX16_SCALE = float(1 << 12)


class BinaryReader:
    """Little endian binary reader over a ``bytes`` or ``bytearray`` buffer"""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes | bytearray, pos: int = 0):
        self.data = data
        self.pos = pos

    def tell(self) -> int:
        return self.pos

    def seek(self, pos: int):
        self.pos = pos

    def skip(self, n: int):
        self.pos += n

    @property
    def length(self):
        return len(self.data)

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def read_u8(self) -> int:
        v = self.data[self.pos]
        self.pos += 1
        return v

    def read_s8(self) -> int:
        v = self.data[self.pos]
        self.pos += 1
        return v - 256 if v >= 128 else v

    def read_u16(self) -> int:
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_s16(self) -> int:
        v = struct.unpack_from("<h", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_u32(self) -> int:
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_s32(self) -> int:
        v = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_f32(self) -> float:
        v = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_fx16(self) -> float:
        return self.read_s16() / FX16_SCALE

    def read_fx32(self) -> float:
        return self.read_s32() / FX32_SCALE

    def read_bytes(self, n: int) -> bytes:
        v = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return v

    def read_u16s(self, n: int) -> list[int]:
        v = struct.unpack_from(f"<{n}H", self.data, self.pos)
        self.pos += n * 2
        return list(v)

    def read_u32s(self, n: int) -> list[int]:
        v = struct.unpack_from(f"<{n}I", self.data, self.pos)
        self.pos += n * 4
        return list(v)

    def read_fx16s(self, n: int) -> list[float]:
        return [self.read_fx16() for _ in range(n)]

    def read_fx32s(self, n: int) -> list[float]:
        return [self.read_fx32() for _ in range(n)]

    def read_str(self, n: int, encoding="ascii") -> str:
        raw = self.read_bytes(n)
        return raw.decode(encoding=encoding, errors="replace")

    def read_name(self, n: int = 16) -> str:
        """Read a fixed-length, 0-padded string"""
        return self.read_str(n).replace("\x00", "")


def _clamp(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


class BinaryWriter:
    """Little endian binary writer. Automatically extends in size when needed"""

    __slots__ = ("buf", "pos")

    def __init__(self):
        self.buf = bytearray()
        self.pos = 0

    def tell(self) -> int:
        return self.pos

    def seek(self, pos: int):
        self.pos = pos

    @property
    def length(self) -> int:
        return len(self.pos)

    def get_bytes(self) -> bytes:
        return bytes(self.buf)

    def write_bytes(self, data: bytes | bytearray):
        end = self.pos + len(data)
        if end > len(self.buf):
            self.buf.extend(b"\x00" * (end - len(self.buf)))
        self.buf[self.pos:end] = data
        self.pos = end

    def _write(self, fmt: str, value):
        self.write_bytes(struct.pack(fmt, value))

    def write_u8(self, v: int):
        self._write("<B", v & 0xFF)

    def write_s8(self, v: int):
        self._write("<b", _clamp(v, -0x80, 0x7F))

    def write_u16(self, v: int):
        self._write("<H", v & 0xFFFF)

    def write_s16(self, v: int):
        self._write("<h", _clamp(v, -0x8000, 0x7FFF))

    def write_u32(self, v: int):
        self._write("<I", v & 0xFFFFFFFF)

    def write_s32(self, v: int):
        self._write("<i", _clamp(v, -0x80000000, 0x7FFFFFFF))

    def write_f32(self, v: float):
        self._write("<f", v)

    def write_fx16(self, v: float):
        self.write_s16(_clamp(int(round(v * FX16_SCALE)), -0x8000, 0x7FFF))

    def write_fx32(self, v: float):
        self.write_s32(
            _clamp(int(round(v * FX32_SCALE)), -0x80000000, 0x7FFFFFFF))

    def write_u16s(self, vs: list[int]):
        for v in vs:
            self.write_u16(v)

    def write_u32s(self, vs: list[int]):
        for v in vs:
            self.write_u32(v)

    def write_fx16s(self, vs: list[int]):
        for v in vs:
            self.write_fx16(v)

    def write_fx32s(self, vs: list[int]):
        for v in vs:
            self.write_fx32(v)

    def write_str(self, s: str, encoding="ascii"):
        self.write_bytes(s.encode(encoding=encoding))

    def align(self, n: int, fill: int = 0):
        while self.pos % n != 0:
            self.write_u8(fill)
