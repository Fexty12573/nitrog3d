
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

    def read_key(self, n: int = 16) -> str:
        """Read a fixed-length, 0-padded string"""
        return self.read_str(n).replace("\x00", "")


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
        return len(self.buf)

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
        self._write("<b", clamp(v, -0x80, 0x7F))

    def write_u16(self, v: int):
        self._write("<H", v & 0xFFFF)

    def write_s16(self, v: int):
        self._write("<h", clamp(v, -0x8000, 0x7FFF))

    def write_u32(self, v: int):
        self._write("<I", v & 0xFFFFFFFF)

    def write_s32(self, v: int):
        self._write("<i", clamp(v, -0x80000000, 0x7FFFFFFF))

    def write_f32(self, v: float):
        self._write("<f", v)

    def write_fx16(self, v: float):
        self.write_s16(clamp(int(round(v * FX16_SCALE)), -0x8000, 0x7FFF))

    def write_fx32(self, v: float):
        self.write_s32(
            clamp(int(round(v * FX32_SCALE)), -0x80000000, 0x7FFFFFFF))

    def write_u8s(self, vs: list[int]):
        for v in vs:
            self.write_u8(v)

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

    def patch_u16(self, offset: int, v: int):
        pos = self.tell()
        self.seek(offset)
        self.write_u16(v)
        self.seek(pos)

    def patch_u32(self, offset: int, v: int):
        pos = self.tell()
        self.seek(offset)
        self.write_u32(v)
        self.seek(pos)

    def write_key(self, s: str, n: int = 16):
        raw = s.encode("ascii", errors="replace")[:n]
        self.write_bytes(raw + b"\x00" * (n - len(raw)))


def read_u32_le(buf: bytes | bytearray, offset: int) -> int:
    """Read a 32-bit unsigned integer from a buffer at the given offset"""
    return struct.unpack_from("<I", buf, offset)[0]


def read_u16_le(buf: bytes | bytearray, offset: int) -> int:
    """Read a 16-bit unsigned integer from a buffer at the given offset"""
    return struct.unpack_from("<H", buf, offset)[0]


def sign_extend(v: int, bits: int) -> int:
    mask = 1 << (bits - 1)
    return (v & (mask - 1)) - (v & mask)


def s16(v: int) -> int:
    return sign_extend(v & 0xFFFF, 16)


def s32(v: int) -> int:
    return sign_extend(v & 0xFFFFFFFF, 32)


def fx16(v: int) -> float:
    return s16(v) / FX16_SCALE


def fx32(v: int) -> float:
    return s32(v) / FX32_SCALE


def expand5(v: int) -> int:
    return (v * 255 + 15) // 31


def unpack3x10(v: int) -> tuple[int, int, int]:
    return (
        sign_extend(v & 0x3FF, 10),
        sign_extend((v >> 10) & 0x3FF, 10),
        sign_extend((v >> 20) & 0x3FF, 10)
    )


def pack3x10(x: int, y: int, z: int) -> int:
    return ((x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20))


def extract_bgr555(bgr: int) -> tuple[int, int, int]:
    b = expand5((bgr >> 10) & 0x1F)
    g = expand5((bgr >> 5) & 0x1F)
    r = expand5(bgr & 0x1F)
    return (r, g, b)


def bgr555_to_float(bgr: int) -> tuple[float, float, float]:
    (r, g, b) = extract_bgr555(bgr)
    return (r / 255.0, g / 255.0, b / 255.0)


def float_to_bgr555(r: float, g: float, b: float) -> int:
    r5 = clamp(int(round(r * 31.0)), 0, 31)
    g5 = clamp(int(round(g * 31.0)), 0, 31)
    b5 = clamp(int(round(b * 31.0)), 0, 31)
    return (b5 << 10) | (g5 << 5) | r5


def clamp(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def to_fx(v: float) -> int:
    x = v * FX32_SCALE
    return int(x + 0.5) if x >= 0 else int(x - 0.5)
