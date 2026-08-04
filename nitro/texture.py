
from .binary import read_u16_le, extract_bgr555
from .tex0 import TexFmt, TexDictData


def decode_texture(tex_entry: TexDictData, pal_bytes: bytes) -> tuple[int, int, bytes]:
    w, h = tex_entry.s, tex_entry.t
    first_transparent = tex_entry.transparent_color
    pal = decode_palette(pal_bytes) if pal_bytes else []

    match tex_entry.fmt:
        case TexFmt.PLTT4:
            pixels = _decode_pltt4(tex_entry.data, pal, first_transparent)
        case TexFmt.PLTT16:
            pixels = _decode_pltt16(tex_entry.data, pal, first_transparent)
        case TexFmt.PLTT256:
            pixels = _decode_pltt256(tex_entry.data, pal, first_transparent)
        case TexFmt.A3I5:
            pixels = _decode_a3i5(tex_entry.data, pal)
        case TexFmt.A5I3:
            pixels = _decode_a5i3(tex_entry.data, pal)
        case TexFmt.DIRECT:
            pixels = _decode_direct(tex_entry.data)
        case TexFmt.COMP4X4:
            pixels = _decode_comp4x4(tex_entry.data, tex_entry.data4x4, pal)
        case _:
            pixels = [0] * w * h

    if len(pixels) < w * h:
        pixels = pixels + [0] * (w * h - len(pixels))
    elif len(pixels) > w * h:
        pixels = pixels[:w * h]

    return w, h, _argb_list_to_rgba_bytes(pixels)


def decode_palette(pal_bytes: bytes) -> list[int]:
    n = len(pal_bytes) // 2
    return [bgr555_to_argb8888(read_u16_le(pal_bytes, i * 2)) for i in range(n)]


def bgr555_to_argb8888(v: int) -> int:
    (r, g, b) = extract_bgr555(v)
    return (0xFF << 24) | (r << 16) | (g << 8) | b


def abgr1555_to_argb8888(v: int) -> int:
    (r, g, b) = extract_bgr555(v)
    a = 0xFF if v & 0x8000 else 0
    return (a << 24) | (r << 16) | (g << 8) | b


def _set_argb8888_alpha(argb: int, a8: int) -> int:
    return (argb & 0x00FFFFFF) | ((a8 & 0xFF) << 24)


def _pal_get(pal: list[int], idx: int) -> int:
    return pal[idx] if idx < len(pal) else 0


def _decode_pltt4(data: bytes, pal: list[int], first_transparent: bool) -> list[int]:
    def get(c): return 0 if c == 0 and first_transparent else _pal_get(pal, c)

    out: list[int] = []
    for byte in data:
        out.append(get((byte >> 0) & 0x3))
        out.append(get((byte >> 2) & 0x3))
        out.append(get((byte >> 4) & 0x3))
        out.append(get((byte >> 6) & 0x3))
    return out


def _decode_pltt16(data: bytes, pal: list[int], first_transparent: bool) -> list[int]:
    def get(c): return 0 if c == 0 and first_transparent else _pal_get(pal, c)

    out: list[int] = []
    for byte in data:
        out.append(get((byte >> 0) & 0xF))
        out.append(get((byte >> 4) & 0xF))
    return out


def _decode_pltt256(data: bytes, pal: list[int], first_transparent: bool) -> list[int]:
    def get(c): return 0 if c == 0 and first_transparent else _pal_get(pal, c)

    out: list[int] = []
    for byte in data:
        out.append(get(byte))
    return out


def _decode_a3i5(data: bytes, pal: list[int]) -> list[int]:
    out: list[int] = []
    for byte in data:
        a3 = byte >> 5
        i5 = byte & 0x1F
        a8 = (((a3 << 2) + (a3 >> 1)) * 255 + 15) // 31
        out.append(_set_argb8888_alpha(_pal_get(pal, i5), a8))
    return out


def _decode_a5i3(data: bytes, pal: list[int]) -> list[int]:
    out: list[int] = []
    for byte in data:
        a5 = byte & 0x1F
        i3 = byte >> 5
        a8 = (a5 * 255 + 15) // 31
        out.append(_set_argb8888_alpha(_pal_get(pal, i3), a8))
    return out


def _decode_direct(data: bytes) -> list[int]:
    out: list[int] = []
    for i in range(len(data) // 2):
        out.append(abgr1555_to_argb8888(read_u16_le(data, i * 2)))
    return out


def _decode_comp4x4(data: bytes, data4x4: bytes, pal: list[int]) -> list[int]:
    raise NotImplementedError()


def _argb_list_to_rgba_bytes(pixels: list[int]) -> bytes:
    out = bytearray(len(pixels) * 4)
    j = 0
    for p in pixels:
        out[j + 0] = (p >> 16) & 0xFF  # R
        out[j + 1] = (p >> 8) & 0xFF   # G
        out[j + 2] = (p >> 0) & 0xFF   # B
        out[j + 3] = (p >> 24) & 0xFF  # A
        j += 4
    return bytes(out)
