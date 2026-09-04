from .tex0 import TexFmt
from .binary import expand5, quantize5, pack_bgr555, extract_bgr555

import struct
from dataclasses import dataclass
from typing import Literal


_MAX_COLORS = {
    TexFmt.PLTT4: 4,
    TexFmt.PLTT16: 16,
    TexFmt.PLTT256: 256,
    TexFmt.A3I5: 32,
    TexFmt.A5I3: 8,
}


@dataclass
class TextureAnalysisResult:
    ncolors555: int
    ncolors888: int
    nalphas: int
    alpha_kind: Literal["opaque", "binary", "graded"]
    suggested: TexFmt


def rgba8_to_abgr1555(r: int, g: int, b: int, a: int) -> int:
    a_bit = 0x8000 if a >= 128 else 0
    return a_bit | pack_bgr555(r, g, b)


def quantize_alpha3(a8: int) -> int:
    return (a8 * 7 + 127) // 255


def quantize_alpha5(a8: int) -> int:
    return quantize5(a8)


def expand_alpha3(a3: int) -> int:
    return (a3 * 255 + 3) // 7


def expand_alpha5(a5: int) -> int:
    return expand5(a5)


PALETTE_ALIGN = 16
PALETTE_FILL = 0x7FFF


def palette_to_bytes(pal: list[int], fmt: TexFmt) -> bytes:
    colors = [c & 0xFFFF for c in pal]
    n = len(colors)

    nbytes = max(
        ((n * 2 + PALETTE_ALIGN - 1) // PALETTE_ALIGN) * PALETTE_ALIGN, PALETTE_ALIGN
    )
    alloc = nbytes // 2
    cap = _MAX_COLORS.get(fmt, alloc)

    colors += [PALETTE_FILL] * max(0, min(alloc, cap) - n)
    colors += [0x0000] * (alloc - len(colors))
    return struct.pack(f"<{len(colors)}H", *colors)


def palette_rgb8(palette: list[int]) -> list[tuple[int, int, int]]:
    return [extract_bgr555(bgr) for bgr in palette]


def encode_rgba(
    rgba: list[int],
    w: int,
    h: int,
    fmt: TexFmt,
    *,
    palette: list[int] | None = None,
    color0_transparent: bool = False,
) -> tuple[bytes, list[int] | None]:
    n = w * h
    if fmt == TexFmt.DIRECT:
        return _pack_direct(rgba, n), None

    if fmt == TexFmt.COMP4X4:
        raise NotImplementedError("COMP4x4 not yet supported")

    reserve0 = color0_transparent and fmt in (
        TexFmt.PLTT4,
        TexFmt.PLTT16,
        TexFmt.PLTT256,
    )

    if palette is not None:
        pal = palette
        idx = _map_indices(rgba, n, pal, reserve0)
    else:
        pal, idx, exact = build_palette(rgba, n, _MAX_COLORS[fmt], reserve0)

    if fmt == TexFmt.PLTT4:
        data = _pack_pltt4(idx, n)
    elif fmt == TexFmt.PLTT16:
        data = _pack_pltt16(idx, n)
    elif fmt == TexFmt.PLTT256:
        data = _pack_pltt256(idx, n)
    elif fmt == TexFmt.A3I5:
        alphas = [quantize_alpha3(rgba[i * 4 + 3]) for i in range(n)]
        data = _pack_a3i5(idx, alphas)
    elif fmt == TexFmt.A5I3:
        alphas = [quantize_alpha5(rgba[i * 4 + 3]) for i in range(n)]
        data = _pack_a5i3(idx, alphas)
    else:
        raise ValueError(f"Unsupported texture format: {fmt}")

    return data, pal


def quantize_rgba(
    rgba: list[int],
    w: int,
    h: int,
    fmt: TexFmt,
    *,
    palette: list[int] | None = None,
    color0_transparent: bool = False,
) -> list[int]:
    n = w * h
    if fmt == TexFmt.DIRECT:
        out: list[int] = []
        for i in range(n):
            base = i * 4
            r, g, b = extract_bgr555(pack_bgr555(*rgba[base : base + 3]))
            out.extend((r, g, b, 255 if rgba[base + 3] >= 128 else 0))
        return out

    if fmt == TexFmt.COMP4X4:
        raise NotImplementedError("COMP4x4 not yet supported")

    reserve0 = color0_transparent and fmt in (
        TexFmt.PLTT4,
        TexFmt.PLTT16,
        TexFmt.PLTT256,
    )

    if palette is not None:
        pal = palette
        idx = _map_indices(rgba, n, pal, reserve0)
    else:
        pal, idx, exact = build_palette(rgba, n, _MAX_COLORS[fmt], reserve0)

    if fmt in (TexFmt.PLTT4, TexFmt.PLTT16, TexFmt.PLTT256):
        alphas = [0 if reserve0 and idx[i] == 0 else 255 for i in range(n)]
    elif fmt == TexFmt.A3I5:
        alphas = [expand_alpha3(quantize_alpha3(rgba[i * 4 + 3])) for i in range(n)]
    elif fmt == TexFmt.A5I3:
        alphas = [expand_alpha5(quantize_alpha5(rgba[i * 4 + 3])) for i in range(n)]
    else:
        raise ValueError(f"Unsupported texture format: {fmt}")

    pal8 = palette_rgb8(pal)
    out = []
    for i in range(n):
        r, g, b = pal8[idx[i]]
        out.extend((r, g, b, alphas[i]))
    return out


def analyze_texture(rgba: list[int], w: int, h: int) -> TextureAnalysisResult:
    n = w * h
    colors555, alphas = _stats(rgba, n)
    colors888: set[tuple[int, int, int]] = set()
    for i in range(n):
        base = i * 4
        colors888.add(tuple(rgba[base : base + 3]))
    opaque = alphas <= {255}
    binary = alphas <= {0, 255}
    return TextureAnalysisResult(
        ncolors555=len(colors555),
        ncolors888=len(colors888),
        nalphas=len(alphas),
        alpha_kind="opaque" if opaque else ("binary" if binary else "graded"),
        suggested=_choose_format(colors555, alphas),
    )


def choose_format(rgba: list[int], w: int, h: int) -> TexFmt:
    colors, alphas = _stats(rgba, w * h)
    return _choose_format(colors, alphas)


def build_palette(
    rgba: list[int], n: int, max_colors: int, reserve0: bool
) -> tuple[list[int], list[int], bool]:
    color_cap = max_colors - (1 if reserve0 else 0)

    order: list[int] = []
    seen: set[int] = set()
    counts: dict[int, int] = {}
    rgb8_of: dict[int, tuple[int, int, int]] = {}

    for i in range(n):
        base = i * 4

        # Transparent
        if reserve0 and rgba[base + 3] < 128:
            continue

        r, g, b = rgba[base + 0], rgba[base + 1], rgba[base + 2]
        c555 = pack_bgr555(r, g, b)
        if c555 not in seen:
            seen.add(c555)
            order.append(c555)
            rgb8_of[c555] = (r, g, b)
        counts[c555] = counts.get(c555, 0) + 1

    if len(order) <= color_cap:
        palette = order
        exact = True
    else:
        reps = _median_cut([(rgb8_of[c], counts[c]) for c in order], color_cap)
        palette: list[int] = []
        dedupe: set[int] = set()
        for rep in reps:
            c555 = pack_bgr555(*rep)
            if c555 not in dedupe:
                dedupe.add(c555)
                palette.append(c555)
        exact = False

    if reserve0:
        palette = [0] + palette

    idx = _map_indices(rgba, n, palette, reserve0)
    return palette, idx, exact


def _stats(rgba: list[int], n: int) -> tuple[set[int], set[int]]:
    colors: set[int] = set()
    alphas: set[int] = set()
    for i in range(n):
        base = i * 4
        colors.add(pack_bgr555(*rgba[base : base + 3]))
        alphas.add(rgba[base + 3])
    return colors, alphas


def _fits(fmt: TexFmt, ncolors: int, alphas: set[int]) -> bool:
    opaque = alphas <= {255}
    binary = alphas <= {0, 255}
    match fmt:
        case TexFmt.DIRECT:
            return opaque or binary
        case TexFmt.PLTT4:
            return (opaque or binary) and ncolors <= (4 if opaque else 3)
        case TexFmt.PLTT16:
            return (opaque or binary) and ncolors <= (16 if opaque else 15)
        case TexFmt.PLTT256:
            return (opaque or binary) and ncolors <= (256 if opaque else 255)
        case TexFmt.A3I5:
            return ncolors <= 32
        case TexFmt.A5I3:
            return ncolors <= 8
    return False


def _choose_format(
    colors: set[int], alphas: set[int], preference: TexFmt | None = None
) -> TexFmt:
    ncolors = len(colors)
    if (
        preference is not None
        and preference != TexFmt.COMP4X4
        and _fits(preference, ncolors, alphas)
    ):
        return preference

    opaque = alphas <= {255}
    binary = alphas <= {0, 255}
    if opaque:
        if ncolors <= 4:
            return TexFmt.PLTT4
        if ncolors <= 16:
            return TexFmt.PLTT16
        if ncolors <= 256:
            return TexFmt.PLTT256
        return TexFmt.DIRECT
    if binary:
        if ncolors <= 3:
            return TexFmt.PLTT4
        if ncolors <= 15:
            return TexFmt.PLTT16
        if ncolors <= 255:
            return TexFmt.PLTT256
        return TexFmt.DIRECT

    nalphas = len(alphas)
    if nalphas <= 8 and ncolors <= 32:
        return TexFmt.A3I5
    if ncolors <= 8:
        return TexFmt.A5I3

    return TexFmt.A3I5


def _median_cut(
    color_counts: list[tuple[tuple[int, int, int], int]], k: int
) -> list[tuple[int, int, int]]:
    if not color_counts or k <= 0:
        return []

    boxes = [color_counts]
    while len(boxes) < k:
        cand: list[tuple[tuple[int, int, int], int]] | None = None
        cand_key: tuple[int, int] | None = None
        for box in boxes:
            if len(box) < 2:
                continue

            ch = _widest_channel(box)
            lo = min(col[ch] for col, _ in box)
            hi = max(col[ch] for col, _ in box)
            key = (hi - lo, sum(count for _, count in box))
            if cand_key is None or key > cand_key:
                cand = box
                cand_key = key

        if cand is None:
            break

        boxes.remove(cand)
        ch = _widest_channel(cand)
        s = sorted(cand, key=lambda cc: (cc[0][ch], cc[0], cc[1]))
        mid = len(s) // 2
        boxes.append(s[:mid])
        boxes.append(s[mid:])

    return [_box_average(b) for b in boxes if b]


def _map_indices(
    rgba: list[int], n: int, palette: list[int], reserve0: bool
) -> list[int]:
    pal8 = palette_rgb8(palette)
    start = 1 if reserve0 else 0
    cache: dict[tuple[int, int, int], int] = {}
    out: list[int] = []

    for i in range(n):
        base = i * 4
        a = rgba[base + 3]
        if reserve0 and a < 128:
            out.append(0)
            continue

        key = (rgba[base + 0], rgba[base + 1], rgba[base + 2])
        j = cache.get(key)
        if j is None:
            r, g, b = key
            best = start
            best_d: int | None = None

            for k in range(start, len(pal8)):
                pr, pg, pb = pal8[k]
                d = (pr - r) ** 2 + (pg - g) ** 2 + (pb - b) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    best = k
                    if d == 0:
                        break

            cache[key] = best
            j = best
        out.append(j)

    return out


def _widest_channel(box: list[tuple[tuple[int, int, int], int]]) -> int:
    lo = [255, 255, 255]
    hi = [0, 0, 0]
    for (r, g, b), _ in box:
        for ch, v in enumerate((r, g, b)):
            if v < lo[ch]:
                lo[ch] = v
            if v > hi[ch]:
                hi[ch] = v
    ranges = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    return ranges.index(max(ranges))


def _box_average(box: list[tuple[tuple[int, int, int], int]]) -> tuple[int, int, int]:
    tr = tg = tb = tc = 0
    for (r, g, b), c in box:
        tr += r * c
        tg += g * c
        tb += b * c
        tc += c

    return (round(tr / tc), round(tg / tc), round(tb / tc))


def _pack_pltt4(idx: list[int], n: int) -> bytes:
    out = bytearray((n + 3) // 4)
    for i in range(n):
        out[i >> 2] |= (idx[i] & 3) << ((i & 3) * 2)
    return bytes(out)


def _pack_pltt16(idx: list[int], n: int) -> bytes:
    out = bytearray((n + 1) // 2)
    for i in range(n):
        if i & 1:
            out[i >> 1] |= (idx[i] & 0xF) << 4
        else:
            out[i >> 1] |= (idx[i] & 0xF) << 0
    return bytes(out)


def _pack_pltt256(idx: list[int], n: int) -> bytes:
    return bytes(idx[i] & 0xFF for i in range(n))


def _pack_a3i5(idx: list[int], alphas: list[int]) -> bytes:
    return bytes(((alphas[i] & 7) << 5) | (idx[i] & 0x1F) for i in range(len(idx)))


def _pack_a5i3(idx: list[int], alphas: list[int]) -> bytes:
    return bytes(((alphas[i] & 0x1F) << 3) | (idx[i] & 7) for i in range(len(idx)))


def _pack_direct(rgba: list[int], n: int) -> bytes:
    out = bytearray(n * 2)
    for i in range(n):
        base = i * 4
        c = rgba8_to_abgr1555(
            rgba[base + 0],
            rgba[base + 1],
            rgba[base + 2],
            rgba[base + 3],
        )
        out[i * 2 + 0] = (c & 0x00FF) >> 0
        out[i * 2 + 1] = (c & 0xFF00) >> 8
    return bytes(out)
