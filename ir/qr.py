"""Minimal QR code encoder (byte mode, versions 1-10, ISO/IEC 18004) - standard library only.

    >>> from ir.qr import qr_matrix, qr_svg
    >>> m = qr_matrix("http://192.168.1.10:8090/")      # list of rows, True = dark module
    >>> svg = qr_svg("http://192.168.1.10:8090/")       # <svg> string with a 4-module quiet zone

Used by the web UI to show a scannable link to the classroom (/api/qr.svg?text=...).
Error-correction level M by default; the mask with the lowest penalty score is chosen.
"""

from __future__ import annotations

# --- GF(256) arithmetic with the QR primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D) ---------------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_remainder(data: list[int], n: int) -> list[int]:
    # polynomial long division; generator coefficients highest degree first
    gen = [1]
    for i in range(n):
        nxt = [0] * (len(gen) + 1)
        for j, c in enumerate(gen):
            nxt[j] ^= c
            nxt[j + 1] ^= _gf_mul(c, _EXP[i])
        gen = nxt
    # gen has n + 1 coefficients, highest first, leading 1
    res = [0] * n
    for b in data:
        factor = b ^ res[0]
        res = res[1:] + [0]
        if factor:
            for j in range(n):
                res[j] ^= _gf_mul(gen[j + 1], factor)
    return res


# --- tables (versions 1-10) -------------------------------------------------------------------------------
# (ec codewords per block, [(number of blocks, data codewords per block), ...])
_EC_TABLE = {
    "L": {1: (7, [(1, 19)]), 2: (10, [(1, 34)]), 3: (15, [(1, 55)]), 4: (20, [(1, 80)]), 5: (26, [(1, 108)]),
          6: (18, [(2, 68)]), 7: (20, [(2, 78)]), 8: (24, [(2, 97)]), 9: (30, [(2, 116)]), 10: (18, [(2, 68), (2, 69)])},
    "M": {1: (10, [(1, 16)]), 2: (16, [(1, 28)]), 3: (26, [(1, 44)]), 4: (18, [(2, 32)]), 5: (24, [(2, 43)]),
          6: (16, [(4, 27)]), 7: (18, [(4, 31)]), 8: (22, [(2, 38), (2, 39)]), 9: (22, [(3, 36), (2, 37)]),
          10: (26, [(4, 43), (1, 44)])},
}
_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42],
          9: [6, 26, 46], 10: [6, 28, 50]}
_ECL_BITS = {"L": 1, "M": 0, "Q": 3, "H": 2}


def _data_capacity(version: int, ecl: str) -> int:
    return sum(n * k for n, k in _EC_TABLE[ecl][version][1])


class _Bits:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def append(self, value: int, length: int) -> None:
        self.bits.extend((value >> (length - 1 - i)) & 1 for i in range(length))


def _encode_codewords(data: bytes, ecl: str) -> tuple[int, list[int]]:
    """Choose the smallest version that fits and return (version, final interleaved codewords)."""
    version = None
    for v in range(1, 11):
        count_bits = 8 if v <= 9 else 16
        needed = 4 + count_bits + 8 * len(data)
        if needed <= _data_capacity(v, ecl) * 8:
            version = v
            break
    if version is None:
        raise ValueError("text too long for a version-10 QR code (%d bytes)" % len(data))
    cap = _data_capacity(version, ecl)
    bb = _Bits()
    bb.append(0b0100, 4)                                   # byte mode
    bb.append(len(data), 8 if version <= 9 else 16)
    for b in data:
        bb.append(b, 8)
    bb.append(0, min(4, cap * 8 - len(bb.bits)))           # terminator
    while len(bb.bits) % 8:
        bb.bits.append(0)
    pad = (0xEC, 0x11)
    i = 0
    while len(bb.bits) < cap * 8:
        bb.append(pad[i % 2], 8)
        i += 1
    codewords = [int("".join(map(str, bb.bits[i:i + 8])), 2) for i in range(0, len(bb.bits), 8)]

    ec_len, groups = _EC_TABLE[ecl][version]
    blocks: list[list[int]] = []
    pos = 0
    for n, k in groups:
        for _ in range(n):
            blocks.append(codewords[pos:pos + k])
            pos += k
    ec_blocks = [_rs_remainder(b, ec_len) for b in blocks]
    out: list[int] = []
    longest = max(len(b) for b in blocks)
    for i in range(longest):
        for b in blocks:
            if i < len(b):
                out.append(b[i])
    for i in range(ec_len):
        for e in ec_blocks:
            out.append(e[i])
    return version, out


# --- matrix ------------------------------------------------------------------------------------------------
def _make_matrix(version: int, ecl: str, codewords: list[int], mask: int) -> list[list[bool]]:
    return _build(version, ecl, codewords, mask)[0]


def _build(version: int, ecl: str, codewords: list[int], mask: int) -> tuple[list[list[bool]], list[list[bool]]]:
    size = version * 4 + 17
    mod = [[False] * size for _ in range(size)]
    fun = [[False] * size for _ in range(size)]

    def setf(x: int, y: int, dark: bool) -> None:   # x = column, y = row
        mod[y][x] = dark
        fun[y][x] = True

    # timing patterns
    for i in range(size):
        setf(6, i, i % 2 == 0)
        setf(i, 6, i % 2 == 0)
    # finder patterns + separators
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    d = max(abs(dx), abs(dy))
                    setf(x, y, d != 2 and d != 4)
    # alignment patterns
    al = _ALIGN[version]
    for i, ay in enumerate(al):
        for j, ax in enumerate(al):
            if (i == 0 and j == 0) or (i == 0 and j == len(al) - 1) or (i == len(al) - 1 and j == 0):
                continue
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    setf(ax + dx, ay + dy, max(abs(dx), abs(dy)) != 1)
    # format information
    _draw_format(mod, setf, size, ecl, mask)
    # version information
    if version >= 7:
        rem = version
        for _ in range(12):
            rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
        bits = version << 12 | rem
        for i in range(18):
            bit = bool((bits >> i) & 1)
            a, b = size - 11 + i % 3, i // 3
            setf(a, b, bit)
            setf(b, a, bit)
    # data placement (zigzag)
    i = 0
    total_bits = len(codewords) * 8
    right = size - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(size):
            for j in range(2):
                x = right - j
                upward = ((right + 1) & 2) == 0
                y = size - 1 - vert if upward else vert
                if not fun[y][x] and i < total_bits:
                    mod[y][x] = bool((codewords[i >> 3] >> (7 - (i & 7))) & 1)
                    i += 1
        right -= 2
    # mask
    for y in range(size):
        for x in range(size):
            if fun[y][x]:
                continue
            if mask == 0:
                inv = (x + y) % 2 == 0
            elif mask == 1:
                inv = y % 2 == 0
            elif mask == 2:
                inv = x % 3 == 0
            elif mask == 3:
                inv = (x + y) % 3 == 0
            elif mask == 4:
                inv = (x // 3 + y // 2) % 2 == 0
            elif mask == 5:
                inv = x * y % 2 + x * y % 3 == 0
            elif mask == 6:
                inv = (x * y % 2 + x * y % 3) % 2 == 0
            else:
                inv = ((x + y) % 2 + x * y % 3) % 2 == 0
            if inv:
                mod[y][x] = not mod[y][x]
    return mod, fun


def _draw_format(mod, setf, size: int, ecl: str, mask: int) -> None:
    data = _ECL_BITS[ecl] << 3 | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    bits = (data << 10 | rem) ^ 0x5412
    bit = lambda i: bool((bits >> i) & 1)  # noqa: E731
    for i in range(6):
        setf(8, i, bit(i))
    setf(8, 7, bit(6))
    setf(8, 8, bit(7))
    setf(7, 8, bit(8))
    for i in range(9, 15):
        setf(14 - i, 8, bit(i))
    for i in range(8):
        setf(size - 1 - i, 8, bit(i))
    for i in range(8, 15):
        setf(8, size - 15 + i, bit(i))
    setf(8, size - 8, True)                                # the dark module


def _penalty(mod: list[list[bool]]) -> int:
    size = len(mod)
    n1, n2, n3, n4 = 3, 3, 40, 10
    score = 0

    def line_score(line: list[bool]) -> int:
        s = 0
        # runs of the same colour
        run = 1
        for i in range(1, size):
            if line[i] == line[i - 1]:
                run += 1
                if run == 5:
                    s += n1
                elif run > 5:
                    s += 1
            else:
                run = 1
        # finder-like patterns 1011101 preceded or followed by 4 light modules (the border counts as light)
        pat = [True, False, True, True, True, False, True]
        i = 0
        while i <= size - 7:
            if line[i:i + 7] != pat:
                i += 1
                continue
            if i == 0 or i == size - 7 or not any(line[max(i - 4, 0):i]) or not any(line[i + 7:i + 11]):
                s += n3
                i += 7
            else:
                i += 4
        return s

    for y in range(size):
        score += line_score(mod[y])
    for x in range(size):
        score += line_score([mod[y][x] for y in range(size)])
    for y in range(size - 1):
        for x in range(size - 1):
            c = mod[y][x]
            if c == mod[y][x + 1] == mod[y + 1][x] == mod[y + 1][x + 1]:
                score += n2
    dark = sum(sum(row) for row in mod)
    total = size * size
    score += n4 * int(abs(dark * 100 / total - 50) / 5)
    return score


def qr_matrix(text: str, ecl: str = "M", mask: int | None = None) -> list[list[bool]]:
    """Return the module matrix (rows of booleans, True = dark) for *text* (UTF-8, byte mode)."""
    version, codewords = _encode_codewords(text.encode("utf-8"), ecl)
    if mask is not None:
        return _make_matrix(version, ecl, codewords, mask)
    best, best_score = None, None
    for m in range(8):
        mat = _make_matrix(version, ecl, codewords, m)
        s = _penalty(mat)
        if best_score is None or s < best_score:
            best, best_score = mat, s
    return best  # type: ignore[return-value]


def qr_svg(text: str, ecl: str = "M", module: int = 8, dark: str = "#000", light: str = "#fff") -> str:
    """Scalable SVG of the QR code with the standard 4-module quiet zone."""
    mat = qr_matrix(text, ecl)
    n = len(mat)
    q = 4
    side = (n + 2 * q) * module
    path = []
    for y, row in enumerate(mat):
        x = 0
        while x < n:
            if row[x]:
                x0 = x
                while x < n and row[x]:
                    x += 1
                path.append("M%d %dh%dv%dh-%dz" % ((x0 + q) * module, (y + q) * module, (x - x0) * module, module, (x - x0) * module))
            else:
                x += 1
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" shape-rendering="crispEdges">'
            '<rect width="100%%" height="100%%" fill="%s"/><path d="%s" fill="%s"/></svg>'
            % (side, side, light, "".join(path), dark))
