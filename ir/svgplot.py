"""Minimal static SVG line plots (standard library only) for the written
report: `python3 cli.py zipf --svg report/` saves the Zipf figures."""

from __future__ import annotations

import math

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _ticks(lo: float, hi: float, log: bool) -> list[tuple[float, str]]:
    if log:
        return [(e, f"{10 ** e:,.0f}") for e in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1)]
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = (5 if raw / mag >= 5 else 2 if raw / mag >= 2 else 1) * mag
    decimals = max(0, -math.floor(math.log10(step)))        # 0.5 -> 1 decimal, 0.05 -> 2
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-9:
        out.append((v, f"{v:,.{decimals}f}"))
        v += step
    return out


def line_plot(series: list[dict], title: str, xlabel: str, ylabel: str, xlog: bool = False, ylog: bool = False,
              vlines: list[tuple[float, str]] = (), width: int = 760, height: int = 440) -> str:
    """series: [{"name", "points": [(x, y)], "color"?, "dash"?}] -> SVG text."""
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    xs = [tx(x) for s in series for x, _ in s["points"]]
    ys = [ty(y) for s in series for _, y in s["points"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x1 += (x1 - x0) * 0.03
    y1 += (y1 - y0) * 0.05
    if not xlog:
        x0 = min(0, x0)
    if not ylog:
        y0 = min(0, y0)
    L, R, T = 70, 20, 40
    rows, used = 1, 0.0                                     # legend rows needed
    for s_ in series:
        w = 46 + len(s_["name"]) * 6.6
        if used + w > width - L - R and used:
            rows += 1
            used = 0.0
        used += w
    B = 60 + 18 * rows
    sx = lambda v: L + (tx(v) - x0) / (x1 - x0) * (width - L - R)      # noqa: E731
    sy = lambda v: height - B - (ty(v) - y0) / (y1 - y0) * (height - T - B)  # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold" fill="#1d2333">{title}</text>']
    for v, lab in _ticks(y0, y1, ylog):
        y = height - B - (v - y0) / (y1 - y0) * (height - T - B)
        o.append(f'<line x1="{L}" x2="{width - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7ef"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{lab}</text>')
    for v, lab in _ticks(x0, x1, xlog):
        x = L + (v - x0) / (x1 - x0) * (width - L - R)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{height - B}" stroke="#e5e7ef"/>'
                 f'<text x="{x:.1f}" y="{height - B + 16}" font-size="11" text-anchor="middle" fill="#6b7280">{lab}</text>')
    for xv, lab in vlines:
        o.append(f'<line x1="{sx(xv):.1f}" x2="{sx(xv):.1f}" y1="{T}" y2="{height - B}" stroke="#6b7280" stroke-dasharray="3 4"/>'
                 f'<text x="{sx(xv) + 4:.1f}" y="{T + 12}" font-size="11" fill="#6b7280">{lab}</text>')
    o.append(f'<text x="{(L + width - R) / 2}" y="{height - B + 36}" font-size="12" text-anchor="middle" fill="#6b7280">{xlabel}</text>'
             f'<text transform="translate(16 {(T + height - B) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#6b7280">{ylabel}</text>')
    lx, ly = L, height - 14 - 18 * (rows - 1)
    for i, s in enumerate(series):
        color = s.get("color") or COLORS[i % len(COLORS)]
        d = " ".join(f"{'L' if j else 'M'}{sx(x):.1f} {sy(y):.1f}" for j, (x, y) in enumerate(s["points"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} stroke-linejoin="round"/>')
        w = 46 + len(s["name"]) * 6.6
        if lx + w > width - R and lx > L:
            lx, ly = L, ly + 18
        o.append(f'<line x1="{lx}" x2="{lx + 20}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"{dash}/>'
                 f'<text x="{lx + 26}" y="{ly + 4}" font-size="12" fill="#1d2333">{s["name"]}</text>')
        lx += w
    o.append("</svg>")
    return "".join(o)


def scatter_plot(points: list[dict], title: str, width: int = 760, height: int = 560) -> str:
    """Labelled 2-D scatter for the embedding maps: points = [{word, x, y, group?, seed?}]."""
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x1 - x0 or 1) * 0.06, (y1 - y0 or 1) * 0.06
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    L, R, T, B = 20, 20, 40, 20
    sx = lambda v: L + (v - x0) / (x1 - x0) * (width - L - R)       # noqa: E731
    sy = lambda v: height - B - (v - y0) / (y1 - y0) * (height - T - B)  # noqa: E731
    groups = sorted({p.get("group", "") for p in points})
    color = {g: COLORS[i % len(COLORS)] for i, g in enumerate(groups)}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold" fill="#1d2333">{title}</text>',
         f'<rect x="{L}" y="{T}" width="{width - L - R}" height="{height - T - B}" fill="none" stroke="#e5e7ef"/>']
    boxes: list[tuple[float, float, float, float]] = []
    for p in sorted(points, key=lambda q: not q.get("seed")):       # seed words get their labels first
        c = color[p.get("group", "")]
        r = 5 if p.get("seed") else 3.5
        o.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="{r}" fill="{c}" fill-opacity="0.85"/>')
        x, y, w = sx(p["x"]) + 6, sy(p["y"]) + 4, len(p["word"]) * 5.6
        box = (x, y - 9, x + w, y + 2)
        if any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes):
            continue                                                 # label would overlap one already drawn
        boxes.append(box)
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" {"font-weight=\"bold\"" if p.get("seed") else ""} fill="#1d2333">{p["word"]}</text>')
    lx = L
    for g in groups:
        if not g:
            continue
        o.append(f'<circle cx="{lx + 6}" cy="{height - 8}" r="5" fill="{color[g]}"/><text x="{lx + 16}" y="{height - 4}" font-size="12" fill="#1d2333">{g}</text>')
        lx += 30 + len(g) * 7
    o.append("</svg>")
    return "".join(o)
