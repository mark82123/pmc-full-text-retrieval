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
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-9:
        out.append((v, f"{v:,.0f}"))
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
    L, R, T, B = 70, 20, 40, 76
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
    lx = L
    for i, s in enumerate(series):
        color = s.get("color") or COLORS[i % len(COLORS)]
        d = " ".join(f"{'L' if j else 'M'}{sx(x):.1f} {sy(y):.1f}" for j, (x, y) in enumerate(s["points"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} stroke-linejoin="round"/>')
        o.append(f'<line x1="{lx}" x2="{lx + 20}" y1="{height - 14}" y2="{height - 14}" stroke="{color}" stroke-width="3"{dash}/>'
                 f'<text x="{lx + 26}" y="{height - 10}" font-size="12" fill="#1d2333">{s["name"]}</text>')
        lx += 40 + len(s["name"]) * 6.6
    o.append("</svg>")
    return "".join(o)
