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


def bar_chart(groups: list[str], series: list[dict], title: str, ylabel: str = "", width: int = 760, height: int = 360,
              ymax: float | None = None, decimals: int = 3) -> str:
    """Grouped vertical bars: series = [{"name", "values": [per group], "color"?}]."""
    L, R, T, B = 56, 16, 40, 70
    top = ymax or max((v for s in series for v in s["values"]), default=1.0) * 1.12
    gw = (width - L - R) / max(len(groups), 1)
    bw = (gw - 8) / max(len(series), 1)
    sy = lambda v: height - B - v / top * (height - T - B)       # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold" fill="#1d2333">{title}</text>']
    for v, lab in _ticks(0, top, False):
        o.append(f'<line x1="{L}" x2="{width - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#e5e7ef"/>'
                 f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{lab}</text>')
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + 4
        for si, s in enumerate(series):
            v = s["values"][gi]
            c = s.get("color") or COLORS[si % len(COLORS)]
            o.append(f'<rect x="{x0 + si * bw:.1f}" y="{sy(v):.1f}" width="{max(bw - 2, 1):.1f}" height="{height - B - sy(v):.1f}" fill="{c}"/>')
            if len(series) <= 2:
                o.append(f'<text x="{x0 + si * bw + bw / 2:.1f}" y="{sy(v) - 3:.1f}" font-size="9" text-anchor="middle" fill="#1d2333">{v:.{decimals}f}</text>')
        o.append(f'<text x="{x0 + gw / 2 - 4:.1f}" y="{height - B + 15}" font-size="11" text-anchor="middle" fill="#1d2333">{g}</text>')
    o.append(f'<text transform="translate(16 {(T + height - B) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#6b7280">{ylabel}</text>')
    lx = L
    for si, s in enumerate(series):
        c = s.get("color") or COLORS[si % len(COLORS)]
        o.append(f'<rect x="{lx}" y="{height - 22}" width="14" height="14" fill="{c}"/><text x="{lx + 20}" y="{height - 11}" font-size="12" fill="#1d2333">{s["name"]}</text>')
        lx += 34 + len(s["name"]) * 6.6
    o.append("</svg>")
    return "".join(o)


def vsm_figures(engine) -> dict[str, str]:
    """Figures of the Project #3 report (file name -> SVG)."""
    from .tfidf import tf_weight
    info = engine.vsm_info()
    ev = engine.vsm_eval("", 10)
    out: dict[str, str] = {}
    tfs = list(range(1, 41))
    out["tfidf_tf_transforms.svg"] = line_plot(
        [{"name": "n  raw tf", "points": [(t, float(t)) for t in tfs]},
         {"name": "l  1 + ln tf", "points": [(t, tf_weight("l", t)) for t in tfs]},
         {"name": "a  0.5 + 0.5 tf / max tf (max tf = 10)", "points": [(t, tf_weight("a", t, 10)) for t in tfs]},
         {"name": "b  boolean", "points": [(t, 1.0) for t in tfs]}],
        "tf transforms (RQ1)", "tf", "tf weight", width=520, height=320)
    out["tfidf_idf_curve.svg"] = line_plot(
        [{"name": "t  idf = ln(N / df)", "points": [(c["df"], c["idf"]) for c in info["idf_curve"]]},
         {"name": "p  prob idf", "points": [(c["df"], c["prob_idf"]) for c in info["idf_curve"] if c["prob_idf"] > 0] or [(1, 0)]}],
        f"idf against df (N = {info['documents']}) (RQ2)", "df (log scale)", "idf", xlog=True, width=520, height=320)
    schemes = [s for s in ev["schemes"]]
    names = [s["scheme"] for s in schemes]
    out["tfidf_eval.svg"] = bar_chart(names, [{"name": "MAP", "values": [s["mean"]["MAP"] for s in schemes]},
                                              {"name": "nDCG@10", "values": [s["mean"]["nDCG@10"] for s in schemes]}],
                                      f"MAP and nDCG@10 per weighting scheme ({len(ev['queries'])} queries)", "mean over queries", width=760, height=330, ymax=0.8)
    out["tfidf_eval_p10.svg"] = bar_chart(names, [{"name": "P@10", "values": [s["mean"]["P@10"] for s in schemes]},
                                                  {"name": "MRR", "values": [s["mean"]["MRR"] for s in schemes]}],
                                          "P@10 and MRR per weighting scheme", "mean over queries", width=760, height=330, ymax=1.15)
    # RQ4: mean length of the top-10 documents per scheme, averaged over the queries
    vs = engine.vsm()
    qs = engine.vsm_queries()
    lens = []
    for s in names:
        tot = 0.0
        for q in qs.queries:
            top = vs.ranking_ids(q.text, s)[:10]
            tot += sum(vs.length[d] for d in top) / max(len(top), 1)
        lens.append(tot / len(qs.queries))
    out["tfidf_top10_length.svg"] = bar_chart(names, [{"name": "mean length of the top-10 documents", "values": lens}],
                                              f"Length of the retrieved documents (collection mean {vs.avg_length:.0f} index terms) (RQ4)",
                                              "index terms", width=760, height=300, decimals=0)
    return out
