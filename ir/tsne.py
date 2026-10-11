"""Dimensionality reduction for the embedding maps (standard library only).

    pca2(rows)                     principal components by power iteration
    tsne(rows, perplexity, iters)  t-SNE (van der Maaten & Hinton 2008), exact
                                   O(n^2) gradient, early exaggeration, momentum
                                   and per-coordinate adaptive gains as in the
                                   reference implementation
    knn_preservation(rows, pts, k) how many of each point's k nearest neighbours
                                   in the original space are still among its k
                                   nearest neighbours in the 2-D map

PCA is a *linear* projection: it keeps the two directions of largest variance,
so global structure (what is far from what) is faithful but small clusters
are often squashed on top of each other.  t-SNE is *non-linear*: it converts
distances into neighbour probabilities (P in the original space, Q in the
map) and moves the points so that KL(P || Q) is small.  It preserves local
neighbourhoods and tends to show clusters, but distances between clusters and
cluster sizes carry no meaning, and different perplexities / seeds give
different pictures.
"""

from __future__ import annotations

import math
import random
from operator import mul


def pca2(rows: list[list[float]], iters: int = 60, seed: int = 0) -> tuple[list[tuple[float, float]], list[float]]:
    """Centre the rows and project them on the first two principal components.
    Returns (coordinates, explained variance ratio of the two components)."""
    n, dim = len(rows), len(rows[0])
    mean = [sum(col) / n for col in zip(*rows)]
    X = [[x - m for x, m in zip(r, mean)] for r in rows]
    total = sum(x * x for r in X for x in r) / n or 1.0
    rng = random.Random(seed)
    comps: list[list[float]] = []
    ratios: list[float] = []
    for _ in range(2):
        v = [rng.random() - 0.5 for _ in range(dim)]
        for _ in range(iters):
            proj = [sum(map(mul, r, v)) for r in X]
            v = [sum(p * r[d] for p, r in zip(proj, X)) for d in range(dim)]
            for c in comps:                              # deflate: stay orthogonal to earlier components
                dot = sum(map(mul, v, c))
                v = [x - dot * y for x, y in zip(v, c)]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / norm for x in v]
        comps.append(v)
        proj = [sum(map(mul, r, v)) for r in X]
        ratios.append(sum(p * p for p in proj) / n / total)
    pts = [(sum(map(mul, r, comps[0])), sum(map(mul, r, comps[1]))) for r in X]
    return pts, [round(r, 4) for r in ratios]


def _squared_distances(rows: list[list[float]]) -> list[list[float]]:
    n = len(rows)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ri = rows[i]
        for j in range(i + 1, n):
            d = sum((a - b) * (a - b) for a, b in zip(ri, rows[j]))
            D[i][j] = D[j][i] = d
    return D


def _joint_probabilities(D: list[list[float]], perplexity: float, tol: float = 1e-5, steps: int = 50) -> list[list[float]]:
    """P_ij: Gaussian neighbour probabilities, one bandwidth per point found by
    binary search so that the entropy of the row matches log(perplexity);
    then symmetrised and normalised over all pairs."""
    n = len(D)
    target = math.log(perplexity)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Di = D[i]
        beta, lo, hi = 1.0, 0.0, math.inf
        for _ in range(steps):
            ps = [math.exp(-d * beta) if j != i else 0.0 for j, d in enumerate(Di)]
            s = sum(ps) or 1e-12
            H = math.log(s) + beta * sum(map(mul, Di, ps)) / s      # Shannon entropy of the row
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:                                            # too flat -> narrower Gaussian
                lo = beta
                beta = beta * 2 if hi == math.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == 0 else (beta + lo) / 2
        P[i] = [p / s for p in ps]
    scale = 1.0 / (2 * n)
    return [[max((P[i][j] + P[j][i]) * scale, 1e-12) for j in range(n)] for i in range(n)]


def tsne(rows: list[list[float]], perplexity: float = 15.0, iters: int = 400, seed: int = 0,
         learning_rate: float | None = None, exaggeration: float = 12.0, exaggeration_iters: int = 100,
         progress=None) -> dict:
    """2-D t-SNE of the rows.  Returns {"points": [(x, y)], "kl": [KL per 50 iterations], ...}."""
    n = len(rows)
    if n < 4:
        raise ValueError("t-SNE needs at least 4 points")
    perplexity = max(2.0, min(float(perplexity), (n - 1) / 3))
    P = _joint_probabilities(_squared_distances(rows), perplexity)
    init, _ = pca2(rows)                                            # PCA initialisation (deterministic, keeps global layout)
    sd = math.sqrt(sum(x * x + y * y for x, y in init) / n) or 1.0
    xs = [x / sd * 1e-4 for x, _ in init]
    ys = [y / sd * 1e-4 for _, y in init]
    lr = learning_rate or max(n / 12.0, 50.0)
    ux, uy = [0.0] * n, [0.0] * n                                   # momentum terms
    gx_gain, gy_gain = [1.0] * n, [1.0] * n
    kl_history: list[float] = []
    for it in range(iters):
        exag = exaggeration if it < exaggeration_iters else 1.0
        momentum = 0.5 if it < 250 else 0.8
        # Student-t kernel  q_ij ∝ (1 + |y_i - y_j|^2)^-1
        num = [[1.0 / (1.0 + (xi - xj) ** 2 + (yi - yj) ** 2) for xj, yj in zip(xs, ys)] for xi, yi in zip(xs, ys)]
        Z = sum(map(sum, num)) - n                                  # exclude the diagonal (= n ones)
        invZ = 1.0 / Z
        gx, gy = [0.0] * n, [0.0] * n
        for i in range(n):
            coef = [(p * exag - q * invZ) * q for p, q in zip(P[i], num[i])]
            s = sum(coef)
            gx[i] = 4.0 * (xs[i] * s - sum(map(mul, coef, xs)))   # sum_j coef_j (y_i - y_j)
            gy[i] = 4.0 * (ys[i] * s - sum(map(mul, coef, ys)))
        for i in range(n):                                          # adaptive gains (Jacobs 1988) + momentum
            gx_gain[i] = max(gx_gain[i] + 0.2 if (gx[i] > 0) != (ux[i] > 0) else gx_gain[i] * 0.8, 0.01)
            gy_gain[i] = max(gy_gain[i] + 0.2 if (gy[i] > 0) != (uy[i] > 0) else gy_gain[i] * 0.8, 0.01)
            ux[i] = momentum * ux[i] - lr * gx_gain[i] * gx[i]
            uy[i] = momentum * uy[i] - lr * gy_gain[i] * gy[i]
        mx, my = sum(ux) / n, sum(uy) / n
        xs = [x + u - mx for x, u in zip(xs, ux)]
        ys = [y + u - my for y, u in zip(ys, uy)]
        cx, cy = sum(xs) / n, sum(ys) / n
        xs = [x - cx for x in xs]
        ys = [y - cy for y in ys]
        if (it + 1) % 50 == 0 or it + 1 == iters:
            kl = sum(p * math.log(p / (q * invZ)) for Pi, Ni, in zip(P, num) for j, (p, q) in enumerate(zip(Pi, Ni)) if q * invZ > 0 and p > 1e-12)
            kl_history.append(round(kl, 4))
            if progress:
                progress((it + 1) / iters, kl)
    return {"points": list(zip(xs, ys)), "kl": kl_history, "perplexity": perplexity, "iters": iters,
            "learning_rate": lr, "seed": seed}


def _knn_sets(dists: list[list[float]], k: int) -> list[set[int]]:
    out = []
    for i, row in enumerate(dists):
        order = sorted((d, j) for j, d in enumerate(row) if j != i)
        out.append({j for _, j in order[:k]})
    return out


def knn_preservation(rows: list[list[float]], pts: list[tuple[float, float]], ks=(5, 10)) -> dict:
    """Share of each point's k nearest neighbours (cosine distance in the
    original space) that are also among its k nearest neighbours in the 2-D
    map (Euclidean).  1.0 = every neighbourhood preserved.  Returned per k,
    averaged over the points, plus the per-point value for the largest k."""
    n = len(rows)
    normed = []
    for r in rows:
        nn = math.sqrt(sum(x * x for x in r)) or 1.0
        normed.append([x / nn for x in r])
    hi = [[1.0 - sum(map(mul, a, b)) for b in normed] for a in normed]
    lo = [[(x1 - x2) ** 2 + (y1 - y2) ** 2 for x2, y2 in pts] for x1, y1 in pts]
    out: dict = {}
    per_point: list[float] = []
    for k in ks:
        k = min(k, n - 1)
        a, b = _knn_sets(hi, k), _knn_sets(lo, k)
        per_point = [len(x & y) / k for x, y in zip(a, b)]
        out[f"k{k}"] = round(sum(per_point) / n, 4)
    out["per_point"] = [round(v, 3) for v in per_point]
    return out
