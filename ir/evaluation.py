"""Ranked-retrieval evaluation measures (Project #3, task 5).

A *run* is the ranked list of document ids returned for a query; the
*qrels* map document id -> relevance grade (0 = not relevant, 1 = relevant,
2 = highly relevant).  Documents without a judgement count as not relevant
(the usual pooling assumption); `condensed=True` instead drops them from
the ranking before measuring (Sakai's condensed list), which removes the
penalty for unjudged documents.

    P@k      relevant documents among the first k / k
    R@k      relevant documents among the first k / all relevant documents
    RR       1 / rank of the first relevant document      (MRR = mean over queries)
    AP       mean of P@r over the ranks r of the relevant documents (MAP = mean over queries)
    nDCG@k   DCG@k / ideal DCG@k with  DCG@k = sum_{i<=k} (2^grade_i - 1) / log2(i + 1)

Rank-list comparison (RQ4: the same query under two schemes)
    overlap@k     |top-k(A) ∩ top-k(B)| / k
    Kendall tau   over the documents in the union of the two top-k lists,
                  using their ranks in the full lists
"""

from __future__ import annotations

import math
from collections import Counter


def precision_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in run[:k] if qrels.get(d, 0) > 0) / k


def recall_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    n_rel = sum(1 for g in qrels.values() if g > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in run[:k] if qrels.get(d, 0) > 0) / n_rel


def reciprocal_rank(run: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(run, 1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision(run: list[str], qrels: dict[str, int]) -> float:
    n_rel = sum(1 for g in qrels.values() if g > 0)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(run, 1):
        if qrels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


def dcg_at_k(grades: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], 1))


def ndcg_at_k(run: list[str], qrels: dict[str, int], k: int) -> float:
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k([qrels.get(d, 0) for d in run], k) / idcg


def r_precision(run: list[str], qrels: dict[str, int]) -> float:
    n_rel = sum(1 for g in qrels.values() if g > 0)
    return precision_at_k(run, qrels, n_rel) if n_rel else 0.0


def evaluate_run(run: list[str], qrels: dict[str, int], ks: tuple[int, ...] = (5, 10, 20),
                 condensed: bool = False, judged: set[str] | None = None) -> dict:
    """All measures of one query.  With *condensed*, documents that are not
    in *judged* are removed from the ranking first."""
    if condensed and judged is not None:
        run = [d for d in run if d in judged]
    out = {"relevant": sum(1 for g in qrels.values() if g > 0), "retrieved": len(run),
           "RR": reciprocal_rank(run, qrels), "AP": average_precision(run, qrels), "R-prec": r_precision(run, qrels)}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(run, qrels, k)
        out[f"R@{k}"] = recall_at_k(run, qrels, k)
        out[f"nDCG@{k}"] = ndcg_at_k(run, qrels, k)
    first = next((i for i, d in enumerate(run, 1) if qrels.get(d, 0) > 0), 0)
    out["first_rank"] = first
    return out


def mean_measures(per_query: list[dict]) -> dict:
    """Average of every numeric measure over the queries (MRR = mean RR, MAP = mean AP)."""
    if not per_query:
        return {}
    keys = [k for k, v in per_query[0].items() if isinstance(v, (int, float)) and k not in ("relevant", "retrieved", "first_rank")]
    out = {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
    out["MRR"] = out.pop("RR")
    out["MAP"] = out.pop("AP")
    out["queries"] = len(per_query)
    return out


# ---------------------------------------------------------------------- #
# comparing two rankings of the same query
# ---------------------------------------------------------------------- #
def overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / k


def kendall_tau(a: list[str], b: list[str], k: int = 10) -> float:
    """Kendall's tau between two rankings restricted to the documents in
    the union of their top-k lists; a document missing from one list is
    placed after every ranked document of that list."""
    docs = list(dict.fromkeys(a[:k] + b[:k]))
    if len(docs) < 2:
        return 1.0
    ra = {d: i for i, d in enumerate(a)}
    rb = {d: i for i, d in enumerate(b)}
    big = max(len(a), len(b)) + 1
    concordant = discordant = 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            x, y = docs[i], docs[j]
            da = ra.get(x, big) - ra.get(y, big)
            db = rb.get(x, big) - rb.get(y, big)
            s = da * db
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    n = concordant + discordant
    return (concordant - discordant) / n if n else 1.0


def rank_changes(a: list[str], b: list[str], k: int = 10) -> list[dict]:
    """Where each document of the two top-k lists sits in the other ranking."""
    ra = {d: i + 1 for i, d in enumerate(a)}
    rb = {d: i + 1 for i, d in enumerate(b)}
    rows = []
    for d in dict.fromkeys(a[:k] + b[:k]):
        rows.append({"doc_id": d, "rank_a": ra.get(d, 0), "rank_b": rb.get(d, 0)})
    rows.sort(key=lambda r: (r["rank_a"] or 10 ** 9, r["rank_b"] or 10 ** 9))
    return rows


def grade_histogram(qrels: dict[str, int]) -> dict[int, int]:
    return dict(Counter(g for g in qrels.values() if g > 0))
