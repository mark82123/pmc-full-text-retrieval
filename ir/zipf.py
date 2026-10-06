"""Zipf's-law analysis of a document collection (Project #2, part 1).

For each pre-processing condition the collection is turned into terms and

    CF(t) = number of occurrences of t in the whole collection
    DF(t) = number of documents containing t

are counted.  Terms are ranked by CF and a straight line is fitted to the
rank-frequency curve in log-log space (ordinary least squares):

    log10 CF(r) = a - b * log10 r          Zipf exponent k = b

The fit is reported for the whole curve and for three segments (high-,
middle- and low-frequency terms; equal thirds of the log-rank axis).

Conditions
    A  basic        white-space tokenisation + lower-case ("data," != "data")
    B  punctuation  tokens are runs of letters/digits (punctuation removed)
    C  stop words   B + stop-word removal
    D  stemming     C + Porter stemming
    E  stem only    B + Porter stemming, stop words kept (D without step C)

Resolving power of significant words (Luhn, 1958)
    Luhn's qualitative curve says the words that discriminate best between
    documents sit in the middle of the rank-frequency curve: the most
    frequent words are too common, the rarest too rare.  Here the resolving
    power of a term is measured as

        power(t) = CF(t) * idf(t) = sum over documents of tf * idf

    (its total TF-IDF weight in the collection).  The curve is smoothed with
    a running median over the rank axis; the default upper / lower cut-offs
    are where the smoothed curve falls to half of its maximum.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .porter import PorterStemmer
from .stopwords import STOP_WORDS
from .tokenizer import TOKEN_RE

CONDITIONS = [
    ("A", "Basic", "tokenisation (white space) + lower-case"),
    ("B", "Punctuation removed", "A + punctuation handling (tokens = runs of letters/digits)"),
    ("C", "Stop words removed", "B + stop-word removal"),
    ("D", "Stemming", "C + Porter stemming"),
    ("E", "Stemming, stop words kept", "B + Porter stemming (stop words kept)"),
]
SEGMENTS = ("high", "middle", "low")
ZONES = ("common", "significant", "rare")       # above the upper cut-off / between / below the lower cut-off
SMOOTH_SPAN = 10 ** 0.15                        # running-median window: rank / span .. rank * span
SMOOTH_MIN = 4                                  # ... and at least this many ranks on each side

_stemmer = PorterStemmer()
_stem_cache: dict[str, str] = {}


def _stem(word: str) -> str:
    s = _stem_cache.get(word)
    if s is None:
        s = _stem_cache[word] = _stemmer.stem(word)
    return s


def terms_for(text: str, condition: str) -> list[str]:
    """The term sequence of *text* under one pre-processing condition."""
    text = text.lower()
    if condition == "A":
        return text.split()
    toks = TOKEN_RE.findall(text)
    if condition == "B":
        return toks
    if condition == "E":
        return [_stem(t) for t in toks]
    toks = [t for t in toks if t not in STOP_WORDS]
    if condition == "C":
        return toks
    return [_stem(t) for t in toks]


# ---------------------------------------------------------------------- #
# Regression
# ---------------------------------------------------------------------- #
def fit_loglog(freqs: list[int], lo: int = 1, hi: int | None = None) -> dict:
    """OLS fit of log10(freq) on log10(rank) for ranks lo..hi (1-based,
    inclusive).  *freqs* must be sorted in decreasing order."""
    hi = min(hi or len(freqs), len(freqs))
    xs = [math.log10(r) for r in range(lo, hi + 1)]
    ys = [math.log10(freqs[r - 1]) for r in range(lo, hi + 1)]
    n = len(xs)
    if n < 2:
        return {"n": n, "from": lo, "to": hi, "slope": 0.0, "intercept": ys[0] if ys else 0.0,
                "exponent": 0.0, "r2": 0.0, "rmse": 0.0}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {"n": n, "from": lo, "to": hi, "slope": round(slope, 4), "intercept": round(intercept, 4),
            "exponent": round(-slope, 4), "r2": round(1 - sse / syy, 4) if syy else 1.0,
            "rmse": round(math.sqrt(sse / n), 4)}


def segment_bounds(vocab: int) -> list[tuple[str, int, int]]:
    """Split ranks 1..V into three equal parts of the log-rank axis."""
    a = max(2, round(vocab ** (1 / 3)))
    b = max(a + 1, round(vocab ** (2 / 3)))
    return [("high", 1, a - 1), ("middle", a, b - 1), ("low", b, vocab)]


def _residuals(freqs: list[int], fit: dict, lo: int, hi: int) -> tuple[float, float]:
    """(RMSE, mean signed residual) of the *global* fit inside ranks lo..hi."""
    res = [math.log10(freqs[r - 1]) - (fit["intercept"] + fit["slope"] * math.log10(r)) for r in range(lo, hi + 1)]
    if not res:
        return 0.0, 0.0
    return math.sqrt(sum(e * e for e in res) / len(res)), sum(res) / len(res)


def _thin(freqs: list[int], dense: int = 60, per_decade: int = 70) -> list[list[int]]:
    """Down-sample the rank-frequency curve for plotting: every rank up to
    *dense*, then log-spaced ranks (the curve is plotted on a log axis)."""
    n = len(freqs)
    if not n:
        return []
    ranks = set(range(1, min(dense, n) + 1))
    r = float(dense)
    step = 10 ** (1 / per_decade)
    while r < n:
        ranks.add(int(r))
        r *= step
    ranks.add(n)
    return [[r, freqs[r - 1]] for r in sorted(ranks)]


# ---------------------------------------------------------------------- #
# Collection analysis
# ---------------------------------------------------------------------- #
class Collection:
    """CF / DF counts of one collection under one condition."""

    def __init__(self, key: str, texts: list[str]):
        self.key = key
        self.n_docs = len(texts)
        self.cf: Counter = Counter()
        self.df: Counter = Counter()
        self.n_tokens = 0
        for text in texts:
            terms = terms_for(text, key)
            self.n_tokens += len(terms)
            self.cf.update(terms)
            self.df.update(set(terms))
        self.ranked = sorted(self.cf.items(), key=lambda x: (-x[1], x[0]))
        self.freqs = [c for _, c in self.ranked]
        self._power: list[float] | None = None
        self._curve: list[list] | None = None

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log10(self.n_docs / df) if df else 0.0

    def row(self, term: str) -> dict:
        cf, df = self.cf.get(term, 0), self.df.get(term, 0)
        return {"term": term, "cf": cf, "df": df, "cf_per_doc": round(cf / df, 2) if df else 0,
                "df_ratio": round(df / self.n_docs, 4) if self.n_docs else 0, "idf": round(self.idf(term), 4)}

    def summary(self, top_n: int = 50) -> dict:
        _, name, desc = next(c for c in CONDITIONS if c[0] == self.key)
        vocab = len(self.freqs)
        fit = fit_loglog(self.freqs) if vocab else {}
        segments = []
        for seg, lo, hi in (segment_bounds(vocab) if vocab > 3 else []):
            f = fit_loglog(self.freqs, lo, hi)
            g_rmse, g_bias = _residuals(self.freqs, fit, lo, hi)
            f.update(name=seg, global_rmse=round(g_rmse, 4), global_bias=round(g_bias, 4),
                     tokens_share=round(sum(self.freqs[lo - 1:hi]) / max(self.n_tokens, 1), 4))
            segments.append(f)
        hapax = sum(1 for c in self.freqs if c == 1)
        return {
            "key": self.key, "name": name, "description": desc,
            "documents": self.n_docs, "tokens": self.n_tokens, "vocabulary": vocab,
            "avg_tokens": round(self.n_tokens / self.n_docs, 1) if self.n_docs else 0,
            "hapax": hapax, "hapax_share": round(hapax / vocab, 4) if vocab else 0,
            "top10_share": round(sum(self.freqs[:10]) / max(self.n_tokens, 1), 4),
            "top": [[t, c, self.df[t]] for t, c in self.ranked[:top_n]],
            "fit": fit, "segments": segments, "points": _thin(self.freqs),
        }

    # ---- term tables (CF vs DF, IDF) ----------------------------------- #
    def normalise(self, word: str) -> str:
        """Map a user-typed word to this condition's term ('' if dropped)."""
        t = terms_for(word, self.key)
        return t[0] if t else ""

    def auto_terms(self, n: int = 24) -> list[str]:
        """A spread of terms: the most frequent, the most 'bursty' (high CF
        relative to DF) and some from the middle / tail of the DF range."""
        top = [t for t, _ in self.ranked[:n // 3]]
        bursty = sorted((t for t, c in self.cf.items() if c >= 20 and self.df[t] >= 3 and t not in top),
                        key=lambda t: -self.cf[t] / self.df[t])[:n // 3]
        rest = [t for t, _ in self.ranked if t not in top and t not in bursty and t.isalpha()]
        spread = [rest[int(len(rest) * q)] for q in (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3)] if len(rest) > 50 else rest[:8]
        seen, out = set(), []
        for t in top + bursty + spread:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def scatter(self, limit: int = 1500) -> list[list]:
        """(term, cf, df) of the most frequent terms for a CF-vs-DF scatter plot."""
        return [[t, c, self.df[t]] for t, c in self.ranked[:limit]]


    # ---- resolving power of significant words (Luhn) -------------------- #
    def powers(self) -> list[float]:
        """power(t) = CF(t) * idf(t) of every term, in rank order."""
        if self._power is None:
            self._power = [cf * self.idf(t) for t, cf in self.ranked]
        return self._power

    def _running_median(self, values: list[float], digits: int = 2) -> list[list]:
        """[[rank, median of values[] in the window around that rank]] at the
        thinned ranks used for plotting."""
        n, out = len(values), []
        for r, _ in _thin(self.freqs):
            lo = max(1, min(r - SMOOTH_MIN, int(r / SMOOTH_SPAN)))
            hi = min(n, max(r + SMOOTH_MIN, int(r * SMOOTH_SPAN)))
            win = sorted(values[lo - 1:hi])
            out.append([r, round(win[len(win) // 2], digits)])
        return out

    def power_curve(self) -> list[list]:
        """Running median of the resolving power over rank."""
        if self._curve is None:
            self._curve = self._running_median(self.powers())
        return self._curve

    def idf_curve(self) -> list[list]:
        """Running median of idf over rank (the other factor of the power)."""
        return self._running_median([self.idf(t) for t, _ in self.ranked], 4)

    def auto_cutoffs(self, level: float = 0.5) -> tuple[int, int]:
        """(upper, lower) cut-off ranks: the stretch around the peak of the
        smoothed curve that stays at or above *level* x the peak."""
        curve = self.power_curve()
        peak = max(range(len(curve)), key=lambda i: curve[i][1])
        floor = curve[peak][1] * level
        i = j = peak
        while i > 0 and curve[i - 1][1] >= floor:
            i -= 1
        while j < len(curve) - 1 and curve[j + 1][1] >= floor:
            j += 1
        return curve[i][0], curve[j][0]

    def resolving(self, upper: int | None = None, lower: int | None = None, top_n: int = 80) -> dict:
        """Resolving-power report.  Significant words are ranks upper..lower
        (inclusive); ranks above *upper* are too common, below *lower* too
        rare.  Cut-offs left out default to auto_cutoffs()."""
        vocab = len(self.ranked)
        out = {"condition": self.key, "documents": self.n_docs, "vocabulary": vocab, "tokens": self.n_tokens}
        if not vocab:
            return {**out, "auto": {"upper": 0, "lower": 0}, "upper": 0, "lower": 0, "peak": None,
                    "curve": [], "idf_curve": [], "points": [], "zones": [], "top": []}
        auto_upper, auto_lower = self.auto_cutoffs()
        upper = min(max(1, upper or auto_upper), vocab)
        lower = min(max(upper, lower or auto_lower), vocab)
        pw = self.powers()
        total = sum(pw) or 1.0
        row = lambda i: {**self.row(self.ranked[i][0]), "rank": i + 1, "power": round(pw[i], 2)}  # noqa: E731
        zones = []
        for name, lo, hi in zip(ZONES, (1, upper, lower + 1), (upper - 1, lower, vocab)):
            idx = range(lo - 1, hi)
            n = len(idx)
            cf = sum(self.freqs[lo - 1:hi])
            power = sum(pw[lo - 1:hi])
            zone = {"name": name, "from": lo, "to": hi, "terms": n, "vocab_share": round(n / vocab, 4),
                    "tokens_share": round(cf / max(self.n_tokens, 1), 4), "power_share": round(power / total, 4),
                    "mean_power": round(power / n, 2) if n else 0,
                    "mean_idf": round(sum(self.idf(self.ranked[i][0]) for i in idx) / n, 4) if n else 0,
                    "examples": [self.ranked[i][0] for i in idx[:15]]}
            if name == "common":
                zone["stopwords"] = sum(1 for i in idx if self.ranked[i][0] in STOP_WORDS)
            zones.append(zone)
        curve = self.power_curve()
        peak = max(curve, key=lambda p: p[1])
        best = sorted(range(upper - 1, lower), key=lambda i: (-pw[i], i))[:top_n]
        return {**out, "auto": {"upper": auto_upper, "lower": auto_lower}, "upper": upper, "lower": lower,
                "peak": {"rank": peak[0], "power": peak[1]}, "curve": curve, "idf_curve": self.idf_curve(),
                "points": [[r, round(pw[r - 1], 2), self.ranked[r - 1][0], cf, self.df[self.ranked[r - 1][0]]]
                           for r, cf in _thin(self.freqs, dense=400)],
                "zones": zones, "top": [row(i) for i in best]}


def stemming_groups(texts: list[str], top_n: int = 25) -> dict:
    """What Porter stemming merges: stems that conflate the most surface
    forms (condition C words -> condition D stems)."""
    words: Counter = Counter()
    for text in texts:
        words.update(terms_for(text, "C"))
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for w, c in words.items():
        groups[_stem(w)].append((w, c))
    merged = [(s, sorted(ws, key=lambda x: -x[1])) for s, ws in groups.items() if len(ws) > 1]
    merged.sort(key=lambda g: (-len(g[1]), -sum(c for _, c in g[1])))
    return {
        "words": len(words), "stems": len(groups), "merged_stems": len(merged),
        "reduction": round(1 - len(groups) / len(words), 4) if words else 0,
        "groups": [{"stem": s, "cf": sum(c for _, c in ws), "forms": ws[:12], "n_forms": len(ws)} for s, ws in merged[:top_n]],
    }


def analyse(texts: list[str]) -> dict[str, Collection]:
    return {key: Collection(key, texts) for key, _, _ in CONDITIONS}
