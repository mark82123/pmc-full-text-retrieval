"""Term weighting and the vector space model (Project #3).

Every document (title + abstract) is turned into a vector over the index
vocabulary; the weight of a term in a document is the product of three
factors, written with the SMART triplet notation  "tf df norm":

    tf   n  natural      tf                       (raw term frequency)
         l  logarithm    1 + ln tf
         a  augmented    0.5 + 0.5 * tf / max_tf  (max_tf = the largest tf in the document)
         b  boolean      1
    df   n  none         1
         t  idf          ln(N / df)
         p  prob idf     max(0, ln((N - df) / df))
    norm n  none         1
         c  cosine       1 / sqrt(sum of squared weights)   (L2 normalisation)

so "ltc" is (1 + ln tf) * ln(N / df), L2-normalised; "nnn" is the raw term
frequency.  The query is weighted the same way (default "nnc": raw query term
frequency, no idf, L2-normalised -- the query normalisation only scales the
scores, it never changes the ranking) and

    score(q, d) = sum over the query terms of  w(t, q) * w(t, d)

With "c" on both sides this is exactly the cosine similarity between the two
vectors; with "n" on the document side it is the un-normalised dot product,
which grows with the length of the document (RQ4).  "bm25" is offered as a
reference outside the vector-space family.

Document vectors are stored as an inverted structure (term -> {doc: weight})
so that a query only touches the postings of its own terms.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .pmc_parser import Document
from .tokenizer import Tokenizer

TF_VARIANTS = {"n": "natural (raw tf)", "l": "logarithm (1 + ln tf)", "a": "augmented (0.5 + 0.5 tf / max tf)", "b": "boolean (1)"}
DF_VARIANTS = {"n": "none (1)", "t": "idf = ln(N / df)", "p": "prob idf = max(0, ln((N - df) / df))"}
NORM_VARIANTS = {"n": "none", "c": "cosine (L2)"}

# the schemes used in the controlled experiments (each changes ONE factor of its neighbour)
PRESETS = [
    ("nnn", "raw TF", "tf"),
    ("lnn", "log TF", "1 + ln tf"),
    ("btn", "IDF only", "idf (boolean tf)"),
    ("ntn", "raw TF × IDF", "tf · idf"),
    ("ltn", "log TF × IDF", "(1 + ln tf) · idf"),
    ("nnc", "raw TF, L2", "tf, normalised"),
    ("lnc", "log TF, L2", "(1 + ln tf), normalised"),
    ("ntc", "raw TF × IDF, L2", "tf · idf, normalised"),
    ("ltc", "log TF × IDF, L2", "(1 + ln tf) · idf, normalised  [standard]"),
    ("atc", "augmented TF × IDF, L2", "(0.5 + 0.5 tf / max tf) · idf, normalised"),
    ("lpc", "log TF × prob IDF, L2", "(1 + ln tf) · max(0, ln((N − df) / df)), normalised"),
    ("bm25", "BM25 (reference)", "Okapi BM25, k1 = 1.2, b = 0.75"),
]
PRESET_NAMES = {k: name for k, name, _ in PRESETS}
DEFAULT_SCHEME = "ltc"
DEFAULT_QUERY_SCHEME = "nnc"


def parse_scheme(scheme: str) -> tuple[str, str, str]:
    s = (scheme or DEFAULT_SCHEME).strip().lower()
    if s == "bm25":
        return ("bm25", "", "")
    if len(s) != 3 or s[0] not in TF_VARIANTS or s[1] not in DF_VARIANTS or s[2] not in NORM_VARIANTS:
        raise ValueError(f"unknown weighting scheme '{scheme}' (expected e.g. ltc, nnn, ntc or bm25)")
    return (s[0], s[1], s[2])


def describe_scheme(scheme: str) -> dict:
    tf, df, norm = parse_scheme(scheme)
    if tf == "bm25":
        return {"scheme": "bm25", "name": PRESET_NAMES["bm25"], "tf": "BM25 saturation", "df": "BM25 idf", "norm": "length (b = 0.75)"}
    return {"scheme": tf + df + norm, "name": PRESET_NAMES.get(tf + df + norm, tf + df + norm),
            "tf": TF_VARIANTS[tf], "df": DF_VARIANTS[df], "norm": NORM_VARIANTS[norm]}


def tf_weight(variant: str, tf: int, max_tf: int = 1) -> float:
    if tf <= 0:
        return 0.0
    if variant == "n":
        return float(tf)
    if variant == "l":
        return 1.0 + math.log(tf)
    if variant == "a":
        return 0.5 + 0.5 * tf / max(max_tf, 1)
    return 1.0                                                   # boolean


def df_weight(variant: str, df: int, n_docs: int) -> float:
    if variant == "n":
        return 1.0
    if df <= 0:
        return 0.0
    if variant == "t":
        return math.log(n_docs / df)
    return max(0.0, math.log((n_docs - df) / df)) if df < n_docs else 0.0   # prob idf


class VectorSpace:
    """Documents as term-weight vectors; several weighting schemes side by side."""

    def __init__(self, docs: list[Document], tokenizer: Tokenizer | None = None,
                 remove_stopwords: bool = True, stemming: bool = True):
        self.tokenizer = tokenizer or Tokenizer(remove_stopwords=remove_stopwords, stemming=stemming)
        self.docs: dict[str, Document] = {}
        self.tf: dict[str, Counter] = {}           # doc -> term -> raw tf
        self.max_tf: dict[str, int] = {}
        self.length: dict[str, int] = {}           # doc -> number of index terms
        self.n_tokens: dict[str, int] = {}         # doc -> tokens before stop-word removal
        self.df: Counter = Counter()
        self.cf: Counter = Counter()
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)   # term -> {doc: tf}
        self.order: list[str] = []
        self._weights: dict[str, tuple[dict[str, dict[str, float]], dict[str, float]]] = {}   # scheme -> (term -> {doc: w}, doc -> norm)
        for d in docs:
            self._add(d)
        self.N = len(self.docs)
        self.total_terms = sum(self.length.values())
        self.avg_length = self.total_terms / self.N if self.N else 0.0
        self._idf_cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    def text_of(self, doc: Document) -> str:
        return doc.title + "\n" + doc.abstract_text()

    def _add(self, doc: Document) -> None:
        toks = self.tokenizer.tokenize(self.text_of(doc))
        terms = [t.stem for t in toks if t.stem]
        tf = Counter(terms)
        self.docs[doc.doc_id] = doc
        self.order.append(doc.doc_id)
        self.tf[doc.doc_id] = tf
        self.max_tf[doc.doc_id] = max(tf.values(), default=1)
        self.length[doc.doc_id] = len(terms)
        self.n_tokens[doc.doc_id] = len(toks)
        for t, c in tf.items():
            self.df[t] += 1
            self.cf[t] += c
            self.postings[t][doc.doc_id] = c

    # ---- term statistics ----------------------------------------------- #
    def idf(self, term: str, variant: str = "t") -> float:
        return df_weight(variant, self.df.get(term, 0), self.N)

    def term_row(self, term: str) -> dict:
        """CF / DF / IDF and the tf distribution of one index term."""
        df = self.df.get(term, 0)
        tfs = sorted(self.postings.get(term, {}).values(), reverse=True)
        return {"term": term, "cf": self.cf.get(term, 0), "df": df, "df_share": round(df / self.N, 4) if self.N else 0.0,
                "idf": round(self.idf(term), 4), "prob_idf": round(self.idf(term, "p"), 4),
                "max_tf": tfs[0] if tfs else 0, "mean_tf": round(sum(tfs) / len(tfs), 2) if tfs else 0.0,
                "log_max_tf": round(tf_weight("l", tfs[0]) if tfs else 0.0, 4)}

    def analyse_terms(self, words: list[str]) -> list[dict]:
        rows = []
        for w in words:
            stem = self.tokenizer.normalize(w)
            if not stem:
                rows.append({"word": w, "term": "", "stopword": True, "cf": 0, "df": 0, "df_share": 0.0, "idf": 0.0,
                             "prob_idf": 0.0, "max_tf": 0, "mean_tf": 0.0, "log_max_tf": 0.0})
            else:
                rows.append({"word": w, "stopword": False, **self.term_row(stem)})
        return rows

    def vocabulary_stats(self) -> dict:
        """Summary of the term statistics of the whole collection."""
        V = len(self.df)
        df_vals = sorted(self.df.values(), reverse=True)
        buckets = [("df = 1", 1, 1), ("2–9", 2, 9), ("10–99", 10, 99), ("100–999", 100, 999), ("≥ 1000", 1000, 10 ** 9)]
        hist = [{"bucket": b, "terms": sum(1 for v in df_vals if lo <= v <= hi),
                 "idf_from": round(df_weight("t", min(hi, self.N), self.N), 3) if self.N else 0.0,
                 "idf_to": round(df_weight("t", lo, self.N), 3) if self.N else 0.0} for b, lo, hi in buckets]
        max_tfs = sorted(self.max_tf.values())
        return {"documents": self.N, "vocabulary": V, "tokens": sum(self.n_tokens.values()), "index_terms": self.total_terms,
                "avg_length": round(self.avg_length, 1), "min_length": min(self.length.values(), default=0),
                "max_length": max(self.length.values(), default=0),
                "stopword_tokens": sum(self.n_tokens.values()) - self.total_terms,
                "hapax_df": sum(1 for v in df_vals if v == 1),
                "df_histogram": hist,
                "median_max_tf": max_tfs[len(max_tfs) // 2] if max_tfs else 0,
                "top_df": [self.term_row(t) for t, _ in self.df.most_common(25)],
                "top_cf": [self.term_row(t) for t, _ in self.cf.most_common(25)]}

    # ---- weighting ------------------------------------------------------ #
    def weights(self, scheme: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """(term -> {doc: weight}, doc -> L2 norm of the *un-normalised* vector)
        for one scheme; built once and cached."""
        tf_v, df_v, norm_v = parse_scheme(scheme)
        key = tf_v + df_v + norm_v
        if key in self._weights:
            return self._weights[key]
        if tf_v == "bm25":
            raise ValueError("bm25 has no term-weight vectors")
        idf = {t: df_weight(df_v, df, self.N) for t, df in self.df.items()}
        raw: dict[str, dict[str, float]] = {}
        sq: Counter = Counter()
        for term, plist in self.postings.items():
            w_idf = idf[term]
            if w_idf == 0.0:
                continue
            row = {}
            for doc_id, tf in plist.items():
                w = tf_weight(tf_v, tf, self.max_tf[doc_id]) * w_idf
                row[doc_id] = w
                sq[doc_id] += w * w
            raw[term] = row
        norms = {d: math.sqrt(sq.get(d, 0.0)) for d in self.docs}
        if norm_v == "c":
            for term, row in raw.items():
                for doc_id in row:
                    if norms[doc_id]:
                        row[doc_id] /= norms[doc_id]
        self._weights[key] = (raw, norms)
        return self._weights[key]

    def doc_vector(self, doc_id: str, scheme: str, top: int = 0) -> dict:
        """The weight of every term of one document under *scheme* (largest first)."""
        if doc_id not in self.docs:
            raise ValueError(f"unknown document {doc_id}")
        tf_v, df_v, norm_v = parse_scheme(scheme)
        if tf_v == "bm25":
            raise ValueError("bm25 has no term-weight vectors")
        w, norms = self.weights(scheme)
        rows = []
        for term, tf in self.tf[doc_id].items():
            rows.append({"term": term, "tf": tf, "df": self.df[term], "idf": round(self.idf(term), 4),
                         "tf_w": round(tf_weight(tf_v, tf, self.max_tf[doc_id]), 4),
                         "df_w": round(df_weight(df_v, self.df[term], self.N), 4),
                         "weight": round(w.get(term, {}).get(doc_id, 0.0), 5)})
        rows.sort(key=lambda r: (-r["weight"], r["term"]))
        total = sum(r["weight"] for r in rows) or 1.0
        for r in rows:
            r["share"] = round(r["weight"] / total, 4)
        return {"doc_id": doc_id, "scheme": tf_v + df_v + norm_v, "terms": len(rows), "length": self.length[doc_id],
                "max_tf": self.max_tf[doc_id], "norm": round(norms[doc_id], 4),
                "top_share": rows[0]["share"] if rows else 0.0,
                "top5_share": round(sum(r["share"] for r in rows[:5]), 4),
                "vector": rows[:top] if top else rows}

    # ---- query ----------------------------------------------------------- #
    def query_terms(self, query: str) -> tuple[Counter, list[dict]]:
        """Query -> (term -> query tf, how each word was processed)."""
        qtf: Counter = Counter()
        steps = []
        for t in self.tokenizer.tokenize(query):
            if t.stem:
                qtf[t.stem] += 1
                steps.append({"word": t.text, "term": t.stem, "df": self.df.get(t.stem, 0),
                              "idf": round(self.idf(t.stem), 4), "role": "term" if t.stem in self.df else "unknown"})
            else:
                steps.append({"word": t.text, "term": "", "df": 0, "idf": 0.0, "role": "stopword"})
        return qtf, steps

    def query_vector(self, qtf: Counter, qscheme: str) -> dict[str, float]:
        tf_v, df_v, norm_v = parse_scheme(qscheme)
        if tf_v == "bm25":
            tf_v, df_v, norm_v = "n", "n", "c"
        max_q = max(qtf.values(), default=1)
        vec = {t: tf_weight(tf_v, c, max_q) * df_weight(df_v, self.df.get(t, 0), self.N) for t, c in qtf.items() if t in self.df}
        vec = {t: w for t, w in vec.items() if w}
        if norm_v == "c" and vec:
            n = math.sqrt(sum(w * w for w in vec.values()))
            vec = {t: w / n for t, w in vec.items()}
        return vec

    def _bm25(self, qtf: Counter, k1: float = 1.2, b: float = 0.75) -> tuple[dict[str, float], dict[str, dict[str, tuple[float, float]]]]:
        scores: Counter = Counter()
        parts: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
        for term, qc in qtf.items():
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for doc_id, tf in plist.items():
                denom = tf + k1 * (1 - b + b * self.length[doc_id] / max(self.avg_length, 1))
                w = idf * tf * (k1 + 1) / denom
                scores[doc_id] += qc * w
                parts[doc_id][term] = (float(qc), w)
        return dict(scores), parts

    def rank(self, query: str, scheme: str = DEFAULT_SCHEME, qscheme: str = DEFAULT_QUERY_SCHEME) -> tuple[list[tuple[str, float]], dict]:
        """Every document with a non-zero score, best first: ([(doc, score)], info).
        info = {terms: query processing, qvec: query weights, contributions: {doc: {term: (qw, dw)}}}"""
        qtf, steps = self.query_terms(query)
        if parse_scheme(scheme)[0] == "bm25":
            scores, parts = self._bm25(qtf)
            qvec = {t: float(c) for t, c in qtf.items() if t in self.df}
        else:
            qvec = self.query_vector(qtf, qscheme)
            w, _ = self.weights(scheme)
            scores = Counter()
            parts = defaultdict(dict)
            for term, qw in qvec.items():
                for doc_id, dw in w.get(term, {}).items():
                    scores[doc_id] += qw * dw
                    parts[doc_id][term] = (qw, dw)
            scores = dict(scores)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked, {"terms": steps, "qvec": qvec, "contributions": parts}

    def search(self, query: str, scheme: str = DEFAULT_SCHEME, qscheme: str = DEFAULT_QUERY_SCHEME, k: int = 10) -> dict:
        """Top-k with scores and the per-term contributions (for display)."""
        ranked, info = self.rank(query, scheme, qscheme)
        tf_v, df_v, norm_v = parse_scheme(scheme)
        out = []
        for i, (doc_id, score) in enumerate(ranked[:k], 1):
            doc = self.docs[doc_id]
            contrib = [{"term": t, "qw": round(qw, 4), "dw": round(dw, 4), "tf": self.tf[doc_id].get(t, 0),
                        "df": self.df.get(t, 0), "part": round(qw * dw, 5)}
                       for t, (qw, dw) in sorted(info["contributions"][doc_id].items(), key=lambda x: -x[1][0] * x[1][1])]
            out.append({"rank": i, "doc_id": doc_id, "score": round(score, 5), "title": doc.title, "year": doc.year,
                        "journal": doc.journal, "length": self.length[doc_id], "max_tf": self.max_tf[doc_id],
                        "matched": len(contrib), "contributions": contrib})
        return {"query": query, "scheme": tf_v + df_v + norm_v if tf_v != "bm25" else "bm25",
                "scheme_info": describe_scheme(scheme), "qscheme": qscheme if tf_v != "bm25" else "–",
                "terms": info["terms"], "qvec": {t: round(w, 4) for t, w in info["qvec"].items()},
                "total": len(ranked), "k": k, "results": out}

    def ranking_ids(self, query: str, scheme: str, qscheme: str = DEFAULT_QUERY_SCHEME) -> list[str]:
        return [d for d, _ in self.rank(query, scheme, qscheme)[0]]
