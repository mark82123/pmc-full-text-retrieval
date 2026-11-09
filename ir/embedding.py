"""Document representations built from word vectors (Project #4).

Three families of document / query vectors are compared on the same
collection, the same tokens and the same cosine ranking:

    A  tfidf      TF-IDF vector over the vocabulary (ltc: (1 + ln tf) · ln(N / df),
                  L2-normalised) — exact word matching only.
    C  w2v-avg    mean of the word2vec vectors of the words of the text
                  (every word counts the same).
    B  w2v-tfidf  TF-IDF weighted mean of the word vectors:
                      d = Σ_t  w(t, d) · v_t   /  ‖…‖     with  w(t, d) = (1 + ln tf) · idf
                  so frequent-but-common words (said, mln, dlrs) are damped and
                  the vector follows the discriminating words.
    D  hybrid     α · cos_tfidf + (1 − α) · cos_w2v-tfidf  (score-level fusion).

The word tokens are the ones word2vec was trained on (`WordTokenizer`: the
WORD_RE of ir/word2vec.py, lower-cased, stop words removed, *no* stemming,
pure numbers dropped) so that the TF-IDF baseline and the embeddings see
exactly the same terms; `tfidf-stem` additionally reports the classic
Porter-stemmed TF-IDF of Project #3 for reference.

Words that are not in the embedding vocabulary (min_count, or unseen in the
train split) are skipped (out of vocabulary, OOV); their share per document
and per query is reported because it is one of the reasons an embedding
query can fail.
"""

from __future__ import annotations

import math
from collections import Counter
from operator import mul

from .porter import PorterStemmer
from .stopwords import STOP_WORDS
from .tfidf import VectorSpace, df_weight, tf_weight
from .tokenizer import Token
from .word2vec import WORD_RE, Word2Vec

METHODS = [
    ("tfidf-stem", "TF-IDF · Porter stems (ltc)", "A₀", "Project #3 pipeline: stemmed terms, (1 + ln tf)·idf, cosine"),
    ("tfidf", "TF-IDF · surface words (ltc)", "A", "same tokens as the embeddings, (1 + ln tf)·idf, cosine"),
    ("w2v-avg", "mean word vector", "C", "unweighted mean of the word vectors, cosine"),
    ("w2v-tfidf", "TF-IDF weighted word vector", "B", "Σ (1 + ln tf)·idf · v(word), L2-normalised, cosine"),
    ("hybrid", "hybrid α·A + (1 − α)·B", "D", "score fusion of TF-IDF cosine and weighted-embedding cosine"),
]
METHOD_NAMES = {k: n for k, n, _, _ in METHODS}
METHOD_LETTERS = {k: l for k, _, l, _ in METHODS}
DEFAULT_METHOD = "w2v-tfidf"
DEFAULT_ALPHA = 0.5


class WordTokenizer:
    """Tokeniser with the word2vec pre-processing (ir/word2vec.py): words
    may keep inner hyphens, pure numbers are dropped, lower-case, stop words
    removed, no stemming (optional).  Duck-types the `Tokenizer` used by
    `VectorSpace` (tokenize / normalize / is_stopword / terms)."""

    def __init__(self, remove_stopwords: bool = True, stemming: bool = False):
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.stemmer = PorterStemmer()

    def normalize(self, word: str) -> str:
        w = word.lower()
        if self.remove_stopwords and (w in STOP_WORDS or len(w) < 2):
            return ""
        return self.stemmer.stem(w) if self.stemming else w

    def is_stopword(self, word: str) -> bool:
        return word.lower() in STOP_WORDS

    def tokenize(self, text: str) -> list[Token]:
        out = []
        for pos, m in enumerate(WORD_RE.finditer(text)):
            surface = m.group(0)
            out.append(Token(surface, surface.lower(), self.normalize(surface), m.start(), m.end(), pos))
        return out

    def terms(self, text: str) -> list[str]:
        return [t.stem for t in self.tokenize(text) if t.stem]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(map(mul, a, b)) / (na * nb) if na and nb else 0.0


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class DocEmbedder:
    """Document and query vectors from a word2vec model over the words of a
    VectorSpace (which supplies tf, df, idf and the postings)."""

    def __init__(self, vs: VectorSpace, model: Word2Vec, name: str = ""):
        self.vs = vs
        self.model = model
        self.name = name or model.name()
        self.dim = model.dim
        self._vectors: dict[str, dict[str, list[float]]] = {}      # method -> doc -> unit vector
        self._coverage: dict[str, dict] | None = None
        self.idf = {t: df_weight("t", df, vs.N) for t, df in vs.df.items()}

    # ---- weights ---------------------------------------------------------- #
    def term_weight(self, term: str, tf: int, method: str) -> float:
        """Weight of one term in the weighted mean: 1 (avg) or (1 + ln tf)·idf."""
        if method == "w2v-avg":
            return float(tf)                                          # a word used twice counts twice (plain mean over tokens)
        return tf_weight("l", tf) * self.idf.get(term, math.log(self.vs.N) if self.vs.N else 0.0)

    def embed_counts(self, counts: Counter, method: str) -> tuple[list[float], list[dict]]:
        """Weighted mean of the word vectors of a bag of words -> (unit vector,
        per-word rows).  Words not in the model are reported with in_model = False."""
        rows = []
        vec = [0.0] * self.dim
        total = 0.0
        idx = self.model.index
        syn0 = self.model.syn0
        for term, tf in counts.items():
            i = idx.get(term)
            w = self.term_weight(term, tf, method)
            rows.append({"word": term, "tf": tf, "df": self.vs.df.get(term, 0), "idf": round(self.idf.get(term, 0.0), 3),
                         "weight": round(w, 4), "in_model": i is not None,
                         "count": self.model.counts[i] if i is not None else 0})
        for r in rows:
            if r["in_model"] and r["weight"]:
                v = syn0[idx[r["word"]]]
                w = r["weight"]
                vec = [a + w * b for a, b in zip(vec, v)]
                total += w
        rows.sort(key=lambda r: (-r["in_model"], -r["weight"], r["word"]))
        if total:
            vec = [x / total for x in vec]
        return _unit(vec), rows

    def doc_vectors(self, method: str) -> dict[str, list[float]]:
        """Unit vector of every document under *method* (built once)."""
        if method not in self._vectors:
            vecs = {}
            for doc_id in self.vs.order:
                v, _ = self.embed_counts(self.vs.tf[doc_id], method)
                vecs[doc_id] = v
            self._vectors[method] = vecs
        return self._vectors[method]

    def doc_vector(self, doc_id: str, method: str) -> list[float]:
        return self.doc_vectors(method)[doc_id]

    def coverage(self) -> dict:
        """How much of the collection the embedding vocabulary covers."""
        if self._coverage is None:
            tok_in = tok_all = 0
            type_in: set[str] = set()
            per_doc = []
            empty = 0
            for doc_id in self.vs.order:
                tf = self.vs.tf[doc_id]
                n_all = sum(tf.values())
                n_in = sum(c for t, c in tf.items() if t in self.model.index)
                tok_all += n_all
                tok_in += n_in
                type_in.update(t for t in tf if t in self.model.index)
                per_doc.append(n_in / n_all if n_all else 0.0)
                if n_in == 0:
                    empty += 1
            per_doc.sort()
            self._coverage = {"model": self.name, "model_vocabulary": len(self.model.vocab), "dim": self.dim,
                              "collection_vocabulary": len(self.vs.df), "types_covered": len(type_in),
                              "types_share": round(len(type_in) / len(self.vs.df), 4) if self.vs.df else 0.0,
                              "tokens": tok_all, "tokens_covered": tok_in, "tokens_share": round(tok_in / tok_all, 4) if tok_all else 0.0,
                              "median_doc_share": round(per_doc[len(per_doc) // 2], 4) if per_doc else 0.0,
                              "documents_without_vector": empty}
        return self._coverage

    # ---- query ------------------------------------------------------------ #
    def query_counts(self, query: str) -> tuple[Counter, list[dict]]:
        qtf: Counter = Counter()
        steps = []
        for t in self.vs.tokenizer.tokenize(query):
            if t.stem:
                qtf[t.stem] += 1
                steps.append({"word": t.text, "term": t.stem, "df": self.vs.df.get(t.stem, 0), "idf": round(self.idf.get(t.stem, 0.0), 3),
                              "in_model": t.stem in self.model.index, "in_collection": t.stem in self.vs.df,
                              "count": self.model.counts[self.model.index[t.stem]] if t.stem in self.model.index else 0})
            else:
                steps.append({"word": t.text, "term": "", "df": 0, "idf": 0.0, "in_model": False, "in_collection": False, "count": 0, "stopword": True})
        return qtf, steps

    def rank(self, query: str, method: str = DEFAULT_METHOD) -> tuple[list[tuple[str, float]], dict]:
        """Every document by cosine with the query vector, best first."""
        qtf, steps = self.query_counts(query)
        qvec, rows = self.embed_counts(qtf, method)
        if not any(qvec):
            return [], {"terms": steps, "qrows": rows, "empty": True}
        docs = self.doc_vectors(method)
        scored = [(d, sum(map(mul, qvec, v))) for d, v in docs.items() if any(v)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored, {"terms": steps, "qrows": rows, "empty": False, "qvec": qvec}

    def explain(self, query_counts: Counter, doc_id: str, method: str = DEFAULT_METHOD, top: int = 3) -> list[dict]:
        """Why does the document match?  For every query word (in the model):
        is it in the document (lexical match) and which document words are
        closest to it in the vector space (semantic match), with their weight."""
        nv = self.model._normalised()
        idx = self.model.index
        dtf = self.vs.tf[doc_id]
        dwords = [(t, c) for t, c in dtf.items() if t in idx]
        out = []
        for q in query_counts:
            if q not in idx:
                out.append({"word": q, "in_model": False, "in_doc": q in dtf, "nearest": []})
                continue
            qv = nv[idx[q]]
            sims = []
            for t, c in dwords:
                s = sum(map(mul, qv, nv[idx[t]]))
                sims.append((s, t, c))
            sims.sort(reverse=True)
            out.append({"word": q, "in_model": True, "in_doc": q in dtf,
                        "nearest": [{"word": t, "similarity": round(s, 3), "tf": c, "weight": round(self.term_weight(t, c, method), 3)}
                                    for s, t, c in sims[:top]]})
        return out


# ---------------------------------------------------------------------- #
def hybrid_scores(tfidf: list[tuple[str, float]], emb: list[tuple[str, float]], alpha: float = DEFAULT_ALPHA) -> list[tuple[str, float]]:
    """α · cos_tfidf + (1 − α) · cos_emb over the union of the two rankings
    (a document missing from one list has cosine 0 there)."""
    a = max(0.0, min(1.0, alpha))
    scores: dict[str, float] = {}
    for d, s in tfidf:
        scores[d] = a * s
    for d, s in emb:
        scores[d] = scores.get(d, 0.0) + (1 - a) * s
    # documents with no positive evidence are not retrieved (so α = 1 is exactly the TF-IDF run)
    return sorted(((d, s) for d, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))


def word_topic_profile(vs: VectorSpace, word: str, topics: list[str]) -> dict:
    """In which topics does a word occur?  Share of the documents containing
    the word that carry each topic (P(topic | word)), against the base rate."""
    plist = vs.postings.get(word, {})
    n = len(plist)
    out = {"word": word, "df": n, "topics": []}
    for t in topics:
        with_t = sum(1 for d in plist if t in vs.docs[d].topics)
        base = sum(1 for d in vs.docs.values() if t in d.topics)
        out["topics"].append({"topic": t, "documents": with_t, "share": round(with_t / n, 3) if n else 0.0,
                              "base_rate": round(base / vs.N, 3) if vs.N else 0.0,
                              "lift": round((with_t / n) / (base / vs.N), 2) if n and base else 0.0})
    return out


def cooccurrence(vs: VectorSpace, a: str, b: str) -> dict:
    """Do two words occur in the same documents?  Document co-occurrence,
    Jaccard of their document sets and PMI over documents."""
    da = set(vs.postings.get(a, {}))
    db = set(vs.postings.get(b, {}))
    both = len(da & db)
    union = len(da | db)
    pmi = math.log((both * vs.N) / (len(da) * len(db))) if both and da and db else None
    return {"a": a, "b": b, "df_a": len(da), "df_b": len(db), "both": both,
            "jaccard": round(both / union, 3) if union else 0.0, "pmi": round(pmi, 2) if pmi is not None else None,
            "share_of_a": round(both / len(da), 3) if da else 0.0}
