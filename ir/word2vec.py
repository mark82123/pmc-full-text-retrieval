"""word2vec in pure Python (Project #2, part 2).

Both architectures of Mikolov et al. (2013) are implemented, trained with
negative sampling:

    Skip-gram (sg)  the centre word predicts each word in its window
    CBOW            the average of the window predicts the centre word

Details follow the original C implementation: dynamic window (the effective
window is drawn uniformly from 1..window), sub-sampling of frequent words,
negatives drawn from the unigram distribution raised to 0.75, a linearly
decaying learning rate and a pre-computed sigmoid table.

Two weight matrices are learned: `syn0` (input / word vectors, the result)
and `syn1` (output / context vectors).  For a training pair with input
vector h and a target word t with label y (1 = observed, 0 = negative):

    g      = (y - sigmoid(h . syn1[t])) * alpha
    syn1[t] += g * h
    h      += sum over targets of g * syn1[t]
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from operator import mul
from pathlib import Path

from .porter import stem as porter_stem
from .sentence import split_sentences
from .stopwords import STOP_WORDS
from .tsne import pca2

# words may keep inner hyphens (glp-1, sglt2, long-term); pure numbers are dropped
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")

_EXP_MAX = 6.0
_EXP_SIZE = 1000
_SIGMOID = [1 / (1 + math.exp(-((i / _EXP_SIZE * 2 - 1) * _EXP_MAX))) for i in range(_EXP_SIZE + 1)]
_TABLE_SIZE = 1_000_000


def sentences_from_texts(texts: list[str], remove_stopwords: bool = True) -> list[list[str]]:
    """Pre-processing for training: sentence split, tokenise, lower-case,
    optional stop-word removal.  One list of words per sentence."""
    out = []
    for text in texts:
        # a line break separates the title from the abstract: the title is a
        # sentence of its own even though it has no full stop
        for sent in (s for line in text.split("\n") for s in split_sentences(line)):
            words = [w.lower() for w in WORD_RE.findall(sent)]
            if remove_stopwords:
                words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
            if len(words) > 1:
                out.append(words)
    return out


class Word2Vec:
    def __init__(self, sg: bool = True, dim: int = 50, window: int = 5, negative: int = 5,
                 min_count: int = 5, epochs: int = 5, alpha: float = 0.025, sample: float = 1e-3, seed: int = 1,
                 remove_stopwords: bool = True):
        self.sg, self.dim, self.window, self.negative = sg, dim, window, negative
        self.min_count, self.epochs, self.alpha, self.sample, self.seed = min_count, epochs, alpha, sample, seed
        self.remove_stopwords = remove_stopwords
        self.vocab: list[str] = []                 # index -> word, most frequent first
        self.counts: list[int] = []
        self.index: dict[str, int] = {}
        self.syn0: list[list[float]] = []
        self.loss_history: list[float] = []        # mean negative-sampling loss per epoch
        self.train_seconds = 0.0
        self.corpus_words = 0
        self._norm: list[list[float]] | None = None

    # ------------------------------------------------------------------ #
    def params(self) -> dict:
        return {"model": "skip-gram" if self.sg else "cbow", "dim": self.dim, "window": self.window,
                "negative": self.negative, "min_count": self.min_count, "epochs": self.epochs,
                "alpha": self.alpha, "sample": self.sample, "seed": self.seed, "stopwords_removed": self.remove_stopwords}

    def name(self) -> str:
        """Short identifier built from the hyper-parameters, e.g. sg-d50-w5-m5-e10."""
        n = f"{'sg' if self.sg else 'cbow'}-d{self.dim}-w{self.window}-m{self.min_count}-e{self.epochs}"
        if self.negative != 5:
            n += f"-n{self.negative}"
        if not self.remove_stopwords:
            n += "-sw"
        if self.seed != 1:
            n += f"-s{self.seed}"
        return n

    def build_vocab(self, sentences: list[list[str]]) -> None:
        counts = Counter(w for s in sentences for w in s)
        items = sorted(((w, c) for w, c in counts.items() if c >= self.min_count), key=lambda x: (-x[1], x[0]))
        self.vocab = [w for w, _ in items]
        self.counts = [c for _, c in items]
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def train(self, sentences: list[list[str]], progress=None) -> "Word2Vec":
        """progress(fraction_done, epoch, alpha, running_loss) is called about
        once per 10 000 words."""
        t0 = time.time()
        rng = random.Random(self.seed)
        self.build_vocab(sentences)
        V, dim = len(self.vocab), self.dim
        if V < 2:
            raise ValueError("vocabulary too small - lower min_count or add documents")
        index = self.index
        corpus = [[index[w] for w in s if w in index] for s in sentences]
        total = sum(len(s) for s in corpus)
        self.corpus_words = total
        # sub-sampling: probability of *keeping* word w
        keep = [1.0] * V
        if self.sample > 0:
            for i, c in enumerate(self.counts):
                f = c / total
                keep[i] = min(1.0, (math.sqrt(f / self.sample) + 1) * self.sample / f)
        # negative-sampling table: word i occupies a share ~ count^0.75
        power = [c ** 0.75 for c in self.counts]
        z = sum(power)
        table: list[int] = []
        for i, p in enumerate(power):
            table.extend([i] * max(1, round(p / z * _TABLE_SIZE)))
        tsize = len(table)

        syn0 = [[(rng.random() - 0.5) / dim for _ in range(dim)] for _ in range(V)]
        syn1 = [[0.0] * dim for _ in range(V)]
        sg, window, negative = self.sg, self.window, self.negative
        sigmoid, emax, half = _SIGMOID, _EXP_MAX, _EXP_SIZE / 2
        rand, randint, log = rng.random, rng.randint, math.log
        zeros = [0.0] * dim
        done, all_words = 0, total * self.epochs
        self.loss_history = []

        def targets_update(h: list[float], target: int, alpha: float) -> tuple[list[float], float]:
            """One positive + `negative` sampled targets for input vector h.
            Updates syn1 in place; returns (gradient for h, loss)."""
            err = zeros
            loss = 0.0
            for d in range(negative + 1):
                if d == 0:
                    t, label = target, 1.0
                else:
                    t = table[int(rand() * tsize)]
                    if t == target:
                        continue
                    label = 0.0
                out = syn1[t]
                f = sum(map(mul, h, out))
                if f > emax:
                    p = 1.0
                elif f < -emax:
                    p = 0.0
                else:
                    p = sigmoid[int((f / emax + 1) * half)]
                loss -= log(max(p if label else 1.0 - p, 1e-7))
                g = (label - p) * alpha
                if g:
                    err = [e + g * o for e, o in zip(err, out)]
                    syn1[t] = [o + g * x for o, x in zip(out, h)]
            return err, loss

        for epoch in range(self.epochs):
            loss_sum, loss_n = 0.0, 0
            order = list(range(len(corpus)))
            rng.shuffle(order)
            for si in order:
                sent = [w for w in corpus[si] if keep[w] >= 1.0 or rand() < keep[w]]
                done += len(corpus[si])
                alpha = max(self.alpha * (1 - done / all_words), self.alpha * 1e-4)
                n = len(sent)
                for pos, word in enumerate(sent):
                    b = randint(1, window)
                    ctx = sent[max(0, pos - b):pos] + sent[pos + 1:pos + b + 1]
                    if not ctx:
                        continue
                    if sg:
                        for c in ctx:                    # (centre -> each context word)
                            h = syn0[word]
                            err, loss = targets_update(h, c, alpha)
                            syn0[word] = [x + e for x, e in zip(h, err)]
                            loss_sum += loss
                            loss_n += 1
                    else:                                # CBOW: mean(context) -> centre
                        inv = 1.0 / len(ctx)
                        h = [sum(col) * inv for col in zip(*(syn0[c] for c in ctx))]
                        err, loss = targets_update(h, word, alpha)
                        for c in ctx:
                            syn0[c] = [x + e for x, e in zip(syn0[c], err)]
                        loss_sum += loss
                        loss_n += 1
                if progress and si % 200 == 0:
                    progress(done / all_words, epoch + 1, alpha, loss_sum / max(loss_n, 1))
            self.loss_history.append(round(loss_sum / max(loss_n, 1), 4))
        self.syn0 = syn0
        self._norm = None
        self.train_seconds = round(time.time() - t0, 1)
        if progress:
            progress(1.0, self.epochs, 0.0, self.loss_history[-1])
        return self

    # ------------------------------------------------------------------ #
    def _normalised(self) -> list[list[float]]:
        if self._norm is None:
            norm = []                                    # built locally, then published: requests run in parallel threads
            for v in self.syn0:
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                norm.append([x / n for x in v])
            self._norm = norm
        return self._norm

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.index

    def similarity(self, a: str, b: str) -> float:
        nv = self._normalised()
        return sum(map(mul, nv[self.index[a.lower()]], nv[self.index[b.lower()]]))

    def _nearest(self, vec: list[float], exclude: set[int], topn: int) -> list[tuple[str, float]]:
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / n for x in vec]
        sims = [(sum(map(mul, vec, v)), i) for i, v in enumerate(self._normalised()) if i not in exclude]
        sims.sort(reverse=True)
        return [(self.vocab[i], round(s, 4)) for s, i in sims[:topn]]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        i = self.index[word.lower()]
        return self._nearest(self._normalised()[i], {i}, topn)

    def analogy(self, a: str, b: str, c: str, topn: int = 10) -> list[tuple[str, float]]:
        """a is to b as c is to ?   (vector b - a + c)"""
        nv = self._normalised()
        ia, ib, ic = (self.index[w.lower()] for w in (a, b, c))
        vec = [y - x + z for x, y, z in zip(nv[ia], nv[ib], nv[ic])]
        return self._nearest(vec, {ia, ib, ic}, topn)

    def vectors(self, words: list[str]) -> tuple[list[str], list[list[float]]]:
        """(words found in the vocabulary, their unit-length vectors)."""
        nv = self._normalised()
        found = [w for w in words if w in self.index]
        return found, [nv[self.index[w]] for w in found]

    def projection(self, words: list[str]) -> list[dict]:
        """2-D PCA of the given words' vectors (power iteration)."""
        found, rows = self.vectors(words)
        if len(found) < 3:
            return []
        pts, _ = pca2(rows)
        return [{"word": w, "count": self.counts[self.index[w]], "x": round(x, 4), "y": round(y, 4)}
                for w, (x, y) in zip(found, pts)]

    # ---- evaluation helpers ------------------------------------------- #
    def frequency_bucket(self, word: str) -> str:
        return bucket_of(self.counts[self.index[word]])

    def stem_groups(self) -> dict[str, list[str]]:
        """Vocabulary words that share a Porter stem (patient / patients,
        treat / treated / treatment ...): morphological variants that a good
        embedding should place close together."""
        groups: dict[str, list[str]] = {}
        for w in self.vocab:
            groups.setdefault(porter_stem(w), []).append(w)
        return {k: v for k, v in groups.items() if len(v) > 1}

    def evaluate(self, k: int = 10, per_bucket: int = 120, seed: int = 0) -> dict:
        """Intrinsic evaluation by word frequency, without any external resource.

        For a sample of words in every frequency bucket:
          sim_at_k        mean cosine similarity of the k nearest neighbours
                          (how tight the neighbourhood is)
          neighbour_count median collection frequency of the neighbours
                          (do rare words get rare or frequent neighbours?)
          variant_hit     share of words that have a morphological variant
                          (same Porter stem) among their k nearest neighbours;
                          only words that *have* a variant in the vocabulary
                          count, and the result is given per bucket
        """
        rng = random.Random(seed)
        groups = self.stem_groups()
        partner = {w: set(g) - {w} for g in groups.values() for w in g}
        by_bucket: dict[str, list[str]] = {}
        for w in self.vocab:
            by_bucket.setdefault(self.frequency_bucket(w), []).append(w)
        out = []
        for b in BUCKETS:
            words = by_bucket.get(b["label"], [])
            if not words:
                continue
            sample = words if len(words) <= per_bucket else rng.sample(words, per_bucket)
            sims, med_counts, hits, has_variant, ranks = [], [], 0, 0, []
            examples = []
            for w in sample:
                nb = self.most_similar(w, k)
                sims.append(sum(s for _, s in nb) / len(nb))
                cs = sorted(self.counts[self.index[x]] for x, _ in nb)
                med_counts.append(cs[len(cs) // 2])
                if partner.get(w):
                    has_variant += 1
                    rank = next((r for r, (x, _) in enumerate(nb, 1) if x in partner[w]), None)
                    if rank:
                        hits += 1
                        ranks.append(1.0 / rank)
                        if len(examples) < 4:
                            examples.append({"word": w, "variant": nb[rank - 1][0], "rank": rank, "similarity": nb[rank - 1][1]})
                    else:
                        ranks.append(0.0)
            out.append({"bucket": b["label"], "range": b["range"], "words": len(words), "sampled": len(sample),
                        "sim_at_k": round(sum(sims) / len(sims), 4),
                        "neighbour_count": sorted(med_counts)[len(med_counts) // 2],
                        "with_variant": has_variant, "variant_hit": round(hits / has_variant, 4) if has_variant else None,
                        "variant_mrr": round(sum(ranks) / len(ranks), 4) if ranks else None, "examples": examples})
        return {"k": k, "buckets": out, "stem_groups": len(groups),
                "words_with_variant": sum(len(g) for g in groups.values())}


    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        return {**self.params(), "vocabulary": len(self.vocab), "corpus_words": self.corpus_words,
                "loss_history": self.loss_history, "train_seconds": self.train_seconds}

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {"params": self.params(), "vocab": self.vocab, "counts": self.counts,
                   "loss_history": self.loss_history, "train_seconds": self.train_seconds,
                   "corpus_words": self.corpus_words, "extra": extra or {},
                   "vectors": [[round(x, 5) for x in v] for v in self.syn0]}
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["Word2Vec", dict]:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = d["params"]
        m = cls(sg=p["model"] == "skip-gram", dim=p["dim"], window=p["window"], negative=p["negative"],
                min_count=p["min_count"], epochs=p["epochs"], alpha=p["alpha"], sample=p["sample"], seed=p.get("seed", 1),
                remove_stopwords=p.get("stopwords_removed", True))
        m.vocab, m.counts, m.syn0 = d["vocab"], d["counts"], d["vectors"]
        m.index = {w: i for i, w in enumerate(m.vocab)}
        m.loss_history, m.train_seconds, m.corpus_words = d.get("loss_history", []), d.get("train_seconds", 0), d.get("corpus_words", 0)
        return m, d.get("extra", {})


BUCKETS = [{"label": "1-4", "range": (1, 4)}, {"label": "5-9", "range": (5, 9)}, {"label": "10-19", "range": (10, 19)},
           {"label": "20-49", "range": (20, 49)}, {"label": "50-199", "range": (50, 199)}, {"label": "200+", "range": (200, 10 ** 9)}]


def bucket_of(count: int) -> str:
    for b in BUCKETS:
        if b["range"][0] <= count <= b["range"][1]:
            return b["label"]
    return BUCKETS[-1]["label"]


def neighbour_overlap(a: "Word2Vec", b: "Word2Vec", k: int = 10, per_bucket: int = 120, seed: int = 0,
                      words: list[str] | None = None) -> dict:
    """How similar are the neighbourhoods of two models?  For a sample of the
    shared vocabulary (or the given words): Jaccard overlap of the k nearest
    neighbours in model a and model b, averaged per frequency bucket (the
    bucket is taken from model a's counts).  Used to compare architectures,
    hyper-parameters and random seeds (stability)."""
    rng = random.Random(seed)
    shared = [w for w in a.vocab if w in b.index]
    if words:
        shared = [w for w in words if w in a.index and w in b.index]
    by_bucket: dict[str, list[str]] = {}
    for w in shared:
        by_bucket.setdefault(a.frequency_bucket(w), []).append(w)
    rows, details = [], []
    for bk in BUCKETS:
        ws = by_bucket.get(bk["label"], [])
        if not ws:
            continue
        sample = ws if len(ws) <= per_bucket else rng.sample(ws, per_bucket)
        vals = []
        for w in sample:
            na = {x for x, _ in a.most_similar(w, k)}
            nb = {x for x, _ in b.most_similar(w, k)}
            j = len(na & nb) / len(na | nb)
            vals.append(j)
            details.append((j, w, a.counts[a.index[w]], sorted(na & nb)))
        rows.append({"bucket": bk["label"], "words": len(ws), "sampled": len(sample), "overlap": round(sum(vals) / len(vals), 4)})
    details.sort()
    fmt = lambda t: {"word": t[1], "count": t[2], "overlap": round(t[0], 3), "shared": t[3]}  # noqa: E731
    total = [d[0] for d in details]
    return {"k": k, "shared_vocabulary": len(shared), "overall": round(sum(total) / len(total), 4) if total else None,
            "buckets": rows, "least_stable": [fmt(t) for t in details[:8]], "most_stable": [fmt(t) for t in details[-8:][::-1]]}


def training_examples(words: list[str], window: int, centre: int | None = None) -> dict:
    """The training examples one sentence yields with a *fixed* window of
    size `window` (the trainer draws the effective window uniformly from
    1..window for every position, so on average half of these are used).

    Skip-gram: one (input = centre word, target = context word) pair per
    context word.   CBOW: one (input = the context words, target = centre)
    example per position."""
    n = len(words)
    sg, cbow = [], []
    for pos, w in enumerate(words):
        lo, hi = max(0, pos - window), min(n, pos + window + 1)
        ctx = [(j, words[j]) for j in range(lo, hi) if j != pos]
        if not ctx:
            continue
        if centre is not None and pos != centre:
            continue
        cbow.append({"position": pos, "centre": w, "context": [c for _, c in ctx], "offsets": [j - pos for j, _ in ctx]})
        for j, c in ctx:
            sg.append({"position": pos, "centre": w, "context": c, "offset": j - pos})
    return {"window": window, "words": words, "skipgram": sg, "cbow": cbow,
            "n_skipgram": len(sg), "n_cbow": len(cbow)}


def count_training_examples(sentences: list[list[str]], window: int) -> dict:
    """How many examples the whole corpus yields per epoch for a fixed
    window, and the expectation under the dynamic window (b ~ U{1..window})."""
    sg_fixed = sg_dyn = 0.0
    cbow = 0
    for s in sentences:
        n = len(s)
        if n < 2:
            continue
        cbow += n
        for pos in range(n):
            left, right = pos, n - 1 - pos
            sg_fixed += min(left, window) + min(right, window)
            sg_dyn += sum(min(left, b) + min(right, b) for b in range(1, window + 1)) / window
    return {"window": window, "skipgram_fixed": int(sg_fixed), "skipgram_dynamic": round(sg_dyn), "cbow": cbow}
