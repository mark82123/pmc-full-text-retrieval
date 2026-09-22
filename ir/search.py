"""Query parsing, matching and ranking.

Query language
--------------
    cancer immunotherapy          free terms (joined with the default operator, AND)
    cancer AND (vaccine OR drug)  boolean operators with parentheses, NOT / -term
    "gut microbiome"              exact phrase (positional match, stop words skipped)
    immun*                        prefix wildcard
    title:crispr                  field restriction (title / abstract / body)

Scoring
-------
    bm25   Okapi BM25 (k1 = 1.5, b = 0.75)                       [default]
    tfidf  cosine-style TF-IDF:  (1 + log tf) * log(N / df) / sqrt(doc length)
    count  raw number of matches (boolean retrieval ordered by hit count)
Title matches receive a small boost.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field as dc_field

from .index import InvertedIndex
from .tokenizer import TOKEN_RE

# ----------------------------------------------------------------------- #
# Lexer
# ----------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<lparen>\()|(?P<rparen>\))|
        (?P<phrase>"(?P<pbody>[^"]*)"|“(?P<pbody2>[^”]*)”)|
        (?P<not>-(?=\S))|
        (?P<word>(?:[a-zA-Z]+:)?[^\s()"“”]+)
    )""",
    re.VERBOSE,
)


@dataclass
class QToken:
    kind: str           # AND OR NOT LPAREN RPAREN TERM PHRASE
    value: str = ""
    field: str = ""


def lex(query: str) -> list[QToken]:
    out: list[QToken] = []
    pos = 0
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        pos = m.end()
        if m.group("lparen"):
            out.append(QToken("LPAREN"))
        elif m.group("rparen"):
            out.append(QToken("RPAREN"))
        elif m.group("phrase") is not None:
            body = m.group("pbody") if m.group("pbody") is not None else m.group("pbody2")
            out.append(QToken("PHRASE", body or ""))
        elif m.group("not"):
            out.append(QToken("NOT"))
        else:
            w = m.group("word")
            up = w.upper()
            if up in ("AND", "&&"):
                out.append(QToken("AND"))
            elif up in ("OR", "||"):
                out.append(QToken("OR"))
            elif up == "NOT":
                out.append(QToken("NOT"))
            else:
                fld = ""
                if ":" in w:
                    f, rest = w.split(":", 1)
                    if f.lower() in ("title", "abstract", "body") and rest:
                        fld, w = f.lower(), rest
                out.append(QToken("TERM", w, fld))
    return out


# ----------------------------------------------------------------------- #
# AST
# ----------------------------------------------------------------------- #
@dataclass
class Node:
    kind: str                      # TERM PHRASE PREFIX AND OR NOT
    value: str = ""
    field: str = ""
    children: list["Node"] = dc_field(default_factory=list)

    def describe(self) -> str:
        if self.kind == "TERM":
            return (self.field + ":" if self.field else "") + self.value
        if self.kind == "PREFIX":
            return (self.field + ":" if self.field else "") + self.value + "*"
        if self.kind == "PHRASE":
            return '"' + self.value + '"'
        if self.kind == "NOT":
            return "NOT " + self.children[0].describe()
        return "(" + f" {self.kind} ".join(c.describe() for c in self.children) + ")"


class QueryParser:
    """Recursive-descent parser.  Precedence: NOT > AND > OR.  Adjacent
    operands are joined with `default_op`."""

    def __init__(self, tokens: list[QToken], default_op: str = "AND"):
        self.toks = tokens
        self.i = 0
        self.default_op = default_op.upper()

    def peek(self) -> QToken | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> QToken:
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self) -> Node | None:
        if not self.toks:
            return None
        node = self.parse_or()
        return node

    def parse_or(self) -> Node | None:
        left = self.parse_and()
        while self.peek() and self.peek().kind == "OR":
            self.take()
            right = self.parse_and()
            if left is None:
                left = right
            elif right is not None:
                left = Node("OR", children=[left, right])
        return left

    def parse_and(self) -> Node | None:
        left = self.parse_not()
        while True:
            t = self.peek()
            if t is None or t.kind in ("RPAREN", "OR"):
                break
            explicit_and = t.kind == "AND"
            if explicit_and:
                self.take()
            right = self.parse_not()
            if right is None:
                if self.peek() is None:
                    break
                continue
            op = "AND" if explicit_and else self.default_op
            if left is None:
                left = right
            else:
                left = Node(op, children=[left, right])
        return left

    def parse_not(self) -> Node | None:
        t = self.peek()
        if t and t.kind == "NOT":
            self.take()
            child = self.parse_not()
            return Node("NOT", children=[child]) if child else None
        return self.parse_atom()

    def parse_atom(self) -> Node | None:
        t = self.peek()
        if t is None:
            return None
        if t.kind == "LPAREN":
            self.take()
            node = self.parse_or()
            if self.peek() and self.peek().kind == "RPAREN":
                self.take()
            return node
        if t.kind == "RPAREN":
            self.take()
            return None
        if t.kind == "PHRASE":
            self.take()
            return Node("PHRASE", t.value)
        if t.kind == "TERM":
            self.take()
            if t.value.endswith("*") and len(t.value) > 1:
                return Node("PREFIX", t.value[:-1], t.field)
            return Node("TERM", t.value, t.field)
        # stray operator
        self.take()
        return self.parse_atom()


def parse_query(query: str, default_op: str = "AND") -> Node | None:
    return QueryParser(lex(query), default_op).parse()


# ----------------------------------------------------------------------- #
# Matching
# ----------------------------------------------------------------------- #
@dataclass
class Match:
    """Per-document match information."""
    hits: dict[str, list[int]] = dc_field(default_factory=dict)   # matched index term -> positions
    label_hits: dict[str, int] = dc_field(default_factory=dict)   # query clause label -> count

    def merge(self, other: "Match") -> "Match":
        m = Match(dict(self.hits), dict(self.label_hits))
        for t, p in other.hits.items():
            m.hits[t] = sorted(set(m.hits.get(t, [])) | set(p))
        for k, v in other.label_hits.items():
            m.label_hits[k] = m.label_hits.get(k, 0) + v
        return m

    def n_hits(self) -> int:
        return sum(len(p) for p in self.hits.values())


class Searcher:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.tokenizer = index.tokenizer

    # ---- helpers ------------------------------------------------------- #
    def _in_field(self, doc_id: str, positions: list[int], fld: str) -> list[int]:
        if not fld:
            return positions
        rng = self.index.docs[doc_id].field_ranges.get(fld)
        if fld == "body":
            rng2 = self.index.docs[doc_id].field_ranges.get("heading")
            if rng and rng2:
                rng = (min(rng[0], rng2[0]), max(rng[1], rng2[1]))
            elif rng2:
                rng = rng2
        if not rng:
            return []
        return [p for p in positions if rng[0] <= p < rng[1]]

    def _term_matches(self, stem: str, fld: str, label: str) -> dict[str, Match]:
        out: dict[str, Match] = {}
        for doc_id, positions in self.index.postings.get(stem, {}).items():
            ps = self._in_field(doc_id, positions, fld)
            if ps:
                out[doc_id] = Match({stem: ps}, {label: len(ps)})
        return out

    def _phrase_matches(self, phrase: str, label: str) -> dict[str, Match]:
        # stems with their relative offsets (stop words keep a slot but are not matched)
        toks = [t for t in self.tokenizer.iter_tokens(phrase)]
        pattern = [(t.stem, t.position) for t in toks if t.stem]
        if not pattern:
            return {}
        if len(pattern) == 1:
            return self._term_matches(pattern[0][0], "", label)
        first_stem, first_off = pattern[0]
        out: dict[str, Match] = {}
        candidates = self.index.postings.get(first_stem, {})
        for doc_id, first_positions in candidates.items():
            plists = []
            ok = True
            for stem, off in pattern[1:]:
                pl = self.index.postings.get(stem, {}).get(doc_id)
                if not pl:
                    ok = False
                    break
                plists.append((set(pl), off - first_off))
            if not ok:
                continue
            starts = [p for p in first_positions if all((p + d) in s for s, d in plists)]
            if starts:
                hits: dict[str, list[int]] = {}
                for stem, off in pattern:
                    hits.setdefault(stem, []).extend(p + off - first_off for p in starts)
                out[doc_id] = Match(hits, {label: len(starts)})
        return out

    def _prefix_matches(self, prefix: str, fld: str, label: str) -> dict[str, Match]:
        out: dict[str, Match] = {}
        for term in self.index.terms_with_prefix(prefix.lower()):
            for doc_id, m in self._term_matches(term, fld, label).items():
                out[doc_id] = out[doc_id].merge(m) if doc_id in out else m
        return out

    # ---- evaluation ---------------------------------------------------- #
    def evaluate(self, node: Node | None) -> dict[str, Match]:
        if node is None:
            return {}
        if node.kind == "TERM":
            if len(self.tokenizer.raw_words(node.value)) > 1:
                # hyphenated / punctuated compound (SARS-CoV-2, COVID-19, T-cell):
                # the tokenizer splits it, so match it as an adjacent phrase
                res = self._phrase_matches(node.value, node.describe())
                if node.field:
                    fielded = {}
                    for d, m in res.items():
                        hits = {t: ps for t, ps in ((t, self._in_field(d, ps, node.field)) for t, ps in m.hits.items()) if ps}
                        if len(hits) == len(m.hits):
                            fielded[d] = Match(hits, m.label_hits)
                    res = fielded
                return res
            stem = self.tokenizer.normalize(node.value)
            if not stem:                       # stop word: matches nothing on its own
                return {}
            return self._term_matches(stem, node.field, node.describe())
        if node.kind == "PREFIX":
            return self._prefix_matches(node.value, node.field, node.describe())
        if node.kind == "PHRASE":
            return self._phrase_matches(node.value, node.describe())
        if node.kind == "NOT":
            excluded = self.evaluate(node.children[0])
            return {d: Match() for d in self.index.docs if d not in excluded}
        if node.kind == "AND":
            results = [self.evaluate(c) for c in node.children]
            # ignore empty clauses that are pure stop words
            common = set.intersection(*(set(r) for r in results)) if results else set()
            out = {}
            for d in common:
                m = Match()
                for r in results:
                    m = m.merge(r[d])
                out[d] = m
            return out
        if node.kind == "OR":
            out: dict[str, Match] = {}
            for c in node.children:
                for d, m in self.evaluate(c).items():
                    out[d] = out[d].merge(m) if d in out else m
            return out
        return {}

    # ---- scoring ------------------------------------------------------- #
    SCOPES = ("all", "title", "abstract", "body")

    def _term_score(self, tf: int, df: int, length: int, avg_length: float,
                    method: str, k1: float, b: float) -> float:
        """Score contribution of one term with tf occurrences in a text of
        `length` index terms (whole document or a single field)."""
        N = self.index.n_docs
        if method == "count":
            return float(tf)
        if method == "tfidf":
            idf = math.log((N + 1) / df)
            return (1 + math.log(tf)) * idf / math.sqrt(max(length, 1)) * 100
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))                      # bm25
        denom = tf + k1 * (1 - b + b * length / max(avg_length, 1))
        return idf * tf * (k1 + 1) / denom

    def score(self, doc_id: str, match: Match, method: str = "bm25", k1: float = 1.5, b: float = 0.75) -> float:
        """Whole-document score (all fields; title matches boosted x1.25)."""
        idoc = self.index.docs[doc_id]
        title_rng = idoc.field_ranges.get("title", (0, 0))
        s = 0.0
        for term, positions in match.hits.items():
            df = self.index.df(term) or 1
            part = self._term_score(len(positions), df, idoc.length, self.index.avg_doc_length, method, k1, b)
            if any(title_rng[0] <= p < title_rng[1] for p in positions):
                part *= 1.25  # title boost
            s += part
        return s

    def field_score(self, doc_id: str, match: Match, fld: str, method: str = "bm25",
                    k1: float = 1.5, b: float = 0.75) -> float:
        """Score using only the hits inside one field (title / abstract / body).
        Length normalisation uses the field's own length and the corpus average
        for that field, so an abstract is compared with abstracts, a body with
        bodies.  No title boost."""
        idoc = self.index.docs[doc_id]
        s = 0.0
        for term, positions in match.hits.items():
            ps = self._in_field(doc_id, positions, fld)
            if not ps:
                continue
            df = self.index.df(term) or 1
            s += self._term_score(len(ps), df, idoc.field_lengths.get(fld, 0),
                                  self.index.avg_field_length(fld), method, k1, b)
        return s

    def field_scores(self, doc_id: str, match: Match, method: str = "bm25") -> dict[str, float]:
        return {"all": self.score(doc_id, match, method),
                **{f: self.field_score(doc_id, match, f, method) for f in ("title", "abstract", "body")}}

    # ---- public API ---------------------------------------------------- #
    def search(self, query: str, method: str = "bm25", default_op: str = "AND",
               scope: str = "all") -> tuple[Node | None, list[tuple[str, float, Match, dict[str, float]]]]:
        """Returns (parsed query, [(doc_id, score, match, {field: score}), …])
        ranked by `scope`: "all" = whole document, or "title" / "abstract" /
        "body" = only the hits inside that field (documents with no hit in
        the field are dropped)."""
        node = parse_query(query, default_op)
        matches = self.evaluate(node)
        scope = scope if scope in self.SCOPES else "all"
        scored = []
        for d, m in matches.items():
            fs = self.field_scores(d, m, method)
            if scope != "all" and fs[scope] <= 0 and m.hits:
                continue
            scored.append((d, fs[scope], m, fs))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return node, scored

    def query_stems(self, node: Node | None) -> tuple[set[str], set[str]]:
        """(stems, prefixes) of the positive clauses, for highlighting."""
        stems: set[str] = set()
        prefixes: set[str] = set()

        def walk(n: Node | None, negated: bool = False) -> None:
            if n is None:
                return
            if n.kind == "NOT":
                walk(n.children[0], True)
                return
            if negated:
                return
            if n.kind == "TERM":
                stems.update(t.stem for t in self.tokenizer.iter_tokens(n.value) if t.stem)
            elif n.kind == "PREFIX":
                prefixes.add(n.value.lower())
            elif n.kind == "PHRASE":
                stems.update(t.stem for t in self.tokenizer.iter_tokens(n.value) if t.stem)
            else:
                for c in n.children:
                    walk(c)

        walk(node)
        return stems, prefixes


# ----------------------------------------------------------------------- #
# Highlighting helpers
# ----------------------------------------------------------------------- #
def highlight_spans(text: str, stems: set[str], prefixes: set[str], tokenizer) -> list[tuple[int, int]]:
    """Character spans of words in `text` whose stem (or prefix) matches the query."""
    spans = []
    for m in TOKEN_RE.finditer(text):
        w = m.group(0).lower()
        st = tokenizer.stemmer.stem(w)
        if st in stems or any(w.startswith(p) or st.startswith(p) for p in prefixes):
            spans.append((m.start(), m.end()))
    return spans
