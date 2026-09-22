"""Positional inverted index.

    postings[term] = {doc_id: [pos1, pos2, ...]}     (positions are word offsets
                                                       in the document token stream,
                                                       stop words keep their slot)
Each document also records its length (number of index terms), the field
ranges (title / abstract / body) inside the token stream, and its title terms
so that title matches can be boosted at query time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .pmc_parser import Document
from .tokenizer import Tokenizer


@dataclass
class IndexedDoc:
    doc_id: str
    length: int                                   # number of index terms (after stop-word removal)
    n_tokens: int                                 # number of tokens incl. stop words
    field_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)   # field -> (start, end) positions
    unit_offsets: list[tuple[str, int, int]] = field(default_factory=list)  # (field, start_pos, end_pos) per unit
    term_freq: Counter = field(default_factory=Counter)
    field_lengths: Counter = field(default_factory=Counter)  # field -> number of index terms (heading counted as body)


class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.postings: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.docs: dict[str, IndexedDoc] = {}
        self.documents: dict[str, Document] = {}
        self.total_length = 0
        self.total_field_length: Counter = Counter()   # field -> sum of field lengths over all docs

    # ------------------------------------------------------------------ #
    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_doc_length(self) -> float:
        return self.total_length / self.n_docs if self.n_docs else 0.0

    def avg_field_length(self, fld: str) -> float:
        return self.total_field_length[fld] / self.n_docs if self.n_docs else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def cf(self, term: str) -> int:
        return sum(len(p) for p in self.postings.get(term, {}).values())

    def vocabulary(self) -> list[str]:
        return sorted(self.postings)

    # ------------------------------------------------------------------ #
    def add_document(self, doc: Document) -> IndexedDoc:
        if doc.doc_id in self.docs:
            self.remove_document(doc.doc_id)
        pos = 0
        tf: Counter = Counter()
        local: dict[str, list[int]] = defaultdict(list)
        field_ranges: dict[str, list[int]] = {}
        unit_offsets: list[tuple[str, int, int]] = []
        field_lengths: Counter = Counter()
        for fld, text in doc.units():
            start = pos
            for tok in self.tokenizer.iter_tokens(text, start_position=pos):
                if tok.stem:
                    local[tok.stem].append(tok.position)
                    tf[tok.stem] += 1
                    field_lengths["body" if fld == "heading" else fld] += 1
                pos = tok.position + 1
            unit_offsets.append((fld, start, pos))
            rng = field_ranges.setdefault(fld, [start, pos])
            rng[1] = pos
        for term, positions in local.items():
            self.postings[term][doc.doc_id] = positions
        idoc = IndexedDoc(
            doc_id=doc.doc_id, length=sum(tf.values()), n_tokens=pos,
            field_ranges={k: (v[0], v[1]) for k, v in field_ranges.items()},
            unit_offsets=unit_offsets, term_freq=tf, field_lengths=field_lengths,
        )
        self.docs[doc.doc_id] = idoc
        self.documents[doc.doc_id] = doc
        self.total_length += idoc.length
        self.total_field_length.update(field_lengths)
        return idoc

    def remove_document(self, doc_id: str) -> None:
        idoc = self.docs.pop(doc_id, None)
        if idoc is None:
            return
        self.documents.pop(doc_id, None)
        self.total_length -= idoc.length
        self.total_field_length.subtract(idoc.field_lengths)
        for term in list(idoc.term_freq):
            plist = self.postings.get(term)
            if plist is not None:
                plist.pop(doc_id, None)
                if not plist:
                    del self.postings[term]

    def build(self, documents: list[Document]) -> None:
        for d in documents:
            self.add_document(d)

    # ------------------------------------------------------------------ #
    def terms_with_prefix(self, prefix: str) -> list[str]:
        return [t for t in self.postings if t.startswith(prefix)]

    def df_counter(self) -> Counter:
        return Counter({t: len(p) for t, p in self.postings.items()})

    def cf_counter(self) -> Counter:
        return Counter({t: sum(len(x) for x in p.values()) for t, p in self.postings.items()})
