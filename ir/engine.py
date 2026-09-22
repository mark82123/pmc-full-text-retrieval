"""High-level engine used by the web application and the CLI."""

from __future__ import annotations

import html
import time
from collections import Counter
from pathlib import Path

from .index import InvertedIndex
from .ncbi import convert_pmids, fetch_pmc_xml, fetch_pubmed_xml, parse_ids, pubmed_to_jats
from .pmc_parser import Document, load_directory, parse_file, parse_xml_string
from .search import Node, Searcher, highlight_spans
from .sentence import split_sentences
from .stats import corpus_stats, document_stats
from .tokenizer import Tokenizer

MATCH_MAP_BINS = 40
ELLIPSIS = '<span class="ellip" title="omitted text between the best-matching sentences">…</span>'


def _mark(text: str, spans: list[tuple[int, int]]) -> str:
    """HTML-escape text and wrap the given spans in <mark>."""
    out = []
    last = 0
    for s, e in spans:
        out.append(html.escape(text[last:s]))
        out.append("<mark>" + html.escape(text[s:e]) + "</mark>")
        last = e
    out.append(html.escape(text[last:]))
    return "".join(out)


class Engine:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex(self.tokenizer)
        self.searcher = Searcher(self.index)
        self.stats: dict[str, dict] = {}
        self._sent_cache: dict[str, list[tuple[str, str]]] = {}
        self.build_time = 0.0

    # ------------------------------------------------------------------ #
    def load(self) -> int:
        t0 = time.time()
        docs = load_directory(self.data_dir) if self.data_dir.exists() else []
        for d in docs:
            self._add(d)
        self.build_time = time.time() - t0
        return len(docs)

    def _add(self, doc: Document) -> None:
        self.index.add_document(doc)
        self.stats[doc.doc_id] = document_stats(doc, self.tokenizer)
        self._sent_cache.pop(doc.doc_id, None)

    def add_file(self, path: str | Path) -> list[str]:
        ids = []
        for d in parse_file(path):
            self._add(d)
            ids.append(d.doc_id)
        return ids

    def add_xml(self, xml_text: str, filename: str = "upload.xml", persist: bool = True) -> list[str]:
        docs = parse_xml_string(xml_text, filename)
        if persist:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            target = self.data_dir / Path(filename).name
            target.write_text(xml_text, encoding="utf-8")
            for d in docs:
                d.source = str(target)
        ids = []
        for d in docs:
            self._add(d)
            ids.append(d.doc_id)
        return ids

    def remove(self, doc_id: str, delete_file: bool = False) -> bool:
        """Drop a document from the index.  With *delete_file* the XML it was
        loaded from is deleted too (only if it lives inside the data directory),
        so it does not come back at the next start."""
        if doc_id not in self.index.docs:
            return False
        source = self.index.documents[doc_id].source
        self.index.remove_document(doc_id)
        self.stats.pop(doc_id, None)
        self._sent_cache.pop(doc_id, None)
        if delete_file and source:
            path = Path(source)
            try:
                if path.is_file() and self.data_dir.resolve() in path.resolve().parents:
                    path.unlink()
            except OSError:
                pass
        return True

    def fetch_ids(self, text: str, persist: bool = True, replace: bool = False) -> list[dict]:
        """Resolve PMIDs / PMC ids found in *text*, download them from NCBI and
        index them.  Full text (JATS XML from PMC) is used when the article is
        in PMC; otherwise the PubMed record (title + abstract) is indexed.
        One result dict per id, in input order:
            {input, kind, number, pmid, pmcid, status, content, message, doc_id, title}
        status:  added | updated | exists | error
        content: full text | abstract only | ""
        """
        items = parse_ids(text)
        results = [{"input": it["input"], "kind": it["kind"], "number": it["number"],
                    "pmid": it["number"] if it["kind"] == "pmid" else "",
                    "pmcid": "PMC" + it["number"] if it["kind"] == "pmc" else "",
                    "status": "", "content": "", "message": "", "doc_id": "", "title": ""} for it in items]
        pmids = [r["pmid"] for r in results if r["pmid"]]
        mapping = convert_pmids(pmids) if pmids else {}
        for r in results:
            not_in_pmc = ""
            if r["pmid"]:
                conv = mapping.get(r["pmid"], {"error": "no answer from ID converter"})
                if "error" in conv:
                    not_in_pmc = conv["error"]
                else:
                    r["pmcid"] = conv["pmcid"]
            if r["pmcid"]:
                target, content, fetch = r["pmcid"], "full text", lambda: fetch_pmc_xml(r["pmcid"])
            else:
                target, content = "PMID" + r["pmid"], "abstract only"
                fetch = lambda: pubmed_to_jats(fetch_pubmed_xml(r["pmid"]))
            if target in self.index.docs and not replace:
                doc = self.index.documents[target]
                r.update(status="exists", doc_id=target, title=doc.title, content=content,
                         pmid=r["pmid"] or doc.pmid, message="already indexed (tick re-download to refresh)")
                continue
            try:
                xml_text = fetch()
            except ValueError as e:
                r["status"], r["message"] = "error", (f"{not_in_pmc}; PubMed: {e}" if not_in_pmc else str(e))
                continue
            existed = target in self.index.docs
            if existed:
                self.remove(target)
            try:
                ids = self.add_xml(xml_text, f"{target}.xml", persist=persist)
            except Exception as e:  # malformed XML etc.
                r["status"], r["message"] = "error", f"could not parse article: {e}"
                continue
            if not ids:
                r["status"], r["message"] = "error", "no article found in the XML"
                continue
            doc = self.index.documents[ids[0]]
            if content == "full text" and doc.pmid and "PMID" + doc.pmid in self.index.docs:
                self.remove("PMID" + doc.pmid)  # an older abstract-only copy of the same article
            r.update(doc_id=ids[0], title=doc.title, content=content, pmid=r["pmid"] or doc.pmid,
                     status="updated" if existed else "added")
            if content == "abstract only":
                r["message"] = "no full text in PMC – title and abstract indexed from PubMed"
            elif not doc.sections:
                r["message"] = "no full text body (front matter only)"
        return results

    # ------------------------------------------------------------------ #
    def query_analysis(self, query: str, node: Node | None) -> dict:
        """How the query was interpreted (for display)."""
        toks = self.tokenizer.tokenize(query.replace('"', " ").replace("(", " ").replace(")", " "))
        steps = []
        for t in toks:
            if t.norm.upper() in ("AND", "OR", "NOT") and t.text.isupper():
                steps.append({"word": t.text, "role": "operator"})
            elif t.norm in ("title", "abstract", "body"):
                steps.append({"word": t.text, "role": "field"})
            elif not t.stem:
                steps.append({"word": t.text, "role": "stopword"})
            else:
                steps.append({"word": t.text, "role": "term", "stem": t.stem, "df": self.index.df(t.stem)})
        return {"tokens": steps, "parsed": node.describe() if node else ""}

    def _sentences_with_offsets(self, doc: Document) -> list[tuple[str, str]]:
        cached = self._sent_cache.get(doc.doc_id)
        if cached is None:
            cached = [(fld, s) for fld, text in doc.units() for s in split_sentences(text)]
            self._sent_cache[doc.doc_id] = cached
        return cached

    def _snippet(self, doc: Document, stems: set[str], prefixes: set[str], max_sents: int = 2) -> str:
        best: list[tuple[int, int, str]] = []
        for i, (fld, sent) in enumerate(self._sentences_with_offsets(doc)):
            if fld == "title":
                continue
            spans = highlight_spans(sent, stems, prefixes, self.tokenizer)
            if spans:
                distinct = len({self.tokenizer.stemmer.stem(sent[s:e].lower()) for s, e in spans})
                best.append((distinct * 1000 + len(spans), i, sent))
        if not best:
            fallback = doc.abstract_text() or doc.body_text()
            return html.escape(fallback[:300]) + (ELLIPSIS if len(fallback) > 300 else "")
        best.sort(key=lambda x: (-x[0], x[1]))
        chosen = sorted(best[:max_sents], key=lambda x: x[1])
        parts = []
        for _, _, sent in chosen:
            if len(sent) > 400:
                sent = sent[:400] + "\u2026"
            parts.append(_mark(sent, highlight_spans(sent, stems, prefixes, self.tokenizer)))
        return (" " + ELLIPSIS + " ").join(p.replace("\u2026", ELLIPSIS) for p in parts)

    FIELD_BINS = {"title": 4, "abstract": 8, "body": 28}      # 40 bins in total

    def _match_map(self, doc_id: str, match) -> dict[str, list[int]]:
        """Hit density per field: title / abstract / body each get their own
        row of bins spanning that field's token range (start -> end)."""
        idoc = self.index.docs[doc_id]
        ranges = dict(idoc.field_ranges)
        if "heading" in ranges:                       # headings belong to the body
            b, h = ranges.get("body"), ranges["heading"]
            ranges["body"] = (min(b[0], h[0]), max(b[1], h[1])) if b else h
        out: dict[str, list[int]] = {}
        for fld, nbins in self.FIELD_BINS.items():
            rng = ranges.get(fld)
            bins = [0] * nbins
            if rng and rng[1] > rng[0]:
                span = rng[1] - rng[0]
                for positions in match.hits.values():
                    for p in positions:
                        if rng[0] <= p < rng[1]:
                            bins[min(int((p - rng[0]) * nbins / span), nbins - 1)] += 1
            out[fld] = bins
        return out

    def search(self, query: str, method: str = "bm25", default_op: str = "AND", limit: int = 50,
               scope: str = "all") -> dict:
        t0 = time.time()
        node, scored = self.searcher.search(query, method, default_op, scope)
        stems, prefixes = self.searcher.query_stems(node)
        results = []
        for doc_id, score, match, fscores in scored[:limit]:
            doc = self.index.documents[doc_id]
            st = self.stats[doc_id]
            title_spans = highlight_spans(doc.title, stems, prefixes, self.tokenizer)
            per_term = {t: len(p) for t, p in match.hits.items()}
            fields = Counter()
            term_fields: dict[str, Counter] = {}
            idoc = self.index.docs[doc_id]
            for term, positions in match.hits.items():
                tf_fields = term_fields.setdefault(term, Counter())
                for p in positions:
                    for fld, s, e in idoc.unit_offsets:
                        if s <= p < e:
                            f = "body" if fld == "heading" else fld
                            fields[f] += 1
                            tf_fields[f] += 1
                            break
            results.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "scores": {k: round(v, 4) for k, v in fscores.items()},
                "title_html": _mark(doc.title, title_spans),
                "journal": doc.journal, "year": doc.year,
                "authors": doc.authors[:4] + (["et al."] if len(doc.authors) > 4 else []),
                "snippet_html": self._snippet(doc, stems, prefixes),
                "term_hits": per_term,
                "term_field_hits": {t: dict(c) for t, c in term_fields.items()},
                "clause_hits": match.label_hits,
                "total_hits": match.n_hits(),
                "field_hits": dict(fields),
                "match_map": self._match_map(doc_id, match),
                "stats": {k: st[k] for k in ("characters", "words", "sentences", "unique_terms", "index_terms")},
                "field_stats": st["fields"],
            })
        return {
            "query": query, "method": method, "default_op": default_op,
            "scope": scope if scope in self.searcher.SCOPES else "all",
            "analysis": self.query_analysis(query, node),
            "total": len(scored), "time_ms": round((time.time() - t0) * 1000, 2),
            "results": results,
        }

    # ------------------------------------------------------------------ #
    def document(self, doc_id: str, query: str = "", default_op: str = "AND") -> dict | None:
        doc = self.index.documents.get(doc_id)
        if doc is None:
            return None
        stems: set[str] = set()
        prefixes: set[str] = set()
        node = None
        if query:
            from .search import parse_query
            node = parse_query(query, default_op)
            stems, prefixes = self.searcher.query_stems(node)
        units = []
        sent_no = 0
        total_hits = 0
        for fld, text in doc.units():
            sents = []
            for s in split_sentences(text):
                sent_no += 1
                spans = highlight_spans(s, stems, prefixes, self.tokenizer) if stems or prefixes else []
                total_hits += len(spans)
                sents.append({"n": sent_no, "html": _mark(s, spans), "hits": len(spans)})
            units.append({"field": fld, "sentences": sents})
        st = self.stats[doc_id]
        return {
            "meta": doc.meta(),
            "stats": st,
            "units": units,
            "query": query,
            "query_hits": total_hits,
            "query_terms": sorted(stems | {p + "*" for p in prefixes}),
            "term_freq_for_query": {s: self.index.docs[doc_id].term_freq.get(s, 0) for s in stems},
        }

    def corpus(self) -> dict:
        all_stats = list(self.stats.values())
        cs = corpus_stats(all_stats, self.index.df_counter(), self.index.cf_counter())
        cs["build_time_s"] = round(self.build_time, 2)
        cs["documents_list"] = [
            {**{k: v for k, v in self.index.documents[d].meta().items() if k != "authors"},
             "n_authors": len(self.index.documents[d].authors),
             "content": "full text" if self.index.documents[d].sections else "abstract only",
             **{k: self.stats[d][k] for k in ("characters", "words", "unique_words", "sentences",
                                               "sentences_naive", "paragraphs", "avg_words_per_sentence",
                                               "avg_word_length", "index_terms", "unique_terms")}}
            for d in sorted(self.index.documents)
        ]
        return cs

    def vocabulary(self, prefix: str = "", limit: int = 200) -> list[dict]:
        terms = self.index.terms_with_prefix(prefix.lower()) if prefix else list(self.index.postings)
        rows = [{"term": t, "df": self.index.df(t), "cf": self.index.cf(t)} for t in terms]
        rows.sort(key=lambda r: (-r["cf"], r["term"]))
        return rows[:limit]
