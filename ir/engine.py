"""High-level engine used by the web application and the CLI."""

from __future__ import annotations

import html
import json
import re
import random
import threading
import time
from collections import Counter
from operator import mul
from pathlib import Path

from . import zipf
from .index import InvertedIndex
from .ncbi import (convert_pmids, fetch_pmc_xml, fetch_pubmed_batch, fetch_pubmed_xml, parse_ids, pubmed_to_jats,
                   search_pubmed)
from .pmc_parser import Document, load_directory, load_jsonl, parse_file, parse_xml_string
from .search import Node, Searcher, highlight_spans, parse_query
from .sentence import split_sentences
from .spell import SpellIndex
from .stopwords import STOP_WORDS
from .stats import corpus_stats, document_stats
from .tokenizer import TOKEN_RE, Tokenizer
from .tfidf import DEFAULT_QUERY_SCHEME, DEFAULT_SCHEME, PRESETS, VectorSpace, describe_scheme, df_weight, parse_scheme, tf_weight
from .relevance import QuerySet
from . import evaluation
from .tsne import knn_preservation, pca2, tsne
from .word2vec import WORD_RE, Word2Vec, count_training_examples, neighbour_overlap, sentences_from_texts, training_examples

W2V_FILE = "word2vec.json"                     # legacy single-model file
W2V_DIR = "w2v"                                # data/w2v/<name>.json, one file per trained model
W2V_DEFAULT = "sg-d50-w5-m5-e10"               # active model at start-up when it exists
VSM_DIR = "vsm"                                # data/vsm/: Project #3 abstract collection(s) + evaluation queries
VSM_QUERIES = "queries_glp1.json"
# query words that are expanded by the engine itself (prefix* / fuzzy~2) are not spell-checked
_EXPANDED_WORD_RE = re.compile(r"\S*(?:\*|~[1-3]?)(?=[\s)]|$)")

MATCH_MAP_BINS = 40
ELLIPSIS = '<span class="ellip" title="omitted text between the best-matching sentences">…</span>'


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


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
        self.spell = SpellIndex()                      # surface words of the collection (spelling correction)
        self.searcher.fuzzy = lambda word, k: [r["word"] for r in self.spell.within(word, k or None, limit=30)]
        self._version = 0                              # bumped whenever the index changes
        self._zipf: tuple[int, dict] | None = None     # cached Zipf analysis
        self.w2v: Word2Vec | None = None               # active model
        self.w2v_meta: dict = {}
        self.w2v_models: dict[str, tuple[Word2Vec, dict]] = {}
        self.w2v_jobs: list[dict] = []                 # training jobs, oldest first
        self._w2v_job_seq = 0
        self._w2v_queue: list = []
        self._w2v_worker: threading.Thread | None = None
        self._w2v_lock = threading.Lock()
        self._w2v_sent: dict = {}                      # cached pre-processed sentences
        self._w2v_cache: dict = {}                     # projections, evaluations, vocabulary stats
        self._vsm: dict = {}                           # Project #3 vector space models by (options), see vsm()
        self._vsm_queries: QuerySet | None = None
        self._vsm_eval_cache: dict = {}

    # ------------------------------------------------------------------ #
    def load(self) -> int:
        t0 = time.time()
        docs = load_directory(self.data_dir) if self.data_dir.exists() else []
        for f in sorted(self.data_dir.glob("*.jsonl")) if self.data_dir.exists() else []:
            docs.extend(load_jsonl(f))                 # PubMed abstract collections
        for d in docs:
            self._add(d)
        self.build_time = time.time() - t0
        self._load_w2v_models()
        return self.index.n_docs

    @staticmethod
    def _word_counts(doc: Document) -> Counter:
        """Lower-cased surface words of a document (the spelling dictionary)."""
        return Counter(w for w in TOKEN_RE.findall(doc.full_text().lower()) if len(w) > 1 and not w.isdigit())

    def _add(self, doc: Document) -> None:
        if doc.doc_id in self.index.docs:
            self.spell.remove(self._word_counts(self.index.documents[doc.doc_id]))
        self.index.add_document(doc)
        self.stats[doc.doc_id] = document_stats(doc, self.tokenizer)
        self._sent_cache.pop(doc.doc_id, None)
        self.spell.add(self._word_counts(doc))
        self._version += 1

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
        doc = self.index.documents[doc_id]
        source = doc.source
        self.spell.remove(self._word_counts(doc))
        self.index.remove_document(doc_id)
        self.stats.pop(doc_id, None)
        self._sent_cache.pop(doc_id, None)
        self._version += 1
        if delete_file and source:
            path = Path(source)
            try:
                if path.is_file() and self.data_dir.resolve() in path.resolve().parents:
                    if path.suffix == ".jsonl":      # a collection file: drop only this record
                        keep = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                                if ln.strip() and "PMID" + json.loads(ln)["pmid"] != doc_id]
                        path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
                    else:
                        path.unlink()
            except OSError:
                pass
        return True

    def collect(self, term: str, n: int = 1000, name: str = "", progress=None) -> dict:
        """Build a PubMed abstract collection: esearch *term*, efetch the
        newest *n* English records that have an abstract, save them as
        data/pubmed_<name>.jsonl (replacing an older collection of that
        name) and index them."""
        n = max(1, min(int(n), 5000))
        pmids, total = search_pubmed(term, n + max(20, n // 25))   # a few spare: some records have no abstract text
        if not pmids:
            raise ValueError(f"PubMed has no English abstracts for: {term}")
        records = fetch_pubmed_batch(pmids, progress=progress)[:n]
        slug = re.sub(r"[^a-z0-9]+", "_", (name or term).lower()).strip("_")[:40] or "collection"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / f"pubmed_{slug}.jsonl"
        for doc_id in [d for d, doc in self.index.documents.items() if doc.source == str(path)]:
            self.remove(doc_id)
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps({**r, "query": term}, ensure_ascii=False) + "\n")
        docs = load_jsonl(path)
        for d in docs:
            self._add(d)
        return {"query": term, "pubmed_matches": total, "requested": n, "fetched": len(docs),
                "file": path.name, "documents": self.index.n_docs}

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
        toks = self.tokenizer.tokenize(re.sub(r"~[1-3]?(?=[\s)]|$)", " ", query).replace('"', " ").replace("(", " ").replace(")", " "))
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
        fuzzy = []

        def walk(n: Node | None) -> None:
            if n is None:
                return
            if n.kind == "FUZZY":
                fuzzy.append({"term": n.describe(), "words": self.spell.within(n.value, n.dist or None, limit=30)})
            for c in n.children:
                walk(c)

        walk(node)
        return {"tokens": steps, "parsed": node.describe() if node else "", "fuzzy": fuzzy}

    def spell_check(self, query: str) -> dict:
        """Query words that are not in the collection, each with its closest
        vocabulary words (edit distance, most frequent first), and the query
        with every such word replaced by its best suggestion."""
        plain = _EXPANDED_WORD_RE.sub(" ", query)
        plain = re.sub(r"\b(?:title|abstract|body):", " ", plain, flags=re.I)
        corrections, corrected = [], query
        seen = set()
        for t in self.tokenizer.tokenize(plain):
            w = t.norm
            if (not t.stem or w in seen or w in self.spell or len(w) < 3 or w.isdigit()
                    or (t.text.isupper() and w in ("and", "or", "not")) or self.index.df(t.stem)):
                continue
            seen.add(w)
            sugg = self.spell.suggest(w, 5)
            corrections.append({"word": t.text, "suggestions": sugg})
            if sugg:
                corrected = re.sub(rf"(?<![\w]){re.escape(t.text)}(?![\w*~])", sugg[0]["word"], corrected, flags=re.I)
        return {"corrections": corrections,
                "corrected_query": corrected if corrected != query else ""}

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
               scope: str = "all", autocorrect: bool = True) -> dict:
        """With *autocorrect*, a query that matches nothing and contains
        words unknown to the collection is re-run with the spelling
        corrections applied (reported in result["spell"])."""
        t0 = time.time()
        node, scored = self.searcher.search(query, method, default_op, scope)
        spell = self.spell_check(query)
        spell["auto"] = False
        if autocorrect and not scored and spell["corrected_query"]:
            node2, scored2 = self.searcher.search(spell["corrected_query"], method, default_op, scope)
            if scored2:
                spell.update(auto=True, original_query=query)
                query, node, scored = spell["corrected_query"], node2, scored2
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
            "analysis": self.query_analysis(query, node), "spell": spell,
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

    # ------------------------------------------------------------------ #
    # Project #2: Zipf analysis, word2vec
    # ------------------------------------------------------------------ #
    def collection_docs(self) -> tuple[str, list[Document]]:
        """The collection analysed by the Zipf / word2vec tabs: the PubMed
        abstract collection(s) when present, otherwise every indexed
        document.  Only title + abstract are used, so every document
        contributes the same kind of text."""
        docs = [d for d in self.index.documents.values() if d.article_type == "pubmed-abstract"]
        if docs:
            names = sorted({Path(d.source).stem for d in docs})
            return "PubMed abstract collection (" + ", ".join(names) + ")", sorted(docs, key=lambda d: d.doc_id)
        return "all indexed documents", sorted(self.index.documents.values(), key=lambda d: d.doc_id)

    def collection_texts(self) -> list[str]:
        return [d.title + "\n" + d.abstract_text() for d in self.collection_docs()[1]]

    def _zipf_state(self) -> dict:
        if self._zipf is None or self._zipf[0] != self._version:
            name, docs = self.collection_docs()
            texts = self.collection_texts()
            cols = zipf.analyse(texts)
            self._zipf = (self._version, {
                "name": name, "cols": cols,
                "summaries": [cols[k].summary() for k, _, _ in zipf.CONDITIONS],
                "stemming": zipf.stemming_groups(texts),
                "sample_ids": [d.doc_id for d in docs[:5]],
            })
        return self._zipf[1]

    def zipf(self) -> dict:
        st = self._zipf_state()
        return {"collection": st["name"], "sample_ids": st["sample_ids"], "conditions": st["summaries"],
                "stemming": st["stemming"]}

    # terms compared in the CF-vs-DF / IDF tables when the user gives none
    DEFAULT_TERMS = ("the of patients glp diabetes obesity weight semaglutide liraglutide tirzepatide insulin glucose "
                     "cardiovascular receptor agonists kidney mice cells placebo nausea pancreatitis retinopathy "
                     "alzheimer bariatric surgery").split()

    def zipf_terms(self, condition: str = "C", terms: str = "", scatter: bool = False) -> dict:
        col = self._zipf_state()["cols"].get(condition.upper()) or self._zipf_state()["cols"]["C"]
        words = [w for w in re.split(r"[\s,;]+", terms.strip()) if w] or list(self.DEFAULT_TERMS)
        rows, missing, seen = [], [], set()
        for w in words:
            t = col.normalise(w)
            if t and t in col.cf:
                if t not in seen:
                    seen.add(t)
                    rows.append(col.row(t))
            else:
                missing.append(w)
        if not terms.strip() and len(rows) < 20:           # collection on another topic: pick terms automatically
            for t in col.auto_terms():
                if t not in seen and len(rows) < 24:
                    seen.add(t)
                    rows.append(col.row(t))
        out = {"condition": col.key, "documents": col.n_docs, "rows": rows, "missing": missing}
        if scatter:
            out["scatter"] = col.scatter()
        return out

    def zipf_resolving(self, condition: str = "C", upper: int | None = None, lower: int | None = None) -> dict:
        """Resolving power of significant words (Luhn): CF x IDF over the
        rank axis with an upper / lower cut-off (None = automatic)."""
        cols = self._zipf_state()["cols"]
        return (cols.get(condition.upper()) or cols["C"]).resolving(upper, lower)

    def zipf_cfdf(self, condition: str = "B", **thresholds) -> dict:
        """CF-DF map: the vocabulary sorted into function / boilerplate /
        topic / keyword / number zones, with the Poisson reference curve."""
        cols = self._zipf_state()["cols"]
        return (cols.get(condition.upper()) or cols["B"]).cfdf_map(**thresholds)

    # ---- word2vec ------------------------------------------------------ #
    # Several trained models live side by side in data/w2v/<name>.json so
    # that architectures (CBOW vs skip-gram), hyper-parameters and random
    # seeds can be compared.  `self.w2v` is the *active* model (the one the
    # simple endpoints use when no model name is given).
    def _w2v_dir(self) -> Path:
        return self.data_dir / W2V_DIR

    def _load_w2v_models(self) -> None:
        self.w2v_models = {}
        paths = sorted(self._w2v_dir().glob("*.json")) if self._w2v_dir().is_dir() else []
        legacy = self.data_dir / W2V_FILE
        if legacy.is_file():
            paths.append(legacy)
        for path in paths:
            try:
                m, meta = Word2Vec.load(path)
            except (ValueError, KeyError, OSError) as e:
                print(f"[warn] could not load {path}: {e}")
                continue
            name = path.stem if path != legacy else m.name()
            meta = {**meta, "name": name, "file": str(path.relative_to(self.data_dir))}
            self.w2v_models.setdefault(name, (m, meta))
        if self.w2v_models:
            name = self.w2v_meta.get("name") if self.w2v_meta.get("name") in self.w2v_models else None
            if name is None:                       # default: the baseline model, else the most recently trained
                name = W2V_DEFAULT if W2V_DEFAULT in self.w2v_models else max(self.w2v_models, key=lambda n: self.w2v_models[n][1].get("trained", ""))
            self.w2v, self.w2v_meta = self.w2v_models[name]
        else:
            self.w2v, self.w2v_meta = None, {}

    def w2v_model(self, name: str | None = None) -> Word2Vec:
        """The named model, or the active one."""
        if name:
            if name not in self.w2v_models:
                raise ValueError(f"no model called '{name}' (available: {', '.join(sorted(self.w2v_models)) or 'none'})")
            return self.w2v_models[name][0]
        if not self.w2v:
            raise ValueError("no word2vec model yet - train one first")
        return self.w2v

    def w2v_select(self, name: str) -> dict:
        self.w2v_model(name)
        self.w2v, self.w2v_meta = self.w2v_models[name]
        return {"active": name}

    def w2v_delete(self, name: str) -> dict:
        self.w2v_model(name)
        _, meta = self.w2v_models.pop(name)
        try:
            (self.data_dir / meta["file"]).unlink()
        except OSError:
            pass
        if self.w2v_meta.get("name") == name:
            self.w2v_meta = {}
            self._load_w2v_models()
        return {"deleted": name, "active": self.w2v_meta.get("name")}

    def _w2v_sentences(self, remove_stopwords: bool = True) -> list[list[str]]:
        key = (self._version, remove_stopwords)
        if key not in self._w2v_sent:
            self._w2v_sent.clear()
            self._w2v_sent[key] = sentences_from_texts(self.collection_texts(), remove_stopwords)
        return self._w2v_sent[key]

    @staticmethod
    def _w2v_params(params: dict) -> Word2Vec:
        clamp = lambda k, lo, hi, d: max(lo, min(hi, int(params.get(k, d) or d)))  # noqa: E731
        flag = lambda v: str(v).lower() not in ("0", "false", "no", "off", "")  # noqa: E731
        return Word2Vec(sg=str(params.get("model", "sg")).lower() in ("sg", "skip-gram", "skipgram"),
                        dim=clamp("dim", 10, 300, 50), window=clamp("window", 1, 15, 5),
                        negative=clamp("negative", 1, 20, 5), min_count=clamp("min_count", 1, 100, 5),
                        epochs=clamp("epochs", 1, 50, 5), seed=clamp("seed", 1, 10 ** 6, 1),
                        remove_stopwords=flag(params.get("remove_stopwords", True)))

    def w2v_train(self, params: dict, block: bool = False) -> dict:
        """Queue a training job (several can be queued; they run one after
        another in a background thread).  With *block* the job runs now."""
        model = self._w2v_params(params)
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(params.get("name", "")).strip())[:60] or model.name()
        name_, docs = self.collection_docs()
        job = {"id": self._w2v_job_seq + 1, "name": name, "state": "queued", "progress": 0.0, "epoch": 0, "loss": None,
               "params": model.params(), "documents": len(docs), "collection": name_}
        self._w2v_job_seq += 1
        self.w2v_jobs.append(job)
        self.w2v_jobs[:] = self.w2v_jobs[-20:]

        def run() -> None:
            job.update(state="running", started=time.time())
            try:
                sentences = self._w2v_sentences(model.remove_stopwords)
                job["sentences"] = len(sentences)

                def progress(frac: float, epoch: int, alpha: float, loss: float) -> None:
                    job.update(progress=round(frac, 4), epoch=epoch, loss=round(loss, 4), alpha=round(alpha, 5))

                model.train(sentences, progress)
                meta = {"name": name, "collection": name_, "documents": len(docs), "sentences": len(sentences),
                        "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "file": f"{W2V_DIR}/{name}.json"}
                self._w2v_dir().mkdir(parents=True, exist_ok=True)
                model.save(self._w2v_dir() / f"{name}.json", {k: v for k, v in meta.items() if k != "file"})
                self.w2v_models[name] = (model, meta)
                self.w2v, self.w2v_meta = model, meta
                self._w2v_cache.clear()
                job.update(state="done", progress=1.0, finished=time.time())
            except Exception as e:  # reported to the UI
                job.update(state="error", message=f"{type(e).__name__}: {e}", finished=time.time())

        if block:
            run()
            return job
        with self._w2v_lock:
            self._w2v_queue.append(run)
            if not self._w2v_worker or not self._w2v_worker.is_alive():
                def worker() -> None:
                    while True:
                        with self._w2v_lock:
                            if not self._w2v_queue:
                                return
                            fn = self._w2v_queue.pop(0)
                        fn()
                self._w2v_worker = threading.Thread(target=worker, daemon=True)
                self._w2v_worker.start()
        return job

    def w2v_status(self) -> dict:
        jobs = []
        for j in self.w2v_jobs:
            j = dict(j)
            if j.get("started"):
                j["elapsed"] = round((j.get("finished") or time.time()) - j.pop("started"), 1)
            j.pop("finished", None)
            jobs.append(j)
        running = next((j for j in jobs if j["state"] == "running"), None)
        models = [{**m.info(), **meta, "final_loss": m.loss_history[-1] if m.loss_history else None,
                   "pairs_per_epoch": self._w2v_pairs(m)}
                  for name, (m, meta) in sorted(self.w2v_models.items())]
        out = {"jobs": jobs, "job": running or (jobs[-1] if jobs else {"state": "idle"}),
               "queued": sum(j["state"] == "queued" for j in jobs),
               "models": models, "active": self.w2v_meta.get("name"), "model": None,
               "collection": self.collection_docs()[0], "documents": len(self.collection_docs()[1])}
        if self.w2v:
            out["model"] = {**self.w2v.info(), **self.w2v_meta,
                            "top_words": list(zip(self.w2v.vocab[:40], self.w2v.counts[:40]))}
        return out

    def _w2v_pairs(self, m: Word2Vec) -> int | None:
        """Expected number of training examples per epoch (dynamic window)."""
        key = ("pairs", m.window, m.remove_stopwords, m.min_count, self._version)
        if key not in self._w2v_cache:
            sentences = [[w for w in s if w in m.index] for s in self._w2v_sentences(m.remove_stopwords)]
            c = count_training_examples(sentences, m.window)
            self._w2v_cache[key] = c["skipgram_dynamic"] if m.sg else c["cbow"]
        return self._w2v_cache[key]

    def _w2v_check(self, m: Word2Vec, *words: str) -> None:
        for w in words:
            if w not in m:
                near = [r["word"] for r in self.spell.within(w.lower(), None) if r["word"] in m][:5]
                hint = f" - did you mean: {', '.join(near)}?" if near else ""
                raise ValueError(f"'{w}' is not in the model vocabulary{hint}")

    def w2v_similar(self, word: str, topn: int = 15, model: str | None = None) -> dict:
        m = self.w2v_model(model)
        self._w2v_check(m, word)
        return {"word": word.lower(), "count": m.counts[m.index[word.lower()]], "model": self._w2v_name(m),
                "neighbours": [{"word": w, "similarity": s, "count": m.counts[m.index[w]]} for w, s in m.most_similar(word, topn)]}

    def _w2v_name(self, m: Word2Vec) -> str:
        return next((n for n, (mm, _) in self.w2v_models.items() if mm is m), m.name())

    def w2v_analogy(self, a: str, b: str, c: str, topn: int = 10, model: str | None = None) -> dict:
        m = self.w2v_model(model)
        self._w2v_check(m, a, b, c)
        return {"a": a, "b": b, "c": c, "model": self._w2v_name(m),
                "results": [{"word": w, "similarity": s} for w, s in m.analogy(a, b, c, topn)]}

    @staticmethod
    def _words(text: str) -> list[str]:
        seen: dict[str, None] = {}
        for w in re.split(r"[\s,;]+", text.strip().lower()):
            if w:
                seen.setdefault(w, None)
        return list(seen)

    def w2v_compare(self, words: str, models: str = "", n: int = 5) -> dict:
        """Nearest neighbours of each word in every model, side by side, with
        the overlap of the neighbour sets between the models."""
        names = [x for x in self._words(models) if x] or sorted(self.w2v_models)
        ms = [(nm, self.w2v_model(nm)) for nm in names]
        if not ms:
            raise ValueError("no word2vec model yet - train one first")
        rows = []
        for w in self._words(words)[:40]:
            per = {}
            for nm, m in ms:
                if w in m:
                    per[nm] = {"count": m.counts[m.index[w]],
                               "neighbours": [{"word": x, "similarity": s} for x, s in m.most_similar(w, n)]}
                else:
                    per[nm] = None
            overlap = {}
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = per[ms[i][0]], per[ms[j][0]]
                    if a and b:
                        sa, sb = {x["word"] for x in a["neighbours"]}, {x["word"] for x in b["neighbours"]}
                        overlap[f"{ms[i][0]}|{ms[j][0]}"] = {"shared": sorted(sa & sb), "jaccard": round(len(sa & sb) / len(sa | sb), 3)}
            rows.append({"word": w, "models": per, "overlap": overlap})
        return {"n": n, "models": [{"name": nm, **m.params()} for nm, m in ms], "words": rows}

    def w2v_pairs(self, groups: str, model: str | None = None) -> dict:
        """Cosine similarity between every pair of the given words, grouped
        (groups separated by ';'), with the mean within-group and
        between-group similarity: does the model keep related words together?"""
        m = self.w2v_model(model)
        parsed = [self._words(g) for g in groups.split(";") if g.strip()]
        words, group_of, missing = [], {}, []
        for gi, g in enumerate(parsed):
            for w in g:
                if w in m and w not in group_of:
                    words.append(w)
                    group_of[w] = gi
                elif w not in m:
                    missing.append(w)
        if len(words) < 2:
            raise ValueError("fewer than two of the words are in the model vocabulary" + (f" (missing: {', '.join(missing)})" if missing else ""))
        _, vecs = m.vectors(words)
        matrix = [[round(sum(map(mul, a, b)), 4) for b in vecs] for a in vecs]
        within, between = [], []
        pairs = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                same = group_of[words[i]] == group_of[words[j]]
                (within if same else between).append(matrix[i][j])
                pairs.append({"a": words[i], "b": words[j], "similarity": matrix[i][j], "same_group": same})
        pairs.sort(key=lambda p: -p["similarity"])
        # the background: similarity of random vocabulary pairs
        rng = random.Random(0)
        base = [m.similarity(rng.choice(m.vocab), rng.choice(m.vocab)) for _ in range(400)]
        per_group = []
        for gi, g in enumerate(parsed):
            ws = [w for w in words if group_of[w] == gi]
            sims = [matrix[words.index(a)][words.index(b)] for a in ws for b in ws if a < b]
            per_group.append({"words": ws, "mean_within": round(sum(sims) / len(sims), 4) if sims else None})
        return {"model": self._w2v_name(m), "words": words, "group": [group_of[w] for w in words], "groups": per_group,
                "counts": [m.counts[m.index[w]] for w in words], "matrix": matrix, "missing": missing,
                "mean_within": round(sum(within) / len(within), 4) if within else None,
                "mean_between": round(sum(between) / len(between), 4) if between else None,
                "mean_random": round(sum(base) / len(base), 4), "pairs": pairs}

    def w2v_projection(self, words: str = "", n: int = 120, model: str | None = None, method: str = "pca",
                       perplexity: float = 15.0, iters: int = 400, seed: int = 0, neighbours: int = 12) -> dict:
        """2-D map of word vectors by PCA or t-SNE.  With *words*: each given
        word plus its nearest neighbours (one colour group per seed word);
        otherwise the n most frequent words.  Also reports how well the map
        preserves the high-dimensional neighbourhoods (k-NN preservation)."""
        m = self.w2v_model(model)
        seeds = self._words(words)
        if seeds:
            self._w2v_check(m, *seeds)
            group: dict[str, str] = {}
            for s in seeds:
                group.setdefault(s, s)
                for w, _ in m.most_similar(s, neighbours):
                    group.setdefault(w, s)
            chosen = list(group)
        else:
            chosen = m.vocab[:max(10, min(n, 400))]
            group = {}
        found, rows = m.vectors(chosen)
        if len(found) < 4:
            raise ValueError("need at least 4 words")
        method = method.lower()
        key = ("proj", self._w2v_name(m), method, tuple(found), perplexity, iters, seed)
        if key not in self._w2v_cache:
            t0 = time.time()
            if method == "tsne":
                r = tsne(rows, perplexity=perplexity, iters=max(50, min(iters, 1000)), seed=seed)
                pts, extra = r["points"], {"kl": r["kl"], "perplexity": r["perplexity"], "iters": r["iters"], "learning_rate": round(r["learning_rate"], 1)}
            else:
                pts, ratios = pca2(rows)
                extra = {"explained_variance": ratios}
            quality = knn_preservation(rows, pts)
            self._w2v_cache[key] = (pts, extra, quality, round(time.time() - t0, 2))
            if len(self._w2v_cache) > 60:
                self._w2v_cache.pop(next(iter(self._w2v_cache)))
        pts, extra, quality, secs = self._w2v_cache[key]
        points = [{"word": w, "count": m.counts[m.index[w]], "x": round(x, 4), "y": round(y, 4),
                   "group": group.get(w, ""), "seed": w in seeds, "knn": quality["per_point"][i]}
                  for i, (w, (x, y)) in enumerate(zip(found, pts))]
        return {"model": self._w2v_name(m), "method": method, "points": points, "groups": seeds, "seconds": secs,
                "knn_preservation": {k: v for k, v in quality.items() if k != "per_point"}, **extra}

    def w2v_vocab(self, min_count: int = 5, remove_stopwords: bool = True) -> dict:
        """Pre-processing and vocabulary statistics of the training corpus:
        documents, sentences, tokens, types, what a minimum count keeps,
        the most frequent words and how many training examples a window
        produces.  Also one sample document walked through the pipeline."""
        min_count = max(1, min(int(min_count), 100))
        key = ("vocab", self._version, min_count, remove_stopwords)
        if key in self._w2v_cache:
            return self._w2v_cache[key]
        name, docs = self.collection_docs()
        sentences = self._w2v_sentences(remove_stopwords)
        counts = Counter(w for s in sentences for w in s)
        tokens = sum(counts.values())
        kept = {w: c for w, c in counts.items() if c >= min_count}
        thresholds = []
        for t in (1, 2, 3, 5, 10, 20, 50):
            ks = sum(1 for c in counts.values() if c >= t)
            thresholds.append({"min_count": t, "types": ks, "types_share": round(ks / len(counts), 4),
                               "tokens": sum(c for c in counts.values() if c >= t),
                               "token_share": round(sum(c for c in counts.values() if c >= t) / tokens, 4)})
        top = counts.most_common(40)
        hapax = sum(1 for c in counts.values() if c == 1)
        # one document through the pipeline
        sample = docs[0] if docs else None
        walk = None
        if sample:
            sents = [s for line in (sample.title, sample.abstract_text()) for s in split_sentences(line)]
            steps = []
            for sent in sents[:3]:
                raw = WORD_RE.findall(sent)
                lower = [w.lower() for w in raw]
                nostop = [w for w in lower if w not in STOP_WORDS and len(w) > 1]
                steps.append({"sentence": sent, "tokens": raw, "lower": lower, "no_stopwords": nostop,
                              "kept": [w for w in (nostop if remove_stopwords else [w for w in lower if len(w) > 1]) if counts.get(w, 0) >= min_count]})
            walk = {"doc_id": sample.doc_id, "title": sample.title, "sentences": len(sents), "steps": steps}
        kept_sent = [[w for w in s if w in kept] for s in sentences]
        examples = [count_training_examples(kept_sent, w) for w in (2, 5, 10)]
        out = {"collection": name, "documents": len(docs), "sentences": len(sentences), "tokens": tokens,
               "types": len(counts), "hapax": hapax, "hapax_share": round(hapax / len(counts), 4),
               "min_count": min_count, "stopwords_removed": remove_stopwords,
               "vocabulary": len(kept), "vocabulary_tokens": sum(kept.values()),
               "coverage": round(sum(kept.values()) / tokens, 4), "thresholds": thresholds,
               "avg_sentence_length": round(tokens / len(sentences), 2) if sentences else 0,
               "top_words": top, "examples": examples, "walk": walk,
               "stopword_tokens": None}
        if remove_stopwords:
            all_tokens = sum(len(s) for s in self._w2v_sentences(False))
            out["stopword_tokens"] = all_tokens - tokens
            out["tokens_with_stopwords"] = all_tokens
        self._w2v_cache[key] = out
        return out

    def w2v_examples(self, text: str, window: int = 2, remove_stopwords: bool = True, centre: int | None = None) -> dict:
        """Training examples (skip-gram pairs and CBOW examples) generated
        from one sentence of text, after the same pre-processing as training."""
        window = max(1, min(int(window), 15))
        sents = sentences_from_texts([text[:2000]], remove_stopwords) or [[]]
        words = sents[0]
        if len(words) < 2:
            raise ValueError("the text has fewer than two words after pre-processing")
        return {**training_examples(words, window, centre), "sentences": len(sents), "original": text[:2000]}

    def w2v_eval(self, models: str = "", k: int = 10) -> dict:
        names = self._words(models) or ([self.w2v_meta["name"]] if self.w2v_meta.get("name") else [])
        out = []
        for nm in names:
            m = self.w2v_model(nm)
            key = ("eval", nm, k)
            if key not in self._w2v_cache:
                self._w2v_cache[key] = m.evaluate(k)
            out.append({"name": nm, **m.params(), "vocabulary": len(m.vocab), **self._w2v_cache[key]})
        if not out:
            raise ValueError("no word2vec model yet - train one first")
        return {"k": k, "models": out}

    def w2v_overlap(self, models: str, k: int = 10, words: str = "") -> dict:
        names = self._words(models)
        if len(names) != 2:
            raise ValueError("give exactly two model names, e.g. models=sg-d50-w5-m5-e10,cbow-d50-w5-m5-e10")
        a, b = self.w2v_model(names[0]), self.w2v_model(names[1])
        ws = self._words(words) or None
        key = ("overlap", names[0], names[1], k, tuple(ws or []))
        if key not in self._w2v_cache:
            self._w2v_cache[key] = neighbour_overlap(a, b, k, words=ws)
        return {"a": names[0], "b": names[1], **self._w2v_cache[key]}

    # ------------------------------------------------------------------ #
    # Project #3: term weighting / vector space model
    # ------------------------------------------------------------------ #
    def _vsm_dir(self) -> Path:
        return self.data_dir / VSM_DIR

    def vsm_docs(self) -> tuple[str, list[Document]]:
        """The collection of the TF-IDF tab: data/vsm/*.jsonl when present
        (the 3,000-abstract GLP-1 collection with MeSH headings), otherwise
        the collection of the Zipf / word2vec tabs."""
        files = sorted(self._vsm_dir().glob("*.jsonl")) if self._vsm_dir().is_dir() else []
        if files:
            docs = []
            for f in files:
                docs.extend(load_jsonl(f))
            return "PubMed abstract collection (" + ", ".join(f.stem for f in files) + ")", docs
        return self.collection_docs()

    def vsm(self, remove_stopwords: bool = True, stemming: bool = True) -> VectorSpace:
        key = (remove_stopwords, stemming)
        hit = self._vsm.get(key)
        if hit is None or (not self._vsm_dir().is_dir() and hit[0] != self._version):
            name, docs = self.vsm_docs()
            t0 = time.time()
            vs = VectorSpace(docs, remove_stopwords=remove_stopwords, stemming=stemming)
            vs.name, vs.build_time = name, time.time() - t0            # type: ignore[attr-defined]
            self._vsm[key] = (self._version, vs)
            self._vsm_eval_cache = {}
            if key == (True, True):
                self._vsm_queries = None
            hit = self._vsm[key]
        return hit[1]

    def vsm_queries(self) -> QuerySet:
        if self._vsm_queries is None:
            self._vsm_queries = QuerySet(self._vsm_dir() / VSM_QUERIES, self.vsm().docs)
        return self._vsm_queries

    def vsm_info(self) -> dict:
        vs = self.vsm()
        st = vs.vocabulary_stats()
        tf_curve = [{"tf": tf, "raw": tf, "log": round(tf_weight("l", tf), 4), "aug_max10": round(tf_weight("a", tf, 10), 4),
                     "boolean": 1} for tf in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]]
        idf_curve = [{"df": df, "idf": round(df_weight("t", df, vs.N), 4), "prob_idf": round(df_weight("p", df, vs.N), 4),
                      "share": round(df / vs.N, 4)} for df in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, vs.N] if df <= vs.N]
        return {"collection": getattr(vs, "name", ""), "build_time_s": round(getattr(vs, "build_time", 0.0), 2),
                "stopwords_removed": vs.tokenizer.remove_stopwords, "stemming": vs.tokenizer.stemming,
                **st, "tf_curve": tf_curve, "idf_curve": idf_curve,
                "schemes": [{"scheme": k, "name": n, "formula": f, **({} if k == "bm25" else describe_scheme(k))} for k, n, f in PRESETS],
                "default_scheme": DEFAULT_SCHEME, "default_qscheme": DEFAULT_QUERY_SCHEME,
                "sample_ids": vs.order[:5], "labelled": len(self.vsm_queries().judged()),
                "with_mesh": sum(1 for d in vs.docs.values() if d.mesh), "with_keywords": sum(1 for d in vs.docs.values() if d.keywords)}

    def vsm_terms(self, words: str = "") -> dict:
        vs = self.vsm()
        ws = self._words(words)
        if not ws:
            ws = ["glp", "receptor", "agonist", "diabetes", "obesity", "semaglutide", "tirzepatide", "liraglutide", "kidney",
                  "microbiota", "alzheimer", "pcos", "apnea", "cost", "mice", "patients", "study", "nausea"]
        return {"documents": vs.N, "rows": vs.analyse_terms(ws)}

    def vsm_doc(self, doc_id: str, schemes: str = "", top: int = 15) -> dict:
        vs = self.vsm()
        if not doc_id:                                  # default: the labelled document with the most repeated term (shows the tf effect best)
            core = [d for d in vs.order if any(m.get("d") in ("Semaglutide", "Glucagon-Like Peptide-1 Receptor Agonists") for m in vs.docs[d].mesh)]
            doc_id = max(core or vs.order, key=lambda d: (vs.max_tf[d], d))
        if doc_id.isdigit():
            doc_id = "PMID" + doc_id
        if doc_id not in vs.docs:
            raise ValueError(f"document {doc_id} is not in the collection")
        names = self._words(schemes) or ["nnn", "lnn", "ntn", "ltn", "ltc"]
        doc = vs.docs[doc_id]
        return {"doc_id": doc_id, "title": doc.title, "year": doc.year, "journal": doc.journal, "abstract": doc.abstract,
                "keywords": doc.keywords, "mesh": doc.mesh, "length": vs.length[doc_id], "tokens": vs.n_tokens[doc_id],
                "unique_terms": len(vs.tf[doc_id]), "max_tf": vs.max_tf[doc_id],
                "vectors": [vs.doc_vector(doc_id, s, top) for s in names if parse_scheme(s)[0] != "bm25"]}

    def _vsm_labels(self, query_text: str, doc_ids: list[str]) -> tuple[str, dict[str, int], dict[str, list[str]]]:
        qs = self.vsm_queries()
        q = qs.by_text(query_text)
        if q is None:
            return "", {}, {}
        rel = qs.qrels(q.id)
        return q.id, {d: rel.get(d, 0) for d in doc_ids}, {d: qs.matched_labels(q.id, d) for d in doc_ids}

    def vsm_search(self, query: str, scheme: str = DEFAULT_SCHEME, qscheme: str = DEFAULT_QUERY_SCHEME, k: int = 10) -> dict:
        if not query.strip():
            raise ValueError("empty query")
        vs = self.vsm()
        t0 = time.time()
        res = vs.search(query, scheme, qscheme, k)
        qid, grades, matched = self._vsm_labels(query, [r["doc_id"] for r in res["results"]])
        for r in res["results"]:
            d = vs.docs[r["doc_id"]]
            r.update(abstract=d.abstract, keywords=d.keywords, mesh=[m["d"] + ("*" if m.get("major") else "") for m in d.mesh],
                     grade=grades.get(r["doc_id"], 0) if qid else None, matched_labels=matched.get(r["doc_id"], []))
        res["time_ms"] = round((time.time() - t0) * 1000, 2)
        res["query_id"] = qid
        res["collection"] = getattr(vs, "name", "")
        if qid:
            res["measures"] = evaluation.evaluate_run(vs.ranking_ids(query, scheme, qscheme), self.vsm_queries().qrels(qid), ks=(5, 10, 20))
        return res

    def vsm_compare(self, query: str, schemes: str = "", qscheme: str = DEFAULT_QUERY_SCHEME, k: int = 10) -> dict:
        """The same query under several schemes: top-k lists, overlap / Kendall
        tau against the first scheme, mean length of the top-k documents."""
        if not query.strip():
            raise ValueError("empty query")
        vs = self.vsm()
        names = self._words(schemes) or ["nnn", "lnn", "ltn", "ltc"]
        runs = {s: vs.rank(query, s, qscheme)[0] for s in names}
        ids = {s: [d for d, _ in r] for s, r in runs.items()}
        base = names[0]
        qid, grades, _ = self._vsm_labels(query, sorted({d for r in ids.values() for d in r[:k]}))
        qrels = self.vsm_queries().qrels(qid) if qid else {}
        out = []
        for s in names:
            top = runs[s][:k]
            out.append({"scheme": s, "info": describe_scheme(s), "total": len(runs[s]),
                        "overlap": round(evaluation.overlap_at_k(ids[base], ids[s], k), 3),
                        "tau": round(evaluation.kendall_tau(ids[base], ids[s], k), 3),
                        "mean_length": round(sum(vs.length[d] for d, _ in top) / max(len(top), 1), 1),
                        "mean_max_tf": round(sum(vs.max_tf[d] for d, _ in top) / max(len(top), 1), 1),
                        "measures": evaluation.evaluate_run(ids[s], qrels, ks=(5, 10, 20)) if qid else None,
                        "top": [{"rank": i, "doc_id": d, "score": round(sc, 5), "title": vs.docs[d].title, "length": vs.length[d],
                                 "max_tf": vs.max_tf[d], "grade": grades.get(d, 0) if qid else None,
                                 "rank_base": next((j for j, (x, _) in enumerate(runs[base], 1) if x == d), 0)}
                                for i, (d, sc) in enumerate(top, 1)]})
        return {"query": query, "qscheme": qscheme, "k": k, "base": base, "query_id": qid, "schemes": out,
                "terms": vs.query_terms(query)[1], "avg_length": round(vs.avg_length, 1)}

    def vsm_eval(self, schemes: str = "", k: int = 10, condensed: bool = False, qscheme: str = DEFAULT_QUERY_SCHEME) -> dict:
        vs = self.vsm()
        qs = self.vsm_queries()
        names = self._words(schemes) or [s for s, _, _ in PRESETS]
        judged = qs.judged()
        rows = []
        for s in names:
            key = (s, k, condensed, qscheme)
            if key not in self._vsm_eval_cache:
                per = []
                for q in qs.queries:
                    run = vs.ranking_ids(q.text, s, qscheme)
                    m = evaluation.evaluate_run(run, qs.qrels(q.id), ks=(5, 10, 20), condensed=condensed, judged=judged)
                    per.append({"id": q.id, **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()}})
                mean = {kk: round(v, 4) for kk, v in evaluation.mean_measures(per).items()}
                self._vsm_eval_cache[key] = {"scheme": s, "info": describe_scheme(s), "mean": mean, "per_query": per}
            rows.append(self._vsm_eval_cache[key])
        return {"k": k, "condensed": condensed, "qscheme": qscheme, "queries": qs.summary(), "labelled": len(judged),
                "documents": vs.N, "schemes": rows, "notes": qs.notes}

    def vsm_query_list(self) -> dict:
        qs = self.vsm_queries()
        return {"queries": qs.summary(), "labelled": len(qs.judged()), "documents": self.vsm().N, "notes": qs.notes,
                "file": _relpath(qs.path, self.data_dir.parent)}

    def vsm_judge(self, qid: str, doc_id: str, grade: int | None) -> dict:
        qs = self.vsm_queries()
        if doc_id.isdigit():
            doc_id = "PMID" + doc_id
        r = qs.set_manual(qid, doc_id, grade)
        self._vsm_eval_cache = {}
        return r

    def vsm_relevant(self, qid: str, scheme: str = DEFAULT_SCHEME, qscheme: str = DEFAULT_QUERY_SCHEME) -> dict:
        """The relevant set of one query with the rank each document got under *scheme*."""
        qs = self.vsm_queries()
        q = qs.get(qid)
        if q is None:
            raise ValueError(f"unknown query {qid}")
        vs = self.vsm()
        run = vs.ranking_ids(q.text, scheme, qscheme)
        pos = {d: i for i, d in enumerate(run, 1)}
        rel = qs.qrels(q.id)
        rows = [{"doc_id": d, "grade": g, "rank": pos.get(d, 0), "title": vs.docs[d].title, "length": vs.length[d],
                 "matched": qs.matched_labels(q.id, d), "manual": d in q.manual} for d, g in rel.items()]
        rows.sort(key=lambda r: (r["rank"] or 10 ** 9))
        return {"query": q.spec(), "scheme": scheme, "relevant": rows, "retrieved": len(run),
                "measures": evaluation.evaluate_run(run, rel, ks=(5, 10, 20))}

    def vsm_collect(self, term: str, n: int = 3000, name: str = "", progress=None) -> dict:
        """Download a PubMed abstract collection (with MeSH headings) into
        data/vsm/ for the TF-IDF tab; replaces a collection of the same name."""
        n = max(1, min(int(n), 5000))
        pmids, total = search_pubmed(term, n + max(20, n // 20))
        if not pmids:
            raise ValueError(f"PubMed has no English abstracts for: {term}")
        records = fetch_pubmed_batch(pmids, progress=progress)[:n]
        slug = re.sub(r"[^a-z0-9]+", "_", (name or term).lower()).strip("_")[:40] or "collection"
        self._vsm_dir().mkdir(parents=True, exist_ok=True)
        path = self._vsm_dir() / f"pubmed_{slug}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps({**r, "query": term}, ensure_ascii=False) + "\n")
        self._vsm, self._vsm_queries, self._vsm_eval_cache = {}, None, {}
        return {"query": term, "pubmed_matches": total, "requested": n, "fetched": len(records), "file": str(path),
                "with_mesh": sum(1 for r in records if r.get("mesh")), "documents": self.vsm().N}
