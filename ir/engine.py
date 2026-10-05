"""High-level engine used by the web application and the CLI."""

from __future__ import annotations

import html
import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

from . import zipf
from .index import InvertedIndex
from .ncbi import (convert_pmids, fetch_pmc_xml, fetch_pubmed_batch, fetch_pubmed_xml, parse_ids, pubmed_to_jats,
                   search_pubmed)
from .pmc_parser import Document, load_directory, load_jsonl, parse_file, parse_xml_string
from .search import Node, Searcher, highlight_spans, parse_query
from .sentence import split_sentences
from .spell import SpellIndex
from .stats import corpus_stats, document_stats
from .tokenizer import TOKEN_RE, Tokenizer
from .word2vec import Word2Vec, sentences_from_texts

W2V_FILE = "word2vec.json"
# query words that are expanded by the engine itself (prefix* / fuzzy~2) are not spell-checked
_EXPANDED_WORD_RE = re.compile(r"\S*(?:\*|~[1-3]?)(?=[\s)]|$)")

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
        self.spell = SpellIndex()                      # surface words of the collection (spelling correction)
        self.searcher.fuzzy = lambda word, k: [r["word"] for r in self.spell.within(word, k or None, limit=30)]
        self._version = 0                              # bumped whenever the index changes
        self._zipf: tuple[int, dict] | None = None     # cached Zipf analysis
        self.w2v: Word2Vec | None = None
        self.w2v_meta: dict = {}
        self.w2v_job: dict = {"state": "idle"}

    # ------------------------------------------------------------------ #
    def load(self) -> int:
        t0 = time.time()
        docs = load_directory(self.data_dir) if self.data_dir.exists() else []
        for f in sorted(self.data_dir.glob("*.jsonl")) if self.data_dir.exists() else []:
            docs.extend(load_jsonl(f))                 # PubMed abstract collections
        for d in docs:
            self._add(d)
        self.build_time = time.time() - t0
        model = self.data_dir / W2V_FILE
        if model.is_file():
            try:
                self.w2v, self.w2v_meta = Word2Vec.load(model)
            except (ValueError, KeyError, OSError) as e:
                print(f"[warn] could not load {model}: {e}")
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

    # ---- word2vec ------------------------------------------------------ #
    def w2v_train(self, params: dict, block: bool = False) -> dict:
        """Train word2vec on the collection in a background thread."""
        if self.w2v_job.get("state") == "running":
            raise ValueError("a training job is already running")
        clamp = lambda k, lo, hi, d: max(lo, min(hi, int(params.get(k, d))))  # noqa: E731
        model = Word2Vec(sg=str(params.get("model", "sg")).lower() in ("sg", "skip-gram", "skipgram"),
                         dim=clamp("dim", 10, 200, 50), window=clamp("window", 1, 15, 5),
                         negative=clamp("negative", 1, 20, 5), min_count=clamp("min_count", 1, 100, 5),
                         epochs=clamp("epochs", 1, 50, 5))
        name, docs = self.collection_docs()
        texts = self.collection_texts()
        job = {"state": "running", "progress": 0.0, "epoch": 0, "loss": None, "started": time.time(),
               "params": model.params(), "documents": len(docs)}
        self.w2v_job = job

        def run() -> None:
            try:
                sentences = sentences_from_texts(texts)
                job["sentences"] = len(sentences)

                def progress(frac: float, epoch: int, alpha: float, loss: float) -> None:
                    job.update(progress=round(frac, 4), epoch=epoch, loss=round(loss, 4), alpha=round(alpha, 5))

                model.train(sentences, progress)
                meta = {"collection": name, "documents": len(docs), "sentences": len(sentences)}
                self.data_dir.mkdir(parents=True, exist_ok=True)
                model.save(self.data_dir / W2V_FILE, meta)
                self.w2v, self.w2v_meta = model, meta
                job.update(state="done", progress=1.0)
            except Exception as e:  # reported to the UI
                job.update(state="error", message=f"{type(e).__name__}: {e}")

        if block:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()
        return job

    def w2v_status(self) -> dict:
        job = dict(self.w2v_job)
        if job.get("started"):
            job["elapsed"] = round(time.time() - job.pop("started"), 1)
        out = {"job": job, "model": None, "collection": self.collection_docs()[0],
               "documents": len(self.collection_docs()[1])}
        if self.w2v:
            out["model"] = {**self.w2v.info(), **self.w2v_meta,
                            "top_words": list(zip(self.w2v.vocab[:40], self.w2v.counts[:40]))}
        return out

    def _w2v_check(self, *words: str) -> None:
        if not self.w2v:
            raise ValueError("no word2vec model yet - train one first")
        for w in words:
            if w not in self.w2v:
                near = [r["word"] for r in self.spell.within(w.lower(), None) if r["word"] in self.w2v][:5]
                hint = f" - did you mean: {', '.join(near)}?" if near else ""
                raise ValueError(f"'{w}' is not in the model vocabulary{hint}")

    def w2v_similar(self, word: str, topn: int = 15) -> dict:
        self._w2v_check(word)
        m = self.w2v
        return {"word": word.lower(), "count": m.counts[m.index[word.lower()]],
                "neighbours": [{"word": w, "similarity": s, "count": m.counts[m.index[w]]} for w, s in m.most_similar(word, topn)]}

    def w2v_analogy(self, a: str, b: str, c: str, topn: int = 10) -> dict:
        self._w2v_check(a, b, c)
        return {"a": a, "b": b, "c": c, "results": [{"word": w, "similarity": s} for w, s in self.w2v.analogy(a, b, c, topn)]}

    def w2v_projection(self, words: str = "", n: int = 120) -> dict:
        """2-D PCA map.  With *words*: each given word plus its nearest
        neighbours (one colour group per seed); otherwise the n most frequent words."""
        self._w2v_check()
        m = self.w2v
        seeds = [w.lower() for w in re.split(r"[\s,;]+", words.strip()) if w]
        if seeds:
            self._w2v_check(*seeds)
            group: dict[str, str] = {}
            for s in seeds:
                group.setdefault(s, s)
                for w, _ in m.most_similar(s, 12):
                    group.setdefault(w, s)
            pts = m.projection(list(group))
            for p in pts:
                p["group"] = group[p["word"]]
                p["seed"] = p["word"] in seeds
            return {"points": pts, "groups": seeds}
        return {"points": m.projection(m.vocab[:max(10, min(n, 400))]), "groups": []}
