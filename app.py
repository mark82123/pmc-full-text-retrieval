#!/usr/bin/env python3
"""Web front-end for the full-text retrieval tool (standard library only).

    python3 app.py [--data data] [--port 8000]

Endpoints
    GET  /                       single-page UI
    GET  /api/search?q=..&method=bm25|tfidf|count&op=AND|OR&scope=all|title|abstract|body&limit=50
    GET  /api/doc/<id>?q=..      document view with sentence boundaries + highlights
    GET  /api/corpus             corpus statistics + per-document table
    GET  /api/vocab?prefix=..    vocabulary browser
    GET  /api/stopwords          the Snowball list and the biomedical additions
    GET  /api/stem?word=a,b      Porter stem of each word with the step-by-step trace
    GET  /api/analyze?text=..    tokenise / lower-case / stop-word / stem a text; sentence split
    GET  /api/zipf               Zipf analysis of the abstract collection (4 pre-processing conditions)
    GET  /api/zipf/terms?cond=C&terms=a,b&scatter=1   CF / DF / IDF of chosen terms
    GET  /api/zipf/cfdf?cond=B&common_df=&boiler_df=&boiler_burst=&topic_df=&topic_burst=&key_df=&key_burst=
                                                      CF-DF map: function / boilerplate / topic / keyword / number zones
    GET  /api/edit?a=..&b=..     edit distance: DP table + alignment
    GET  /api/spell?word=..&k=2  vocabulary words within k edits
    GET  /api/w2v/status         trained models, active model, training jobs
    GET  /api/w2v/vocab?min_count=5&stopwords=1      pre-processing + vocabulary statistics, training-example counts
    GET  /api/w2v/examples?text=..&window=2&stopwords=1&centre=   skip-gram pairs / CBOW examples of one sentence
    GET  /api/w2v/similar?word=..&n=15&model=        /api/w2v/analogy?a=..&b=..&c=..&model=
    GET  /api/w2v/compare?words=a,b&models=m1,m2&n=5 nearest neighbours side by side + overlap
    GET  /api/w2v/pairs?groups=a,b;c,d&model=        cosine matrix, within- / between-group means
    GET  /api/w2v/projection?words=..&model=&method=pca|tsne&perplexity=15&iters=400&seed=0
    GET  /api/w2v/eval?models=m1,m2&k=10             neighbourhood quality per frequency bucket
    GET  /api/w2v/overlap?models=m1,m2&k=10          neighbour overlap between two models per frequency bucket
    POST /api/w2v/train          (body: JSON {model: sg|cbow, dim, window, negative, min_count, epochs, seed,
                                  remove_stopwords, name}) - queued; several jobs run one after another
    POST /api/w2v/select?name=   make a model the active one      POST /api/w2v/delete?name=
    GET  /api/vsm/info           Project #3: collection + term statistics, tf / idf curves, weighting schemes
    GET  /api/vsm/terms?terms=a,b          CF / DF / IDF / max tf of chosen terms
    GET  /api/vsm/doc?id=PMID..&schemes=nnn,lnn,ltc&top=15   one document's term-weight vector under several schemes
    GET  /api/vsm/search?q=..&scheme=ltc&qscheme=nnc&k=10    top-k by dot product / cosine with per-term contributions
    GET  /api/vsm/compare?q=..&schemes=ltn,ltc&k=10          the same query under several schemes (overlap, tau, lengths)
    GET  /api/vsm/eval?schemes=..&k=10&condensed=0           P@k / R@k / MRR / MAP / nDCG@k over the evaluation queries
    GET  /api/vsm/queries        the evaluation queries and their relevant sets;  /api/vsm/relevant?id=q01&scheme=ltc
    POST /api/vsm/judge?id=q01&doc=PMID..&grade=0|1|2|   manual relevance judgement (empty grade = remove)
    POST /api/vsm/collect        (body: JSON {term, n, name}) download a collection with MeSH into data/vsm/
    POST /api/collect            (body: JSON {term, n, name}) build a PubMed abstract collection
    POST /api/upload?filename=x.xml   (body: raw XML)  add a document to the index
    POST /api/fetch?replace=0|1  (body: JSON {"ids": "..."} or plain text)
                                 fetch PMIDs / PMC ids from NCBI and index them
    POST /api/remove?id=PMC..&delete=1   remove a document from the index (and its file in data/)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ir import zipf  # noqa: E402
from ir.engine import Engine  # noqa: E402
from ir.porter import stem_trace  # noqa: E402
from ir.sentence import naive_split, split_sentences  # noqa: E402
from ir.spell import edit_matrix  # noqa: E402
from ir.stopwords import BIOMEDICAL_STOP_WORDS, SMART_STOP_WORDS, SNOWBALL_STOP_WORDS, STOP_WORDS  # noqa: E402

STATIC = ROOT / "static"
engine: Engine


class Handler(BaseHTTPRequestHandler):
    server_version = "PMC-IR/1.0"

    # ---- helpers ------------------------------------------------------ #
    def _json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        if not path.is_file() or STATIC not in path.resolve().parents:
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quieter log
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- GET ----------------------------------------------------------- #
    def do_GET(self) -> None:
        url = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                return self._file(STATIC / path[len("/static/"):])
            if path == "/api/search":
                q = qs.get("q", "").strip()
                if not q:
                    return self._json({"error": "empty query"}, 400)
                limit = max(1, min(int(qs.get("limit", 50)), 500))
                return self._json(engine.search(q, qs.get("method", "bm25"), qs.get("op", "AND"), limit,
                                                qs.get("scope", "all"), autocorrect=qs.get("spell", "1") == "1"))
            if path.startswith("/api/doc/"):
                doc_id = path[len("/api/doc/"):]
                d = engine.document(doc_id, qs.get("q", ""), qs.get("op", "AND"))
                return self._json(d if d else {"error": "not found"}, 200 if d else 404)
            if path == "/api/corpus":
                return self._json(engine.corpus())
            if path == "/api/stopwords":
                return self._json({"snowball": sorted(SNOWBALL_STOP_WORDS), "biomedical": sorted(BIOMEDICAL_STOP_WORDS),
                                   "smart": sorted(SMART_STOP_WORDS),
                                   "smart_only": sorted(SMART_STOP_WORDS - STOP_WORDS),
                                   "used_not_in_smart": sorted(STOP_WORDS - SMART_STOP_WORDS)})
            if path == "/api/stem":
                words = [w for w in re.split(r"[\s,;]+", qs.get("word", "").strip().lower()) if w][:50]
                return self._json([{"word": w, "stem": engine.tokenizer.stemmer.stem(w), "trace": stem_trace(w),
                                    "stopword": engine.tokenizer.is_stopword(w)} for w in words])
            if path == "/api/analyze":
                text = qs.get("text", "")[:5000]
                toks = [{"text": t.text, "lower": t.norm, "stem": t.stem, "stopword": not t.stem and engine.tokenizer.is_stopword(t.norm),
                         "position": t.position} for t in engine.tokenizer.tokenize(text)]
                return self._json({"tokens": toks, "sentences": split_sentences(text), "sentences_naive": naive_split(text)})
            if path == "/api/zipf":
                return self._json(engine.zipf())
            if path == "/api/zipf/terms":
                return self._json(engine.zipf_terms(qs.get("cond", "C"), qs.get("terms", ""), qs.get("scatter") == "1"))
            if path == "/api/zipf/resolving":
                cut = lambda k: int(qs[k]) if qs.get(k, "").isdigit() else None  # noqa: E731
                return self._json(engine.zipf_resolving(qs.get("cond", "C"), cut("upper"), cut("lower")))
            if path == "/api/zipf/cfdf":
                num = lambda k: float(qs[k]) if re.match(r"^\d*\.?\d+$", qs.get(k, "")) else None  # noqa: E731
                return self._json(engine.zipf_cfdf(qs.get("cond", "B"), **{k: num(k) for k in zipf.MAP_DEFAULTS}))
            if path == "/api/edit":
                a, b = qs.get("a", "").strip().lower()[:40], qs.get("b", "").strip().lower()[:40]
                return self._json(edit_matrix(a, b, qs.get("transpose", "1") == "1"))
            if path == "/api/spell":
                w = qs.get("word", "").strip().lower()
                k = int(qs["k"]) if qs.get("k", "").isdigit() else None
                return self._json({"word": w, "in_vocabulary": w in engine.spell, "cf": engine.spell.words.get(w, 0),
                                   "vocabulary": len(engine.spell.words),
                                   "candidates": len(engine.spell.candidates(w, k or 2)) if w else 0,
                                   "suggestions": engine.spell.suggest(w, 15, k) if w else []})
            if path == "/api/w2v/status":
                return self._json(engine.w2v_status())
            if path == "/api/w2v/similar":
                return self._json(engine.w2v_similar(qs.get("word", "").strip(), max(1, min(int(qs.get("n", 15)), 50)), qs.get("model") or None))
            if path == "/api/w2v/analogy":
                return self._json(engine.w2v_analogy(*(qs.get(k, "").strip() for k in "abc"), model=qs.get("model") or None))
            if path == "/api/w2v/projection":
                return self._json(engine.w2v_projection(qs.get("words", ""), int(qs.get("n", 120)), qs.get("model") or None,
                                                        qs.get("method", "pca"), float(qs.get("perplexity", 15)),
                                                        int(qs.get("iters", 400)), int(qs.get("seed", 0)),
                                                        max(3, min(int(qs.get("neighbours", 12)), 40))))
            if path == "/api/w2v/vocab":
                return self._json(engine.w2v_vocab(int(qs.get("min_count", 5)), qs.get("stopwords", "1") != "0"))
            if path == "/api/w2v/examples":
                c = qs.get("centre", "")
                return self._json(engine.w2v_examples(qs.get("text", ""), int(qs.get("window", 2)), qs.get("stopwords", "1") != "0",
                                                      int(c) if c.isdigit() else None))
            if path == "/api/w2v/compare":
                return self._json(engine.w2v_compare(qs.get("words", ""), qs.get("models", ""), max(1, min(int(qs.get("n", 5)), 20))))
            if path == "/api/w2v/pairs":
                return self._json(engine.w2v_pairs(qs.get("groups", ""), qs.get("model") or None))
            if path == "/api/w2v/eval":
                return self._json(engine.w2v_eval(qs.get("models", ""), max(1, min(int(qs.get("k", 10)), 30))))
            if path == "/api/w2v/overlap":
                return self._json(engine.w2v_overlap(qs.get("models", ""), max(1, min(int(qs.get("k", 10)), 30)), qs.get("words", "")))
            if path == "/api/vsm/info":
                return self._json(engine.vsm_info())
            if path == "/api/vsm/terms":
                return self._json(engine.vsm_terms(qs.get("terms", "")))
            if path == "/api/vsm/doc":
                return self._json(engine.vsm_doc(qs.get("id", "").strip(), qs.get("schemes", ""), max(1, min(int(qs.get("top", 15)), 200))))
            if path == "/api/vsm/search":
                return self._json(engine.vsm_search(qs.get("q", ""), qs.get("scheme", "ltc"), qs.get("qscheme", "nnc"),
                                                    max(1, min(int(qs.get("k", 10)), 100))))
            if path == "/api/vsm/compare":
                return self._json(engine.vsm_compare(qs.get("q", ""), qs.get("schemes", ""), qs.get("qscheme", "nnc"),
                                                     max(1, min(int(qs.get("k", 10)), 50))))
            if path == "/api/vsm/eval":
                return self._json(engine.vsm_eval(qs.get("schemes", ""), max(1, min(int(qs.get("k", 10)), 50)),
                                                  qs.get("condensed", "0") == "1", qs.get("qscheme", "nnc")))
            if path == "/api/vsm/queries":
                return self._json(engine.vsm_query_list())
            if path == "/api/vsm/relevant":
                return self._json(engine.vsm_relevant(qs.get("id", ""), qs.get("scheme", "ltc"), qs.get("qscheme", "nnc")))
            if path == "/api/vocab":
                return self._json(engine.vocabulary(qs.get("prefix", ""), int(qs.get("limit", 200))))
            self.send_error(404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # ---- POST ---------------------------------------------------------- #
    def do_POST(self) -> None:
        url = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            if url.path == "/api/upload":
                filename = Path(qs.get("filename", "upload.xml")).name
                ids = engine.add_xml(body.decode("utf-8", errors="replace"), filename, persist=qs.get("persist", "1") == "1")
                return self._json({"added": ids, "documents": engine.index.n_docs})
            if url.path == "/api/fetch":
                text = body.decode("utf-8", errors="replace")
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    payload = json.loads(text or "{}")
                    text = payload.get("ids", "") if isinstance(payload, dict) else str(payload)
                results = engine.fetch_ids(text, persist=qs.get("persist", "1") == "1",
                                           replace=qs.get("replace", "0") == "1")
                return self._json({"results": results, "documents": engine.index.n_docs})
            if url.path == "/api/w2v/train":
                payload = json.loads(body.decode("utf-8") or "{}")
                jobs = payload if isinstance(payload, list) else [payload]        # a list queues several jobs
                return self._json({"jobs": [engine.w2v_train(j) for j in jobs[:24]]})
            if url.path == "/api/w2v/select":
                return self._json(engine.w2v_select(qs.get("name", "")))
            if url.path == "/api/w2v/delete":
                return self._json(engine.w2v_delete(qs.get("name", "")))
            if url.path == "/api/vsm/judge":
                g = qs.get("grade", "")
                return self._json(engine.vsm_judge(qs.get("id", ""), qs.get("doc", ""), int(g) if g.strip().lstrip("-").isdigit() else None))
            if url.path == "/api/vsm/collect":
                p = json.loads(body.decode("utf-8") or "{}")
                if not str(p.get("term", "")).strip():
                    raise ValueError("empty PubMed query")
                return self._json(engine.vsm_collect(str(p["term"]).strip(), int(p.get("n", 3000)), str(p.get("name", ""))))
            if url.path == "/api/collect":
                p = json.loads(body.decode("utf-8") or "{}")
                if not str(p.get("term", "")).strip():
                    raise ValueError("empty PubMed query")
                return self._json(engine.collect(str(p["term"]).strip(), int(p.get("n", 1000)), str(p.get("name", ""))))
            if url.path == "/api/remove":
                ok = engine.remove(qs.get("id", ""), delete_file=qs.get("delete", "1") == "1")
                return self._json({"removed": ok, "documents": engine.index.n_docs}, 200 if ok else 404)
            self.send_error(404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main() -> None:
    global engine
    ap = argparse.ArgumentParser(description="PMC full-text retrieval tool - web UI")
    ap.add_argument("--data", default=str(ROOT / "data"), help="directory with PMC XML files")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    engine = Engine(args.data)
    n = engine.load()
    print(f"Indexed {n} documents ({len(engine.index.postings)} terms) in {engine.build_time:.2f}s", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
