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
    GET  /api/edit?a=..&b=..     edit distance: DP table + alignment
    GET  /api/spell?word=..&k=2  vocabulary words within k edits
    GET  /api/w2v/status         word2vec model + training job
    GET  /api/w2v/similar?word=..&n=15      /api/w2v/analogy?a=..&b=..&c=..      /api/w2v/projection?words=..
    POST /api/w2v/train          (body: JSON {model: sg|cbow, dim, window, negative, min_count, epochs})
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
                return self._json(engine.w2v_similar(qs.get("word", "").strip(), max(1, min(int(qs.get("n", 15)), 50))))
            if path == "/api/w2v/analogy":
                return self._json(engine.w2v_analogy(*(qs.get(k, "").strip() for k in "abc")))
            if path == "/api/w2v/projection":
                return self._json(engine.w2v_projection(qs.get("words", ""), int(qs.get("n", 120))))
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
                return self._json(engine.w2v_train(json.loads(body.decode("utf-8") or "{}")))
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
