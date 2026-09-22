#!/usr/bin/env python3
"""Command-line interface for the PMC full-text retrieval tool.

    python3 cli.py search "cancer immunotherapy" [--method bm25|tfidf|count] [--op AND|OR] [--scope all|title|abstract|body] [-n 10]
    python3 cli.py stats [PMC7536103]          document statistics (all documents if no id)
    python3 cli.py sentences PMC7536103 [-n 20] show detected sentence boundaries
    python3 cli.py corpus                       corpus statistics
    python3 cli.py vocab [prefix]               vocabulary with df / cf
    python3 cli.py analyze "some text"          tokenise / stem / stop-word demo
    python3 cli.py fetch 32895440 PMC7536103    download from NCBI by PMID / PMC id and index
    python3 cli.py fetch --file ids.txt         (one id per line; --replace re-downloads)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ir.engine import Engine  # noqa: E402
from ir.sentence import naive_split, split_sentences  # noqa: E402
from ir.tokenizer import Tokenizer  # noqa: E402

_TAG = re.compile(r"</?mark>")


def _plain(html_text: str) -> str:
    import html
    return html.unescape(_TAG.sub(lambda m: "[" if m.group(0) == "<mark>" else "]", html_text))


def main() -> None:
    ap = argparse.ArgumentParser(description="PMC full-text retrieval tool (CLI)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--method", default="bm25", choices=["bm25", "tfidf", "count"])
    s.add_argument("--op", default="AND", choices=["AND", "OR"]); s.add_argument("-n", type=int, default=10)
    s.add_argument("--scope", default="all", choices=["all", "title", "abstract", "body"],
                   help="rank by the whole document or by the hits in one field only")
    st = sub.add_parser("stats"); st.add_argument("doc_id", nargs="?")
    se = sub.add_parser("sentences"); se.add_argument("doc_id"); se.add_argument("-n", type=int, default=30)
    sub.add_parser("corpus")
    v = sub.add_parser("vocab"); v.add_argument("prefix", nargs="?", default=""); v.add_argument("-n", type=int, default=40)
    an = sub.add_parser("analyze"); an.add_argument("text")
    fe = sub.add_parser("fetch"); fe.add_argument("ids", nargs="*"); fe.add_argument("--file")
    fe.add_argument("--replace", action="store_true", help="re-download ids that are already indexed")
    args = ap.parse_args()

    if args.cmd == "analyze":
        tok = Tokenizer()
        print(f"{'token':<22}{'lower':<22}{'stem':<20}note")
        for t in tok.tokenize(args.text):
            print(f"{t.text:<22}{t.norm:<22}{t.stem or '-':<20}{'stop word' if not t.stem else ''}")
        print("\nSentences (rule-based):")
        for i, s_ in enumerate(split_sentences(args.text), 1):
            print(f"  {i}. {s_}")
        print(f"rule-based: {len(split_sentences(args.text))}   naive: {len(naive_split(args.text))}")
        return

    eng = Engine(args.data)
    n = eng.load()
    print(f"[index] {n} documents, {len(eng.index.postings)} terms, built in {eng.build_time:.2f}s\n")

    if args.cmd == "search":
        r = eng.search(args.query, args.method, args.op, args.n, args.scope)
        a = r["analysis"]
        print("query analysis:", ", ".join(
            f"{t['word']}->{t['stem']}(df={t['df']})" if t["role"] == "term" else f"{t['word']}[{t['role']}]" for t in a["tokens"]))
        print(f"parsed: {a['parsed']}\n{r['total']} documents matched in {r['time_ms']} ms "
              f"(ranked by {args.method}, scope={r['scope']})\n")
        for i, d in enumerate(r["results"], 1):
            sc = d["scores"]
            print(f"#{i}  {d['doc_id']}  score={d['score']}  "
                  f"[title={sc['title']} abstract={sc['abstract']} body={sc['body']} all={sc['all']}]  hits={d['total_hits']}  {d['field_hits']}")
            print(f"    {_plain(d['title_html'])}")
            print(f"    {d['journal']} ({d['year']}) – {', '.join(d['authors'])}")
            print(f"    terms: {d['term_hits']}")
            print(f"    …{_plain(d['snippet_html'])}…")
            print(f"    chars={d['stats']['characters']} words={d['stats']['words']} sentences={d['stats']['sentences']}\n")
    elif args.cmd == "stats":
        ids = [args.doc_id] if args.doc_id else sorted(eng.stats)
        for i in ids:
            s_ = eng.stats.get(i)
            if not s_:
                print("unknown document", i); continue
            d = eng.index.documents[i]
            print(f"{i}: {d.title}")
            for k, v in s_.items():
                if k in ("doc_id",):
                    continue
                if k == "top_terms":
                    v = ", ".join(f"{t}({c})" for t, c in v)
                print(f"    {k:<24}{v}")
            print()
    elif args.cmd == "sentences":
        d = eng.document(args.doc_id)
        if not d:
            print("unknown document"); return
        print(f"{d['meta']['title']}\nrule-based sentences: {d['stats']['sentences']}   naive: {d['stats']['sentences_naive']}\n")
        shown = 0
        for u in d["units"]:
            for s_ in u["sentences"]:
                if shown >= args.n:
                    return
                print(f"[{s_['n']:>4}] ({u['field']}) {_plain(s_['html'])}")
                shown += 1
    elif args.cmd == "corpus":
        c = eng.corpus()
        for k, v in c.items():
            if k == "documents_list":
                continue
            if k in ("top_terms", "top_df"):
                v = ", ".join(f"{t}({c_})" for t, c_ in v[:20])
            print(f"{k:<24}{v}")
        print("\nper document:")
        print(f"{'id':<13}{'chars':>9}{'words':>8}{'sent':>6}{'naive':>7}{'paras':>6}  title")
        for r in c["documents_list"]:
            print(f"{r['doc_id']:<13}{r['characters']:>9}{r['words']:>8}{r['sentences']:>6}{r['sentences_naive']:>7}{r['paragraphs']:>6}  {r['title'][:60]}")
    elif args.cmd == "vocab":
        for r in eng.vocabulary(args.prefix, args.n):
            print(f"{r['term']:<28}df={r['df']:<5}cf={r['cf']}")
    elif args.cmd == "fetch":
        text = " ".join(args.ids)
        if args.file:
            text += "\n" + Path(args.file).read_text(encoding="utf-8")
        results = eng.fetch_ids(text, replace=args.replace)
        if not results:
            print("no PMID / PMC id found in the input"); return
        for r in results:
            tag = {"added": "+", "updated": "~", "exists": "=", "error": "!"}[r["status"]]
            ident = " / ".join(x for x in (r["pmid"] and "PMID " + r["pmid"], r["pmcid"]) if x)
            print(f"{tag} {r['input']:<18} {ident:<28} {r['status']:<8} {r['content']:<14} {r['title'][:60] or r['message']}")
            if r["title"] and r["message"]:
                print(f"    note: {r['message']}")
        print(f"\n[index] now {eng.index.n_docs} documents, {len(eng.index.postings)} terms")


if __name__ == "__main__":
    main()
