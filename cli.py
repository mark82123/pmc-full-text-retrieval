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

Project #2
    python3 cli.py collect "GLP-1" -n 1000 --name glp1   build a PubMed abstract collection (data/pubmed_glp1.jsonl)
    python3 cli.py zipf [--cond B] [--top 50] [--svg report/]   Zipf analysis: vocabulary, CF/DF, regression, 4 conditions
    python3 cli.py cfdf [--cond B]                       CF-DF map: function / boilerplate / topic / keyword / number zones
    python3 cli.py resolving [--cond C] [--upper R] [--lower R] Resolving power of significant words (Luhn cut-offs)
    python3 cli.py terms glp obesity nausea [--cond C]   CF / DF / IDF of chosen terms (default set if none given)
    python3 cli.py edit semaglutide samegluitde          edit distance with the DP table
    python3 cli.py spell diabtes                         spelling suggestions from the collection vocabulary
    python3 cli.py w2v train [--model sg|cbow] [--dim 50] [--window 5] [--epochs 10] [--min-count 5] [--seed 1] [--keep-stopwords] [--name X]
    python3 cli.py w2v list                              trained models (data/w2v/*.json) and the active one
    python3 cli.py w2v use NAME  |  w2v delete NAME      choose / remove a model
    python3 cli.py w2v vocab [--min-count 5]             pre-processing + vocabulary statistics, training-example counts
    python3 cli.py w2v examples "GLP-1 receptor agonists reduce body weight" [--window 2]   skip-gram pairs / CBOW examples
    python3 cli.py w2v similar semaglutide [--use NAME]  |  w2v analogy kidney ckd liver
    python3 cli.py w2v compare semaglutide nausea kidney --models sg-d50-w5-m5-e10,cbow-d50-w5-m5-e10 [-n 5]
    python3 cli.py w2v pairs "semaglutide,liraglutide;nausea,vomiting" [--use NAME]      cosine matrix within / between groups
    python3 cli.py w2v map semaglutide nausea kidney mice [--method pca|tsne] [--svg report/]   2-D map (+ k-NN preservation)
    python3 cli.py w2v eval [--models a,b] [-k 10]       neighbourhood quality per frequency bucket (rare-word behaviour)
    python3 cli.py w2v overlap --models a,b [-k 10]      neighbour overlap between two models per frequency bucket

Project #3 (term weighting / vector space model; collection = data/vsm/*.jsonl)
    python3 cli.py vsm info                              collection, vocabulary, tf / idf curves, weighting schemes
    python3 cli.py vsm terms glp obesity semaglutide     CF / DF / IDF / max tf of terms
    python3 cli.py vsm doc PMID42892933 --schemes nnn,lnn,ltn,ltc [--top 15]   one document's weight vector per scheme
    python3 cli.py vsm search "semaglutide weight loss" [--scheme ltc] [--qscheme nnc] [-k 10]   top-k with contributions
    python3 cli.py vsm compare "semaglutide weight loss" --schemes nnn,lnn,ltn,ltc [-k 10]       controlled experiment
    python3 cli.py vsm eval [--schemes nnn,lnn,ltc,bm25] [-k 10] [--condensed]   P@k / R@k / MRR / MAP / nDCG@k
    python3 cli.py vsm queries  |  vsm relevant q01 [--scheme ltc]           evaluation queries and relevant sets
    python3 cli.py vsm judge q01 PMID42892933 2          manual relevance judgement (grade 0 / 1 / 2, "-" removes)
    python3 cli.py vsm collect "GLP-1" -n 3000 --name glp1_3000   download the collection (with MeSH) into data/vsm/
    python3 cli.py vsm svg report/                       figures for the report
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


def _vsm(eng: Engine, args) -> None:
    """Project #3 sub-commands."""
    a = args.action
    if a == "collect":
        term = " ".join(args.words) or "GLP-1"
        r = eng.vsm_collect(term, args.n, args.name, progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True))
        print(f"\n{r['fetched']} records ({r['with_mesh']} with MeSH) of {r['pubmed_matches']} PubMed matches -> {r['file']}")
        return
    if a == "info":
        i = eng.vsm_info()
        print(f"{i['collection']}: {i['documents']} documents, {i['tokens']} tokens, {i['index_terms']} index terms "
              f"(stop words removed: {i['stopword_tokens']}), vocabulary {i['vocabulary']} (df = 1: {i['hapax_df']}), "
              f"length {i['min_length']}–{i['max_length']} (mean {i['avg_length']}), built in {i['build_time_s']} s")
        print(f"labelled documents: {i['labelled']} (MeSH {i['with_mesh']}, keywords {i['with_keywords']})\n")
        print("df histogram:     " + "   ".join(f"{h['bucket']}: {h['terms']} terms (idf {h['idf_from']}–{h['idf_to']})" for h in i["df_histogram"]))
        print("\ntf -> raw / 1+ln tf / augmented(max tf = 10):")
        print("   " + "  ".join(f"{c['tf']:>4}:{c['log']:.2f}/{c['aug_max10']:.2f}" for c in i["tf_curve"]))
        print("\ndf -> idf = ln(N/df):  " + "  ".join(f"{c['df']}:{c['idf']:.2f}" for c in i["idf_curve"]))
        print("\nweighting schemes:")
        for sch in i["schemes"]:
            print(f"  {sch['scheme']:5s} {sch['name']:28s} {sch['formula']}")
        return
    if a == "terms":
        r = eng.vsm_terms(" ".join(args.words))
        print(f"{'word':<16}{'term':<14}{'cf':>7}{'df':>7}{'df/N':>8}{'idf':>8}{'p-idf':>8}{'max tf':>8}{'1+ln':>7}{'mean tf':>9}")
        for t in r["rows"]:
            if t["stopword"]:
                print(f"{t['word']:<16}{'(stop word)':<14}"); continue
            print(f"{t['word']:<16}{t['term']:<14}{t['cf']:>7}{t['df']:>7}{t['df_share']:>8.3f}{t['idf']:>8.3f}{t['prob_idf']:>8.3f}{t['max_tf']:>8}{t['log_max_tf']:>7.2f}{t['mean_tf']:>9.2f}")
        return
    if a == "doc":
        d = eng.vsm_doc(args.words[0] if args.words else "", args.schemes, args.top)
        print(f"{d['doc_id']}: {d['title']}\n{d['tokens']} tokens, {d['length']} index terms, {d['unique_terms']} distinct, max tf = {d['max_tf']}\n")
        for v in d["vectors"]:
            print(f"[{v['scheme']}]  |d| = {v['norm']}   top term share = {v['top_share']:.3f}   top-5 share = {v['top5_share']:.3f}")
            for r in v["vector"]:
                print(f"   {r['term']:<16} tf={r['tf']:<3} df={r['df']:<5} tf_w={r['tf_w']:<8} df_w={r['df_w']:<8} w={r['weight']:<9} share={r['share']:.3f}")
        return
    if a == "search":
        r = eng.vsm_search(" ".join(args.words), args.scheme, args.qscheme, args.k)
        print("query terms: " + ", ".join(f"{t['word']}->{t['term']}(df={t['df']}, idf={t['idf']})" if t["term"] else f"{t['word']}[stop]" for t in r["terms"]))
        print(f"query vector ({r['qscheme']}): {r['qvec']}\n{r['total']} documents with a non-zero score in {r['time_ms']} ms; "
              f"scheme {r['scheme']} = {r['scheme_info']['name']}" + (f"; query {r['query_id']}: P@10 = {r['measures']['P@10']:.2f}" if r.get("measures") else "") + "\n")
        for d in r["results"]:
            g = f"  rel={d['grade']}" if d.get("grade") is not None else ""
            print(f"#{d['rank']:<3} {d['doc_id']}  score={d['score']}{g}  len={d['length']}  {d['title'][:90]}")
            print("      " + "  ".join(f"{c['term']}: {c['qw']}×{c['dw']}={c['part']}" for c in d["contributions"]))
        return
    if a == "compare":
        r = eng.vsm_compare(" ".join(args.words), args.schemes, args.qscheme, args.k)
        print(f"query: {r['query']}   (mean document length {r['avg_length']})\n")
        print(f"{'scheme':<7}{'overlap@k':>10}{'tau':>7}{'mean len':>10}{'max tf':>8}" + ("  P@10   nDCG@10" if r["query_id"] else ""))
        for s_ in r["schemes"]:
            m = s_["measures"]
            print(f"{s_['scheme']:<7}{s_['overlap']:>10.2f}{s_['tau']:>7.2f}{s_['mean_length']:>10.1f}{s_['mean_max_tf']:>8.1f}" + (f"  {m['P@10']:.2f}   {m['nDCG@10']:.3f}" if m else ""))
        for s_ in r["schemes"]:
            print(f"\n[{s_['scheme']}] {s_['info']['name']}")
            for t in s_["top"]:
                print(f"  #{t['rank']:<3}{t['doc_id']:<14}{t['score']:<9} len={t['length']:<4} (rank under {r['base']}: {t['rank_base'] or '-'})"
                      + (f" rel={t['grade']}" if t["grade"] is not None else "") + f"  {t['title'][:70]}")
        return
    if a == "eval":
        r = eng.vsm_eval(args.schemes, args.k, args.condensed, args.qscheme)
        keys = ["P@5", "P@10", "R@10", "R@20", "MRR", "MAP", "nDCG@10", "nDCG@20", "R-prec"]
        print(f"{len(r['queries'])} queries, {r['documents']} documents ({r['labelled']} labelled)" + (", condensed lists" if r["condensed"] else "") + "\n")
        print(f"{'scheme':<7}" + "".join(f"{k_:>9}" for k_ in keys) + "  name")
        for s_ in r["schemes"]:
            print(f"{s_['scheme']:<7}" + "".join(f"{s_['mean'][k_]:>9.3f}" for k_ in keys) + f"  {s_['info']['name']}")
        print("\nper query (P@10 / nDCG@10):")
        print(f"{'query':<6}" + "".join(f"{s_['scheme']:>13}" for s_ in r["schemes"]))
        for i, q in enumerate(r["queries"]):
            print(f"{q['id']:<6}" + "".join(f"{s_['per_query'][i]['P@10']:>6.2f}/{s_['per_query'][i]['nDCG@10']:.3f}" for s_ in r["schemes"]) + f"   {q['text']}")
        return
    if a == "queries":
        r = eng.vsm_query_list()
        print(f"{len(r['queries'])} queries, {r['labelled']} of {r['documents']} documents carry topic labels ({r['file']})\n")
        for q in r["queries"]:
            print(f"{q['id']}  {q['text']}\n      relevant {q['relevant']} (grade 2: {q['grade2']}, grade 1: {q['grade1']}; via MeSH {q['from_mesh']}, keywords only {q['from_keyword_only']}; manual {q['manual']})")
            print("      rules: " + "  AND  ".join("(" + " | ".join(g) + ")" for g in q["groups"]))
        return
    if a == "relevant":
        r = eng.vsm_relevant(args.words[0] if args.words else "q01", args.scheme, args.qscheme)
        m = r["measures"]
        print(f"{r['query']['id']}: {r['query']['text']}  — {m['relevant']} relevant, P@10 {m['P@10']:.2f}, R@10 {m['R@10']:.3f}, AP {m['AP']:.3f}, nDCG@10 {m['nDCG@10']:.3f}\n")
        for d in r["relevant"]:
            print(f"  rank {d['rank'] or '-':>5}  grade {d['grade']}  len={d['length']:<4} {d['doc_id']}  {d['title'][:70]}\n         {', '.join(d['matched'])}")
        return
    if a == "judge":
        if len(args.words) < 2:
            raise SystemExit("usage: vsm judge QUERY_ID DOC_ID [0|1|2|-]")
        g = args.words[2] if len(args.words) > 2 else "-"
        print(eng.vsm_judge(args.words[0], args.words[1], None if g == "-" else int(g)))
        return
    if a == "svg":
        from ir.svgplot import vsm_figures
        out = Path(args.words[0] if args.words else "report")
        out.mkdir(parents=True, exist_ok=True)
        for name, svg in vsm_figures(eng).items():
            (out / name).write_text(svg, encoding="utf-8")
            print("wrote", out / name)
        return


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
    co = sub.add_parser("collect"); co.add_argument("term"); co.add_argument("-n", type=int, default=1000); co.add_argument("--name", default="")
    zp = sub.add_parser("zipf"); zp.add_argument("--cond", default="B", choices=["A", "B", "C", "D", "E"]); zp.add_argument("--top", type=int, default=50)
    zp.add_argument("--svg", help="directory to write the figures (SVG) to")
    mp = sub.add_parser("cfdf"); mp.add_argument("--cond", default="B", choices=["A", "B", "C", "D", "E"]); mp.add_argument("--top", type=int, default=12)
    rp = sub.add_parser("resolving"); rp.add_argument("--cond", default="C", choices=["A", "B", "C", "D", "E"]); rp.add_argument("--top", type=int, default=30)
    rp.add_argument("--upper", type=int, help="upper cut-off rank (default: automatic)"); rp.add_argument("--lower", type=int, help="lower cut-off rank (default: automatic)")
    tm = sub.add_parser("terms"); tm.add_argument("words", nargs="*"); tm.add_argument("--cond", default="C", choices=["A", "B", "C", "D", "E"])
    ed = sub.add_parser("edit"); ed.add_argument("a"); ed.add_argument("b")
    sp = sub.add_parser("spell"); sp.add_argument("word"); sp.add_argument("-k", type=int)
    wv = sub.add_parser("w2v"); wv.add_argument("action", choices=["train", "list", "use", "delete", "vocab", "examples", "similar", "analogy",
                                                                     "compare", "pairs", "map", "eval", "overlap"]); wv.add_argument("words", nargs="*")
    wv.add_argument("--model", default="sg", choices=["sg", "cbow"]); wv.add_argument("--dim", type=int, default=50)
    wv.add_argument("--window", type=int, default=5); wv.add_argument("--negative", type=int, default=5)
    wv.add_argument("--min-count", type=int, default=5); wv.add_argument("--epochs", type=int, default=10)
    wv.add_argument("--seed", type=int, default=1); wv.add_argument("--keep-stopwords", action="store_true"); wv.add_argument("--name", default="")
    wv.add_argument("--use", default="", help="model name (default: the active model)"); wv.add_argument("--models", default="", help="comma-separated model names")
    wv.add_argument("-n", type=int, default=5); wv.add_argument("-k", type=int, default=10)
    wv.add_argument("--method", default="pca", choices=["pca", "tsne"]); wv.add_argument("--perplexity", type=float, default=15)
    wv.add_argument("--svg", help="directory to write the map (SVG) to")
    vm = sub.add_parser("vsm"); vm.add_argument("action", choices=["info", "terms", "doc", "search", "compare", "eval", "queries",
                                                                     "relevant", "judge", "collect", "svg"]); vm.add_argument("words", nargs="*")
    vm.add_argument("--scheme", default="ltc"); vm.add_argument("--qscheme", default="nnc"); vm.add_argument("--schemes", default="")
    vm.add_argument("-k", type=int, default=10); vm.add_argument("--top", type=int, default=15); vm.add_argument("--condensed", action="store_true")
    vm.add_argument("-n", type=int, default=3000); vm.add_argument("--name", default="")
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

    if args.cmd == "edit":
        from ir.spell import edit_matrix
        r = edit_matrix(args.a.lower(), args.b.lower())
        print("      " + "  ".join("ε" + r["b"]))
        for i, row in enumerate(r["matrix"]):
            print(f"  {('ε' + r['a'])[i]}  " + " ".join(f"{v:>2}" for v in row))
        print(f"\nedit distance = {r['distance']}")
        print("alignment: " + ", ".join(f"{o['op']}({o['a'] or '-'}→{o['b'] or '-'})" for o in r["ops"] if o["op"] != "match"))
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
    elif args.cmd == "collect":
        r = eng.collect(args.term, args.n, args.name, progress=lambda a, b: print(f"  fetched {a}/{b}", flush=True))
        print(f"{r['fetched']} abstracts for '{r['query']}' saved to {args.data}/{r['file']} "
              f"(PubMed has {r['pubmed_matches']} matches); index now holds {r['documents']} documents")
    elif args.cmd == "zipf":
        z = eng.zipf()
        c = next(x for x in z["conditions"] if x["key"] == args.cond)
        print(f"collection: {z['collection']}\ncondition {c['key']}: {c['description']}\n")
        print(f"documents {c['documents']}   tokens {c['tokens']}   unique terms {c['vocabulary']}   avg tokens/doc {c['avg_tokens']}"
              f"   hapax {c['hapax']} ({c['hapax_share']:.1%})\n")
        print(f"{'range':<10}{'ranks':>16}{'slope':>9}{'intercept':>11}{'exponent':>10}{'R2':>8}{'RMSE':>8}")
        for name, f in [("whole", c["fit"])] + [(sg["name"], sg) for sg in c["segments"]]:
            print(f"{name:<10}{str(f['from']) + '-' + str(f['to']):>16}{f['slope']:>9}{f['intercept']:>11}{f['exponent']:>10}{f['r2']:>8}{f['rmse']:>8}")
        print(f"\n{'rank':>4}  {'term':<22}{'CF':>8}{'DF':>7}{'r*CF':>9}")
        for i, (t, cf, df) in enumerate(c["top"][:args.top], 1):
            print(f"{i:>4}  {t:<22}{cf:>8}{df:>7}{i * cf:>9}")
        print(f"\n{'cond':<5}{'tokens':>9}{'vocab':>8}{'avg/doc':>9}{'k':>8}{'R2':>8}{'RMSE':>8}{'k(mid)':>8}  top terms")
        for x in z["conditions"]:
            print(f"{x['key']:<5}{x['tokens']:>9}{x['vocabulary']:>8}{x['avg_tokens']:>9}{x['fit']['exponent']:>8}{x['fit']['r2']:>8}"
                  f"{x['fit']['rmse']:>8}{x['segments'][1]['exponent']:>8}  {' '.join(t[0] for t in x['top'][:8])}")
        if args.svg:
            from ir.svgplot import line_plot
            out = Path(args.svg); out.mkdir(parents=True, exist_ok=True)
            pts = [tuple(p) for p in c["points"]]
            f = c["fit"]
            fit = [(1, 10 ** f["intercept"]), (c["vocabulary"], 10 ** (f["intercept"] + f["slope"] * __import__("math").log10(c["vocabulary"])))]
            fit = [(x, max(y, 0.5)) for x, y in fit]
            vl = [(sg["from"], sg["name"]) for sg in c["segments"]]
            figs = {
                f"rank_frequency_linear_{c['key']}.svg": line_plot([{"name": "CF", "points": pts}], f"Rank vs frequency (linear axes) - condition {c['key']}", "Rank r", "Collection frequency CF"),
                f"rank_frequency_semilog_{c['key']}.svg": line_plot([{"name": "CF", "points": pts}], f"Rank vs frequency (log frequency axis) - condition {c['key']}", "Rank r", "Collection frequency CF (log scale)", ylog=True),
                f"zipf_loglog_{c['key']}.svg": line_plot([{"name": "Observed CF", "points": pts}, {"name": f"Regression line (k = {f['exponent']:.3f}, R2 = {f['r2']:.3f})", "points": fit, "color": "#1d2333", "dash": True}],
                                                          f"Log-log rank-frequency plot - condition {c['key']}", "Rank r (log scale)", "Collection frequency CF (log scale)", xlog=True, ylog=True, vlines=vl),
                "zipf_conditions.svg": line_plot([{"name": f"{x['key']} {x['name']}", "points": [tuple(p) for p in x["points"]]} for x in z["conditions"]],
                                                 "Rank-frequency curves of the four pre-processing conditions", "Rank r (log scale)", "Collection frequency CF (log scale)", xlog=True, ylog=True),
            }
            for name, svg in figs.items():
                (out / name).write_text(svg, encoding="utf-8")
            print(f"\nfigures written to {out}/: {', '.join(figs)}")
    elif args.cmd == "cfdf":
        r = eng.zipf_cfdf(args.cond)
        t = r["thresholds"]
        print(f"condition {r['condition']}, N = {r['documents']} documents, {r['vocabulary']} terms; burst = CF / DF, poisson = expected DF at random / DF")
        print(f"thresholds: function DF/N >= {t['common_df']}; boilerplate DF/N >= {t['boiler_df']} & burst < {t['boiler_burst']}; "
              f"topic DF/N >= {t['topic_df']} & burst >= {t['topic_burst']}; keyword DF/N < {t['key_df']} & burst >= {t['key_burst']}\n")
        for z in r["zones"]:
            print(f"{z['name']:<12}{z['terms']:>7} terms  {z['vocab_share']:>6.1%} of vocab  {z['tokens_share']:>6.1%} of tokens  mean idf {z['mean_idf']:.3f}")
            print("             " + "  ".join(f"{x['term']}({x['cf']}/{x['df']})" for x in z["examples"][:args.top]))
    elif args.cmd == "resolving":
        r = eng.zipf_resolving(args.cond, args.upper, args.lower)
        print(f"condition {r['condition']}, N = {r['documents']} documents, {r['vocabulary']} terms; resolving power = CF * idf, idf = log10(N / df)")
        if not r["vocabulary"]:
            return
        print(f"smoothed peak at rank {r['peak']['rank']} ({r['peak']['power']}); automatic cut-offs (half of the peak): ranks {r['auto']['upper']}-{r['auto']['lower']}; "
              f"applied: ranks {r['upper']}-{r['lower']}\n")
        print(f"{'zone':<13}{'ranks':>14}{'terms':>8}{'%vocab':>9}{'%tokens':>9}{'%power':>9}{'mean pw':>9}{'mean idf':>10}  highest-ranked terms")
        for z in r["zones"]:
            ranks = f"{z['from']}-{z['to']}" if z["terms"] else "-"
            print(f"{z['name']:<13}{ranks:>14}{z['terms']:>8}{z['vocab_share']:>9.1%}{z['tokens_share']:>9.1%}{z['power_share']:>9.1%}"
                  f"{z['mean_power']:>9}{z['mean_idf']:>10}  {' '.join(z['examples'][:8])}")
        print(f"\n{'significant word':<22}{'rank':>6}{'CF':>8}{'DF':>7}{'IDF':>8}{'CF*IDF':>10}")
        for x in r["top"][:args.top]:
            print(f"{x['term']:<22}{x['rank']:>6}{x['cf']:>8}{x['df']:>7}{x['idf']:>8.3f}{x['power']:>10.1f}")
    elif args.cmd == "terms":
        r = eng.zipf_terms(args.cond, " ".join(args.words))
        print(f"condition {r['condition']}, N = {r['documents']} documents, idf = log10(N / df)\n")
        print(f"{'term':<20}{'CF':>8}{'DF':>7}{'CF/DF':>8}{'DF/N':>8}{'IDF':>8}")
        for x in sorted(r["rows"], key=lambda x: -x["cf"]):
            print(f"{x['term']:<20}{x['cf']:>8}{x['df']:>7}{x['cf_per_doc']:>8}{x['df_ratio']:>8.3f}{x['idf']:>8.3f}")
        if r["missing"]:
            print("\nnot in the vocabulary under this condition:", ", ".join(r["missing"]))
    elif args.cmd == "spell":
        w = args.word.lower()
        print(f"'{w}' {'is' if w in eng.spell else 'is not'} in the dictionary ({len(eng.spell.words)} words)")
        for r in eng.spell.suggest(w, 10, args.k):
            print(f"  {r['word']:<24}distance={r['distance']}  cf={r['cf']}")
    elif args.cmd == "vsm":
        _vsm(eng, args)
    elif args.cmd == "w2v":
        _w2v(eng, args)


def _w2v(eng: Engine, args) -> None:
    import threading
    import time
    a = args.action
    if a == "train":
        params = {"model": args.model, "dim": args.dim, "window": args.window, "negative": args.negative, "min_count": args.min_count,
                  "epochs": args.epochs, "seed": args.seed, "remove_stopwords": not args.keep_stopwords, "name": args.name}
        job = eng.w2v_train(params)                       # queued; the engine's worker thread trains it
        while job["state"] in ("queued", "running"):
            time.sleep(5)
            if job.get("epoch"):
                print(f"  {job['progress']:.0%}  epoch {job['epoch']}  loss {job['loss']}", flush=True)
        if job["state"] != "done":
            print("training failed:", job.get("message")); return
        m = eng.w2v.info()
        print(f"{job['name']}: vocabulary {m['vocabulary']}, {m['corpus_words']} training words, {m['train_seconds']}s, "
              f"loss per epoch {m['loss_history']}\nsaved to {args.data}/w2v/{job['name']}.json")
    elif a == "list":
        st = eng.w2v_status()
        print(f"{'name':<28}{'model':<10}{'dim':>4}{'win':>4}{'min':>4}{'ep':>4}{'vocab':>7}{'loss':>8}{'secs':>7}  trained")
        for m in st["models"]:
            print(f"{m['name']:<28}{m['model']:<10}{m['dim']:>4}{m['window']:>4}{m['min_count']:>4}{m['epochs']:>4}{m['vocabulary']:>7}"
                  f"{(m['final_loss'] or 0):>8.4f}{m['train_seconds']:>7}  {m.get('trained', '')}{'   <- active' if m['name'] == st['active'] else ''}")
        if not st["models"]:
            print("no models yet - python3 cli.py w2v train")
    elif a in ("use", "delete"):
        if len(args.words) != 1:
            print(f"usage: w2v {a} NAME"); return
        print(eng.w2v_select(args.words[0]) if a == "use" else eng.w2v_delete(args.words[0]))
    elif a == "vocab":
        v = eng.w2v_vocab(args.min_count, not args.keep_stopwords)
        print(f"{v['collection']}\ndocuments {v['documents']}, sentences {v['sentences']}, tokens {v['tokens']}"
              + (f" (+{v['stopword_tokens']} stop-word tokens removed)" if v['stopword_tokens'] is not None else "")
              + f"\ntypes {v['types']} (hapax {v['hapax']} = {v['hapax_share']:.1%}); min_count {v['min_count']} keeps {v['vocabulary']} types "
              f"covering {v['coverage']:.1%} of the tokens; {v['avg_sentence_length']} tokens per sentence")
        print("\nmin_count   types  share   tokens  share")
        for t in v["thresholds"]:
            print(f"{t['min_count']:>9}{t['types']:>8}{t['types_share']:>7.1%}{t['tokens']:>9}{t['token_share']:>7.1%}")
        print("\ntraining examples per epoch (fixed window / dynamic window expectation / CBOW):")
        for e in v["examples"]:
            print(f"  window {e['window']:>2}: skip-gram {e['skipgram_fixed']:>9} / {e['skipgram_dynamic']:>9}   CBOW {e['cbow']:>7}")
        print("\ntop words: " + ", ".join(f"{w} {c}" for w, c in v["top_words"][:30]))
        if v["walk"]:
            st = v["walk"]["steps"][0]
            print(f"\nexample ({v['walk']['doc_id']}): {st['sentence']}\n  tokens      {st['tokens']}\n  lower-case  {st['lower']}\n  no stopword {st['no_stopwords']}\n  kept        {st['kept']}")
    elif a == "examples":
        r = eng.w2v_examples(" ".join(args.words), args.window, not args.keep_stopwords)
        print(f"sentence after pre-processing: {r['words']}\nwindow {r['window']}: {r['n_skipgram']} skip-gram pairs, {r['n_cbow']} CBOW examples\n")
        print("skip-gram (input centre -> target context)")
        for e in r["skipgram"]:
            print(f"  {e['centre']:<20} -> {e['context']:<20} (offset {e['offset']:+d})")
        print("CBOW (input context words -> target centre)")
        for e in r["cbow"]:
            print(f"  {' '.join(e['context']):<50} -> {e['centre']}")
    elif a == "similar":
        for w in args.words:
            r = eng.w2v_similar(w, 15, args.use or None)
            print(f"{r['word']} (count {r['count']}, model {r['model']}):")
            for nb in r["neighbours"]:
                print(f"    {nb['word']:<28}{nb['similarity']:.3f}   count {nb['count']}")
    elif a == "analogy":
        if len(args.words) != 3:
            print("usage: w2v analogy a b c   (a is to b as c is to ?)"); return
        for x in eng.w2v_analogy(*args.words, model=args.use or None)["results"]:
            print(f"    {x['word']:<28}{x['similarity']:.3f}")
    elif a == "compare":
        r = eng.w2v_compare(" ".join(args.words), args.models, args.n)
        names = [m["name"] for m in r["models"]]
        for row in r["words"]:
            print(f"\n{row['word']}")
            for nm in names:
                m = row["models"][nm]
                print(f"  {nm:<26} " + ("(not in vocabulary)" if not m else ", ".join(f"{x['word']} {x['similarity']:.2f}" for x in m["neighbours"])))
            for k, o in row["overlap"].items():
                print(f"  overlap {k}: {o['jaccard']:.2f}  shared: {', '.join(o['shared']) or '-'}")
    elif a == "pairs":
        r = eng.w2v_pairs(" ".join(args.words), args.use or None)
        w = max(len(x) for x in r["words"]) + 1
        print(" " * w + "".join(f"{x[:7]:>8}" for x in r["words"]))
        for x, row in zip(r["words"], r["matrix"]):
            print(f"{x:<{w}}" + "".join(f"{v:>8.2f}" for v in row))
        print(f"\nmean within groups {r['mean_within']}, between groups {r['mean_between']}, random pairs {r['mean_random']}")
        if r["missing"]:
            print("not in vocabulary:", ", ".join(r["missing"]))
    elif a == "map":
        r = eng.w2v_projection(" ".join(args.words), 120, args.use or None, args.method, args.perplexity)
        print(f"{r['method']} map of {len(r['points'])} words ({r['seconds']}s); k-NN preservation {r['knn_preservation']}"
              + (f"; explained variance {r['explained_variance']}" if r.get("explained_variance") else f"; KL {r['kl']}"))
        for p in sorted(r["points"], key=lambda p: (p["group"], -p["count"])):
            print(f"  {p['group']:<14}{p['word']:<24}{p['x']:>8.3f}{p['y']:>8.3f}")
        if args.svg:
            from ir.svgplot import scatter_plot
            out = Path(args.svg); out.mkdir(parents=True, exist_ok=True)
            path = out / f"w2v_{r['method']}_{r['model']}.svg"
            path.write_text(scatter_plot(r["points"], f"{r['method'].upper()} map - {r['model']}"), encoding="utf-8")
            print("saved", path)
    elif a == "eval":
        r = eng.w2v_eval(args.models, args.k)
        for m in r["models"]:
            print(f"\n{m['name']}  (vocabulary {m['vocabulary']}, {m['stem_groups']} stem groups)")
            print(f"  {'count':<8}{'words':>7}{'sim@k':>8}{'nb.count':>10}{'variant hit':>13}{'MRR':>7}  example")
            for b in m["buckets"]:
                ex = b["examples"][0] if b["examples"] else None
                print(f"  {b['bucket']:<8}{b['words']:>7}{b['sim_at_k']:>8.3f}{b['neighbour_count']:>10}"
                      f"{(b['variant_hit'] if b['variant_hit'] is not None else float('nan')):>13.2f}{(b['variant_mrr'] or 0):>7.2f}  "
                      + (f"{ex['word']} -> {ex['variant']} (rank {ex['rank']})" if ex else ""))
    elif a == "overlap":
        r = eng.w2v_overlap(args.models, args.k)
        print(f"neighbour overlap (Jaccard of top-{r['k']}) between {r['a']} and {r['b']}: overall {r['overall']}")
        for b in r["buckets"]:
            print(f"  count {b['bucket']:<8}{b['words']:>6} words   overlap {b['overlap']:.3f}")
        print("most stable:  " + ", ".join(f"{x['word']} {x['overlap']:.2f}" for x in r["most_stable"]))
        print("least stable: " + ", ".join(f"{x['word']} {x['overlap']:.2f}" for x in r["least_stable"]))

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:          # unknown word, NCBI unreachable, …
        sys.exit(f"error: {e}")
