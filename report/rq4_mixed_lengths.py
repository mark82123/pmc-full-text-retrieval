#!/usr/bin/env python3
"""RQ4 side experiment: the same weighting on a collection of very unequal
document lengths (the 6 full-text PMC articles in data/ mixed with the 1,000
GLP-1 abstracts of Project #2).  Without L2 normalisation the long full-text
articles reach the top of almost every query; with it they compete on
direction, not length.    python3 report/rq4_mixed_lengths.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ir.engine import Engine          # noqa: E402
from ir.tfidf import VectorSpace      # noqa: E402


class FullTextSpace(VectorSpace):
    def text_of(self, doc):
        return doc.full_text()        # title + abstract + body (abstract only for PubMed records)


eng = Engine(ROOT / "data")
eng.load()
docs = list(eng.index.documents.values())
vs = FullTextSpace(docs)
full = {d.doc_id for d in docs if d.sections}
lens = sorted(vs.length[d] for d in full)
print(f"{vs.N} documents: {len(full)} full-text articles ({lens[0]}–{lens[-1]} index terms) + {vs.N - len(full)} abstracts (mean {vs.avg_length:.0f})\n")
queries = ["semaglutide weight loss in obesity", "GLP-1 receptor agonists and cardiovascular outcomes", "chronic kidney disease",
           "gut microbiota and bile acid metabolism", "insulin secretion from pancreatic beta cells", "type 1 diabetes", "nausea vomiting",
           "polycystic ovary syndrome", "bariatric surgery", "cost-effectiveness"]
print(f"{'query':<52}" + "".join(f"{s:>8}" for s in ["nnn", "ltn", "nnc", "ltc", "bm25"]) + "   (full-text articles in the top 10)")
tot = {s: 0 for s in ["nnn", "ltn", "nnc", "ltc", "bm25"]}
for q in queries:
    row = []
    for s in tot:
        n = sum(1 for d in vs.ranking_ids(q, s)[:10] if d in full)
        tot[s] += n
        row.append(n)
    print(f"{q:<52}" + "".join(f"{n:>8}" for n in row))
print(f"{'total of ' + str(len(queries) * 10):<52}" + "".join(f"{tot[s]:>8}" for s in tot))
