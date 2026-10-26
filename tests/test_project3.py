"""Unit tests for Project #3 (term weighting, vector space model, evaluation, relevance judgements):
python3 -m unittest discover -s tests -v"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir import evaluation as ev  # noqa: E402
from ir.pmc_parser import Document, load_jsonl  # noqa: E402
from ir.relevance import QuerySet, doc_labels  # noqa: E402
from ir.tfidf import VectorSpace, describe_scheme, df_weight, parse_scheme, tf_weight  # noqa: E402


def _doc(i, title, abstract, mesh=(), keywords=()):
    return Document(doc_id=f"D{i}", title=title, abstract=[abstract], keywords=list(keywords),
                    mesh=[{"d": d, "major": maj, "q": []} for d, maj in mesh], article_type="pubmed-abstract")


DOCS = [
    _doc(1, "Semaglutide and weight loss", "Semaglutide reduced body weight in adults with obesity. Weight loss was sustained.",
         mesh=[("Semaglutide", False), ("Obesity", True), ("Weight Loss", False)], keywords=["semaglutide", "obesity"]),
    _doc(2, "Tirzepatide in type 2 diabetes", "Tirzepatide improved glycemic control. HbA1c fell in patients with diabetes.",
         mesh=[("Tirzepatide", True), ("Diabetes Mellitus, Type 2", True)]),
    _doc(3, "Gut microbiota", "The gut microbiota and bile acids were changed in mice. Mice mice mice mice mice.",
         keywords=["gut microbiota"]),
    _doc(4, "Kidney outcomes", "Chronic kidney disease progression slowed with GLP-1 receptor agonists in patients with diabetes.",
         mesh=[("Renal Insufficiency, Chronic", True), ("Diabetes Mellitus, Type 2", False)]),
    _doc(5, "A note", "Patients patients patients. Obesity obesity."),
]


class WeightTests(unittest.TestCase):
    def test_tf_variants(self):
        self.assertEqual(tf_weight("n", 7), 7.0)
        self.assertAlmostEqual(tf_weight("l", 1), 1.0)
        self.assertAlmostEqual(tf_weight("l", math.e), 2.0)
        self.assertAlmostEqual(tf_weight("l", 30), 1 + math.log(30))
        self.assertAlmostEqual(tf_weight("a", 5, 10), 0.75)
        self.assertEqual(tf_weight("b", 99), 1.0)
        self.assertEqual(tf_weight("n", 0), 0.0)

    def test_df_variants(self):
        self.assertEqual(df_weight("n", 3, 100), 1.0)
        self.assertAlmostEqual(df_weight("t", 1, 100), math.log(100))
        self.assertAlmostEqual(df_weight("t", 100, 100), 0.0)
        self.assertAlmostEqual(df_weight("p", 10, 100), math.log(9))
        self.assertEqual(df_weight("p", 60, 100), 0.0)        # more than half the documents -> 0, never negative
        self.assertEqual(df_weight("t", 0, 100), 0.0)

    def test_parse_scheme(self):
        self.assertEqual(parse_scheme("ltc"), ("l", "t", "c"))
        self.assertEqual(parse_scheme("BM25"), ("bm25", "", ""))
        self.assertEqual(describe_scheme("nnn")["name"], "raw TF")
        with self.assertRaises(ValueError):
            parse_scheme("xyz")
        with self.assertRaises(ValueError):
            parse_scheme("lt")


class VectorSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vs = VectorSpace(DOCS)

    def test_statistics(self):
        vs = self.vs
        self.assertEqual(vs.N, 5)
        self.assertEqual(vs.df["obes"], 2)                   # D1 and D5
        self.assertEqual(vs.tf["D3"]["mice"], 6)
        self.assertEqual(vs.max_tf["D3"], 6)
        self.assertAlmostEqual(vs.idf("obes"), math.log(5 / 2))
        row = vs.term_row("mice")
        self.assertEqual((row["df"], row["cf"], row["max_tf"]), (1, 6, 6))
        self.assertEqual(vs.analyse_terms(["the"])[0]["stopword"], True)
        st = vs.vocabulary_stats()
        self.assertEqual(st["documents"], 5)
        self.assertEqual(sum(h["terms"] for h in st["df_histogram"]), st["vocabulary"])

    def test_weights_and_normalisation(self):
        vs = self.vs
        w, norms = vs.weights("nnn")
        self.assertEqual(w["mice"]["D3"], 6.0)
        w, _ = vs.weights("lnn")
        self.assertAlmostEqual(w["mice"]["D3"], 1 + math.log(6))
        w, _ = vs.weights("ntn")
        self.assertAlmostEqual(w["mice"]["D3"], 6 * math.log(5))
        w, _ = vs.weights("ltc")
        for d in vs.docs:                                    # every document vector has unit length
            self.assertAlmostEqual(math.sqrt(sum(w[t][d] ** 2 for t in vs.tf[d])), 1.0, places=9)
        vec = vs.doc_vector("D3", "nnn", top=1)
        self.assertEqual(vec["vector"][0]["term"], "mice")
        self.assertAlmostEqual(sum(r["share"] for r in vs.doc_vector("D3", "nnn")["vector"]), 1.0, places=3)

    def test_prob_idf_drops_common_terms(self):
        w, _ = self.vs.weights("lpc")
        self.assertNotIn("patient", w)                       # df 3 of 5 -> prob idf 0 -> not in the vector

    def test_query_and_ranking(self):
        vs = self.vs
        qtf, steps = vs.query_terms("the obesity Obesity")
        self.assertEqual(qtf["obes"], 2)
        self.assertEqual(steps[0]["role"], "stopword")
        qv = vs.query_vector(qtf, "nnc")
        self.assertAlmostEqual(qv["obes"], 1.0)
        ranked, info = vs.rank("mice microbiota", "ltc")
        self.assertEqual(ranked[0][0], "D3")
        self.assertEqual(len(ranked), 1)
        res = vs.search("obesity weight", "ltc", k=10)
        self.assertEqual(res["results"][0]["doc_id"], "D1")
        self.assertEqual(res["scheme"], "ltc")
        parts = {c["term"]: c["part"] for c in res["results"][0]["contributions"]}
        self.assertAlmostEqual(sum(parts.values()), res["results"][0]["score"], places=3)
        # cosine with a normalised query and normalised documents is bounded by 1
        for r in res["results"]:
            self.assertLessEqual(r["score"], 1.0 + 1e-9)

    def test_unnormalised_dot_product_grows_with_length(self):
        filler = " ".join(f"filler{i} filler{i}" for i in range(60))          # a long document that mentions the query terms a bit more often
        long_doc = _doc(9, "Obesity", "obesity weight obesity weight obesity weight obesity weight " + filler)
        vs = VectorSpace(DOCS + [long_doc])
        ranked_n, _ = vs.rank("obesity weight", "ntn")
        ranked_c, _ = vs.rank("obesity weight", "ntc")
        self.assertEqual(ranked_n[0][0], "D9")               # raw tf x idf: the long repetitive document wins
        self.assertEqual(ranked_c[0][0], "D1")               # cosine: the short focused document wins

    def test_bm25(self):
        ranked, info = self.vs.rank("mice", "bm25")
        self.assertEqual(ranked[0][0], "D3")
        self.assertIn("mice", info["qvec"])
        with self.assertRaises(ValueError):
            self.vs.weights("bm25")

    def test_unknown_query_terms(self):
        ranked, info = self.vs.rank("zzzz", "ltc")
        self.assertEqual(ranked, [])
        self.assertEqual(info["terms"][0]["role"], "unknown")


class EvaluationTests(unittest.TestCase):
    QRELS = {"a": 2, "b": 1, "c": 1}

    def test_precision_recall(self):
        run = ["a", "x", "b", "y", "z"]
        self.assertAlmostEqual(ev.precision_at_k(run, self.QRELS, 2), 0.5)
        self.assertAlmostEqual(ev.precision_at_k(run, self.QRELS, 5), 0.4)
        self.assertAlmostEqual(ev.recall_at_k(run, self.QRELS, 5), 2 / 3)
        self.assertAlmostEqual(ev.r_precision(run, self.QRELS), 2 / 3)
        self.assertEqual(ev.precision_at_k([], self.QRELS, 5), 0.0)
        self.assertEqual(ev.recall_at_k(run, {}, 5), 0.0)

    def test_rr_ap(self):
        self.assertAlmostEqual(ev.reciprocal_rank(["x", "y", "b"], self.QRELS), 1 / 3)
        self.assertEqual(ev.reciprocal_rank(["x"], self.QRELS), 0.0)
        # AP = mean of precision at the relevant ranks over ALL relevant docs: (1/1 + 2/3) / 3
        self.assertAlmostEqual(ev.average_precision(["a", "x", "b", "y"], self.QRELS), (1 + 2 / 3) / 3)

    def test_ndcg(self):
        run = ["b", "a", "x"]                                # grades 1, 2, 0
        dcg = (2 ** 1 - 1) / math.log2(2) + (2 ** 2 - 1) / math.log2(3)
        idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(ev.ndcg_at_k(run, self.QRELS, 3), dcg / idcg)
        self.assertAlmostEqual(ev.ndcg_at_k(["a", "b", "c"], self.QRELS, 3), 1.0)
        self.assertEqual(ev.ndcg_at_k(run, {}, 3), 0.0)

    def test_evaluate_run_and_mean(self):
        m = ev.evaluate_run(["a", "x", "b"], self.QRELS, ks=(1, 3))
        self.assertEqual((m["relevant"], m["first_rank"]), (3, 1))
        self.assertAlmostEqual(m["P@3"], 2 / 3)
        mean = ev.mean_measures([m, ev.evaluate_run(["x", "a"], self.QRELS, ks=(1, 3))])
        self.assertAlmostEqual(mean["MRR"], (1 + 0.5) / 2)
        self.assertIn("MAP", mean)
        self.assertEqual(mean["queries"], 2)
        condensed = ev.evaluate_run(["u", "a"], self.QRELS, ks=(1,), condensed=True, judged={"a", "b", "c"})
        self.assertEqual(condensed["RR"], 1.0)                # the unjudged document u is dropped first

    def test_rank_comparison(self):
        a = ["a", "b", "c", "d"]
        self.assertEqual(ev.overlap_at_k(a, ["c", "d", "a", "b"], 4), 1.0)
        self.assertAlmostEqual(ev.overlap_at_k(a, ["x", "y", "a", "b"], 4), 0.5)
        self.assertAlmostEqual(ev.kendall_tau(a, a, 4), 1.0)
        self.assertAlmostEqual(ev.kendall_tau(a, list(reversed(a)), 4), -1.0)
        rows = ev.rank_changes(a, ["b", "z"], 2)
        self.assertEqual([(r["doc_id"], r["rank_a"], r["rank_b"]) for r in rows], [("a", 1, 0), ("b", 2, 1), ("z", 0, 2)])


class RelevanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "queries.json"
        self.path.write_text(json.dumps({"notes": "t", "queries": [
            {"id": "q1", "text": "semaglutide weight loss in obesity", "groups": [["semaglutide"], ["obes", "weight loss"]], "manual": {}},
            {"id": "q2", "text": "type 2 diabetes", "groups": [["diabetes mellitus, type 2"]], "manual": {}},
            {"id": "q3", "text": "gut microbiota", "groups": [["microbio"]], "manual": {}}]}), encoding="utf-8")
        self.docs = {d.doc_id: d for d in DOCS}

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_and_rules(self):
        self.assertIn(("Obesity", "mesh", True), doc_labels(DOCS[0]))
        qs = QuerySet(self.path, self.docs)
        self.assertEqual(qs.qrels("q1"), {"D1": 2})                 # major-topic MeSH match -> grade 2
        self.assertEqual(qs.qrels("q2"), {"D2": 2, "D4": 1})        # D4: non-major match -> grade 1
        self.assertEqual(qs.qrels("q3"), {"D3": 1})                 # keyword only -> grade 1
        self.assertEqual(qs.judged(), {"D1", "D2", "D3", "D4"})     # D5 has no label at all
        self.assertEqual(qs.by_text("  Semaglutide weight loss in OBESITY ").id, "q1")
        self.assertIsNone(qs.by_text("nothing"))
        row = next(r for r in qs.summary() if r["id"] == "q2")
        self.assertEqual((row["relevant"], row["grade2"], row["grade1"], row["from_mesh"]), (2, 1, 1, 2))

    def test_manual_override_persists(self):
        qs = QuerySet(self.path, self.docs)
        r = qs.set_manual("q3", "D5", 2)
        self.assertEqual((r["grade"], r["manual"]), (2, True))
        self.assertEqual(qs.qrels("q3"), {"D3": 1, "D5": 2})
        qs2 = QuerySet(self.path, self.docs)                        # reloaded from disk
        self.assertEqual(qs2.get("q3").manual, {"D5": 2})
        qs2.set_manual("q3", "D5", None)
        self.assertEqual(qs2.qrels("q3"), {"D3": 1})
        qs2.set_manual("q3", "D3", 0)                               # a manual 0 removes a rule-based judgement
        self.assertEqual(qs2.qrels("q3"), {})
        with self.assertRaises(ValueError):
            qs2.set_manual("nope", "D1", 1)


class CollectionTests(unittest.TestCase):
    """The shipped 3,000-abstract collection and its query set (skipped when absent)."""
    PATH = ROOT / "data" / "vsm" / "pubmed_glp1_3000.jsonl"

    def setUp(self):
        if not self.PATH.is_file():
            self.skipTest("collection not present")

    def test_records_carry_mesh(self):
        docs = load_jsonl(self.PATH)
        self.assertGreaterEqual(len(docs), 1000)
        with_mesh = sum(1 for d in docs if d.mesh)
        self.assertGreater(with_mesh / len(docs), 0.4)
        self.assertTrue(all(d.article_type == "pubmed-abstract" for d in docs))

    def test_engine_pipeline(self):
        from ir.engine import Engine
        eng = Engine(ROOT / "data")
        eng.load()
        info = eng.vsm_info()
        self.assertGreaterEqual(info["documents"], 1000)
        qs = eng.vsm_queries()
        self.assertGreaterEqual(len(qs.queries), 10)
        self.assertTrue(all(len(qs.qrels(q.id)) >= 10 for q in qs.queries))
        res = eng.vsm_search(qs.queries[0].text, "ltc", k=10)
        self.assertEqual(len(res["results"]), 10)
        self.assertEqual(res["query_id"], qs.queries[0].id)
        self.assertIn("P@10", res["measures"])
        cmp = eng.vsm_compare(qs.queries[0].text, "ltn,ltc", k=10)
        self.assertEqual(cmp["schemes"][0]["overlap"], 1.0)
        e = eng.vsm_eval("nnn,ltc", 10)
        self.assertEqual(len(e["schemes"][0]["per_query"]), len(qs.queries))
        self.assertGreater(e["schemes"][1]["mean"]["MAP"], e["schemes"][0]["mean"]["MAP"])   # tf-idf + L2 beats raw tf


if __name__ == "__main__":
    unittest.main()
