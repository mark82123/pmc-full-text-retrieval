"""Unit tests for Project #2 (Zipf, edit distance / spelling, word2vec):
python3 -m unittest discover -s tests -v"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir import zipf  # noqa: E402
from ir.engine import Engine  # noqa: E402
from ir.ncbi import pubmed_records  # noqa: E402
from ir.search import parse_query  # noqa: E402
from ir.spell import SpellIndex, bounded_distance, edit_distance, edit_matrix  # noqa: E402
from ir.word2vec import Word2Vec, sentences_from_texts  # noqa: E402


class ZipfTests(unittest.TestCase):
    def test_conditions(self):
        text = "The Data, data. GLP-1 agonists reduced the weights; DATA!"
        self.assertEqual(zipf.terms_for(text, "A"), ["the", "data,", "data.", "glp-1", "agonists", "reduced", "the", "weights;", "data!"])
        self.assertEqual(zipf.terms_for(text, "B"), ["the", "data", "data", "glp", "1", "agonists", "reduced", "the", "weights", "data"])
        self.assertEqual(zipf.terms_for(text, "C"), ["data", "data", "glp", "1", "agonists", "reduced", "weights", "data"])
        self.assertEqual(zipf.terms_for(text, "D"), ["data", "data", "glp", "1", "agonist", "reduc", "weight", "data"])
        self.assertEqual(zipf.terms_for(text, "E"), ["the", "data", "data", "glp", "1", "agonist", "reduc", "the", "weight", "data"])

    def test_fit_recovers_exact_power_law(self):
        freqs = [round(1e6 / r ** 1.2) for r in range(1, 400)]
        f = zipf.fit_loglog(freqs)
        self.assertAlmostEqual(f["exponent"], 1.2, places=2)
        self.assertAlmostEqual(f["slope"], -f["exponent"])
        self.assertAlmostEqual(f["intercept"], 6.0, places=1)
        self.assertGreater(f["r2"], 0.999)
        self.assertLess(f["rmse"], 0.01)
        part = zipf.fit_loglog(freqs, 10, 100)
        self.assertEqual((part["from"], part["to"], part["n"]), (10, 100, 91))

    def test_cf_df_idf(self):
        texts = ["insulin insulin insulin glucose", "glucose weight", "weight loss", "weight"]
        col = zipf.Collection("B", texts)
        self.assertEqual((col.n_docs, col.n_tokens, len(col.cf)), (4, 9, 4))
        self.assertEqual((col.cf["insulin"], col.df["insulin"]), (3, 1))      # high CF, low DF
        self.assertEqual((col.cf["weight"], col.df["weight"]), (3, 3))
        self.assertAlmostEqual(col.idf("insulin"), 0.602, places=3)            # log10(4/1)
        self.assertGreater(col.idf("insulin"), col.idf("weight"))
        self.assertEqual(col.ranked[0], ("insulin", 3))
        s = col.summary()
        self.assertEqual((s["documents"], s["tokens"], s["vocabulary"], s["hapax"]), (4, 9, 4, 1))
        self.assertEqual([b[0] for b in zipf.segment_bounds(1000)], ["high", "middle", "low"])
        self.assertEqual(zipf.segment_bounds(1000), [("high", 1, 9), ("middle", 10, 99), ("low", 100, 1000)])

    def test_stemming_groups(self):
        g = zipf.stemming_groups(["patients patient treated treatment treating", "patient treats obesity"])
        self.assertEqual((g["words"], g["stems"]), (7, 4))      # patient, treat, treatment, obes
        self.assertEqual(g["groups"][0]["stem"], "treat")


class EditDistanceTests(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("diabetes", "diabetes"), 0)
        self.assertEqual(edit_distance("obestiy", "obesity"), 1)                       # transposition
        self.assertEqual(edit_distance("obestiy", "obesity", transpositions=False), 2)
        for a, b in [("semaglutide", "samegluitde"), ("insulin", "insuline"), ("abc", "xyz"), ("ca", "abc")]:
            self.assertEqual(edit_distance(a, b), edit_distance(b, a))
            full = edit_distance(a, b)
            for k in range(5):
                self.assertEqual(bounded_distance(a, b, k), full if full <= k else None, (a, b, k))

    def test_matrix_and_alignment(self):
        r = edit_matrix("kitten", "sitting")
        self.assertEqual(r["distance"], 3)
        self.assertEqual(r["matrix"][0], list(range(8)))
        self.assertEqual(sum(1 for o in r["ops"] if o["op"] != "match"), 3)
        self.assertEqual("".join(o["a"] for o in r["ops"]), "kitten")
        self.assertEqual("".join(o["b"] for o in r["ops"]), "sitting")
        self.assertEqual((r["path"][0], r["path"][-1]), ([0, 0], [6, 7]))
        self.assertEqual([o["op"] for o in edit_matrix("ab", "ba")["ops"]], ["transpose"])

    def test_spell_index(self):
        from collections import Counter
        sp = SpellIndex()
        sp.add(Counter({"diabetes": 50, "diabetic": 9, "obesity": 40, "obese": 5, "insulin": 30, "dates": 2}))
        self.assertEqual(sp.suggest("diabtes")[0], {"word": "diabetes", "distance": 1, "cf": 50})
        self.assertEqual([r["word"] for r in sp.suggest("obesty")], ["obesity", "obese"])
        self.assertEqual(sp.suggest("xylophone"), [])
        self.assertNotIn("insulin", [r["word"] for r in sp.suggest("insulin")])
        sp.remove(Counter({"diabetes": 50}))
        self.assertNotIn("diabetes", sp)
        self.assertEqual(sp.suggest("diabtes")[0]["word"], "dates")


def _records():
    topics = [("Semaglutide reduces body weight in obesity", "Semaglutide treatment reduced body weight in adults with obesity. Nausea was the most common adverse event."),
              ("Liraglutide and glycaemic control in diabetes", "Liraglutide improved glycaemic control in patients with type 2 diabetes. Weight loss was also observed."),
              ("Insulin secretion in mice", "GLP-1 increased insulin secretion in mice. Glucose tolerance improved in treated mice.")]
    return [{"pmid": str(100 + i), "title": t, "abstract": [a], "journal": "J Test", "year": "2026", "authors": [], "keywords": [], "doi": ""}
            for i, (t, a) in enumerate(topics)]


class CollectionEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "pubmed_test.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in _records()), encoding="utf-8")
        self.eng = Engine(self.tmp.name)
        self.eng.load()

    def tearDown(self):
        self.tmp.cleanup()

    def test_jsonl_collection_is_indexed(self):
        self.assertEqual(sorted(self.eng.index.docs), ["PMID100", "PMID101", "PMID102"])
        name, docs = self.eng.collection_docs()
        self.assertIn("pubmed_test", name)
        self.assertEqual(len(docs), 3)
        self.assertEqual(self.eng.search("semaglutide")["results"][0]["doc_id"], "PMID100")

    def test_zipf_api(self):
        z = self.eng.zipf()
        self.assertEqual([c["key"] for c in z["conditions"]], ["A", "B", "C", "D", "E"])
        a, b, c, d, e = z["conditions"]
        self.assertTrue(all(x["documents"] == 3 for x in z["conditions"]))
        self.assertGreater(a["vocabulary"], b["vocabulary"])          # punctuation glued to words inflates the vocabulary
        self.assertGreater(b["tokens"], c["tokens"])                  # stop words removed
        self.assertGreaterEqual(c["vocabulary"], d["vocabulary"])     # stemming merges forms
        self.assertEqual(e["tokens"], b["tokens"])                    # E stems B without removing anything
        self.assertGreaterEqual(b["vocabulary"], e["vocabulary"])
        self.assertGreaterEqual(e["vocabulary"], d["vocabulary"])     # D = E minus the stop-word stems
        rows = {r["term"]: r for r in self.eng.zipf_terms("C", "mice weight zzz the")["rows"]}
        self.assertEqual((rows["mice"]["cf"], rows["mice"]["df"]), (3, 1))
        self.assertEqual(rows["weight"]["df"], 2)
        self.assertEqual(self.eng.zipf_terms("C", "zzz the")["missing"], ["zzz", "the"])

    def test_cfdf_map(self):
        # 'the' in every document (function), 'intro' once in 60% of the documents (boilerplate), 'alpha' bursty in half of the
        # documents (topic), 'zz' three times in one of ten documents (keyword), numbers in their own zone
        texts = [("the alpha alpha alpha 12" if i % 2 else "the beta 12") + (" intro" if i < 6 else "") + (" zz zz zz" if i < 2 else "") for i in range(10)]
        col = zipf.Collection("B", texts)
        m = col.cfdf_map(key_df=0.3)
        zone = {x["term"]: z["name"] for z in m["zones"] for x in z["examples"]}
        self.assertEqual(zone["the"], "function")
        self.assertEqual(zone["intro"], "boilerplate")
        self.assertEqual(zone["alpha"], "topic")
        self.assertEqual(zone["zz"], "keyword")
        self.assertEqual(zone["12"], "number")
        self.assertEqual(sum(z["terms"] for z in m["zones"]), m["vocabulary"])
        self.assertAlmostEqual(col.poisson_df(0), 0)
        self.assertLess(col.poisson_df(30), 10)                           # never more than N documents
        self.assertEqual({p[3] for p in m["points"]} <= set(zipf.MAP_ZONES), True)
        api = self.eng.zipf_cfdf("B")
        self.assertEqual([z["name"] for z in api["zones"]], list(zipf.MAP_ZONES))

    def test_resolving_power(self):
        # 'the' is in every document (idf 0) and each 'rare*' word occurs once: the middle words resolve best
        texts = [f"the alpha alpha alpha beta rare{i}" if i % 2 else f"the gamma gamma gamma beta rare{i}" for i in range(10)]
        col = zipf.Collection("B", texts)
        power = dict(zip((t for t, _ in col.ranked), col.powers()))
        self.assertEqual(power["the"], 0)
        self.assertAlmostEqual(power["alpha"], 15 * math.log10(10 / 5))
        self.assertGreater(power["alpha"], power["rare1"])
        r = col.resolving()
        self.assertEqual((r["upper"], r["lower"]), (r["auto"]["upper"], r["auto"]["lower"]))
        self.assertEqual([z["name"] for z in r["zones"]], ["common", "significant", "rare"])
        self.assertEqual(sum(z["terms"] for z in r["zones"]), r["vocabulary"])
        self.assertAlmostEqual(sum(z["power_share"] for z in r["zones"]), 1, places=3)
        self.assertEqual({x["term"] for x in r["top"][:2]}, {"alpha", "gamma"})
        self.assertEqual([p[0] for p in r["idf_curve"]], [p[0] for p in r["curve"]])   # same thinned ranks as the power curve
        self.assertTrue(all(0 <= v <= math.log10(10) for _, v in r["idf_curve"]))
        self.assertLess(r["idf_curve"][0][1], r["idf_curve"][-1][1])          # idf rises along the rank axis
        r = col.resolving(upper=2, lower=3)                              # explicit cut-offs; ranks are inclusive
        self.assertEqual([(z["from"], z["to"], z["terms"]) for z in r["zones"]], [(1, 1, 1), (2, 3, 2), (4, 14, 11)])
        self.assertEqual(r["zones"][0]["examples"], ["alpha"])
        self.assertEqual(col.resolving(upper=99, lower=1)["lower"], 14)   # clamped to the vocabulary, lower >= upper
        self.assertEqual(zipf.Collection("B", []).resolving()["zones"], [])

    def test_resolving_api(self):
        r = self.eng.zipf_resolving("C")
        self.assertEqual(r["condition"], "C")
        self.assertEqual(sum(z["terms"] for z in r["zones"]), r["vocabulary"])
        self.assertTrue(all(x["power"] >= y["power"] for x, y in zip(r["top"], r["top"][1:])))

    def test_spelling_correction_in_search(self):
        r = self.eng.search("semaglutde obesty")
        self.assertTrue(r["spell"]["auto"])
        self.assertEqual(r["query"], "semaglutide obesity")
        self.assertEqual(r["results"][0]["doc_id"], "PMID100")
        self.assertIn("<mark>", r["results"][0]["snippet_html"])
        r = self.eng.search("semaglutde obesty", autocorrect=False)
        self.assertEqual((r["total"], r["spell"]["corrected_query"]), (0, "semaglutide obesity"))
        self.assertEqual(self.eng.search("insulin")["spell"]["corrections"], [])
        self.assertEqual(self.eng.search("insul*")["spell"]["corrections"], [])    # wildcards are not spell-checked

    def test_fuzzy_query(self):
        node = parse_query("liraglutid~ AND diabets~1")
        self.assertEqual(node.describe(), "(liraglutid~ AND diabets~1)")
        r = self.eng.search("liraglutid~")
        self.assertEqual([d["doc_id"] for d in r["results"]], ["PMID101"])
        self.assertEqual(r["analysis"]["fuzzy"][0]["words"][0]["word"], "liraglutide")
        self.assertEqual(r["spell"]["corrections"], [])

    def test_remove_from_jsonl(self):
        self.assertTrue(self.eng.remove("PMID101", delete_file=True))
        e2 = Engine(self.tmp.name)
        self.assertEqual(e2.load(), 2)
        self.assertNotIn("liraglutide", self.eng.spell)

    def test_word2vec_job(self):
        job = self.eng.w2v_train({"model": "cbow", "dim": 10, "epochs": 2, "min_count": 1}, block=True)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(self.eng.w2v_similar("mice", 3)["neighbours"]), 3)
        with self.assertRaises(ValueError):
            self.eng.w2v_similar("notaword")
        e2 = Engine(self.tmp.name)
        e2.load()
        self.assertEqual(e2.w2v.vocab, self.eng.w2v.vocab)                 # model was saved and reloaded
        self.assertEqual(e2.w2v_status()["model"]["model"], "cbow")


class Word2VecTests(unittest.TestCase):
    def test_preprocessing(self):
        s = sentences_from_texts(["GLP-1 receptor agonists reduce weight. The SGLT2 inhibitors were used in 2024."])
        self.assertEqual(s, [["glp-1", "receptor", "agonists", "reduce", "weight"], ["sglt2", "inhibitors", "used"]])

    def _toy(self):
        import random
        rng = random.Random(0)
        groups = [["cat", "dog", "horse", "cow"], ["red", "blue", "green", "pink"]]
        ctx = [["animal", "pet", "farm", "fur"], ["colour", "paint", "shade", "hue"]]
        return [[rng.choice(groups[g]), rng.choice(ctx[g]), rng.choice(groups[g]), rng.choice(ctx[g])]
                for g in (0, 1) for _ in range(300)]

    def test_both_models_learn_co_occurrence(self):
        for sg in (True, False):
            m = Word2Vec(sg=sg, dim=16, window=3, negative=5, min_count=1, epochs=8, sample=0, alpha=0.05).train(self._toy())
            self.assertEqual(len(m.vocab), 16)
            self.assertGreater(m.similarity("cat", "dog"), m.similarity("cat", "blue"), "sg" if sg else "cbow")
            self.assertIn(m.most_similar("red", 1)[0][0], {"blue", "green", "pink", "colour", "paint", "shade", "hue"})
            self.assertLess(m.loss_history[-1], m.loss_history[0])
            self.assertEqual(len(m.projection(m.vocab)), 16)
            self.assertEqual(len(m.analogy("cat", "dog", "red", 3)), 3)

    def test_save_load(self):
        m = Word2Vec(dim=8, min_count=1, epochs=1).train(self._toy())
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d) / "m.json", {"documents": 5})
            m2, extra = Word2Vec.load(Path(d) / "m.json")
        self.assertEqual((m2.vocab, extra), (m.vocab, {"documents": 5}))
        self.assertEqual(m2.most_similar("cat", 3)[0][0], m.most_similar("cat", 3)[0][0])


class PubmedRecordTests(unittest.TestCase):
    def test_records(self):
        xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article><Journal><Title>J</Title>
          <JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal><ArticleTitle>GLP-1 trial.</ArticleTitle>
          <Abstract><AbstractText Label="BACKGROUND">Obesity is common.</AbstractText><AbstractText>Weight fell.</AbstractText></Abstract>
          <AuthorList><Author><LastName>Hu</LastName><ForeName>Y</ForeName></Author></AuthorList></Article></MedlineCitation></PubmedArticle>
          <PubmedArticle><MedlineCitation><PMID>2</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
          </PubmedArticleSet>"""
        recs = pubmed_records(xml)
        self.assertEqual(len(recs), 1)                                    # records without an abstract are skipped
        self.assertEqual((recs[0]["pmid"], recs[0]["title"], recs[0]["year"], recs[0]["authors"]), ("1", "GLP-1 trial", "2026", ["Y Hu"]))
        self.assertEqual(recs[0]["abstract"], ["Background: Obesity is common.", "Weight fell."])


if __name__ == "__main__":
    unittest.main()
