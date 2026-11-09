"""Unit tests for Project #4 (Reuters-21578 parsing, the ModApte split, topic relevance,
word tokeniser, document embeddings, hybrid fusion, evaluation):
python3 -m unittest discover -s tests -v"""

import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir import evaluation as ev  # noqa: E402
from ir import reuters  # noqa: E402
from ir.embedding import DocEmbedder, WordTokenizer, cooccurrence, cosine, hybrid_scores, word_topic_profile  # noqa: E402
from ir.tfidf import VectorSpace  # noqa: E402
from ir.word2vec import Word2Vec, sentences_from_texts  # noqa: E402

SGML = '''<!DOCTYPE lewis SYSTEM "lewis.dtd">
<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">
<DATE>26-FEB-1987 15:01:01.79</DATE>
<TOPICS><D>crude</D></TOPICS>
<PLACES><D>usa</D><D>uruguay</D></PLACES>
<PEOPLE></PEOPLE>
<ORGS></ORGS>
<EXCHANGES></EXCHANGES>
<COMPANIES></COMPANIES>
<UNKNOWN>
&#5;&#5;&#5;C T
&#22;&#22;&#1;f0704&#31;reute
u f BC-OIL   02-26 0105</UNKNOWN>
<TEXT>&#2;
<TITLE>OIL PRICES RISE</TITLE>
<DATELINE>    NEW YORK, Feb 26 - </DATELINE><BODY>Crude oil prices rose today as OPEC
output fell, traders said. Petroleum stocks were &lt;ABC> lower
than a year ago.
    Spot crude was quoted at 18 dlrs a barrel, up 50 cts.
 Reuter
&#3;</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="5545" NEWID="2">
<DATE>26-FEB-1987 15:02:20.00</DATE>
<TOPICS><D>earn</D><D>acq</D></TOPICS>
<PLACES><D>usa</D></PLACES>
<PEOPLE></PEOPLE>
<ORGS></ORGS>
<EXCHANGES></EXCHANGES>
<COMPANIES></COMPANIES>
<UNKNOWN>x</UNKNOWN>
<TEXT>&#2;
<TITLE>ACME CORP &lt;ACM> 4TH QTR NET</TITLE>
<DATELINE>    CHICAGO, Feb 26 - </DATELINE><BODY>Shr 50 cts vs 40 cts
    Net 1.2 mln vs 1.0 mln
    Revs 10.5 mln vs 9.8 mln
    The company also agreed to acquire Beta Inc for 20 mln dlrs.
 Reuter
&#3;</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="NO" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5546" NEWID="3">
<DATE>26-FEB-1987 15:03:27.51</DATE>
<TOPICS></TOPICS>
<PLACES></PLACES>
<PEOPLE></PEOPLE>
<ORGS></ORGS>
<EXCHANGES></EXCHANGES>
<COMPANIES></COMPANIES>
<UNKNOWN>x</UNKNOWN>
<TEXT TYPE="BRIEF">&#2;
<TITLE>******NO TOPIC STORY</TITLE>
&#3;</TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="5547" NEWID="4">
<DATE>26-FEB-1987 15:04:00.00</DATE>
<TOPICS><D>grain</D></TOPICS>
<PLACES><D>usa</D></PLACES>
<PEOPLE></PEOPLE>
<ORGS></ORGS>
<EXCHANGES></EXCHANGES>
<COMPANIES></COMPANIES>
<UNKNOWN>x</UNKNOWN>
<TEXT>&#2;
<TITLE>WHEAT HARVEST SEEN LARGER</TITLE>
<BODY>The wheat and corn harvest will be larger this year, the
agriculture department said in its grain crop report.
 Reuter
&#3;</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="5548" NEWID="5">
<DATE>26-FEB-1987 15:05:00.00</DATE>
<TOPICS><D>crude</D></TOPICS>
<PLACES><D>uk</D></PLACES>
<PEOPLE></PEOPLE>
<ORGS></ORGS>
<EXCHANGES></EXCHANGES>
<COMPANIES></COMPANIES>
<UNKNOWN>x</UNKNOWN>
<TEXT>&#2;
<TITLE>NORTH SEA OIL OUTPUT</TITLE>
<BODY>North Sea oil output rose, petroleum traders said. Opec crude
prices were steady.
 Reuter
&#3;</BODY></TEXT>
</REUTERS>
'''


class SgmlTests(unittest.TestCase):
    def setUp(self):
        self.recs = reuters.parse_sgml(SGML)

    def test_records_and_attributes(self):
        self.assertEqual([r["newid"] for r in self.recs], [1, 2, 3, 4, 5])
        r = self.recs[0]
        self.assertEqual((r["lewissplit"], r["topics_attr"], r["topics"], r["places"]), ("TRAIN", "YES", ["crude"], ["usa", "uruguay"]))
        self.assertEqual(r["title"], "OIL PRICES RISE")
        self.assertEqual(self.recs[2]["type"], "BRIEF")
        self.assertEqual(self.recs[2]["body"], [])

    def test_body_is_unwrapped_and_cleaned(self):
        body = self.recs[0]["body"]
        self.assertEqual(len(body), 2, body)                               # two paragraphs (the second starts with four spaces)
        self.assertEqual(body[0], "Crude oil prices rose today as OPEC output fell, traders said. Petroleum stocks were <ABC> lower than a year ago.")
        self.assertEqual(body[1], "Spot crude was quoted at 18 dlrs a barrel, up 50 cts.")   # "Reuter" signature and &#3; removed
        self.assertEqual(self.recs[1]["title"], "ACME CORP <ACM> 4TH QTR NET")                 # &lt; decoded
        self.assertEqual(self.recs[1]["body"][:3], ["Shr 50 cts vs 40 cts", "Net 1.2 mln vs 1.0 mln", "Revs 10.5 mln vs 9.8 mln"])

    def test_modapte_split(self):
        ma = reuters.modapte(self.recs)
        self.assertEqual([(r["newid"], r["split"]) for r in ma], [(1, "train"), (2, "test"), (4, "test"), (5, "test")])   # TOPICS="NO" dropped
        self.assertEqual(reuters.modapte_split({"topics_attr": "YES", "lewissplit": "NOT-USED"}), "")

    def test_jsonl_round_trip_and_documents(self):
        ma = reuters.modapte(self.recs)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.jsonl"
            reuters.write_jsonl(ma, path)
            docs = reuters.load_documents(path)
            test = reuters.load_documents(path, "test")
        self.assertEqual([x.doc_id for x in docs], ["R1", "R2", "R4", "R5"])
        self.assertEqual([x.doc_id for x in test], ["R2", "R4", "R5"])
        d2 = docs[1]
        self.assertEqual((d2.topics, d2.split, d2.year, d2.article_type), (["earn", "acq"], "test", "1987", "reuters"))
        self.assertIn("acquire Beta Inc", d2.abstract_text())
        st = reuters.topic_statistics(docs)
        self.assertEqual(st["splits"], {"train": 1, "test": 3})
        crude = next(r for r in st["rows"] if r["topic"] == "crude")
        self.assertEqual((crude["total"], crude["train"], crude["test"], crude["test_only"]), (2, 1, 1, 1))
        self.assertEqual(st["topics_per_document"][0], {"topics": 1, "documents": 3})

    def test_load_dir_requires_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                reuters.load_sgml_dir(d)


class TopicRelevanceTests(unittest.TestCase):
    def setUp(self):
        ma = reuters.modapte(reuters.parse_sgml(SGML))
        self.docs = {d.doc_id: d for d in map(reuters.to_document, ma) if d.split == "test"}

    def test_grades(self):
        q = reuters.TopicQuery({"id": "x", "text": "acquisition of a company", "topic": "acq"})
        self.assertEqual(q.judge(self.docs["R2"]), 1)                     # acq is one of two topics
        self.assertEqual(q.judge(self.docs["R4"]), 0)
        q2 = reuters.TopicQuery({"id": "y", "text": "crude oil prices", "topic": "crude"})
        self.assertEqual(q2.judge(self.docs["R5"]), 2)                    # only topic

    def test_query_set_defaults_and_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.json"
            qs = reuters.TopicQuerySet(path, self.docs)
            self.assertTrue(path.is_file())
            self.assertEqual(len(qs.queries), len(reuters.DEFAULT_QUERIES))
            self.assertEqual(qs.qrels("c01"), {"R5": 2})
            self.assertEqual(qs.qrels("a01"), {"R2": 1})
            self.assertEqual(qs.by_text("  Petroleum PRODUCTION and output ").id, "c03")
            self.assertEqual(qs.judged(), set(self.docs))
            row = next(r for r in qs.summary() if r["id"] == "e01")
            self.assertEqual((row["relevant"], row["grade1"], row["kind_name"]), (1, 1, "topic word"))
            # the file is re-read on the next load
            data = json.loads(path.read_text())
            data["queries"] = data["queries"][:3]
            path.write_text(json.dumps(data))
            self.assertEqual(len(reuters.TopicQuerySet(path, self.docs).queries), 3)

    def test_every_default_query_has_a_known_topic_and_kind(self):
        ids = {q["id"] for q in reuters.DEFAULT_QUERIES}
        self.assertEqual(len(ids), len(reuters.DEFAULT_QUERIES))
        for q in reuters.DEFAULT_QUERIES:
            self.assertIn(q["topic"], reuters.TOPIC_IDS)
            self.assertIn(q["kind"], reuters.QUERY_KIND_NAMES)
        for t in reuters.TOPIC_IDS:                                         # four kinds per topic
            self.assertEqual(sorted(q["kind"] for q in reuters.DEFAULT_QUERIES if q["topic"] == t), ["domain", "semantic", "synonym", "topic"])


class WordTokenizerTests(unittest.TestCase):
    def test_same_tokens_as_word2vec(self):
        tok = WordTokenizer()
        text = "Crude oil prices rose 18 dlrs a barrel; GLP-1 and the U.S. dollar."
        self.assertEqual(tok.terms(text), sentences_from_texts([text])[0])
        self.assertEqual(tok.normalize("The"), "")
        self.assertEqual(tok.normalize("Prices"), "prices")                # no stemming
        self.assertEqual(WordTokenizer(stemming=True).normalize("Prices"), "price")
        toks = tok.tokenize("Oil rose")
        self.assertEqual([(t.text, t.stem, t.position) for t in toks], [("Oil", "oil", 0), ("rose", "rose", 1)])


def _toy_corpus():
    # two "topics" with disjoint vocabularies: oil stories and grain stories
    oil = ["crude oil prices rose as opec output fell", "petroleum traders said crude prices were steady", "opec oil output quota barrel prices",
           "oil petroleum barrel opec crude output", "crude petroleum prices barrel opec"]
    grain = ["wheat corn harvest larger this year agriculture department said", "grain crop report wheat corn exports", "corn wheat grain harvest crop",
             "agriculture department grain report wheat", "wheat harvest corn crop grain exports"]
    return oil, grain


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ir.pmc_parser import Document
        oil, grain = _toy_corpus()
        texts = oil * 6 + grain * 6
        sents = sentences_from_texts(texts)
        cls.model = Word2Vec(sg=True, dim=12, window=3, negative=5, min_count=1, epochs=15, sample=0, alpha=0.05, seed=3).train(sents)
        docs = []
        for i, t in enumerate(oil):
            docs.append(Document(doc_id=f"O{i}", title="", abstract=[t], topics=["crude"] if i < 4 else ["crude", "ship"], split="test"))
        for i, t in enumerate(grain):
            docs.append(Document(doc_id=f"G{i}", title="", abstract=[t], topics=["grain"], split="test"))
        docs.append(Document(doc_id="X", title="", abstract=["zebra quokka"], topics=[], split="test"))     # only OOV words
        cls.vs = VectorSpace(docs, tokenizer=WordTokenizer())
        cls.emb = DocEmbedder(cls.vs, cls.model, "toy")

    def test_vectors_are_unit_and_oov_documents_are_empty(self):
        vecs = self.emb.doc_vectors("w2v-tfidf")
        self.assertEqual(len(vecs), self.vs.N)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vecs["O0"])), 1.0, places=6)
        self.assertEqual(vecs["X"], [0.0] * self.model.dim)
        cov = self.emb.coverage()
        self.assertEqual(cov["documents_without_vector"], 1)
        self.assertLess(cov["types_share"], 1.0)
        self.assertGreater(cov["tokens_share"], 0.9)

    def test_weights(self):
        self.assertEqual(self.emb.term_weight("oil", 3, "w2v-avg"), 3.0)
        w = self.emb.term_weight("oil", 3, "w2v-tfidf")
        self.assertAlmostEqual(w, (1 + math.log(3)) * math.log(self.vs.N / self.vs.df["oil"]))
        _, rows = self.emb.embed_counts(Counter({"oil": 1, "zebra": 1}), "w2v-tfidf")
        self.assertEqual([(r["word"], r["in_model"]) for r in rows], [("oil", True), ("zebra", False)])

    def test_semantic_retrieval_beats_lexical_mismatch(self):
        # "petroleum" never occurs in G*, and the oil documents that lack the word still rank first by embedding
        ranked, info = self.emb.rank("petroleum", "w2v-tfidf")
        self.assertFalse(info["empty"])
        top = [d for d, _ in ranked[:5]]
        self.assertTrue(all(d.startswith("O") for d in top), top)
        ranked_tf, _ = self.vs.rank("petroleum", "ltc", "ltc")
        self.assertTrue(all(d in self.vs.postings["petroleum"] for d, _ in ranked_tf))   # TF-IDF only finds the exact word
        self.assertLess(len(ranked_tf), len(top))

    def test_empty_query_and_oov_query(self):
        ranked, info = self.emb.rank("zebra quokka", "w2v-tfidf")
        self.assertEqual(ranked, [])
        self.assertTrue(info["empty"])
        self.assertEqual([t["in_model"] for t in info["terms"]], [False, False])

    def test_explain(self):
        qtf = Counter({"petroleum": 1})
        ex = self.emb.explain(qtf, "O0", "w2v-tfidf")                         # O0 has no "petroleum"
        self.assertEqual((ex[0]["word"], ex[0]["in_model"], ex[0]["in_doc"]), ("petroleum", True, False))
        self.assertTrue(ex[0]["nearest"])
        self.assertIn(ex[0]["nearest"][0]["word"], self.vs.tf["O0"])

    def test_hybrid_and_cosine(self):
        tf = [("A", 0.8), ("B", 0.2)]
        em = [("B", 0.9), ("C", 0.5)]
        h = hybrid_scores(tf, em, 0.5)
        self.assertEqual(h[0], ("B", 0.55))
        self.assertEqual(dict(h)["C"], 0.25)
        self.assertEqual(hybrid_scores(tf, em, 1.0)[0], ("A", 0.8))
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)

    def test_cooccurrence_and_topic_profile(self):
        co = cooccurrence(self.vs, "crude", "opec")
        self.assertEqual((co["df_a"], co["df_b"]), (self.vs.df["crude"], self.vs.df["opec"]))
        self.assertGreater(co["pmi"], 0)
        self.assertEqual(cooccurrence(self.vs, "crude", "wheat")["both"], 0)
        prof = word_topic_profile(self.vs, "wheat", ["crude", "grain"])
        self.assertEqual([t["share"] for t in prof["topics"]], [0.0, 1.0])
        self.assertGreater(prof["topics"][1]["lift"], 1)

    def test_evaluation_over_topic_queries(self):
        qrels = {d: (2 if self.vs.docs[d].topics == ["crude"] else 1) for d in self.vs.docs if "crude" in self.vs.docs[d].topics}
        run = [d for d, _ in self.emb.rank("petroleum output", "w2v-avg")[0]]
        m = ev.evaluate_run(run, qrels, ks=(5, 10))
        self.assertEqual(m["P@5"], 1.0)
        self.assertEqual(m["RR"], 1.0)
        self.assertGreater(m["nDCG@5"], 0.9)


class EngineTests(unittest.TestCase):
    """The engine with a tiny Reuters collection written to a temporary data directory."""

    @classmethod
    def setUpClass(cls):
        from ir.engine import Engine
        cls.tmp = tempfile.TemporaryDirectory()
        data = Path(cls.tmp.name) / "data"
        (data / "reuters").mkdir(parents=True)
        recs = reuters.parse_sgml(SGML)
        oil, grain = _toy_corpus()
        extra = []
        for i, (t, topic) in enumerate([(x, "crude") for x in oil] + [(x, "grain") for x in grain], 100):
            extra.append({"newid": i, "oldid": i, "lewissplit": "TRAIN" if i % 2 else "TEST", "topics_attr": "YES", "cgisplit": "", "type": "NORM",
                          "date": "1-MAR-1987", "topics": [topic], "places": ["usa"], "people": [], "orgs": [], "exchanges": [], "companies": [],
                          "title": t.upper()[:20], "dateline": "", "author": "", "body": [t]})
        reuters.write_jsonl(reuters.modapte(recs + extra), data / "reuters" / "reuters_modapte.jsonl")
        cls.eng = Engine(data)
        cls.eng.load()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_info_and_tfidf_without_a_model(self):
        info = self.eng.reuters_info()
        self.assertEqual(info["dataset"]["splits"]["train"] + info["dataset"]["splits"]["test"], info["dataset"]["documents"])
        self.assertEqual(info["models"], [])
        r = self.eng.reuters_search("crude oil prices", "tfidf", k=3)
        self.assertEqual(r["query_id"], "c01")
        self.assertTrue(r["results"])
        self.assertIsNone(r["model"])
        with self.assertRaises(ValueError):
            self.eng.reuters_search("crude oil prices", "w2v-tfidf")

    def test_train_search_compare_eval(self):
        job = self.eng.reuters_train({"model": "cbow", "dim": 10, "window": 3, "min_count": 1, "epochs": 2}, block=True)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(self.eng.reuters_active, job["name"])
        self.assertTrue((Path(self.tmp.name) / "data" / "reuters" / "w2v" / f"{job['name']}.json").is_file())
        r = self.eng.reuters_search("petroleum production and output", "w2v-tfidf", k=3)
        self.assertEqual((r["method"], r["model"], r["query_id"]), ("w2v-tfidf", job["name"], "c03"))
        self.assertIn("explain", r["results"][0])
        c = self.eng.reuters_compare("crude oil prices", "tfidf,w2v-avg,w2v-tfidf,hybrid", k=3)
        self.assertEqual([m["method"] for m in c["methods"]], ["tfidf", "w2v-avg", "w2v-tfidf", "hybrid"])
        self.assertEqual(c["methods"][0]["overlap"], 1.0)
        e = self.eng.reuters_eval("tfidf,w2v-tfidf", "", 5)
        self.assertEqual([x["method"] for x in e["runs"]], ["tfidf", "w2v-tfidf"])
        self.assertEqual(len(e["runs"][0]["per_query"]), len(e["queries"]))
        self.assertIn("synonym", e["runs"][1]["by_kind"])
        sw = self.eng.reuters_alpha_sweep(steps=4)
        self.assertEqual([x["alpha"] for x in sw["rows"]], [0.0, 0.25, 0.5, 0.75, 1.0])
        nb = self.eng.reuters_neighbours("oil zebra", "", 3)
        self.assertTrue(nb["words"][0]["models"][0]["in_model"])
        self.assertFalse(nb["words"][1]["models"][0]["in_model"])
        d = self.eng.reuters_doc("2")
        self.assertEqual((d["doc_id"], d["topics"]), ("R2", ["earn", "acq"]))
        # a second engine on the same directory finds the saved model
        from ir.engine import Engine
        e2 = Engine(Path(self.tmp.name) / "data")
        e2.load()
        self.assertIn(job["name"], e2.reuters_models)


if __name__ == "__main__":
    unittest.main()
