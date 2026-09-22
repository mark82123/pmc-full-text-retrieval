"""Unit tests:  python3 -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir.engine import Engine  # noqa: E402
from ir.index import InvertedIndex  # noqa: E402
from ir.ncbi import parse_ids, pubmed_to_jats  # noqa: E402
from ir.pmc_parser import Document, Section, parse_xml_string  # noqa: E402
from ir.porter import stem, stem_trace  # noqa: E402
from ir.stopwords import BIOMEDICAL_STOP_WORDS, SMART_STOP_WORDS, SNOWBALL_STOP_WORDS, STOP_WORDS  # noqa: E402
from ir.search import Searcher, parse_query  # noqa: E402
from ir.sentence import naive_split, split_sentences  # noqa: E402
from ir.stats import document_stats  # noqa: E402
from ir.tokenizer import Tokenizer  # noqa: E402


class PorterTests(unittest.TestCase):
    def test_published_examples(self):
        cases = {"caresses": "caress", "ponies": "poni", "ties": "ti", "cats": "cat", "feed": "feed",
                 "agreed": "agre", "plastered": "plaster", "motoring": "motor", "conflated": "conflat",
                 "hopping": "hop", "happy": "happi", "relational": "relat", "digitizer": "digit",
                 "hopefulness": "hope", "electrical": "electr", "adjustable": "adjust", "replacement": "replac",
                 "adoption": "adopt", "activate": "activ", "generalizations": "gener", "controll": "control",
                 "immunotherapy": "immunotherapi", "diabetes": "diabet", "patients": "patient", "sky": "sky"}
        for w, expected in cases.items():
            self.assertEqual(stem(w), expected, w)


class StopwordAndTraceTests(unittest.TestCase):
    def test_stopword_lists(self):
        self.assertEqual(len(SNOWBALL_STOP_WORDS), 174)
        self.assertEqual(len(BIOMEDICAL_STOP_WORDS), 17)
        self.assertEqual(STOP_WORDS, SNOWBALL_STOP_WORDS | BIOMEDICAL_STOP_WORDS)
        self.assertFalse(SNOWBALL_STOP_WORDS & BIOMEDICAL_STOP_WORDS)
        self.assertIn("the", SNOWBALL_STOP_WORDS); self.assertIn("al", BIOMEDICAL_STOP_WORDS)
        # SMART list is reference material only and must not leak into indexing
        self.assertEqual(len(SMART_STOP_WORDS), 570)
        self.assertIn("novel", SMART_STOP_WORDS); self.assertNotIn("novel", STOP_WORDS)
        self.assertEqual(Tokenizer().terms("a novel significant biomarker"), ["novel", "signific", "biomark"])

    def test_stem_trace_matches_stem(self):
        for w in ("immunotherapies", "generalized", "organization", "analyses", "sars", "rate", "the"):
            tr = stem_trace(w)
            final = tr[-1]["after"] if tr else w
            self.assertEqual(final, stem(w))
            for a, b in zip(tr, tr[1:]):
                self.assertEqual(a["after"], b["before"])
        self.assertEqual([t["step"] for t in stem_trace("generalized")], ["1b", "3", "4"])
        self.assertEqual(stem_trace("rate"), [])


class TokenizerTests(unittest.TestCase):
    def test_pipeline(self):
        tok = Tokenizer()
        self.assertEqual(tok.terms("The T-cells were activated by COVID-19 vaccines."),
                         ["t", "cell", "activ", "covid", "19", "vaccin"])

    def test_positions_keep_stopword_slots(self):
        toks = Tokenizer().tokenize("role of the gut")
        self.assertEqual([t.position for t in toks], [0, 1, 2, 3])
        self.assertEqual([t.stem for t in toks], ["role", "", "", "gut"])


class SentenceTests(unittest.TestCase):
    def test_abbreviations_and_decimals(self):
        t = ("Kim et al. reported p < 0.05 in Fig. 2A. However, E. coli growth (see Table 1) was not affected [12]. "
             "Is this true? Yes! We used 3.5 mg/kg i.e. the standard dose. Dr. Smith agreed.")
        s = split_sentences(t)
        self.assertEqual(len(s), 6)
        self.assertTrue(s[0].endswith("Fig. 2A."))
        self.assertEqual(s[2], "Is this true?")
        self.assertGreater(len(naive_split(t)), len(s))

    def test_quotes_and_brackets(self):
        s = split_sentences('He said "stop." Then (finally) it ended. Really')
        self.assertEqual(s, ['He said "stop."', "Then (finally) it ended.", "Really"])

    def test_lowercase_continuation(self):
        self.assertEqual(len(split_sentences("The approx. value is 3. the end")), 1)

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


XML = """<?xml version="1.0"?><!DOCTYPE article PUBLIC "x" "y">
<article article-type="research-article"><front><journal-meta><journal-title-group><journal-title>Test J</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="pmcid">PMC1</article-id><article-id pub-id-type="pmid">99</article-id>
<title-group><article-title>Gut microbiome and diabetes</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Doe</surname><given-names>J</given-names></name></contrib></contrib-group>
<pub-date pub-type="epub"><year>2021</year></pub-date>
<abstract><p>The gut microbiome matters. It really does.</p></abstract><kwd-group><kwd>gut</kwd></kwd-group></article-meta></front>
<body><sec><title>Introduction</title><p>Diabetes is common <xref ref-type="bibr" rid="r1">[1]</xref>. The role of the gut is debated.</p>
<table-wrap><caption><p>Table caption.</p></caption><table><tr><td>ignored cell</td></tr></table></table-wrap></sec>
<sec><title>Methods</title><p>We studied mice.</p></sec></body>
<back><ref-list><ref id="r1"><mixed-citation>Ignored reference text</mixed-citation></ref></ref-list></back></article>"""


class ParserTests(unittest.TestCase):
    def test_parse(self):
        d = parse_xml_string(XML, "t.xml")[0]
        self.assertEqual(d.doc_id, "PMC1")
        self.assertEqual(d.pmid, "99")
        self.assertEqual(d.title, "Gut microbiome and diabetes")
        self.assertEqual(d.journal, "Test J")
        self.assertEqual(d.year, "2021")
        self.assertEqual(d.authors, ["J Doe"])
        self.assertEqual(d.keywords, ["gut"])
        self.assertEqual([s.heading for s in d.sections], ["Introduction", "Methods"])
        body = d.body_text()
        self.assertIn("Diabetes is common.", body)          # citation removed
        self.assertIn("Table caption.", body)
        self.assertNotIn("ignored cell", body)
        self.assertNotIn("Ignored reference", d.full_text())


def _corpus():
    return [
        Document("D1", title="Gut microbiome", abstract=["The gut microbiome and diabetes."],
                 sections=[Section("Intro", ["Role of the gut microbiome in disease."])]),
        Document("D2", title="Cancer immunotherapy", abstract=["Immunotherapy for lung cancer patients."],
                 sections=[Section("", ["Cancer immunotherapies are promising. The microbiome modulates immunotherapy."])]),
        Document("D3", title="Vaccines", abstract=["COVID-19 vaccine hesitancy."],
                 sections=[Section("", ["Vaccination and pregnancy."])]),
    ]


class IndexSearchTests(unittest.TestCase):
    def setUp(self):
        self.idx = InvertedIndex()
        self.idx.build(_corpus())
        self.s = Searcher(self.idx)

    def ids(self, q, op="AND", scope="all"):
        return [d for d, *_ in self.s.search(q, default_op=op, scope=scope)[1]]

    def test_index_basics(self):
        self.assertEqual(self.idx.n_docs, 3)
        self.assertEqual(self.idx.df("microbiom"), 2)
        self.assertEqual(self.idx.cf("microbiom"), 4)
        self.assertEqual(self.idx.docs["D1"].field_ranges["title"], (0, 2))

    def test_and_or_not(self):
        self.assertEqual(self.ids("microbiome immunotherapy"), ["D2"])
        self.assertEqual(set(self.ids("microbiome immunotherapy", op="OR")), {"D1", "D2"})
        self.assertEqual(self.ids("microbiome NOT cancer"), ["D1"])
        self.assertEqual(self.ids("microbiome -cancer"), ["D1"])
        self.assertEqual(set(self.ids("(gut OR vaccine) AND NOT cancer")), {"D1", "D3"})

    def test_phrase(self):
        self.assertEqual(self.ids('"gut microbiome"'), ["D1"])
        self.assertEqual(self.ids('"role of the gut"'), ["D1"])       # stop words inside phrase
        self.assertEqual(self.ids('"microbiome gut"'), [])

    def test_prefix_and_field(self):
        self.assertEqual(set(self.ids("immun*")), {"D2"})
        self.assertEqual(set(self.ids("vacc*")), {"D3"})
        self.assertEqual(self.ids("title:cancer"), ["D2"])
        self.assertEqual(self.ids("title:microbiome"), ["D1"])       # D2 mentions it only in body

    def test_hyphenated_compound_terms(self):
        # "COVID-19" is tokenised as covid + 19, so the term must match as a phrase
        self.assertEqual(self.ids("COVID-19"), ["D3"])
        self.assertEqual(self.ids("covid-19 vaccine"), ["D3"])
        self.assertEqual(self.ids("abstract:covid-19"), ["D3"])
        self.assertEqual(self.ids("title:covid-19"), [])
        self.assertEqual(self.ids("19-covid"), [])                     # order matters, like a phrase
        stems, _ = self.s.query_stems(parse_query("COVID-19"))
        self.assertEqual(stems, {"covid", "19"})

    def test_stemming_matches_variants(self):
        self.assertEqual(self.ids("vaccination"), ["D3"])
        self.assertEqual(self.ids("immunotherapies"), ["D2"])

    def test_ranking(self):
        ranked = self.ids("microbiome", op="OR")
        self.assertEqual(ranked[0], "D1")        # 3 occurrences incl. title vs 1
        for method in ("bm25", "tfidf", "count"):
            res = self.s.search("microbiome", method=method)[1]
            self.assertGreater(res[0][1], res[1][1])

    def test_field_scores(self):
        # D1 has "microbiome" in title, abstract and body; D2 only in the body
        res = {d: fs for d, _, _, fs in self.s.search("microbiome", default_op="OR")[1]}
        self.assertGreater(res["D1"]["abstract"], 0)
        self.assertGreater(res["D1"]["body"], 0)
        self.assertGreater(res["D1"]["title"], 0)
        self.assertEqual(res["D2"]["abstract"], 0.0)
        self.assertEqual(res["D2"]["title"], 0.0)
        self.assertGreater(res["D2"]["body"], 0)
        self.assertAlmostEqual(res["D1"]["all"], self.s.score("D1", self.s.evaluate(parse_query("microbiome"))["D1"]))
        # ranking on one field drops documents without a hit in that field
        self.assertEqual(self.ids("microbiome", scope="abstract"), ["D1"])
        self.assertEqual(set(self.ids("microbiome", scope="body")), {"D1", "D2"})
        self.assertEqual(self.ids("immunotherapy", scope="title"), ["D2"])
        # per-field lengths are maintained on add / remove
        self.assertEqual(self.idx.docs["D1"].field_lengths["title"], 2)
        total_body = self.idx.total_field_length["body"]
        self.idx.remove_document("D2")
        self.assertEqual(self.idx.total_field_length["body"], total_body - 6)

    def test_remove(self):
        self.idx.remove_document("D1")
        self.assertEqual(self.idx.n_docs, 2)
        self.assertEqual(self.idx.df("gut"), 0)
        self.assertEqual(self.ids("gut"), [])

    def test_parser_precedence(self):
        self.assertEqual(parse_query("a OR b AND c").describe(), "(a OR (b AND c))")
        self.assertEqual(parse_query("NOT a b").describe(), "(NOT a AND b)")
        self.assertIsNone(parse_query(""))


class StatsAndEngineTests(unittest.TestCase):
    def test_document_stats(self):
        d = _corpus()[0]
        st = document_stats(d, Tokenizer())
        self.assertEqual(st["sentences"], 4)                          # title + abstract + heading + body paragraph
        self.assertEqual(st["words"], 15)
        f = st["fields"]
        self.assertEqual(f["title"]["words"] + f["abstract"]["words"] + f["body"]["words"], 15)
        self.assertEqual(f["title"]["sentences"] + f["abstract"]["sentences"] + f["body"]["sentences"], st["sentences"])
        self.assertEqual(f["title"]["index_terms"] + f["abstract"]["index_terms"] + f["body"]["index_terms"], st["index_terms"])
        self.assertGreater(st["characters"], 0)
        self.assertEqual({t for t, _ in st["top_terms"][:2]}, {"gut", "microbiom"})   # 3 occurrences each

    def test_engine_with_real_data(self):
        data = ROOT / "data"
        if not any(data.glob("*.xml")):
            self.skipTest("no sample data")
        eng = Engine(data)
        n = eng.load()
        self.assertGreater(n, 0)
        r = eng.search("cancer immunotherapy")
        self.assertGreater(r["total"], 0)
        top = r["results"][0]
        self.assertIn("<mark>", top["snippet_html"])
        self.assertEqual(sorted(top["match_map"]), ["abstract", "body", "title"])
        self.assertEqual(sum(len(v) for v in top["match_map"].values()), 40)
        self.assertEqual(sum(sum(v) for v in top["match_map"].values()), top["total_hits"])
        fs = top["field_stats"]
        self.assertEqual(fs["title"]["words"] + fs["abstract"]["words"] + fs["body"]["words"], top["stats"]["words"])
        doc = eng.document(top["doc_id"], "cancer immunotherapy")
        self.assertGreater(doc["query_hits"], 0)
        self.assertLess(doc["stats"]["sentences"], doc["stats"]["sentences_naive"])
        c = eng.corpus()
        self.assertEqual(c["documents"], n)
        self.assertEqual(len(c["documents_list"]), n)

    def test_engine_add_and_remove_xml(self):
        eng = Engine(Path("/nonexistent"))
        eng.load()
        ids = eng.add_xml(XML, "t.xml", persist=False)
        self.assertEqual(ids, ["PMC1"])
        self.assertEqual(eng.search('"role of the gut"')["total"], 1)
        self.assertTrue(eng.remove("PMC1"))
        self.assertEqual(eng.index.n_docs, 0)




class NcbiIdTests(unittest.TestCase):
    def test_parse_ids_formats(self):
        ids = parse_ids("32895440\nPMC7536103\nPMID: 33301246, pmc 123\n"
                        "https://pubmed.ncbi.nlm.nih.gov/32895440/\n"
                        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7477542/")
        self.assertEqual([(i["kind"], i["number"]) for i in ids],
                         [("pmid", "32895440"), ("pmc", "7536103"), ("pmid", "33301246"),
                          ("pmc", "123"), ("pmc", "7477542")])

    def test_pubmed_record_to_jats(self):
        pm = """<?xml version="1.0"?><!DOCTYPE PubmedArticleSet PUBLIC "x" "y"><PubmedArticleSet><PubmedArticle>
        <MedlineCitation><PMID Version="1">27741350</PMID><Article>
        <Journal><Title>The Journal of pathology</Title><JournalIssue><PubDate><Year>2017</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Measuring cancer evolution from the genome.</ArticleTitle>
        <ELocationID EIdType="doi">10.1002/path.4821</ELocationID>
        <Abstract><AbstractText Label="BACKGROUND">Tumours evolve.</AbstractText>
        <AbstractText Label="RESULTS">We measured <i>E. coli</i> growth &amp; more.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Graham</LastName><ForeName>Trevor A</ForeName></Author></AuthorList>
        <PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
        </Article><KeywordList><Keyword>subclones</Keyword></KeywordList>
        <MeshHeadingList><MeshHeading><DescriptorName>Neoplasms</DescriptorName></MeshHeading></MeshHeadingList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        docs = parse_xml_string(pubmed_to_jats(pm), "PMID27741350.xml")
        self.assertEqual(len(docs), 1)
        d = docs[0]
        self.assertEqual(d.doc_id, "PMID27741350")
        self.assertEqual(d.pmid, "27741350")
        self.assertEqual(d.doi, "10.1002/path.4821")
        self.assertEqual(d.title, "Measuring cancer evolution from the genome")
        self.assertEqual(d.journal, "The Journal of pathology")
        self.assertEqual(d.year, "2017")
        self.assertEqual(d.authors, ["Trevor A Graham"])
        self.assertEqual(d.article_type, "review-article")
        self.assertIn("subclones", d.keywords); self.assertIn("Neoplasms", d.keywords)
        self.assertEqual(d.sections, [])
        self.assertIn("Results: We measured E. coli growth & more.", d.abstract_text())
        # abstract-only document is searchable and yields statistics
        eng = Engine("/nonexistent-dir")
        eng.add_xml(pubmed_to_jats(pm), "PMID27741350.xml", persist=False)
        self.assertEqual(eng.search("abstract:tumour")["total"], 1)
        self.assertEqual(eng.search("body:tumour")["total"], 0)
        self.assertEqual(eng.stats["PMID27741350"]["body_words"], 0)
        self.assertTrue(eng.document("PMID27741350", "evolution"))

    def test_remove_deletes_persisted_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            eng = Engine(tmp)
            xml = "<article><front><article-meta><article-id pub-id-type='pmcid'>PMC1</article-id>" \
                  "<title-group><article-title>T</article-title></title-group></article-meta></front>" \
                  "<body><p>Some body text here.</p></body></article>"
            eng.add_xml(xml, "PMC1.xml", persist=True)
            self.assertTrue((Path(tmp) / "PMC1.xml").is_file())
            self.assertTrue(eng.remove("PMC1", delete_file=True))
            self.assertFalse((Path(tmp) / "PMC1.xml").exists())
            self.assertEqual(eng.index.n_docs, 0)

    def test_parse_ids_dedup_and_empty(self):
        self.assertEqual(parse_ids(""), [])
        self.assertEqual(parse_ids("no ids here"), [])
        self.assertEqual(len(parse_ids("PMC1 PMC1 pmc1")), 1)

if __name__ == "__main__":
    unittest.main()
