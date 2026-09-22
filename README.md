# PMC Full-Text Retrieval Tool

Author: 胡元楨 (student ID P77141105)

Project #1 for the Biomedical Information Retrieval course: a full-text search
engine for PubMed Central (PMC) XML articles with keyword retrieval, result
visualisation, document statistics and rule-based sentence (EOS) detection.

Everything is implemented from scratch in **pure Python 3 (standard library only)**
plus a small vanilla-JavaScript front-end. No packages need to be installed.

```
python3 app.py            # web UI  ->  http://127.0.0.1:8080/   (use --port N to change)
python3 cli.py search "cancer immunotherapy"
python3 cli.py fetch 32895440 PMC7536103     # download from NCBI by PMID / PMC id and index
python3 -m unittest discover -s tests -v
```

## Features

| Area | What the system does |
|---|---|
| Corpus | 30 open-access PMC articles (JATS XML, fetched with NCBI E-utilities) in `data/`; more can be added from the UI or by dropping files into `data/` |
| Parsing | JATS XML → title, abstract, section headings, body paragraphs, authors, journal, year, keywords, PMID/DOI. Citations, tables and reference lists are stripped |
| Pre-processing | tokenisation, lower-casing, stop-word removal (Snowball 174-word list + 17 biomedical additions), **own Porter stemmer** (verified against Porter's published examples) |
| Indexing | positional inverted index `term → {doc → [positions]}`, document lengths, field ranges, incremental add/remove |
| Query language | free terms, `AND` / `OR` / `NOT` / `-term` with parentheses, `"exact phrase"` (positional, stop words skipped), `prefix*` wildcard, `title:` / `abstract:` / `body:` field restriction |
| Ranking | Okapi BM25 (default), cosine-style TF-IDF, or raw match count; title matches boosted. Every hit is also scored **per field** (title / abstract / body, each length-normalised against its own field) and *Rank by* lets you rank on the abstract or body alone |
| Visualisation | highlighted titles & best-sentence snippets, relevance bars, term-hit chips, hits per field, match-position map, query-term frequency chart across top documents, per-document top-term chart, corpus Zipf-style term charts, sortable document table |
| Statistics | characters (with/without spaces), letters, digits, tokens, words, unique words, stop words, index terms, unique terms, sentences (rule-based **and** naive for comparison), paragraphs, sections, average word length, words per sentence, lexical diversity, field word counts; corpus totals and vocabulary browser |
| EOS detection | rule-based splitter: abbreviations (e.g., i.e., et al., Fig., Dr., vs. …), initials / genus names (`E. coli`), decimals, closing quotes/brackets, lower-case continuation; sentence boundaries can be shown inline in the document view |

## Layout

```
app.py              HTTP server (standard library) + JSON API
cli.py              command-line interface
ir/porter.py        Porter stemming algorithm
ir/stopwords.py     stop-word list
ir/tokenizer.py     tokenisation + normalisation pipeline
ir/sentence.py      rule-based sentence boundary detection (+ naive baseline)
ir/pmc_parser.py    JATS / PMC XML parser
ir/stats.py         document & corpus statistics
ir/index.py         positional inverted index
ir/search.py        query lexer/parser, boolean & phrase matching, BM25 / TF-IDF scoring
ir/engine.py        glue: snippets, highlighting, match maps, document view
ir/ncbi.py          PMID -> PMCID conversion and efetch download from NCBI
static/             index.html, style.css, app.js (front-end)
tests/test_ir.py    unit tests (stemmer, tokenizer, splitter, parser, index, search, engine)
data/               PMC XML corpus
```

## Web UI

* **Search** – enter a query, choose the scoring function and default operator.
  The *query analysis* strip shows how each word was processed (stem, document
  frequency, stop words removed, operators) and the parsed boolean expression.
  Each result shows the highlighted title, the best matching sentences, the
  number of hits per term and per field, a relevance bar and a *match map*
  showing where in the document the hits occur. A grouped bar chart compares
  query-term frequencies across the top documents.
* **Document view** (click a title) – metadata, statistics tiles, top-term
  chart, rule-based vs naive sentence count, and the full text with every
  query term highlighted. Toggle *Show sentence boundaries* to see each detected
  sentence numbered, or *Only sentences with matches* for a KWIC-style view.
* **前處理與演算法** – the pre-processing pipeline explained with live examples:
  tokeniser / case-folding pitfalls (WHO, AIDS, SARS), the Snowball stop-word
  list next to the biomedical additions (and the 571-word SMART list for
  comparison only), Porter's five steps with a
  step-by-step tracer for any word, and curated over- / under-stemming
  counterexamples computed by the system itself.
* **Corpus & Statistics** – corpus totals, most frequent terms (collection and
  document frequency), vocabulary browser, sortable per-document statistics table.
* **Add Documents** – paste a list of PMIDs / PMC IDs and press *Fetch & index*:
  PMIDs are converted to PMCIDs with the NCBI ID Converter, the JATS XML is
  downloaded with E-utilities `efetch`, parsed, indexed and saved to `data/`
  (a per-id status table shows added / updated / already indexed / failed).
  Articles without full text in PMC fall back to the PubMed record: title,
  abstract, authors, keywords and MeSH terms are converted to a minimal JATS
  file (`data/PMID<n>.xml`) and indexed as *abstract only*.
  Alternatively upload PMC XML files by hand. The tab also lists every indexed
  document with a *Remove* button.

## API

```
GET  /api/search?q=<query>&method=bm25|tfidf|count&op=AND|OR&scope=all|title|abstract|body&limit=25
GET  /api/doc/<PMC id>?q=<query>
GET  /api/corpus
GET  /api/vocab?prefix=immun&limit=100
GET  /api/stopwords
GET  /api/stem?word=immunotherapies,sars
GET  /api/analyze?text=WHO+reported+SARS-CoV-2
POST /api/upload?filename=PMC123.xml     (body = raw XML)
POST /api/fetch?replace=0|1              (body = {"ids": "32895440\nPMC7536103"} or plain text)
POST /api/remove?id=PMC123&delete=1     (delete=1 also deletes data/PMC123.xml)
```

## CLI

```
python3 cli.py search 'covid AND vaccine NOT pregnancy' --method tfidf -n 5
python3 cli.py stats PMC7536103
python3 cli.py sentences PMC7536103 -n 15
python3 cli.py corpus
python3 cli.py vocab immun
python3 cli.py analyze "Kim et al. reported p < 0.05 in Fig. 2A. The T-cells were activated."
```

## Getting more PMC data

Any JATS XML from <https://pmc.ncbi.nlm.nih.gov/> works, e.g. via E-utilities:

```
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC7536103&retmode=xml" -o data/PMC7536103.xml
```

## Sentence detection rules

A candidate boundary is `.`, `!`, `?` (or an ellipsis) followed by optional
closing quotes/brackets and white-space. It is rejected when the next character
is lower-case; when the word before a period is a known abbreviation, a single
letter (initials, genus abbreviations) or a dotted abbreviation (`a.m.`,
`U.S.`); decimals never split because no white-space follows the period.
Paragraph ends and headings always close a sentence. On the sample corpus the
rule-based method finds ~6,800 sentences where the naive method reports ~9,000.
