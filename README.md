# PMC Full-Text Retrieval Tool

Author: 胡元楨 (student ID P77141105)

Projects #1 and #2 for the AI Information Retrieval (人工智慧資訊檢索) course.

* **Project #1** – a full-text search engine for PubMed Central (PMC) XML articles with keyword
  retrieval, result visualisation, document statistics and rule-based sentence (EOS) detection.
* **Project #2** – a 1,000-abstract PubMed collection (query *GLP-1*), **Zipf's-law analysis**
  (CF / DF, rank-frequency plots, log-log regression, four pre-processing conditions, Porter
  stemming comparison, IDF), **word2vec** (skip-gram and CBOW, written from scratch) and
  **edit-distance spelling correction / fuzzy matching** in search. The written analysis is in
  [REPORT.md](REPORT.md).

Everything is implemented from scratch in **pure Python 3 (standard library only)**
plus a small vanilla-JavaScript front-end. No packages need to be installed.

```
python3 app.py            # web UI  ->  http://127.0.0.1:8080/   (use --port N to change)
python3 cli.py search "cancer immunotherapy"
python3 cli.py fetch 32895440 PMC7536103     # download from NCBI by PMID / PMC id and index
python3 -m unittest discover -s tests -v

# Project #2
python3 cli.py collect "GLP-1" -n 1000 --name glp1   # build the PubMed abstract collection (already in data/)
python3 cli.py zipf --cond B --svg report/           # Zipf analysis + figures
python3 cli.py terms --cond C                        # CF / DF / IDF table
python3 cli.py resolving --cond B                    # resolving power of significant words (Luhn cut-offs)
python3 cli.py edit semaglutide samegluitde          # edit distance (DP table)
python3 cli.py spell diabtes                         # spelling suggestions
python3 cli.py w2v train --model sg --epochs 10      # train word2vec (about 4.5 min in pure Python)
python3 cli.py w2v similar nausea mice
```

## Features

| Area | What the system does |
|---|---|
| Corpus | 30 open-access PMC articles (JATS XML, fetched with NCBI E-utilities) in `data/`; more can be added from the UI or by dropping files into `data/` |
| Parsing | JATS XML → title, abstract, section headings, body paragraphs, authors, journal, year, keywords, PMID/DOI. Citations, tables and reference lists are stripped |
| Pre-processing | tokenisation, lower-casing, stop-word removal (Snowball 174-word list + 17 biomedical additions), **own Porter stemmer** (verified against Porter's published examples) |
| Indexing | positional inverted index `term → {doc → [positions]}`, document lengths, field ranges, incremental add/remove |
| Query language | free terms, `AND` / `OR` / `NOT` / `-term` with parentheses, `"exact phrase"` (positional, stop words skipped), `prefix*` wildcard, `word~` fuzzy term (edit distance), `title:` / `abstract:` / `body:` field restriction |
| Ranking | Okapi BM25 (default), cosine-style TF-IDF, or raw match count; title matches boosted. Every hit is also scored **per field** (title / abstract / body, each length-normalised against its own field) and *Rank by* lets you rank on the abstract or body alone |
| Visualisation | highlighted titles & best-sentence snippets, relevance bars, term-hit chips, hits per field, match-position map, query-term frequency chart across top documents, per-document top-term chart, corpus Zipf-style term charts, sortable document table |
| Statistics | characters (with/without spaces), letters, digits, tokens, words, unique words, stop words, index terms, unique terms, sentences (rule-based **and** naive for comparison), paragraphs, sections, average word length, words per sentence, lexical diversity, field word counts; corpus totals and vocabulary browser |
| EOS detection | rule-based splitter: abbreviations (e.g., i.e., et al., Fig., Dr., vs. …), initials / genus names (`E. coli`), decimals, closing quotes/brackets, lower-case continuation; sentence boundaries can be shown inline in the document view |

## Project #2 features

| Area | What the system does |
|---|---|
| Collection | `esearch` + `efetch` download of the newest N English PubMed abstracts for a query, stored as `data/pubmed_<name>.jsonl` (one record per line, PMID as document id) and indexed alongside the PMC articles. Shipped: 1,000 abstracts for *GLP-1* |
| Zipf analysis | five pre-processing conditions (A basic → B punctuation removed → C stop words removed → D Porter stemming, plus E = B + stemming with stop words kept, to separate the effect of the stop list from that of stemming); documents / tokens / vocabulary / average length; CF and DF of every term; top-50 table; rank-frequency plot (linear or log frequency axis); log-log plot with OLS regression (slope, intercept, Zipf exponent, R², RMSE) for the whole curve and for the high / middle / low-frequency thirds; overlay of all conditions; what stemming merges |
| CF vs DF vs IDF | table for any terms you type (CF, DF, CF/DF, DF/N, `idf = log10(N/df)`) and a CF-vs-DF scatter plot of the 1,500 most frequent terms |
| Resolving power | Luhn's significant words: `power(t) = CF × idf` (the term's total TF-IDF weight) plotted against rank, smoothed with a running median; upper / lower cut-off sliders (automatic default = where the smoothed curve falls to half of its peak); share of vocabulary, tokens and resolving power in the *too common* / *significant* / *too rare* zones; table of the most discriminating words; the smoothed curve is overlaid on the rank-frequency plot (right-hand axis), a second overlay shows the IDF of every term (and its running median) on the same rank axis, and an A → E chart overlays all conditions to show how the significant-word range shifts |
| word2vec | pure-Python skip-gram and CBOW with negative sampling, dynamic window, sub-sampling, decaying learning rate; background training with a progress bar and loss curve; nearest neighbours, analogies, 2-D PCA map; the model is saved to `data/word2vec.json` |
| Edit distance | dynamic-programming Levenshtein / Damerau distance with the full DP table and alignment shown |
| Spelling correction | dictionary = surface words of the collection; character-bigram index to select candidates, banded DP to verify, ranking by (distance, collection frequency). Search shows *Did you mean* and automatically re-runs a query that matched nothing with the corrected words |
| Fuzzy matching | `word~` / `word~2` expands to every vocabulary word within the edit distance |

The Zipf / word2vec tabs analyse the PubMed abstract collection (title + abstract of each
record); if no collection is present they fall back to all indexed documents.

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
ir/ncbi.py          PMID -> PMCID conversion, efetch download, PubMed search + bulk abstract download
ir/zipf.py          Zipf analysis: conditions, CF / DF, log-log regression, segments, stemming groups
ir/spell.py         edit distance (DP), bigram index, spelling suggestions
ir/word2vec.py      skip-gram / CBOW with negative sampling, similarity, analogy, PCA
ir/svgplot.py       static SVG figures for the report
static/             index.html, style.css, app.js, project2.js (front-end)
tests/test_ir.py    unit tests (stemmer, tokenizer, splitter, parser, index, search, engine)
tests/test_project2.py   unit tests (Zipf, edit distance, spelling, fuzzy search, word2vec, collection)
data/               PMC XML corpus, pubmed_*.jsonl abstract collections, word2vec.json
report/             figures used in REPORT.md
REPORT.md           Project #2 written analysis (Zipf questions, CF/DF, TF-IDF, IR implications)
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
* **Zipf 分析** – choose a pre-processing condition; tiles, rank-frequency and log-log plots,
  regression report, answers to the analysis questions computed from the live numbers,
  top-50 table, comparison of the conditions, stemming groups, CF / DF / IDF table and scatter.
* **Word2Vec** – train (model, dimensions, window, negatives, min count, epochs), then explore
  neighbours, analogies and the PCA map.
* **Edit Distance** – DP table and alignment for two strings; spelling suggestions for a word.
* **Add Documents** – *Build a PubMed abstract collection* (query, size, name); paste a list of PMIDs / PMC IDs and press *Fetch & index*:
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
GET  /api/search?q=<query>&method=bm25|tfidf|count&op=AND|OR&scope=all|title|abstract|body&limit=25&spell=1
GET  /api/doc/<PMC id>?q=<query>
GET  /api/corpus
GET  /api/vocab?prefix=immun&limit=100
GET  /api/stopwords
GET  /api/stem?word=immunotherapies,sars
GET  /api/analyze?text=WHO+reported+SARS-CoV-2
GET  /api/zipf
GET  /api/zipf/terms?cond=C&terms=glp,obesity&scatter=1
GET  /api/edit?a=semaglutide&b=samegluitde
GET  /api/spell?word=diabtes&k=2
GET  /api/w2v/status   /api/w2v/similar?word=nausea   /api/w2v/analogy?a=..&b=..&c=..   /api/w2v/projection?words=..
POST /api/w2v/train                      (body = {"model": "sg", "dim": 50, "window": 5, "negative": 5, "min_count": 5, "epochs": 10})
POST /api/collect                        (body = {"term": "GLP-1", "n": 1000, "name": "glp1"})
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
