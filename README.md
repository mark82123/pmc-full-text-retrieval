# PMC Full-Text Retrieval Tool

Author: 胡元楨 (student ID P77141105)

Projects #1, #2 and #3 for the AI Information Retrieval (人工智慧資訊檢索) course.

* **Project #1** – a full-text search engine for PubMed Central (PMC) XML articles with keyword
  retrieval, result visualisation, document statistics and rule-based sentence (EOS) detection.
* **Project #2** – a 1,000-abstract PubMed collection (query *GLP-1*), **Zipf's-law analysis**
  (CF / DF, rank-frequency plots, log-log regression, four pre-processing conditions, Porter
  stemming comparison, IDF), **word embeddings** (skip-gram and CBOW written from scratch, several
  models side by side, cosine-similarity analysis, PCA and t-SNE maps, frequency-bucket evaluation,
  seed stability) and **edit-distance spelling correction / fuzzy matching** in search. The written
  analyses are in [REPORT.md](REPORT.md) (Zipf) and [REPORT_w2v.md](REPORT_w2v.md) (word embedding,
  RQ1–RQ5).
* **Project #3** – **term weighting and a vector-space search engine** over a 3,000-abstract GLP-1
  collection (with MeSH headings): raw / log / augmented / boolean TF, IDF and probabilistic IDF, L2
  normalisation (SMART `tf·df·norm` triplets such as `ltc`), cosine ranking with per-term score
  contributions, controlled one-factor experiments, and an evaluation with 15 queries whose
  relevance judgements are derived from MeSH / author-keyword labels plus 118 pooled manual judgements (P@k, R@k, MRR, MAP, nDCG@k).
  Screenshots of the tab: `docs/screenshots/project3-vsm-*.png`.
  Written analysis: [REPORT_tfidf.md](REPORT_tfidf.md) (RQ1–RQ5); one-page report in `docs/`.

Everything is implemented from scratch in **pure Python 3 (standard library only)**
plus a small vanilla-JavaScript front-end. No packages need to be installed.

```
python3 app.py            # web UI  ->  http://127.0.0.1:8090/   (default port 8090; use --port N to change)
python3 app.py --host 0.0.0.0   # also reachable from other machines / the Windows host, e.g. http://<WSL-IP>:8090/
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
python3 cli.py w2v train --model sg --epochs 10      # train word2vec (about 4 min in pure Python; cbow 1.5 min)
python3 cli.py w2v list                              # every trained model in data/w2v/
python3 cli.py w2v similar nausea mice
python3 cli.py w2v compare semaglutide nausea cancer --models sg-d50-w5-m5-e10,cbow-d50-w5-m5-e10
python3 cli.py w2v pairs "semaglutide,liraglutide;nausea,vomiting;mice,rats"   # cosine matrix
python3 cli.py w2v map semaglutide nausea kidney mice --method tsne --svg report/
python3 cli.py w2v eval ; python3 cli.py w2v overlap --models sg-d50-w5-m5-e10,sg-d50-w5-m5-e10-s2

# Project #3  (web UI: the TF-IDF 檢索 tab, /#vsm)
python3 cli.py vsm info                                      # collection, vocabulary, tf / idf curves, weighting schemes
python3 cli.py vsm terms glp obesity semaglutide apnea       # cf / df / idf / max tf of terms
python3 cli.py vsm doc PMID42233552 --schemes nnn,lnn,ltn,ltc    # one document's weight vector per scheme
python3 cli.py vsm search "semaglutide weight loss in obesity" --scheme ltc -k 10   # top-10 with per-term contributions
python3 cli.py vsm compare "polycystic ovary syndrome" --schemes ltn,ltc,nnn,nnc    # one factor at a time
python3 cli.py vsm eval                                      # P@k / R@k / MRR / MAP / nDCG@k of every scheme, 15 queries
python3 cli.py vsm queries ; python3 cli.py vsm relevant q10 # the queries, their rules and relevant sets
python3 cli.py vsm collect "GLP-1" -n 3000 --name glp1_3000  # (re)download the collection with MeSH into data/vsm/
python3 cli.py vsm svg report/                               # figures of the report
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
| CF–DF map | the vocabulary sorted by burstiness (CF / DF) and spread (DF / N) into function / boilerplate / topic / keyword / number zones with adjustable thresholds; scatter with the Poisson reference curve DF* = N(1 − e^(−CF/N)); lists of stop-word candidates, numeric artefacts, collection-defining and document-specific terms (`cli.py cfdf`, `/api/zipf/cfdf`) |
| Resolving power | Luhn's significant words: `power(t) = CF × idf` (the term's total TF-IDF weight) plotted against rank, smoothed with a running median; upper / lower cut-off sliders (automatic default = where the smoothed curve falls to half of its peak); share of vocabulary, tokens and resolving power in the *too common* / *significant* / *too rare* zones; table of the most discriminating words; the smoothed curve is overlaid on the rank-frequency plot (right-hand axis), a second overlay shows the IDF of every term (and its running median) on the same rank axis, and an A → E chart overlays all conditions to show how the significant-word range shifts |
| word2vec | pure-Python skip-gram and CBOW with negative sampling, dynamic window, sub-sampling, decaying learning rate; a training queue with progress bars; every model is kept in `data/w2v/<name>.json` so architectures, hyper-parameters and seeds can be compared (9 models ship with the repo) |
| Embedding analysis | pre-processing and vocabulary statistics (min-count coverage table, training examples per window), skip-gram / CBOW training-example generator for any sentence, nearest neighbours of ≥ 10 words side by side across models with Jaccard overlap, cosine matrix of word groups (within / between / random baseline), analogies, **PCA and t-SNE** (`ir/tsne.py`, with k-NN preservation of each map), intrinsic evaluation per frequency bucket (sim@k, neighbour frequency, same-stem variant hit@k / MRR) and neighbour overlap between two models (parameter effect, seed stability) |
| Edit distance | dynamic-programming Levenshtein / Damerau distance with the full DP table and alignment shown |
| Spelling correction | dictionary = surface words of the collection; character-bigram index to select candidates, banded DP to verify, ranking by (distance, collection frequency). Search shows *Did you mean* and automatically re-runs a query that matched nothing with the corrected words |
| Fuzzy matching | `word~` / `word~2` expands to every vocabulary word within the edit distance |

The Zipf / word2vec tabs analyse the PubMed abstract collection (title + abstract of each
record); if no collection is present they fall back to all indexed documents.

## Project #3 features

| Area | What the system does |
|---|---|
| Collection | `data/vsm/pubmed_glp1_3000.jsonl`: the 3,000 newest English PubMed abstracts for *GLP-1* downloaded with `efetch`, each record with title, abstract, journal, year, authors, author keywords, **MeSH headings** (descriptor, major-topic flag, qualifiers) and publication types. 1,704 records carry MeSH, 2,534 keywords, 2,837 at least one label. The file lives outside the main index so the Project #2 analyses are unchanged |
| Pre-processing + term statistics | the Project #1 pipeline (tokenise → lower-case → stop words → Porter stemming) applied to title + abstract; per document tf, length and max tf; per term cf, df, idf = ln(N / df), probabilistic idf, max / mean tf; df histogram; tf-transform and idf curves |
| Term weighting | SMART triplets `tf·df·norm`: tf **n**atural / **l**og (1 + ln tf) / **a**ugmented (0.5 + 0.5 tf / max tf) / **b**oolean; df **n**one / **t** = ln(N / df) / **p** = max(0, ln((N − df) / df)); norm **n**one / **c**osine (L2). Twelve presets from `nnn` (raw TF) to `ltc` (standard) plus `atc`, `lpc` and Okapi BM25 as a reference. Weight vectors are stored as an inverted structure (term → {doc: weight}) and built lazily per scheme |
| Search engine | query → same pipeline → query vector (default `nnc`) → dot product with every document that shares a term → ranking → top-k with the score, the per-term contributions w(t,q) × w(t,d) (stacked bar), document length, the abstract, MeSH and keywords. With `c` on both sides the score *is* the cosine similarity |
| Controlled experiments | the same query under several schemes side by side: overlap@k and Kendall τ against the base scheme, mean length and mean max tf of the top-k documents, P@k / AP / nDCG@k when the query is an evaluation query; one-click sets for RQ1 (raw vs log TF), RQ2 (with / without IDF), RQ3 (TF vs IDF vs TF-IDF) and RQ4 (L2 normalisation) |
| Relevance judgements | `data/vsm/queries_glp1.json`: 15 queries, each a list of concept groups (regular expressions matched against MeSH headings and keywords). A document is relevant when every group matches a label; grade 2 when a matching MeSH heading is a major topic. Manual judgements (UI buttons 0 / 1 / 2 under each result, or `cli.py vsm judge`) override the rule and are saved to the file |
| Evaluation | `ir/evaluation.py`: P@k, R@k, RR / MRR, AP / MAP, R-precision, nDCG@k (graded, log₂ discount), condensed-list option (unlabelled documents removed); mean table with the best value per column, 2 × 2 × 2 factor grid (tf × idf × norm), per-query table, bar chart; rank-list comparison (overlap@k, Kendall τ, rank changes) |
| Research questions | the RQ1–RQ5 section of the tab is written from the live numbers (tf curve, idf of the most common terms, MAP / nDCG differences between neighbouring schemes, mean length of the top-10) |

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
ir/word2vec.py      skip-gram / CBOW with negative sampling, similarity, analogy, training examples, evaluation
ir/tsne.py          PCA (power iteration), t-SNE, k-NN preservation
ir/tfidf.py         term weighting (SMART tf·df·norm), vector space model, cosine ranking, BM25 reference
ir/evaluation.py    P@k, R@k, MRR, MAP, nDCG@k, R-precision; overlap@k, Kendall tau between rankings
ir/relevance.py     label-based (MeSH / keyword) relevance judgements with manual overrides
ir/svgplot.py       static SVG figures for the reports
static/             index.html, style.css, app.js, project2.js, w2v.js, tfidf.js (front-end)
tests/test_ir.py    unit tests (stemmer, tokenizer, splitter, parser, index, search, engine)
tests/test_project2.py   unit tests (Zipf, edit distance, spelling, fuzzy search, word2vec, collection)
tests/test_project3.py   unit tests (tf / idf variants, vectors, cosine ranking, measures, judgements, collection)
data/               PMC XML corpus, pubmed_*.jsonl abstract collections, w2v/<name>.json trained embeddings
data/vsm/           Project #3: pubmed_glp1_3000.jsonl (3,000 abstracts with MeSH), queries_glp1.json (queries + judgements)
report/             figures used in the reports; rq4_mixed_lengths.py (RQ4 side experiment)
REPORT.md           Project #2 written analysis (Zipf questions, CF/DF, TF-IDF, IR implications)
REPORT_w2v.md       Project #2 word-embedding report (pre-processing, vocabulary, training, similarity, PCA / t-SNE, RQ1–RQ5)
REPORT_tfidf.md     Project #3 term-weighting report (tasks 1–5, RQ1–RQ5, limitations of the judgements)
docs/               A4 one-page reports (HTML + PDF) of Projects #2 and #3
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
* **Word2Vec** – eight numbered sections that follow the assignment: pre-processing + vocabulary,
  training-example generator, training queue + model registry with loss curves, CBOW vs skip-gram
  table with measured values, nearest neighbours side by side, cosine matrix of word groups,
  PCA and t-SNE maps with k-NN preservation, evaluation per frequency bucket and neighbour overlap
  between models. Deep link: `/#w2v`.
* **TF-IDF 檢索** – six numbered sections that follow Project #3: term statistics with the tf / idf
  curves, one document's vector under several weighting schemes, the vector-space search engine
  (top-10 with score contributions and relevance badges), controlled experiments, evaluation
  (queries, relevant sets, measure table, factor grid, per-query table) and the RQ1–RQ5 summary.
  Deep link: `/#vsm`.
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
GET  /api/vsm/info   /api/vsm/terms?terms=glp,obesity   /api/vsm/doc?id=PMID42233552&schemes=nnn,ltc&top=15
GET  /api/vsm/search?q=..&scheme=ltc&qscheme=nnc&k=10   /api/vsm/compare?q=..&schemes=ltn,ltc&k=10
GET  /api/vsm/eval?schemes=nnn,ltc,bm25&k=10&condensed=0   /api/vsm/queries   /api/vsm/relevant?id=q01&scheme=ltc
POST /api/vsm/judge?id=q01&doc=PMID..&grade=2        (empty grade removes the manual judgement)
POST /api/vsm/collect                    (body = {"term": "GLP-1", "n": 3000, "name": "glp1_3000"})
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
