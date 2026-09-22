"""Full-text retrieval tool for PubMed Central (PMC) XML documents.

Modules
-------
porter      - Porter stemming algorithm (pure Python implementation)
stopwords   - English stop-word list
tokenizer   - tokenisation + normalisation pipeline (lower-case, stop words, stemming)
sentence    - rule-based sentence boundary (EOS) detection
pmc_parser  - JATS/PMC XML -> Document
stats       - document / corpus statistics
index       - positional inverted index
search      - query parsing (boolean, phrase, wildcard), scoring (BM25 / TF-IDF)
engine      - glue layer used by the web app and the CLI
"""
