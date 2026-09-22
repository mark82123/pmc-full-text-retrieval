"""Document and corpus statistics."""

from __future__ import annotations

from collections import Counter

from .pmc_parser import Document
from .sentence import naive_split, split_sentences
from .tokenizer import TOKEN_RE, WORD_RE, Tokenizer


def document_stats(doc: Document, tokenizer: Tokenizer, top_n: int = 15) -> dict:
    units = doc.units()
    text = "\n".join(t for _, t in units)
    tokens = TOKEN_RE.findall(text)
    words = WORD_RE.findall(text)
    sentences = [s for _, t in units for s in split_sentences(t)]
    naive = [s for _, t in units for s in naive_split(t)]
    terms = tokenizer.terms(text)                      # stemmed, stop words removed
    lowered = [w.lower() for w in words]
    stop_count = sum(1 for w in lowered if tokenizer.is_stopword(w))
    freq = Counter(terms)
    body_paras = sum(1 for f, _ in units if f == "body")
    n_sent = len(sentences)
    fields = {}
    for fld in ("title", "abstract", "body"):
        ftext = "\n".join(t for f, t in units if (f == fld or (fld == "body" and f == "heading")))
        fterms = tokenizer.terms(ftext)
        fields[fld] = {
            "characters": len(ftext),
            "words": len(WORD_RE.findall(ftext)),
            "sentences": sum(len(split_sentences(t)) for f, t in units if (f == fld or (fld == "body" and f == "heading"))),
            "index_terms": len(fterms),
            "unique_terms": len(set(fterms)),
        }
    return {
        "fields": fields,                                 # per-field breakdown of the numbers below
        "doc_id": doc.doc_id,
        "characters": len(text),
        "characters_no_spaces": sum(1 for c in text if not c.isspace()),
        "letters": sum(1 for c in text if c.isalpha()),
        "digits": sum(1 for c in text if c.isdigit()),
        "tokens": len(tokens),
        "words": len(words),
        "unique_words": len(set(lowered)),
        "stop_words": stop_count,
        "index_terms": len(terms),
        "unique_terms": len(freq),
        "sentences": n_sent,
        "sentences_naive": len(naive),
        "paragraphs": body_paras,
        "abstract_paragraphs": len(doc.abstract),
        "sections": len([s for s in doc.sections if s.heading]),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
        "avg_words_per_sentence": round(len(words) / n_sent, 2) if n_sent else 0,
        "avg_sentence_chars": round(sum(len(s) for s in sentences) / n_sent, 1) if n_sent else 0,
        "lexical_diversity": round(len(freq) / len(terms), 3) if terms else 0,
        "top_terms": freq.most_common(top_n),
        "title_words": len(WORD_RE.findall(doc.title)),
        "abstract_words": len(WORD_RE.findall(doc.abstract_text())),
        "body_words": len(WORD_RE.findall(doc.body_text())),
    }


def corpus_stats(all_stats: list[dict], term_df: Counter, term_cf: Counter, top_n: int = 30) -> dict:
    n = len(all_stats)
    tot = lambda k: sum(s[k] for s in all_stats)  # noqa: E731
    return {
        "documents": n,
        "characters": tot("characters"),
        "words": tot("words"),
        "sentences": tot("sentences"),
        "sentences_naive": tot("sentences_naive"),
        "index_terms": tot("index_terms"),
        "vocabulary": len(term_df),
        "avg_words_per_doc": round(tot("words") / n, 1) if n else 0,
        "avg_sentences_per_doc": round(tot("sentences") / n, 1) if n else 0,
        "top_terms": term_cf.most_common(top_n),           # collection frequency
        "top_df": term_df.most_common(top_n),              # document frequency
    }
