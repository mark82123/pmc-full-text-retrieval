"""Tokenisation and term normalisation.

Pipeline (classic IR pre-processing):
    raw text -> word tokens -> lower-case -> stop-word removal -> Porter stemming

Tokens are maximal runs of Unicode letters/digits.  Hyphens and apostrophes
split tokens, so "T-cell" -> ["t", "cell"] and "COVID-19" -> ["covid", "19"];
this keeps matching robust to inconsistent hyphenation in the literature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from .porter import PorterStemmer
from .stopwords import STOP_WORDS

# [^\W_]+  == one or more "word" characters excluding the underscore
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)          # letters only (for word counts)
# Same, but a hyphenated compound counts as ONE word ("Glucagon-like", "all-cause").
# Only used for statistics, never for indexing -- see WORD_RE's note above.
WORD_HYPHEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str        # original surface form
    norm: str        # lower-cased form
    stem: str        # Porter stem ("" if stop word)
    start: int       # character offset in source text
    end: int
    position: int    # word position in the token stream (stop words included)


class Tokenizer:
    def __init__(self, remove_stopwords: bool = True, stemming: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.stemmer = PorterStemmer()

    # ------------------------------------------------------------------ #
    def normalize(self, word: str) -> str:
        """lower-case + stem a single word; '' means the word is a stop word
        (and should be dropped)."""
        w = word.lower()
        if self.remove_stopwords and w in STOP_WORDS:
            return ""
        return self.stemmer.stem(w) if self.stemming else w

    def is_stopword(self, word: str) -> bool:
        return word.lower() in STOP_WORDS

    # ------------------------------------------------------------------ #
    def iter_tokens(self, text: str, start_position: int = 0) -> Iterator[Token]:
        pos = start_position
        for m in TOKEN_RE.finditer(text):
            surface = m.group(0)
            norm = surface.lower()
            yield Token(surface, norm, self.normalize(surface), m.start(), m.end(), pos)
            pos += 1

    def tokenize(self, text: str) -> list[Token]:
        return list(self.iter_tokens(text))

    def terms(self, text: str) -> list[str]:
        """Index terms for a text: stems with stop words removed."""
        return [t.stem for t in self.iter_tokens(text) if t.stem]

    def raw_words(self, text: str) -> list[str]:
        return [m.group(0) for m in TOKEN_RE.finditer(text)]
