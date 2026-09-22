"""Rule-based sentence boundary (end-of-sentence, EOS) detection.

A naive splitter breaks on every '.', '!' or '?', which badly over-counts in
biomedical prose ("Fig. 2", "et al.", "p < 0.05", "i.e.", "E. coli", "J. R.
Smith").  This module applies a set of hand-written rules to decide whether a
candidate punctuation mark really terminates a sentence:

  1. A candidate is [.!?] (or an ellipsis), optionally followed by closing
     quotes/brackets, followed by white-space and a next character.
  2. No split if the token before a '.' is a known abbreviation
     (e.g., i.e., et al., Fig., Dr., vs., approx., No., ...).
  3. No split if the token before a '.' is a single letter (an initial such
     as "J." or a genus abbreviation such as "E. coli") or a lower-case
     abbreviation-looking token with internal dots ("e.g.", "i.e.", "a.m.").
  4. No split if the next character is lower-case (sentences start with an
     upper-case letter, digit, quote, bracket or a Greek/other symbol).
  5. '?' and '!' always end a sentence unless followed by lower-case text.
  6. Decimal numbers ("0.05") never split because there is no white-space
     after the period.
  7. The end of a paragraph / heading always ends a sentence.

`split_sentences` returns the sentences; `naive_split` is kept so the two
methods can be compared in the UI.
"""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset("""
e.g i.e etc et al fig figs dr prof mr mrs ms vs approx no nos ref refs eq eqs
tab vol vols pp cf viz inc ltd co jr sr st mt ca resp suppl dept univ ed eds
jan feb mar apr jun jul aug sep sept oct nov dec min max sec hr hrs wk wks mo yr
yrs kg mg ml mm cm nm um mol conc spp sp subsp var ph.d m.d u.s u.k u.s.a a.m
p.m b.i.d q.d t.i.d i.v i.m s.c p.o approx ver rev ch chap sect para
""".split())

# candidate boundary: terminal punctuation, optional closers, whitespace, look-ahead at next char
_CANDIDATE_RE = re.compile(
    r"""(?P<punct>\.\.\.|[.!?]+)          # terminal punctuation (or ellipsis)
        (?P<close>["'”’)\]]*)             # closing quotes / brackets
        (?P<space>\s+)                    # white-space
        (?=(?P<next>\S))                  # next non-space char (not consumed)
    """,
    re.VERBOSE,
)
_PREV_TOKEN_RE = re.compile(r"([^\W_]+(?:\.[^\W_]+)*)$", re.UNICODE)   # word, possibly dotted (e.g / i.e / u.s)
_WS_RE = re.compile(r"\s+")


def _is_boundary(text: str, m: re.Match) -> bool:
    punct = m.group("punct")
    nxt = m.group("next")

    # rule 4/5: the next sentence should not start with a lower-case letter
    if nxt.islower():
        return False

    if "?" in punct or "!" in punct or punct == "...":
        return True

    # Only '.' remains.  Inspect the token immediately before it.
    before = text[: m.start("punct")]
    pm = _PREV_TOKEN_RE.search(before)
    if pm:
        tok = pm.group(1).lower()
        if tok in ABBREVIATIONS:                       # rule 2
            return False
        if len(tok) == 1 and tok.isalpha():            # rule 3: initials, "E. coli"
            return False
        if "." in tok and tok.replace(".", "").isalpha() and len(tok) <= 5:  # a.m., e.g.
            return False
        # "Fig. 3" style: previous word capitalised abbreviation followed by digit
        if nxt.isdigit() and tok in ABBREVIATIONS:
            return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split one paragraph (or any text block) into sentences using the rules."""
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return []
    sentences = []
    start = 0
    for m in _CANDIDATE_RE.finditer(text):
        if _is_boundary(text, m):
            end = m.end("close")
            sent = text[start:end].strip()
            if sent:
                sentences.append(sent)
            start = m.end("space")
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def naive_split(text: str) -> list[str]:
    """Baseline: split on every . ! ? (for comparison with the rule-based method)."""
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))
