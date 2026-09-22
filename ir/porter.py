"""Porter stemming algorithm.

A straightforward implementation of M.F. Porter, "An algorithm for suffix
stripping", Program 14(3):130-137, 1980.  The algorithm is in the public
domain; this implementation follows the original five-step description and
uses string slicing (rather than index juggling) for readability.
"""

from __future__ import annotations

_VOWELS = set("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    """True if word[i] is a consonant.  'y' is a consonant when it starts the
    word or follows a vowel (e.g. 'yellow', 'toy'), otherwise a vowel ('syzygy')."""
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's measure m: number of VC sequences in the stem,
    [C](VC){m}[V].  e.g. m(tr)=0, m(trouble)=1, m(troubles)=2."""
    m = 0
    i = 0
    n = len(stem)
    # skip leading consonants
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        # inside a vowel run
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        # consonant run -> completes a VC
        while i < n and _is_consonant(stem, i):
            i += 1
        m += 1
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _cvc(stem: str) -> bool:
    """consonant-vowel-consonant ending where the last consonant is not w, x or y."""
    if len(stem) < 3:
        return False
    n = len(stem)
    return (
        _is_consonant(stem, n - 1)
        and not _is_consonant(stem, n - 2)
        and _is_consonant(stem, n - 3)
        and stem[-1] not in "wxy"
    )


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP4 = [
    "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism",
    "ate", "iti", "ous", "ive", "ize", "ion", "al", "er", "ic", "ou",
]


def _apply_rules(word: str, rules, min_measure: int) -> str:
    """Apply the first rule whose suffix matches.  Per Porter, once a suffix
    matches the step terminates even if the measure condition fails."""
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > min_measure:
                return stem + replacement
            return word
    return word


def stem(word: str) -> str:
    """Return the Porter stem of an (already lower-cased) word."""
    return _stem(word, None)


def stem_trace(word: str) -> list[dict]:
    """Stem a word and return the intermediate results, one entry per step
    that changed something: [{"step": "1a", "before": "patients",
    "after": "patient", "rule": "…"}, …]."""
    trace: list[dict] = []
    _stem(word.lower(), trace)
    return trace


def _note(trace, step, before, after, rule):
    if trace is not None and before != after:
        trace.append({"step": step, "before": before, "after": after, "rule": rule})


def _stem(word: str, trace) -> str:
    w = word
    if len(w) <= 2:
        return w

    # ---- Step 1a: plurals -------------------------------------------------
    b = w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    _note(trace, "1a", b, w, "plural: sses→ss, ies→i, s→(drop)")

    # ---- Step 1b: -ed / -ing ---------------------------------------------
    b = w
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        stripped = False
        if w.endswith("ed") and _contains_vowel(w[:-2]):
            w = w[:-2]
            stripped = True
        elif w.endswith("ing") and _contains_vowel(w[:-3]):
            w = w[:-3]
            stripped = True
        if stripped:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif _ends_double_consonant(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif _measure(w) == 1 and _cvc(w):
                w += "e"
    _note(trace, "1b", b, w, "-ed / -ing removed (stem must contain a vowel), ending repaired (at/bl/iz→+e, double consonant→single, cvc→+e)")

    # ---- Step 1c: y -> i --------------------------------------------------
    b = w
    if w.endswith("y") and _contains_vowel(w[:-1]):
        w = w[:-1] + "i"
    _note(trace, "1c", b, w, "y→i when the stem has a vowel")

    # ---- Step 2 -----------------------------------------------------------
    b = w
    w = _apply_rules(w, _STEP2, 0)
    _note(trace, "2", b, w, "long suffix mapped (m>0): ational→ate, ization→ize, iveness→ive, biliti→ble …")
    # ---- Step 3 -----------------------------------------------------------
    b = w
    w = _apply_rules(w, _STEP3, 0)
    _note(trace, "3", b, w, "suffix mapped (m>0): icate→ic, ative→(drop), alize→al, ical→ic, ful/ness→(drop)")

    # ---- Step 4: remove suffix if m > 1 ----------------------------------
    b = w
    for suffix in _STEP4:
        if w.endswith(suffix):
            stem_ = w[: -len(suffix)]
            if _measure(stem_) > 1:
                if suffix == "ion" and not stem_.endswith(("s", "t")):
                    break
                w = stem_
            break
    _note(trace, "4", b, w, "suffix dropped when m>1: ement, ance, ence, able, ment, ant, ent, ism, ate, iti, ous, ive, ize, ion(after s/t), al, er, ic, ou")

    # ---- Step 5a: remove trailing e ---------------------------------------
    b = w
    if w.endswith("e"):
        a = w[:-1]
        m = _measure(a)
        if m > 1 or (m == 1 and not _cvc(a)):
            w = a
    _note(trace, "5a", b, w, "trailing e dropped when m>1 (or m=1 and not cvc)")
    # ---- Step 5b: ll -> l --------------------------------------------------
    b = w
    if w.endswith("ll") and _measure(w) > 1:
        w = w[:-1]
    _note(trace, "5b", b, w, "ll→l when m>1")
    return w


class PorterStemmer:
    """Memoising wrapper: stemming the same word repeatedly is the hot path
    when indexing a corpus, so results are cached."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is None:
            cached = stem(word)
            self._cache[word] = cached
        return cached

    __call__ = stem
