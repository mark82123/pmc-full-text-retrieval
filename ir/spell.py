"""Edit distance (dynamic programming) and spelling correction.

* `edit_distance` / `edit_matrix` - Levenshtein distance with the optional
  Damerau extension (transposition of two adjacent characters counts as one
  edit).  `edit_matrix` also returns the DP table and one optimal alignment
  so the computation can be displayed.
* `SpellIndex` - dictionary of the surface words of the collection with
  their collection frequency.  Candidates for a misspelled word are found
  through a character-bigram index (a correction within k edits must share
  most bigrams with the query), verified with a banded DP and ranked by
  (distance, -frequency).
"""

from __future__ import annotations

from collections import Counter, defaultdict


def edit_distance(a: str, b: str, transpositions: bool = True) -> int:
    """Levenshtein distance between a and b (O(|a|*|b|) time, three rows)."""
    if a == b:
        return 0
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            d = min(prev[j] + 1,                      # deletion
                    cur[j - 1] + 1,                   # insertion
                    prev[j - 1] + (ca != cb))         # substitution / match
            if transpositions and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                d = min(d, prev2[j - 2] + 1)          # transposition
            cur.append(d)
        prev2, prev = prev, cur
    return prev[-1]


def bounded_distance(a: str, b: str, k: int) -> int | None:
    """Edit distance if it is <= k, else None.  Only the diagonal band of
    width 2k+1 is filled and the scan stops as soon as a row exceeds k."""
    if abs(len(a) - len(b)) > k:
        return None
    big = k + 1
    prev2: list[int] = []
    prev = [j if j <= k else big for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [i if i <= k else big] + [big] * len(b)
        lo, hi = max(1, i - k), min(len(b), i + k)
        for j in range(lo, hi + 1):
            cb = b[j - 1]
            d = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                d = min(d, prev2[j - 2] + 1)
            cur[j] = d
        if min(cur[lo - 1:hi + 1]) > k:
            return None
        prev2, prev = prev, cur
    return prev[-1] if prev[-1] <= k else None


def edit_matrix(a: str, b: str, transpositions: bool = True) -> dict:
    """Full DP table plus one optimal alignment (for display).

    ops: [{"op": match|substitute|insert|delete|transpose, "a": .., "b": .., "i": row, "j": col}]
    path: the (i, j) cells of the table on the optimal path.
    """
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if transpositions and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d = min(d, D[i - 2][j - 2] + 1)
            D[i][j] = d
    ops, path = [], [(n, m)]
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            ops.append({"op": "match", "a": a[i - 1], "b": b[j - 1]})
            i, j = i - 1, j - 1
        elif (transpositions and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]
              and D[i][j] == D[i - 2][j - 2] + 1):
            ops.append({"op": "transpose", "a": a[i - 2:i], "b": b[j - 2:j]})
            i, j = i - 2, j - 2
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append({"op": "substitute", "a": a[i - 1], "b": b[j - 1]})
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append({"op": "delete", "a": a[i - 1], "b": ""})
            i -= 1
        else:
            ops.append({"op": "insert", "a": "", "b": b[j - 1]})
            j -= 1
        path.append((i, j))
    ops.reverse()
    return {"a": a, "b": b, "distance": D[n][m], "matrix": D, "ops": ops, "path": [list(p) for p in reversed(path)]}


def _bigrams(word: str) -> set[str]:
    w = f"${word}$"
    return {w[i:i + 2] for i in range(len(w) - 1)}


def default_max_distance(word: str) -> int:
    return 1 if len(word) <= 4 else 2


class SpellIndex:
    def __init__(self) -> None:
        self.words: Counter = Counter()                  # surface word (lower-case) -> collection frequency
        self._grams: dict[str, set[str]] | None = None   # bigram -> words, built lazily

    def add(self, counts: Counter) -> None:
        self.words.update(counts)
        self._grams = None

    def remove(self, counts: Counter) -> None:
        self.words.subtract(counts)
        for w in [w for w in counts if self.words[w] <= 0]:
            del self.words[w]
        self._grams = None

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def _index(self) -> dict[str, set[str]]:
        if self._grams is None:
            grams: dict[str, set[str]] = defaultdict(set)
            for w in self.words:
                for g in _bigrams(w):
                    grams[g].add(w)
            self._grams = grams
        return self._grams

    def candidates(self, word: str, k: int) -> set[str]:
        """Words that could be within k edits: one edit destroys at most two
        (three for a transposition) of the query's bigrams."""
        grams = _bigrams(word)
        need = len(grams) - 3 * k
        if need <= 0:                                    # short word: bigram filter is useless
            return {w for w in self.words if abs(len(w) - len(word)) <= k}
        index = self._index()
        shared: Counter = Counter()
        for g in grams:
            for w in index.get(g, ()):
                shared[w] += 1
        return {w for w, c in shared.items() if c >= need and abs(len(w) - len(word)) <= k}

    def within(self, word: str, k: int | None = None, limit: int = 0) -> list[dict]:
        """Dictionary words within k edits of *word*, best first:
        [{"word", "distance", "cf"}]."""
        word = word.lower()
        k = default_max_distance(word) if k is None else k
        out = []
        for w in self.candidates(word, k):
            d = bounded_distance(word, w, k)
            if d is not None:
                out.append({"word": w, "distance": d, "cf": self.words[w]})
        out.sort(key=lambda r: (r["distance"], -r["cf"], r["word"]))
        return out[:limit] if limit else out

    def suggest(self, word: str, limit: int = 5, k: int | None = None) -> list[dict]:
        """Corrections for a word (the word itself is not returned)."""
        word = word.lower()
        return [r for r in self.within(word, k) if r["word"] != word][:limit]
