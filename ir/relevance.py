"""Relevance judgements for the evaluation queries (Project #3, task 5).

The abstract collection carries no relevance labels of its own, so the
judgements are derived from the *topic labels* of each record: the MeSH
headings assigned by NLM indexers (with the major-topic flag) and the
author keywords.  A query is a set of *concept groups*; each group is a
list of case-insensitive regular expressions matched against every label
of a document.  A document is relevant when **every** group matches at
least one of its labels:

    grade 2   relevant, and at least one matching MeSH heading is a major
              topic (the article is *about* the concept)
    grade 1   relevant (labels mention every concept)
    grade 0   otherwise

Labels are only compared with the *labels*, never with the text, so the
judgements are independent of the term weighting under test.  Manual
judgements (added from the UI or by editing the JSON file) override the
rule for a document: {"PMID123": 2, "PMID456": 0}.

Limitations (stated in the report): ~5 % of the records have neither MeSH
nor keywords and can never be judged relevant; MeSH indexing is not yet
complete for recent records; author keywords are free text; the rules were
written by the author, not by independent assessors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .pmc_parser import Document


def doc_labels(doc: Document) -> list[tuple[str, str, bool]]:
    """(label text, source, major) for every MeSH heading (+ qualifiers) and keyword."""
    out: list[tuple[str, str, bool]] = []
    for m in doc.mesh:
        out.append((m.get("d", ""), "mesh", bool(m.get("major"))))
        for q in m.get("q", []):
            out.append((f"{m.get('d', '')}/{q}", "mesh", bool(m.get("major"))))
    for k in doc.keywords:
        out.append((k.replace("‐", "-").replace("‑", "-").replace("’", "'"), "keyword", False))
    return out


class Query:
    def __init__(self, spec: dict):
        self.id: str = spec["id"]
        self.text: str = spec["text"]
        self.description: str = spec.get("description", "")
        self.groups: list[list[str]] = spec.get("groups", [])
        self.manual: dict[str, int] = {k: int(v) for k, v in spec.get("manual", {}).items()}
        self._res = [[re.compile(p, re.I) for p in g] for g in self.groups]

    def spec(self) -> dict:
        return {"id": self.id, "text": self.text, "description": self.description, "groups": self.groups, "manual": self.manual}

    def judge(self, doc: Document) -> tuple[int, list[str]]:
        """(grade, matched labels) of one document under the rules + manual overrides."""
        if doc.doc_id in self.manual:
            return self.manual[doc.doc_id], ["manual"]
        labels = doc_labels(doc)
        if not labels or not self._res:
            return 0, []
        matched: list[str] = []
        major = False
        for group in self._res:
            hit = [(text, src, maj) for text, src, maj in labels if any(r.search(text) for r in group)]
            if not hit:
                return 0, []
            matched.extend(f"{src}:{text}" + ("*" if maj else "") for text, src, maj in hit)
            major = major or any(maj for _, _, maj in hit)
        return (2 if major else 1), sorted(set(matched))


class QuerySet:
    def __init__(self, path: str | Path, docs: dict[str, Document]):
        self.path = Path(path)
        self.docs = docs
        self.queries: list[Query] = []
        self.notes: str = ""
        self._qrels: dict[str, dict[str, int]] = {}
        self._matched: dict[str, dict[str, list[str]]] = {}
        if self.path.is_file():
            self.load()
        self.rebuild()

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.notes = data.get("notes", "")
        self.queries = [Query(q) for q in data.get("queries", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"notes": self.notes, "queries": [q.spec() for q in self.queries]},
                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def rebuild(self) -> None:
        self._qrels, self._matched = {}, {}
        for q in self.queries:
            rel, matched = {}, {}
            for doc in self.docs.values():
                g, m = q.judge(doc)
                if g > 0:
                    rel[doc.doc_id] = g
                    matched[doc.doc_id] = m
            self._qrels[q.id] = rel
            self._matched[q.id] = matched

    # ------------------------------------------------------------------ #
    def get(self, qid: str) -> Query | None:
        return next((q for q in self.queries if q.id == qid), None)

    def by_text(self, text: str) -> Query | None:
        t = " ".join(text.lower().split())
        return next((q for q in self.queries if " ".join(q.text.lower().split()) == t), None)

    def qrels(self, qid: str) -> dict[str, int]:
        return self._qrels.get(qid, {})

    def matched_labels(self, qid: str, doc_id: str) -> list[str]:
        return self._matched.get(qid, {}).get(doc_id, [])

    def judged(self) -> set[str]:
        """Documents that carry at least one topic label (MeSH or keyword) and so *could* be judged."""
        return {d.doc_id for d in self.docs.values() if d.mesh or d.keywords}

    def set_manual(self, qid: str, doc_id: str, grade: int | None) -> dict:
        q = self.get(qid)
        if q is None:
            raise ValueError(f"unknown query {qid}")
        if doc_id not in self.docs:
            raise ValueError(f"unknown document {doc_id}")
        if grade is None:
            q.manual.pop(doc_id, None)
        else:
            q.manual[doc_id] = max(0, min(2, int(grade)))
        self.save()
        self.rebuild()
        g, m = q.judge(self.docs[doc_id])
        return {"query": qid, "doc_id": doc_id, "grade": g, "matched": m, "manual": doc_id in q.manual}

    def summary(self) -> list[dict]:
        rows = []
        for q in self.queries:
            rel = self._qrels[q.id]
            rows.append({"id": q.id, "text": q.text, "description": q.description, "groups": q.groups,
                         "relevant": len(rel), "grade2": sum(1 for g in rel.values() if g == 2),
                         "grade1": sum(1 for g in rel.values() if g == 1), "manual": len(q.manual),
                         "from_mesh": sum(1 for d in rel if any(x.startswith("mesh:") for x in self._matched[q.id][d])),
                         "from_keyword_only": sum(1 for d in rel if self._matched[q.id][d]
                                                  and all(x.startswith("keyword:") for x in self._matched[q.id][d]))})
        return rows
