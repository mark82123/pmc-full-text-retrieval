"""Reuters-21578 for Project #4: SGML parsing, the ModApte split and
topic-based relevance judgements.

The collection (Lewis, "Reuters-21578, Distribution 1.0") is 22 SGML files
reut2-000.sgm … reut2-021.sgm with 21,578 newswire stories from 1987.  Each
story is a <REUTERS> element whose attributes give the split
(LEWISSPLIT = TRAIN / TEST / NOT-USED, TOPICS = YES / NO / BYPASS) and whose
children hold the categories (<TOPICS><D>earn</D>…) and the text
(<TITLE>, <DATELINE>, <BODY>).  The standard **ModApte** split keeps the
stories with TOPICS="YES":

    train   LEWISSPLIT="TRAIN"  TOPICS="YES"    9,603 documents
    test    LEWISSPLIT="TEST"   TOPICS="YES"    3,299 documents
    unused  everything else                     8,676 documents

(a TOPICS="YES" story may still carry zero topics: the indexers looked at it
and found none of the 135 categories).

Relevance for the retrieval experiments comes from the TOPICS field: a
query is written for a topic (earn, acq, crude, grain, trade …) and a test
document is relevant when it carries that topic.  The grades are

    2   the topic is the document's only topic (the story is *about* it)
    1   the topic is one of several topics of the document
    0   the document does not carry the topic

Topics are assigned by Reuters / Carnegie Group indexers and never looked
at the query text, so the judgements are independent of the document
representation under test (TF-IDF or word embeddings).

Text clean-up: the SGML is Latin-1, entities (&lt; &amp; &#3;) are decoded,
control characters removed, the trailing "Reuter" signature dropped and the
hard-wrapped lines of a paragraph re-joined (paragraphs start with an
indentation of four spaces in the source), so that sentence splitting and
word2vec training see whole sentences.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

from .pmc_parser import Document

# the five topics recommended for the experiments, with the official category description
TOPICS = [
    ("earn", "Earnings and earnings forecasts", "企業獲利與財務結果"),
    ("acq", "Mergers and acquisitions", "企業收購與併購"),
    ("crude", "Crude oil", "原油"),
    ("grain", "Grain", "穀物"),
    ("trade", "Trade", "國際貿易"),
]
TOPIC_IDS = [t for t, _, _ in TOPICS]
TOPIC_NAMES = {t: n for t, n, _ in TOPICS}
TOPIC_NAMES_ZH = {t: z for t, _, z in TOPICS}

_REUTERS = re.compile(r"<REUTERS([^>]*)>(.*?)</REUTERS>", re.S)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_TAG = re.compile(r"<(TITLE|DATELINE|BODY|TOPICS|PLACES|PEOPLE|ORGS|EXCHANGES|COMPANIES|DATE|AUTHOR)>(.*?)</\1>", re.S)
_D = re.compile(r"<D>(.*?)</D>", re.S)
_TEXT_TYPE = re.compile(r'<TEXT\s+TYPE="([^"]+)"')
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS = re.compile(r"[ \t]+")
_SIGNATURE = re.compile(r"\s*(?:Reuter|REUTER)\s*$")
_LEADING_SPACES = "    "


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = _CTRL.sub("", text)
    return text


def unwrap_body(body: str) -> list[str]:
    """Body text -> paragraphs.  Reuters hard-wraps lines at ~70 characters
    and starts a new paragraph with four spaces, e.g.

        Showers continued throughout the week in
        the Bahia cocoa zone, …
            The dry period means the temporao will be late this year.

    Lines that start with the indentation open a paragraph; the other lines
    continue the current one."""
    body = _SIGNATURE.sub("", _clean(body).replace("\r", ""))
    paras: list[str] = []
    cur: list[str] = []
    for line in body.split("\n"):
        if not line.strip():
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if line.startswith(_LEADING_SPACES) and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(_WS.sub(" ", line.strip()))
    if cur:
        paras.append(" ".join(cur))
    return [p for p in paras if p]


def parse_sgml(text: str) -> list[dict]:
    """Every <REUTERS> element of one SGML file as a dict (all 21,578 stories
    over the 22 files; nothing is filtered here)."""
    out = []
    for m in _REUTERS.finditer(text):
        attrs = dict(_ATTR.findall(m.group(1)))
        inner = m.group(2)
        fields = {k: v for k, v in _TAG.findall(inner)}
        tt = _TEXT_TYPE.search(inner)
        lists = {k: [_clean(d).strip() for d in _D.findall(fields.get(k, ""))] for k in ("TOPICS", "PLACES", "PEOPLE", "ORGS", "EXCHANGES", "COMPANIES")}
        title = _WS.sub(" ", _clean(fields.get("TITLE", "")).strip())
        body = unwrap_body(fields.get("BODY", ""))
        if not body and not title and tt and tt.group(1) == "UNPROC":
            # un-processed stories keep their text directly inside <TEXT>
            raw = re.search(r"<TEXT[^>]*>(.*?)</TEXT>", inner, re.S)
            body = unwrap_body(re.sub(r"<[^>]+>", "", raw.group(1))) if raw else []
        out.append({
            "newid": int(attrs.get("NEWID", "0")), "oldid": int(attrs.get("OLDID", "0") or 0),
            "lewissplit": attrs.get("LEWISSPLIT", ""), "topics_attr": attrs.get("TOPICS", ""), "cgisplit": attrs.get("CGISPLIT", ""),
            "type": tt.group(1) if tt else "NORM",
            "date": _clean(fields.get("DATE", "")).strip(),
            "topics": lists["TOPICS"], "places": lists["PLACES"], "people": lists["PEOPLE"], "orgs": lists["ORGS"],
            "exchanges": lists["EXCHANGES"], "companies": lists["COMPANIES"],
            "title": title, "dateline": _WS.sub(" ", _clean(fields.get("DATELINE", "")).strip()),
            "author": _clean(fields.get("AUTHOR", "")).strip(),
            "body": body,
        })
    return out


def load_sgml_dir(path: str | Path) -> list[dict]:
    """All stories of a directory with reut2-*.sgm files, in NEWID order."""
    files = sorted(Path(path).glob("reut2-*.sgm"))
    if not files:
        raise ValueError(f"no reut2-*.sgm files in {path}")
    recs: list[dict] = []
    for f in files:
        recs.extend(parse_sgml(f.read_bytes().decode("latin-1")))
    recs.sort(key=lambda r: r["newid"])
    return recs


def modapte_split(rec: dict) -> str:
    """'train' / 'test' for the ModApte documents, '' for the unused ones."""
    if rec.get("topics_attr") != "YES":
        return ""
    if rec.get("lewissplit") == "TRAIN":
        return "train"
    if rec.get("lewissplit") == "TEST":
        return "test"
    return ""


def modapte(records: list[dict]) -> list[dict]:
    """The 12,902 ModApte documents with a 'split' field (train / test)."""
    out = []
    for r in records:
        s = modapte_split(r)
        if s:
            out.append({**r, "split": s})
    return out


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ("newid", "oldid", "split", "type", "date", "topics", "places", "title", "dateline", "body")
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def to_document(rec: dict) -> Document:
    """A Reuters record as the Document used by the index / vector space:
    the title is the title, the body paragraphs take the place of the abstract."""
    return Document(doc_id=f"R{rec['newid']}", title=rec.get("title", ""), abstract=list(rec.get("body", [])),
                    journal="Reuters", year=(rec.get("date") or "")[7:11] or "1987",
                    keywords=[], article_type="reuters", source=f"reuters-{rec.get('split', '')}",
                    topics=list(rec.get("topics", [])), places=list(rec.get("places", [])), split=rec.get("split", ""))


def load_documents(path: str | Path, split: str | None = None) -> list[Document]:
    return [to_document(r) for r in read_jsonl(path) if split is None or r.get("split") == split]


def topic_statistics(docs: list[Document], topics: list[str] | None = None) -> dict:
    """Counts per topic and split, number of topics per document, documents
    without a body."""
    topics = topics or TOPIC_IDS
    by_split: dict[str, list[Document]] = {}
    for d in docs:
        by_split.setdefault(d.split or "all", []).append(d)
    rows = []
    all_counts = Counter(t for d in docs for t in d.topics)
    for t in topics:
        row = {"topic": t, "name": TOPIC_NAMES.get(t, t), "name_zh": TOPIC_NAMES_ZH.get(t, ""), "total": all_counts.get(t, 0)}
        for s, ds in by_split.items():
            row[s] = sum(1 for d in ds if t in d.topics)
            row[s + "_only"] = sum(1 for d in ds if d.topics == [t])
        rows.append(row)
    n_topics = Counter(len(d.topics) for d in docs)
    wanted = set(topics)
    # distinct documents carrying at least one of the listed topics (the rows overlap, so they do not add up to this)
    any_of = {"total": sum(1 for d in docs if wanted.intersection(d.topics))}
    for s, ds in by_split.items():
        any_of[s] = sum(1 for d in ds if wanted.intersection(d.topics))
    return {"documents": len(docs), "splits": {s: len(ds) for s, ds in by_split.items()}, "any_of": any_of,
            "topics_total": len(all_counts), "rows": rows,
            "topics_per_document": [{"topics": k, "documents": v} for k, v in sorted(n_topics.items())],
            "without_body": sum(1 for d in docs if not d.abstract),
            "most_common": [{"topic": t, "documents": c} for t, c in all_counts.most_common(20)]}


# ---------------------------------------------------------------------- #
# evaluation queries judged by topic membership
# ---------------------------------------------------------------------- #
QUERY_KINDS = [
    ("topic", "topic word", "查詢直接使用主題詞本身（oil、grain、trade…）"),
    ("domain", "Reuters vocabulary", "查詢使用路透新聞慣用詞（takeover、barrel、tariff…）"),
    ("synonym", "synonym / paraphrase", "查詢改用同義詞或換句話說（petroleum、cereal、firms…），與文件的用字不同"),
    ("semantic", "semantic description", "查詢只描述情境，幾乎不含主題詞（energy exploration、crop report…）"),
]
QUERY_KIND_NAMES = {k: n for k, n, _ in QUERY_KINDS}

DEFAULT_QUERIES = [
    # earn
    {"id": "e01", "topic": "earn", "kind": "topic", "text": "quarterly net profit results",
     "description": "earnings report vocabulary (net, profit, qtr)"},
    {"id": "e02", "topic": "earn", "kind": "domain", "text": "company reports earnings per share",
     "description": "Reuters earnings tables use shr / net / revs"},
    {"id": "e03", "topic": "earn", "kind": "synonym", "text": "corporate financial performance for the year",
     "description": "paraphrase: none of the words net / profit / earnings"},
    {"id": "e04", "topic": "earn", "kind": "semantic", "text": "income statement with revenue and sales figures",
     "description": "describes the content of an earnings story"},
    # acq
    {"id": "a01", "topic": "acq", "kind": "topic", "text": "acquisition of a company",
     "description": "topic word acquisition"},
    {"id": "a02", "topic": "acq", "kind": "domain", "text": "takeover bid for shares",
     "description": "Reuters uses takeover, bid, stake"},
    {"id": "a03", "topic": "acq", "kind": "synonym", "text": "merger agreement between two firms",
     "description": "merger / firms instead of acquisition / company"},
    {"id": "a04", "topic": "acq", "kind": "semantic", "text": "buying a controlling stake in a business",
     "description": "describes an acquisition without saying so"},
    # crude
    {"id": "c01", "topic": "crude", "kind": "topic", "text": "crude oil prices",
     "description": "topic words crude oil"},
    {"id": "c02", "topic": "crude", "kind": "domain", "text": "OPEC barrel output quota",
     "description": "Reuters oil-market vocabulary (opec, bpd, barrel)"},
    {"id": "c03", "topic": "crude", "kind": "synonym", "text": "petroleum production and output",
     "description": "petroleum instead of oil — the lexical-mismatch example of the assignment"},
    {"id": "c04", "topic": "crude", "kind": "semantic", "text": "energy exploration and drilling",
     "description": "describes the oil business without oil / crude"},
    # grain
    {"id": "g01", "topic": "grain", "kind": "topic", "text": "grain exports",
     "description": "topic word grain"},
    {"id": "g02", "topic": "grain", "kind": "domain", "text": "wheat and corn harvest",
     "description": "sub-topics of grain (wheat, corn) — the word grain itself is absent"},
    {"id": "g03", "topic": "grain", "kind": "synonym", "text": "cereal crop production forecast",
     "description": "cereal instead of grain"},
    {"id": "g04", "topic": "grain", "kind": "semantic", "text": "agriculture department crop report",
     "description": "USDA crop reports are grain stories"},
    # trade
    {"id": "t01", "topic": "trade", "kind": "topic", "text": "international trade dispute",
     "description": "topic word trade"},
    {"id": "t02", "topic": "trade", "kind": "domain", "text": "trade deficit and surplus",
     "description": "Reuters trade-balance vocabulary"},
    {"id": "t03", "topic": "trade", "kind": "synonym", "text": "import tariffs and export restrictions",
     "description": "tariffs / imports / exports instead of trade"},
    {"id": "t04", "topic": "trade", "kind": "semantic", "text": "protectionism and talks between countries",
     "description": "describes a trade story without the word trade"},
]

QUERY_NOTES = ("Evaluation queries for the Reuters-21578 ModApte test split. Every query is written for one TOPICS category; "
               "a document is relevant when it carries that topic (grade 2 when the topic is its only topic, grade 1 otherwise). "
               "The 'kind' field groups the queries by how far their wording is from the vocabulary of the relevant stories: "
               "topic word / Reuters vocabulary / synonym or paraphrase / semantic description. Topics were assigned by Reuters "
               "and Carnegie Group indexers in 1987, independently of any retrieval method.")


class TopicQuery:
    def __init__(self, spec: dict):
        self.id: str = spec["id"]
        self.text: str = spec["text"]
        self.topic: str = spec["topic"]
        self.kind: str = spec.get("kind", "topic")
        self.description: str = spec.get("description", "")

    def spec(self) -> dict:
        return {"id": self.id, "text": self.text, "topic": self.topic, "kind": self.kind, "description": self.description}

    def judge(self, doc: Document) -> int:
        if self.topic not in doc.topics:
            return 0
        return 2 if len(doc.topics) == 1 else 1


class TopicQuerySet:
    """The queries of data/reuters/queries_reuters.json and their relevant
    sets over the given documents (the test split)."""

    def __init__(self, path: str | Path, docs: dict[str, Document]):
        self.path = Path(path)
        self.docs = docs
        self.notes = QUERY_NOTES
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.notes = data.get("notes", QUERY_NOTES)
            self.queries = [TopicQuery(q) for q in data.get("queries", [])]
        else:
            self.queries = [TopicQuery(q) for q in DEFAULT_QUERIES]
            self.save()
        self._qrels: dict[str, dict[str, int]] = {}
        for q in self.queries:
            self._qrels[q.id] = {d.doc_id: g for d in docs.values() if (g := q.judge(d)) > 0}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"notes": self.notes, "queries": [q.spec() for q in self.queries]},
                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def get(self, qid: str) -> TopicQuery | None:
        return next((q for q in self.queries if q.id == qid), None)

    def by_text(self, text: str) -> TopicQuery | None:
        t = " ".join(text.lower().split())
        return next((q for q in self.queries if " ".join(q.text.lower().split()) == t), None)

    def qrels(self, qid: str) -> dict[str, int]:
        return self._qrels.get(qid, {})

    def topic_qrels(self, topic: str) -> dict[str, int]:
        """Relevant set of an ad hoc query judged against one TOPICS category
        (same grading as the evaluation queries)."""
        key = "topic:" + topic
        if key not in self._qrels:
            tq = TopicQuery({"id": key, "text": "", "topic": topic})
            self._qrels[key] = {d.doc_id: g for d in self.docs.values() if (g := tq.judge(d)) > 0}
        return self._qrels[key]

    def judged(self) -> set[str]:
        return set(self.docs)                         # every ModApte document was indexed for topics

    def summary(self) -> list[dict]:
        rows = []
        for q in self.queries:
            rel = self._qrels[q.id]
            rows.append({"id": q.id, "text": q.text, "topic": q.topic, "topic_name": TOPIC_NAMES.get(q.topic, q.topic),
                         "kind": q.kind, "kind_name": QUERY_KIND_NAMES.get(q.kind, q.kind), "description": q.description,
                         "relevant": len(rel), "grade2": sum(1 for g in rel.values() if g == 2),
                         "grade1": sum(1 for g in rel.values() if g == 1)})
        return rows
