"""Parse PubMed Central (JATS) XML articles into Document objects.

Handles both a single <article> file and a <pmc-articleset> (as returned by
NCBI E-utilities efetch).  Bibliographic citations (<xref ref-type="bibr">),
tables and the reference list are dropped from the body text because they
add noise (thousands of bare numbers and author names) to the index.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_WS = re.compile(r"\s+")

# elements whose text should not be part of the retrievable body text
_SKIP_TAGS = {"table", "ref-list", "table-wrap-foot", "alternatives", "graphic", "inline-graphic",
              "supplementary-material", "media", "object-id", "mml:math", "tex-math", "disp-formula",
              "inline-formula"}


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str                 # e.g. "PMC7536103"
    title: str = ""
    abstract: list[str] = field(default_factory=list)      # paragraphs
    sections: list[Section] = field(default_factory=list)
    pmid: str = ""
    doi: str = ""
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    article_type: str = ""
    source: str = ""            # file path

    # -------------------------------------------------------------- #
    def units(self) -> list[tuple[str, str]]:
        """Ordered text units as (field, text): title, abstract paragraphs,
        section headings and body paragraphs."""
        out = [("title", self.title)]
        out += [("abstract", p) for p in self.abstract]
        for sec in self.sections:
            if sec.heading:
                out.append(("heading", sec.heading))
            out += [("body", p) for p in sec.paragraphs]
        return [(f, t) for f, t in out if t]

    def body_text(self) -> str:
        return "\n".join(t for f, t in self.units() if f in ("heading", "body"))

    def abstract_text(self) -> str:
        return "\n".join(self.abstract)

    def full_text(self) -> str:
        return "\n".join(t for _, t in self.units())

    def meta(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title, "pmid": self.pmid, "doi": self.doi,
            "journal": self.journal, "year": self.year, "authors": self.authors,
            "keywords": self.keywords, "article_type": self.article_type, "source": self.source,
        }


# ------------------------------------------------------------------ #
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


_EMPTY_CITE = re.compile(r"\s*[\[(]\s*[–\-,;\s]*[\])]")   # residue left after dropping citations: "[–]", "[, ]", "()"


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")


def _clean(text: str) -> str:
    text = _EMPTY_CITE.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)      # "common [1]." -> "common." not "common ."
    return _WS.sub(" ", text).strip()


def _text_of(el: ET.Element | None) -> str:
    """Concatenated text of an element, skipping citation cross-references and
    other non-prose descendants."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        tag = _local(node.tag)
        if tag in _SKIP_TAGS or (tag == "xref" and node.get("ref-type") == "bibr"):
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node.tail:
            parts.append(node.tail)

    if el.text:
        parts.append(el.text)
    for child in el:
        walk(child)
    return _clean("".join(parts))


def _paragraphs(container: ET.Element | None) -> list[str]:
    """All <p> (and figure/table captions) under a container, in document order,
    without descending into nested <sec> (those are handled separately)."""
    if container is None:
        return []
    out: list[str] = []
    for child in container:
        tag = _local(child.tag)
        if tag == "p":
            t = _text_of(child)
            if t:
                out.append(t)
        elif tag in ("fig", "table-wrap", "boxed-text", "list", "disp-quote", "statement"):
            cap = child.find("caption")
            label = child.find("label")
            t = " ".join(x for x in (_text_of(label), _text_of(cap)) if x)
            if tag in ("list", "boxed-text", "disp-quote", "statement"):
                for p in child.iter():
                    if _local(p.tag) == "p":
                        pt = _text_of(p)
                        if pt:
                            out.append(pt)
            elif t:
                out.append(t)
    return out


def _sections(container: ET.Element | None, depth: int = 0) -> list[Section]:
    if container is None:
        return []
    out: list[Section] = []
    title_el = container.find("title")
    heading = _text_of(title_el) if depth > 0 else ""
    paras = _paragraphs(container)
    if heading or paras:
        out.append(Section(heading, paras))
    for child in container:
        if _local(child.tag) == "sec":
            out.extend(_sections(child, depth + 1))
    return out


def _find(root: ET.Element, path: str) -> ET.Element | None:
    return root.find(path)


def parse_article(article: ET.Element, source: str = "") -> Document:
    meta = _find(article, "front/article-meta")
    jmeta = _find(article, "front/journal-meta")
    ids = {}
    if meta is not None:
        for aid in meta.findall("article-id"):
            ids[aid.get("pub-id-type", "")] = (aid.text or "").strip()
    doc_id = ids.get("pmcid") or ("PMC" + ids["pmcaid"] if ids.get("pmcaid") else "") \
        or ids.get("pmid") or Path(source).stem or "unknown"
    if doc_id and doc_id.isdigit() and ids.get("pmid") == doc_id:
        doc_id = "PMID" + doc_id

    doc = Document(doc_id=doc_id, source=source)
    doc.pmid = ids.get("pmid", "")
    doc.doi = ids.get("doi", "")
    doc.article_type = article.get("article-type", "")
    if meta is not None:
        doc.title = _text_of(meta.find("title-group/article-title"))
        for ab in meta.findall("abstract"):
            if ab.get("abstract-type") in ("graphical", "teaser"):
                continue
            paras = _paragraphs(ab)
            for sec in ab.findall("sec"):
                h = _text_of(sec.find("title"))
                for p in _paragraphs(sec):
                    paras.append(f"{h}: {p}" if h else p)
            doc.abstract.extend(paras)
        for c in meta.iter("contrib"):
            if c.get("contrib-type") == "author":
                name = c.find("name")
                if name is not None:
                    sur = _text_of(name.find("surname"))
                    giv = _text_of(name.find("given-names"))
                    doc.authors.append(" ".join(x for x in (giv, sur) if x))
                else:
                    coll = c.find("collab")
                    if coll is not None:
                        doc.authors.append(_text_of(coll))
        for k in meta.iter("kwd"):
            kw = _text_of(k)
            if kw:
                doc.keywords.append(kw)
        years = {}
        for pd in meta.findall("pub-date"):
            y = pd.find("year")
            if y is not None and y.text:
                years[pd.get("pub-type") or pd.get("date-type") or "x"] = y.text.strip()
        doc.year = years.get("epub") or years.get("ppub") or years.get("pub") or \
            (next(iter(years.values())) if years else "")
    if jmeta is not None:
        doc.journal = _text_of(jmeta.find("journal-title-group/journal-title")) or \
            _text_of(jmeta.find("journal-title")) or _text_of(jmeta.find("journal-id"))

    body = article.find("body")
    doc.sections = _sections(body, 0)
    return doc


def parse_xml_string(xml_text: str, source: str = "") -> list[Document]:
    # strip DOCTYPE so ElementTree does not try to fetch the DTD
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, count=1)
    root = ET.fromstring(xml_text)
    articles = [root] if _local(root.tag) == "article" else [a for a in root.iter() if _local(a.tag) == "article"]
    docs = []
    for i, art in enumerate(articles):
        d = parse_article(art, source)
        if len(articles) > 1 and not d.doc_id:
            d.doc_id = f"{Path(source).stem}#{i}"
        docs.append(d)
    return docs


def parse_file(path: str | Path) -> list[Document]:
    path = Path(path)
    return parse_xml_string(path.read_text(encoding="utf-8", errors="replace"), str(path))


def load_directory(directory: str | Path, patterns: Iterable[str] = ("*.xml", "*.nxml")) -> list[Document]:
    directory = Path(directory)
    docs: list[Document] = []
    files = sorted({p for pat in patterns for p in directory.glob(pat)})
    for f in files:
        try:
            docs.extend(parse_file(f))
        except ET.ParseError as e:
            print(f"[warn] could not parse {f}: {e}")
    return docs
