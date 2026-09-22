"""Fetch PMC full-text XML from NCBI by PMID or PMC ID (standard library only).

Two public services are used:

* PMC ID Converter  https://pmc.ncbi.nlm.nih.gov/tools/idconv/   PMID -> PMCID
* E-utilities efetch (db=pmc, retmode=xml)                        PMCID -> JATS XML
* E-utilities efetch (db=pubmed, retmode=xml)                     PMID -> PubMed record
  (title + abstract only; used when the article has no full text in PMC.
  The record is converted to a minimal JATS <article> so the normal parser
  and index treat it like any other document, just without a <body>.)

NCBI allows 3 requests per second without an API key; a small delay is kept
between requests so a batch of ids never trips the limit.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL = "pmc-full-text-retrieval-tool"
MIN_INTERVAL = 0.35  # seconds between requests (NCBI: max 3 req/s without key)
TIMEOUT = 30

# "PMC7536103", "pmc 7536103", "PMID:32895440", "PMID 32895440", "32895440",
# also inside URLs such as https://pubmed.ncbi.nlm.nih.gov/32895440/
_ID_RE = re.compile(r"(?<![A-Za-z0-9])(PMC|PMID)?\s*:?\s*(\d{1,12})(?![A-Za-z0-9])", re.I)

_lock = threading.Lock()
_last_request = 0.0


def parse_ids(text: str) -> list[dict]:
    """Extract identifiers from free text (one per line, comma separated, URLs…).

    Returns [{"input": "PMC7536103", "kind": "pmc", "number": "7536103"}, …] in
    order of appearance, duplicates removed. A bare number is taken as a PMID.
    """
    out, seen = [], set()
    for m in _ID_RE.finditer(text or ""):
        prefix, number = (m.group(1) or "").upper(), m.group(2).lstrip("0") or "0"
        kind = "pmc" if prefix == "PMC" else "pmid"
        key = (kind, number)
        if key in seen:
            continue
        seen.add(key)
        out.append({"input": m.group(0).strip(), "kind": kind, "number": number})
    return out


def _throttle() -> None:
    global _last_request
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()


def _get(url: str, params: dict) -> bytes:
    _throttle()
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def convert_pmids(pmids: list[str]) -> dict[str, dict]:
    """PMID -> {"pmcid": "PMC…"} or {"error": "…"} using the PMC ID Converter."""
    result: dict[str, dict] = {}
    pmids = [p for p in pmids if p]
    for i in range(0, len(pmids), 200):  # the service accepts up to 200 ids per call
        chunk = pmids[i:i + 200]
        try:
            raw = _get(IDCONV_URL, {"ids": ",".join(chunk), "format": "json", "tool": TOOL})
            data = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, ValueError, OSError) as e:
            for p in chunk:
                result[p] = {"error": f"ID converter request failed: {e}"}
            continue
        if data.get("status") != "ok":
            msg = "; ".join(e.get("message", "") for e in data.get("errors", [])) or "ID converter error"
            for p in chunk:
                result[p] = {"error": msg}
            continue
        for rec in data.get("records", []):
            key = str(rec.get("requested-id") or rec.get("pmid") or "")
            if rec.get("status") == "error" or not rec.get("pmcid"):
                result[key] = {"error": rec.get("errmsg") or "not found in PMC (no full text in PubMed Central)"}
            else:
                result[key] = {"pmcid": rec["pmcid"], "doi": rec.get("doi", "")}
        for p in chunk:
            result.setdefault(p, {"error": "no answer from ID converter"})
    return result


def fetch_pmc_xml(pmcid: str) -> str:
    """Download the JATS XML of one PMC article. Raises ValueError if unavailable."""
    number = pmcid.upper().replace("PMC", "").strip()
    if not number.isdigit():
        raise ValueError(f"not a PMC id: {pmcid}")
    try:
        raw = _get(EFETCH_URL, {"db": "pmc", "id": number, "retmode": "xml", "tool": TOOL})
    except (urllib.error.URLError, OSError) as e:
        raise ValueError(f"efetch request failed: {e}") from e
    text = raw.decode("utf-8", errors="replace")
    m = re.search(r"<error[^>]*>([^<]*)</error>", text)
    if m:
        raise ValueError(m.group(1).strip() or "PMC reports an error for this id")
    if "<article" not in text:
        raise ValueError("no article in the efetch response")
    return text


# ---------------------------------------------------------------------- #
# PubMed (abstract-only) fallback
# ---------------------------------------------------------------------- #
def fetch_pubmed_xml(pmid: str) -> str:
    """Download the PubMed record of one PMID (PubmedArticleSet XML)."""
    number = pmid.upper().replace("PMID", "").strip(": ")
    if not number.isdigit():
        raise ValueError(f"not a PMID: {pmid}")
    try:
        raw = _get(EFETCH_URL, {"db": "pubmed", "id": number, "retmode": "xml", "tool": TOOL})
    except (urllib.error.URLError, OSError) as e:
        raise ValueError(f"efetch (pubmed) request failed: {e}") from e
    text = raw.decode("utf-8", errors="replace")
    if "<PubmedArticle" not in text:
        raise ValueError("PMID not found in PubMed")
    return text


def _t(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def pubmed_to_jats(xml_text: str) -> str:
    """Convert a PubmedArticleSet record into a minimal JATS article set
    (front matter + abstract, no body) that ir.pmc_parser understands."""
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, count=1)
    root = ET.fromstring(xml_text)
    out = ['<?xml version="1.0"?><pmc-articleset>']
    for pa in root.iter("PubmedArticle"):
        cit = pa.find("MedlineCitation")
        art = cit.find("Article") if cit is not None else None
        if cit is None or art is None:
            continue
        pmid = _t(cit.find("PMID"))
        doi = ""
        for eid in art.findall("ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = _t(eid)
        for aid in pa.iter("ArticleId"):
            if aid.get("IdType") == "doi" and not doi:
                doi = _t(aid)
        ptypes = [_t(x) for x in art.iter("PublicationType")]
        atype = "review-article" if any("Review" in x for x in ptypes) else "research-article"
        journal = _t(art.find("Journal/Title")) or _t(cit.find("MedlineJournalInfo/MedlineTA"))
        title = _t(art.find("ArticleTitle")).rstrip(".")
        year = _t(art.find("Journal/JournalIssue/PubDate/Year")) or _t(art.find("ArticleDate/Year")) \
            or _t(art.find("Journal/JournalIssue/PubDate/MedlineDate"))[:4]
        x = out.append
        x(f'<article article-type="{atype}" xml:lang="en"><front>')
        x(f"<journal-meta><journal-title-group><journal-title>{escape(journal)}</journal-title></journal-title-group></journal-meta>")
        x("<article-meta>")
        if pmid:
            x(f'<article-id pub-id-type="pmid">{escape(pmid)}</article-id>')
        if doi:
            x(f'<article-id pub-id-type="doi">{escape(doi)}</article-id>')
        x(f"<title-group><article-title>{escape(title)}</article-title></title-group>")
        x('<contrib-group>')
        for au in art.iter("Author"):
            sur, giv, coll = _t(au.find("LastName")), _t(au.find("ForeName")), _t(au.find("CollectiveName"))
            if sur:
                x(f'<contrib contrib-type="author"><name><surname>{escape(sur)}</surname>'
                  f'<given-names>{escape(giv)}</given-names></name></contrib>')
            elif coll:
                x(f'<contrib contrib-type="author"><collab>{escape(coll)}</collab></contrib>')
        x('</contrib-group>')
        if year:
            x(f'<pub-date pub-type="ppub"><year>{escape(year)}</year></pub-date>')
        abstract = art.find("Abstract")
        if abstract is not None:
            x("<abstract>")
            for at in abstract.findall("AbstractText"):
                label, text = at.get("Label", ""), _t(at)
                if not text:
                    continue
                if label:
                    x(f"<sec><title>{escape(label.title())}</title><p>{escape(text)}</p></sec>")
                else:
                    x(f"<p>{escape(text)}</p>")
            x("</abstract>")
        kws = [_t(k) for k in cit.iter("Keyword")] + [_t(m) for m in cit.iter("DescriptorName")]
        kws = [k for k in kws if k]
        if kws:
            x("<kwd-group>" + "".join(f"<kwd>{escape(k)}</kwd>" for k in kws) + "</kwd-group>")
        x("</article-meta></front></article>")
    out.append("</pmc-articleset>")
    return "".join(out)
