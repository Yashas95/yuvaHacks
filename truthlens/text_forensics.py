"""Text & context analysis: claim extraction and language-level red flags."""
from __future__ import annotations

import datetime as dt
import re

from dateutil import parser as dparser

from .config import RELIABILITY as R
from .evidence import Indicator

MONTHS = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_PATTERNS = [
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\,?\s+\d{{4}}\b",
    rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\,?\s+\d{{4}}\b",
]
RECENCY = ["today", "tonight", "yesterday", "this morning", "this evening", "just now", "live", "right now",
           "breaking", "minutes ago", "hours ago", "last night", "happening now"]
SENSATIONAL = ["100% real", "100% genuine", "not fake", "leaked", "exposed", "shocking", "they don't want you", "share before",
               "deleted soon", "viral", "must watch", "unbelievable", "secret", "banned", "wake up"]
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being this that these those it its
he she they we you i his her their our your my me him them us not no yes do does did have has had will would can could should may
might just also very so than then there here what which who whom when where why how all any some more most other such only own same
too into over after before about up down out off again further once said says video photo image clip audio""".split())


def extract_claims(text: str) -> dict:
    t = text or ""
    low = t.lower()
    dates = []
    for p in DATE_PATTERNS:
        for m in re.finditer(p, t, flags=re.I):
            try:
                d = dparser.parse(m.group(0), dayfirst=True, fuzzy=True)
                dates.append({"text": m.group(0), "date": d.date().isoformat()})
            except Exception:
                pass
    years = sorted({int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", t)})
    numbers = sorted(set(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|crore|lakh|million|billion|people|dead|killed|injured|km|kg|rs|₹|\$)?", t, re.I)))
    numbers = [n.strip() for n in numbers if len(n.strip()) > 0 and not re.fullmatch(r"(19|20)\d\d", n.strip())]
    ents = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", t)
    entities = sorted({e for e in ents if e.lower() not in STOP and len(e) > 2})
    words = [w for w in re.findall(r"[a-zA-Zऀ-ॿ]{3,}", low) if w not in STOP]
    return {"dates": dates, "years": years, "numbers": numbers[:20], "entities": entities[:25],
            "recency_terms": [r for r in RECENCY if r in low], "keywords": sorted(set(words))[:80],
            "sensational": [s for s in SENSATIONAL if s in low],
            "exclamations": t.count("!"), "caps_ratio": round(sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t)), 3)}


def analyze_text(eid: str, text: str, nid, label="context") -> dict:
    claims = extract_claims(text)
    inds = []
    if claims["sensational"] or claims["exclamations"] >= 3 or claims["caps_ratio"] > 0.35:
        bits = []
        if claims["sensational"]:
            bits.append("phrases: " + ", ".join(f'"{s}"' for s in claims["sensational"]))
        if claims["exclamations"] >= 3:
            bits.append(f"{claims['exclamations']} exclamation marks")
        if claims["caps_ratio"] > 0.35:
            bits.append(f"{claims['caps_ratio'] * 100:.0f}% capital letters")
        inds.append(Indicator(nid(), eid, "text", "text_language", "Persuasion / virality language", "context",
                              "The accompanying text uses " + "; ".join(bits) + ".",
                              "Misleading media is often packaged with urgency and authenticity claims to discourage verification.",
                              0.4, R["text_language"], {"type": "file"},
                              ["Genuine breaking news can also be written emotionally. This says nothing about the media itself."]))
    return {"indicators": inds, "claims": claims, "text": text}


def parse_media_date(s: str | None):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(s[:26] if "%f" in fmt else s, fmt).date()
        except Exception:
            continue
    try:
        return dparser.parse(s).date()
    except Exception:
        return None
