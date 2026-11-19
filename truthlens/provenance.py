"""Content Credentials (C2PA) and IPTC "digital source type" reader, shared by image, video and audio analysis.

AI generators from Google (Gemini, Veo, Imagen, Lyria), OpenAI, Adobe Firefly, Microsoft and others now embed a C2PA
manifest that states how the file was made. The standard field is IPTC `digitalSourceType`; the value
`trainedAlgorithmicMedia` is the official code for "created by generative AI". This is direct provenance evidence,
not a statistical guess, and it survives as long as the file isn't re-encoded or stripped.

Found by testing (2026-11-20): a song generated in the Gemini app scored 40 % Inconclusive because the audio path
never looked at metadata. Its MP3 carries a Google-signed C2PA manifest in an ID3 GEOB frame that says
"Created by Google Generative AI", "Applied imperceptible SynthID watermark", digitalSourceType=trainedAlgorithmicMedia.

We decode the manifest's CBOR text fields exactly (by their length prefix) instead of keyword-searching, so a word
appearing by chance in compressed data can't trigger it. We do NOT verify the cryptographic signature: a manifest
can be copied into another file, which the report states as an alternative explanation.
"""
from __future__ import annotations

import re
from pathlib import Path

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
# IPTC codes that mean generative AI made the whole file, or part of it.
AI_SOURCE_TYPES = {"trainedAlgorithmicMedia": "created by generative AI",
                   "compositeWithTrainedAlgorithmicMedia": "partly created by generative AI (composite)"}
OTHER_SOURCE_TYPES = {"algorithmicMedia": "made by a non-AI algorithm", "digitalCapture": "captured by a camera/recorder",
                      "algorithmicallyEnhanced": "enhanced by an algorithm", "compositeSynthetic": "composite with synthetic parts",
                      "digitalCreation": "created digitally by a person", "minorHumanEdits": "minor human edits",
                      "compositeCapture": "composite of captures", "screenCapture": "screen capture"}


def _read(path: Path, head=16 << 20, tail=4 << 20) -> bytes:
    with open(path, "rb") as f:
        data = f.read(head)
        f.seek(0, 2)
        size = f.tell()
        if size > head:
            f.seek(max(head, size - tail))
            data += f.read()
    return data


def _cbor_texts(data: bytes, key: bytes) -> list[str]:
    """Values of CBOR text strings that directly follow the CBOR text key `key`."""
    out = []
    for m in re.finditer(re.escape(key), data):
        i = m.end()
        if i >= len(data):
            continue
        h = data[i]
        if 0x60 <= h <= 0x77:              # short text: length in the header byte
            n, s = h - 0x60, i + 1
        elif h == 0x78 and i + 1 < len(data):  # text with 1-byte length
            n, s = data[i + 1], i + 2
        elif h == 0x79 and i + 2 < len(data):  # text with 2-byte length
            n, s = int.from_bytes(data[i + 1:i + 3], "big"), i + 3
        else:
            continue
        v = data[s:s + n]
        if len(v) == n and n:
            try:
                out.append(v.decode("utf-8"))
            except UnicodeDecodeError:
                pass
    return out


def _cert_orgs(data: bytes) -> list[str]:
    """Organisation names (X.509 O=, OID 2.5.4.10) in the manifest's signing certificates."""
    out = []
    for m in re.finditer(rb"\x06\x03\x55\x04\x0a[\x0c\x13]", data):
        i = m.end()
        n = data[i] if i < len(data) else 0
        v = data[i + 1:i + 1 + n]
        if 1 < n < 80 and len(v) == n:
            out.append(v.decode("utf-8", "ignore"))
    return list(dict.fromkeys(out))


def scan(path: Path) -> dict:
    data = _read(Path(path))
    low = data.lower()
    c2pa = b"c2pa" in low and (b"jumb" in low or b"c2pa.claim" in low or b"c2pa.signature" in low)
    # Match only the published IPTC codes, longest first: in CBOR the next field's name follows without a separator.
    # (The XMP form <Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/... uses the same URI and is covered too.)
    known = sorted(list(AI_SOURCE_TYPES) + list(OTHER_SOURCE_TYPES), key=len, reverse=True)
    canon = {t.lower(): t for t in known}
    pat = rb"digitalsourcetype/(" + b"|".join(k.encode() for k in known) + rb")"
    types = sorted({canon[m.group(1).decode().lower()] for m in re.finditer(pat, data, flags=re.I)})
    ai_types = [t for t in types if t in AI_SOURCE_TYPES]
    actions = [d.strip() for d in _cbor_texts(data, b"description") if 6 <= len(d.strip()) <= 200] if c2pa else []
    generators = [g for g in _cbor_texts(data, b"name") if 3 <= len(g) <= 120 and re.search(r"[A-Za-z]{3}", g)] if c2pa else []
    signers = _cert_orgs(data) if c2pa else []
    return {"c2pa_present": c2pa, "source_types": types, "ai_source_types": ai_types,
            "actions": list(dict.fromkeys(actions))[:8], "generators": list(dict.fromkeys(generators))[:5],
            "signers": signers[:4], "synthid_declared": any("synthid" in a.lower() for a in actions)}


def summary(p: dict) -> str:
    """One readable line, e.g. for an indicator observation."""
    bits = []
    if p["ai_source_types"]:
        bits.append("digitalSourceType = " + ", ".join(p["ai_source_types"]) + " (" +
                    "; ".join(AI_SOURCE_TYPES[t] for t in p["ai_source_types"]) + ")")
    if p["actions"]:
        bits.append("actions: " + "; ".join(f"“{a}”" for a in p["actions"][:4]))
    if p["signers"]:
        bits.append("signed by: " + ", ".join(p["signers"][:2]))
    return ". ".join(bits)


def indicators(ev, p: dict, nid, modality: str, R: dict) -> list:
    """Provenance indicators from a scan(): AI source type → strong; other C2PA → neutral note."""
    from .evidence import Indicator
    out = []
    if p["ai_source_types"]:
        out.append(Indicator(
            nid(), ev.id, modality, "metadata_ai_signature", "Content Credentials say: made with generative AI", "synthetic",
            "C2PA manifest in the file: " + summary(p) + ".",
            "The file carries a Content Credentials (C2PA) manifest whose IPTC digital-source-type is the official code "
            "for AI-generated media. This is a statement written by the generating tool, not a statistical guess."
            + (" It also declares an invisible SynthID watermark, which Google's SynthID Detector can confirm." if p["synthid_declared"] else ""),
            0.97, R["metadata_ai_signature"], {"type": "file"},
            ["We read the manifest but do not verify its cryptographic signature: a manifest can be copied into another "
             "file. Verify it at contentcredentials.org/verify.",
             "“Composite” means only part of the file was AI-generated."]))
    elif p["c2pa_present"]:
        out.append(Indicator(
            nid(), ev.id, modality, "metadata_editor", "Content Credentials (C2PA) manifest present", "metadata",
            "A C2PA provenance manifest was found" + (": " + summary(p) if summary(p) else "") + ".",
            "Content Credentials record which tools created or edited the file. This one does not declare AI generation.",
            0.3, R["metadata_editor"], {"type": "file"},
            ["Cameras, phones and editors also sign genuine media with C2PA; presence alone is neutral."]))
    return out
