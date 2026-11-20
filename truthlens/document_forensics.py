"""Documents (.docx, .pptx, .xlsx, .pdf): which software produced the file, and what that says.

TruthLens does not try to judge whether the *wording* was written by AI: text detectors misfire on ordinary
writing (non-native English, formal reports), so a score would be a guess. What a document file does record is the
software that wrote it:

- A document typed in Word, Google Docs or LibreOffice names that app, and Office files usually carry an edit
  history (number of saves, minutes of editing).
- A document made by a program carries the fingerprint of a document library instead, and no edit history. AI chat
  assistants such as ChatGPT hand you a .docx, .pptx, .xlsx or .pdf this way (they write it with code), but so do
  websites, apps and scripts. So this is a lead, reported with that alternative, not proof.
- Some tools write their own name (e.g. "ChatGPT") into the properties: direct evidence.
- Pictures inside the file are checked for AI labels (C2PA), like uploaded pictures.
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from .ai_text import check_ai_writing
from .config import RELIABILITY as R
from .evidence import Findings
from .provenance import scan as scan_provenance

OOXML = {".docx": "Word document", ".pptx": "PowerPoint presentation", ".xlsx": "Excel workbook"}
MAX_XML = 40 * 1024 * 1024   # don't inflate absurdly large parts (zip bombs)

# Programming libraries that write documents. Matched on the file's own metadata, never on its text.
LIBRARIES = [
    (r"python-docx", "python-docx"), (r"python-pptx", "python-pptx"),
    (r"\bopenpyxl\b", "openpyxl"), (r"XlsxWriter", "XlsxWriter"),
    (r"PptxGenJS", "PptxGenJS"), (r"SheetJS", "SheetJS"),
    (r"ReportLab", "ReportLab"), (r"\b(?:Py)?FPDF2?\b|\bfpdf2?\b", "FPDF"),
    (r"pdf-lib", "pdf-lib"), (r"\bjsPDF\b", "jsPDF"), (r"\bpdfmake\b", "pdfmake"),
    (r"WeasyPrint", "WeasyPrint"), (r"^PDFKit\b", "PDFKit"),
]
# The docx JavaScript library writes this placeholder as author when the program sets none.
DOCXJS_AUTHOR = "Un-named"
EDITORS = [
    (r"Microsoft.{0,12}Word", "Microsoft Word"), (r"Microsoft.{0,12}PowerPoint", "Microsoft PowerPoint"),
    (r"Microsoft.{0,12}Excel", "Microsoft Excel"), (r"LibreOffice|OpenOffice", "LibreOffice"),
    (r"Google Docs", "Google Docs"), (r"Google Slides", "Google Slides"), (r"Google Sheets", "Google Sheets"),
    (r"WPS Office|Kingsoft", "WPS Office"), (r"\bCanva\b", "Canva"), (r"Quartz PDFContext", "macOS export / print to PDF"),
    (r"Skia/PDF", "a browser's Print to PDF"), (r"Acrobat|Adobe PDF", "Adobe Acrobat"),
]
# AI tools that name themselves in metadata. Searched only in software / description fields: "Claude" or "Gemini"
# as an author name is a person.
AI_TOOLS = r"\b(ChatGPT|OpenAI|Google Gemini|Gemini AI|Copilot|Anthropic|Claude AI|Perplexity|Grok|DeepSeek|Mistral AI)\b"


def _read(z: zipfile.ZipFile, name: str) -> str:
    try:
        info = z.getinfo(name)
    except KeyError:
        return ""
    if info.file_size > MAX_XML:
        return ""
    return z.read(name).decode("utf-8", "replace")


def _tag(xml: str, tag: str) -> str:
    m = re.search(rf"<(?:\w+:)?{tag}\b[^>]*>(.*?)</(?:\w+:)?{tag}>", xml, re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def _ooxml(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        core, app, custom = _read(z, "docProps/core.xml"), _read(z, "docProps/app.xml"), _read(z, "docProps/custom.xml")
        if path.suffix.lower() == ".docx":
            parts = ["word/document.xml"]
            text_re = r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|(</w:p>)"
        elif path.suffix.lower() == ".pptx":
            parts = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                           key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
            text_re = r"<a:t>([^<]*)</a:t>|(</a:p>)"
        else:
            parts = ["xl/sharedStrings.xml"]
            text_re = r"<t(?:\s[^>]*)?>([^<]*)</t>|(</si>)"
        chunks = []
        for p in parts:
            for t, br in re.findall(text_re, _read(z, p)):
                chunks.append("\n" if br else html.unescape(t))
        media = [n for n in names if re.match(r"(word|ppt|xl)/media/", n) and not n.endswith("/")]
        pics = []
        for n in media[:40]:
            info = z.getinfo(n)
            if info.file_size <= 30 * 1024 * 1024:
                pics.append((n, z.read(n)))
    props = {k: _tag(core, k) for k in ("creator", "lastModifiedBy", "revision", "created", "modified",
                                         "title", "subject", "description", "keywords")}
    props.update({k: _tag(app, k) for k in ("Application", "AppVersion", "TotalTime", "Company", "Template")})
    props["custom"] = " ".join(html.unescape(v) for v in re.findall(r"<vt:lpwstr>([^<]*)</vt:lpwstr>", custom))
    text = re.sub(r"[ \t]+\n", "\n", "".join(chunks)).strip()
    return {"props": props, "text": text, "pictures": pics}


def _pdf_str(raw: bytes) -> str:
    if raw.startswith(b"<"):
        try:
            raw = bytes.fromhex(re.sub(rb"[^0-9A-Fa-f]", b"", raw[1:-1]).decode())
        except ValueError:
            return ""
    else:
        raw = re.sub(rb"\\([()\\])", rb"\1", raw[1:-1])
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace").strip()
    return raw.decode("latin-1", "replace").strip()


def _pdf(path: Path) -> dict:
    with open(path, "rb") as f:
        head = f.read(8 * 1024 * 1024)
        size = f.seek(0, 2)
        f.seek(max(len(head), size - 2 * 1024 * 1024))
        data = head + f.read()
    props = {}
    for key, raw in re.findall(rb"/(Producer|Creator|Author|Title|CreationDate|ModDate)\s*(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>)", data):
        props.setdefault(key.decode(), _pdf_str(raw))
    for key, pat in (("Producer", rb"<pdf:Producer>(.*?)</pdf:Producer>"), ("CreatorTool", rb"<xmp:CreatorTool>(.*?)</xmp:CreatorTool>")):
        m = re.search(pat, data, re.S)
        if m and key not in props:
            props[key] = html.unescape(m.group(1).decode("utf-8", "replace")).strip()
    pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", data))
    return {"props": props, "text": "", "pictures": [], "pages": pages}


def read_document(path: Path) -> dict:
    return _pdf(path) if path.suffix.lower() == ".pdf" else _ooxml(path)


def document_text(path: Path) -> str:
    try:
        return read_document(path)["text"]
    except Exception:
        return ""


def analyze_document(ev, case, nid) -> dict:
    out = Findings(ev, nid, "document")
    ext = ev.path.suffix.lower()
    doc = read_document(ev.path)
    p = doc["props"]
    pdf = ext == ".pdf"
    software = {k: v for k, v in p.items() if v and k in (("Producer", "Creator", "CreatorTool") if pdf else ("Application", "AppVersion", "Template"))}
    descriptive = {k: v for k, v in p.items() if v and k in ("title", "subject", "description", "keywords", "custom", "Title")}
    people = {k: v for k, v in p.items() if v and k in ("creator", "lastModifiedBy", "Author", "Company")}
    everything = {**software, **descriptive, **people}

    library = next(((name, k, v) for pat, name in LIBRARIES for k, v in everything.items() if re.search(pat, v, re.I)), None)
    if not library and not pdf and p.get("creator") == DOCXJS_AUTHOR:
        library = ("docx.js", "creator", DOCXJS_AUTHOR)
    editor = next((name for pat, name in EDITORS for v in software.values() if re.search(pat, v, re.I)), None)
    ai = next(((m.group(1), k, v) for k, v in {**software, **descriptive}.items()
               for m in [re.search(AI_TOOLS, v, re.I)] if m), None)
    words = len(doc["text"].split())
    m = {"kind": OOXML.get(ext, "PDF document"), "words": words, "made_with": None, "library": None, "ai_tool": None,
         "revision": p.get("revision") or None, "editing_minutes": p.get("TotalTime") or None,
         "created": p.get("created") or p.get("CreationDate") or None, "modified": p.get("modified") or p.get("ModDate") or None,
         "pictures": len(doc["pictures"]), "pictures_with_ai_label": 0, "pages": doc.get("pages")}

    if ai:
        tool, k, v = ai
        m["ai_tool"] = tool
        out.add("metadata_ai_signature", f"Metadata names an AI tool: {tool}", "synthetic",
                f"The file's “{k}” property reads “{v[:120]}”.",
                "Document files record the software that made them. This one names an AI assistant, which is direct "
                "evidence from the file rather than a guess about its wording.",
                0.95, R["metadata_ai_signature"], {"type": "file"},
                ["Someone typed the tool's name into the document properties by hand."])
    if library:
        name, k, v = library
        m["library"] = name
        how = (f"The author and last-editor fields read “{DOCXJS_AUTHOR}”, the placeholder the docx library writes when a "
               "program sets no author" if v == DOCXJS_AUTHOR else f"The file's “{k}” property reads “{v[:120]}”")
        out.add("document_generator", f"Made by a program, not typed in an editor: {name}", "synthetic",
                how + ("; no editing app is recorded." if not editor else f"; it also mentions {editor}."),
                "Documents typed in Word, Google Docs or LibreOffice record that app. This one carries the fingerprint of a "
                "programming library that writes documents automatically. That is how AI chat assistants such as ChatGPT "
                "create the files they let you download.",
                0.85, R["document_generator"], {"type": "file"},
                ["Websites, apps and scripts (invoices, exported reports, forms) use the same libraries.",
                 "A person may have written the text and only used a program to lay it out."])
    m["made_with"] = (library and library[0]) or editor

    if not pdf:
        rev = int(p["revision"]) if p.get("revision", "").isdigit() else None
        mins = int(p["TotalTime"]) if p.get("TotalTime", "").isdigit() else None
        same = bool(p.get("created")) and p.get("created") == p.get("modified")
        if words >= 150 and rev is not None and rev <= 1 and (mins in (None, 0)) and (same or mins == 0):
            facts = [f"saved {rev} time" + ("s" if rev != 1 else ""), "no editing time recorded" if mins is None else "0 minutes of editing"]
            if same:
                facts.append("created and last changed at the same moment")
            out.add("document_history", "No editing history", "synthetic",
                    f"{words} words, but the file records: " + ", ".join(facts) + ".",
                    "Writing a document of this length in an editor normally leaves several saves and some editing time. "
                    "A single save with none is what you get when a program writes the whole file at once.",
                    0.45, R["document_history"], {"type": "file"},
                    ["“Save as” copies, exports from Google Docs and email attachments can reset this history."])

    labelled = []
    tmp = case.artifacts_dir / f"{ev.id}_pictures"
    for name, data in doc["pictures"]:
        tmp.mkdir(exist_ok=True)
        f = tmp / Path(name).name
        f.write_bytes(data)
        prov = scan_provenance(f)
        if prov.get("ai_source_types"):
            labelled.append(Path(name).name)
    m["pictures_with_ai_label"] = len(labelled)
    if labelled:
        out.add("document_ai_image", f"{len(labelled)} picture(s) inside carry an AI label", "synthetic",
                "Content Credentials (C2PA) in " + ", ".join(labelled[:5]) + " declare them AI-generated.",
                "The pictures' own labels say a generative AI tool made them. That says nothing on its own about who "
                "wrote the text.", 0.9, R["document_ai_image"], {"type": "file"},
                ["The author may have deliberately added an AI illustration to a human-written document."])
    m["writing"] = check_ai_writing(out, doc["text"]) if doc["text"] else {"skipped": "the text of PDF files is not read yet"}
    meta = {"properties": {k: v for k, v in p.items() if v}}
    return {"indicators": out, "metrics": m, "metadata": meta, "visuals": {}, "text_excerpt": doc["text"][:600]}
