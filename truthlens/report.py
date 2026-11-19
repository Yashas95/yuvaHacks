"""Self-contained HTML forensic report (images embedded, print-to-PDF ready).

Layout follows the problem statement's "What the forensic report should show":
flagged evidence, indicator + why, cross-modal context, confidence, source trace, summary.
Screen theme is dark; print switches to a light, paper-friendly theme.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .plain import build as build_plain

# Styles and scripts live in truthlens/assets/ and are inlined so every report.html stays self-contained.
_ASSETS = Path(__file__).resolve().parent / "assets"
CSS = (_ASSETS / "report.css").read_text(encoding="utf-8")       # base + simple view + light + crisp layers
JS = (_ASSETS / "report.js").read_text(encoding="utf-8")         # tabs, filters, Simple/Expert switch, lightbox
JS_CHAT = (_ASSETS / "chat.js").read_text(encoding="utf-8")      # offline case assistant

E = html.escape
SEV = {"green": "#2a9d63", "amber": "#d99a00", "orange": "#f76b15", "red": "#e5484d"}
REL_BADGE = {"corroborates": ("#e5484d", "Corroborates"), "inconsistent": ("#f76b15", "Inconsistent"),
             "consistent": ("#2a9d63", "Consistent"), "links": ("#2f6fed", "Links files")}
TYPE_LABEL = {"image": "Images", "video": "Videos", "audio": "Audio", "text": "Text"}

FAMILIES = [
    ("provenance", "Provenance & metadata", "Generator signatures, editing software, capture dates"),
    ("pixel", "Pixel forensics", "Error-level, sensor noise, copy-move, frequency artifacts"),
    ("temporal", "Temporal (video)", "Cuts, splices, duplicated or frozen frames"),
    ("audio_signal", "Audio signal", "Silence gaps, noise-floor jumps, band limits"),
    ("ml", "ML classifiers", "AI-image, face-swap and synthetic-voice models"),
    ("context", "Claim & context", "Caption vs content, dates, persuasion language"),
    ("corroboration", "Cross-modal corroboration", "Independent channels agreeing on one moment"),
]

ICON = {
    "image": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 17l-5-5-9 8"/></svg>',
    "video": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="13" height="14" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
    "audio": '<svg viewBox="0 0 24 24"><path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/></svg>',
    "text": '<svg viewBox="0 0 24 24"><path d="M5 5h14M5 10h14M5 15h9M5 20h6"/></svg>',
    "cross": '<svg viewBox="0 0 24 24"><path d="M7 7h10l-3-3M17 17H7l3 3"/></svg>',
    "shield": '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>',
}


def _ico(kind: str) -> str:
    return f'<span class="ico">{ICON.get(kind, ICON["text"])}</span>'


def _img(case_dir: Path, rel: str) -> str:
    p = case_dir / rel
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def fmt_loc(loc: dict) -> str:
    """Plain-text location summary (used in key findings)."""
    t = (loc or {}).get("type")
    if t == "regions":
        return "; ".join(f"{r.get('label', 'R')} @ x={r['bbox'][0]}, y={r['bbox'][1]} ({r['bbox'][2]}×{r['bbox'][3]}px)"
                         for r in loc["regions"][:3])
    if t == "frames":
        return ", ".join(f"{f['time']:.2f}s" for f in loc["frames"][:6]) + (" …" if len(loc["frames"]) > 6 else "")
    if t == "segments":
        return ", ".join(f"{s[0]:.1f}–{s[1]:.1f}s" for s in loc["segments"][:6])
    if t == "points":
        return ", ".join(f"{x:.2f}s" for x in loc["times"][:8])
    if t == "cross":
        return "across evidence"
    return "whole file"


def _loc_chips(loc: dict) -> str:
    t = (loc or {}).get("type")
    chips = []
    if t == "regions":
        chips = [f"{E(str(r.get('label', 'R')))} · x{r['bbox'][0]} y{r['bbox'][1]} · {r['bbox'][2]}×{r['bbox'][3]}px" for r in loc["regions"][:5]]
    elif t == "frames":
        chips = [f"{f['time']:.2f}s · frame {f['frame']}" + (f" · {f['p'] * 100:.0f}%" if f.get("p") is not None else "")
                 for f in loc["frames"][:8]]
        if len(loc["frames"]) > 8:
            chips.append(f"+{len(loc['frames']) - 8} more")
    elif t == "segments":
        chips = [f"{s[0]:.1f}–{s[1]:.1f}s" + (f" · {s[2] * 100:.0f}%" if len(s) > 2 else "") for s in loc["segments"][:8]]
    elif t == "points":
        chips = [f"t = {x:.2f}s" for x in loc["times"][:8]]
    elif t == "cross":
        chips = ["Across evidence"]
    else:
        chips = ["Whole file"]
    return "".join(f'<span class="loc">{c}</span>' for c in chips)


def _meter(label: str, v: float, color: str, hint: str) -> str:
    return (f'<div class="meter" title="{E(hint)}"><div class="mlab"><span>{E(label)}</span><b>{v * 100:.0f}%</b></div>'
            f'<div class="mbar"><i style="width:{max(2, min(100, v * 100)):.0f}%;background:{color}"></i></div></div>')


def _sev_color(w: float) -> str:
    return SEV["red"] if w >= 0.35 else SEV["orange"] if w >= 0.2 else SEV["amber"] if w >= 0.08 else "#a3a8af"


def _ind_cards(inds, highlight=()) -> str:
    out = []
    for i in sorted(inds, key=lambda i: (i["supports"] != "manipulation", -i["score"] * i["reliability"])):
        auth = i["supports"] == "authenticity"
        w = i["score"] * i["reliability"]
        col = SEV["green"] if auth else _sev_color(w)
        alts = i.get("alternatives", [])
        alts_html = ("<details class='alts'><summary>Alternative explanations "
                     f"<span class='cnt'>{len(alts)}</span></summary><ul>" + "".join(f"<li>{E(a)}</li>" for a in alts) +
                     "</ul></details>") if alts else ""
        out.append(f"""<article class="ind {'auth' if auth else ''} {'key' if i['id'] in highlight else ''}" id="{E(i['id'])}" style="--sev:{col}">
<div class="ihead"><span class="iid">{E(i['id'])}</span><h4>{E(i['title'])}</h4>
<span class="chip">{E(i.get('category', ''))}</span>{'<span class="chip ok">supports authenticity</span>' if auth else ''}
{'<span class="chip key">key finding</span>' if i['id'] in highlight else ''}</div>
<div class="locs"><span class="lk">Flagged at</span>{_loc_chips(i['location'])}</div>
<div class="igrid"><div><div class="lk">Observed</div><p>{E(i['observation'])}</p></div>
<div><div class="lk">Why it was flagged</div><p>{E(i['why'])}</p></div></div>
<div class="meters">{_meter('Signal strength', i['score'], col, 'How strong the anomaly is')}
{_meter('Detector reliability', i['reliability'], '#9aa0a6', 'How far this detector family can be trusted')}
{_meter('Weight in score', w, col, 'Signal × reliability')}</div>
{alts_html}</article>""")
    return "".join(out)


def _gauge(p: int, color: str, size: int = 150) -> str:
    r = 58
    c = 2 * 3.14159 * r
    off = c * (1 - p / 100)
    return (f'<svg class="gauge" viewBox="0 0 140 140" width="{size}" height="{size}">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="var(--line)" stroke-width="12"/>'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 70 70)"/>'
            f'<text x="70" y="70" text-anchor="middle" class="gnum" fill="{color}">{p}%</text>'
            f'<text x="70" y="92" text-anchor="middle" class="gsub">likelihood</text></svg>')


def _range_bar(a: dict, color: str) -> str:
    lo, hi = a["range"]
    p = a["likelihood"]
    ticks = "".join(f'<span style="left:{t}%"></span>' for t in (20, 45, 70))
    return (f'<div class="rbar"><div class="zones"><i class="z1"></i><i class="z2"></i><i class="z3"></i><i class="z4"></i></div>'
            f'<div class="rng" style="left:{lo}%;width:{max(1, hi - lo)}%;border-color:{color}"></div>'
            f'<div class="pin" style="left:{p}%;background:{color}"></div><div class="ticks">{ticks}</div></div>'
            f'<div class="rlabels"><span>No signal</span><span>Inconclusive</span><span>Moderate</span><span>Strong</span></div>')


def _mini_assess(a: dict) -> str:
    col = SEV.get(a["color"], "#a3a8af")
    return f'<span class="vpill" style="--c:{col}"><b>{a["likelihood"]}%</b> {E(a["band"].split(" —")[0])}</span>'


def _duration(ev: dict):
    m = ev.get("metrics") or {}
    for k in ("duration_s", "duration"):
        if isinstance(m.get(k), (int, float)) and m[k] > 0:
            return float(m[k])
    a = m.get("audio") if isinstance(m.get("audio"), dict) else {}
    for k in ("duration_s", "duration"):
        if isinstance(a.get(k), (int, float)) and a[k] > 0:
            return float(a[k])
    return None


def _timeline(ev: dict) -> str:
    """Horizontal track showing where on the time axis each indicator fired."""
    marks = []
    for i in ev["indicators"]:
        if i["supports"] != "manipulation":
            continue
        loc = i["location"] or {}
        t = loc.get("type")
        col = _sev_color(i["score"] * i["reliability"])
        if t == "frames":
            marks += [("pt", f["time"], f["time"], col, i) for f in loc["frames"]]
        elif t == "points":
            marks += [("pt", x, x, col, i) for x in loc["times"]]
        elif t == "segments":
            marks += [("seg", s[0], s[1], col, i) for s in loc["segments"]]
    if not marks:
        return ""
    dur = _duration(ev) or max(m[2] for m in marks) * 1.05 + 0.5
    rows, lanes = [], {}
    for kind, a, b, col, i in marks:
        lane = lanes.setdefault(i["id"], len(lanes))
        left = max(0, min(100, a / dur * 100))
        width = max(0.6, min(100 - left, (b - a) / dur * 100)) if kind == "seg" else 0
        tip = E(f"{i['id']} · {i['title']} · {a:.2f}s" + (f"–{b:.2f}s" if kind == 'seg' else ""))
        cls = "tseg" if kind == "seg" else "tpt"
        style = f"left:{left:.2f}%;top:{lane * 22 + 6}px;background:{col}" + (f";width:{width:.2f}%" if kind == "seg" else "")
        rows.append(f'<a href="#{E(i["id"])}" class="{cls}" style="{style}" title="{tip}"></a>')
    legend = "".join(f'<div class="tlab" style="top:{n * 22}px">{E(k)}</div>' for k, n in lanes.items())
    h = len(lanes) * 22 + 10
    ticks = "".join(f'<span style="left:{x * 25}%">{dur * x / 4:.1f}s</span>' for x in range(5))
    return (f'<div class="tline"><div class="lk">Flag timeline <span class="muted">— click a mark to jump to its indicator</span></div>'
            f'<div class="tl-wrap"><div class="tl-lab" style="height:{h}px">{legend}</div>'
            f'<div class="tl-track" style="height:{h}px">{"".join(rows)}</div></div><div class="tl-ticks">{ticks}</div></div>')


def _facts(ev: dict) -> list:
    facts = []
    md = ev.get("metadata") or {}
    m = ev.get("metrics") or {}
    if ev["modality"] == "image":
        ex = md.get("exif", {})
        facts += [("Format", md.get("format")), ("Dimensions", "×".join(map(str, md.get("size", [])))),
                  ("Camera", " ".join(x for x in [ex.get("Make"), ex.get("Model")] if x) or "none recorded"),
                  ("Software", ex.get("Software")), ("Captured", ex.get("DateTimeOriginal")), ("Modified", ex.get("DateTime")),
                  ("GPS present", "yes" if md.get("has_gps") else "no"), ("C2PA manifest", "yes" if md.get("c2pa_present") else "no"),
                  ("AI signatures", ", ".join(md.get("ai_signatures", [])) or "none"),
                  ("AI-image classifier", None if m.get("ml", {}).get("image_ai_p") is None else f"{m['ml']['image_ai_p'] * 100:.0f}% ({m['ml'].get('image_ai_model')})"),
                  ("Faces detected", m.get("ml", {}).get("faces"))]
    elif ev["modality"] == "video":
        facts += [("Resolution", m.get("resolution")), ("FPS", m.get("fps")), ("Duration", f"{m.get('duration_s')}s"),
                  ("Frames scanned", f"{m.get('frames_scanned')} / {m.get('frames_total')}"),
                  ("Creation time", md.get("creation_time")), ("Device", md.get("device")),
                  ("Editors / encoders", ", ".join(md.get("editors", [])) or "none"),
                  ("AI signatures", ", ".join(md.get("ai_signatures", [])) or "none"),
                  ("Discontinuities", len(m.get("discontinuities", []))),
                  ("Audio track", "yes" if m.get("audio") else "no")]
    if ev["modality"] == "audio" or (m.get("audio") if ev["modality"] == "video" else None):
        am = m if ev["modality"] == "audio" else m["audio"]
        facts += [("Audio codec", am.get("codec")), ("Sample rate", am.get("sample_rate")),
                  ("Effective bandwidth", f"{am.get('effective_bandwidth_hz')} Hz"),
                  ("Synthetic-speech mean", None if am.get("synthetic_mean_p") is None else f"{am['synthetic_mean_p'] * 100:.0f}% ({am.get('audio_model')})" + (" — model failed evaluation, ignored" if am.get('audio_model_unreliable') else "")),
                  ("Noise-floor jumps", am.get("noise_floor_jumps")), ("Digital-silence gaps", am.get("digital_silence_gaps"))]
    if ev.get("scene"):
        facts.append(("Visual content (CLIP)", "; ".join(ev["scene"])))
    if ev.get("caption_similarity") is not None:
        facts.append(("Claim ↔ visual similarity", ev["caption_similarity"]))
    if ev.get("capture_date"):
        facts.append(("Capture date (metadata)", ev["capture_date"]))
    return [(k, v) for k, v in facts if v not in (None, "", "None", "Nones", "None / None")]


def _evidence_card(ev: dict, case_dir: Path, top_ids) -> str:
    eid = ev["id"]
    seen, vis = set(), []
    for k, p in ev.get("visuals", {}).items():
        if p in seen:
            continue
        seen.add(p)
        src = _img(case_dir, p)
        if src:
            vis.append(f"<figure><img src='{src}' alt='{E(k)}' onclick='lb(this)'><figcaption>{E(k)}</figcaption></figure>")
    tr = ev.get("transcript") or {}
    flagged_seg = [(s[0], s[1]) for i in ev["indicators"]
                   if i["location"].get("type") == "segments" and i["supports"] == "manipulation"
                   for s in i["location"]["segments"]]
    tr_html = ""
    if tr.get("chunks"):
        lines = []
        for c in tr["chunks"]:
            s, e = c.get("start") or 0, c.get("end") or (c.get("start") or 0)
            hit = any(a - 0.3 <= e and s <= b + 0.3 for a, b in flagged_seg)
            lines.append(f"<div class='tr {'hit' if hit else ''}'><span class='ts'>{s:6.1f}s</span><span>{E(c['text'])}</span></div>")
        tr_html = ("<p class='muted small'>Whisper transcript with timestamps. Lines in red overlap a flagged audio segment.</p>"
                   "<div class='transcript'>" + "".join(lines) + "</div>")
    elif tr.get("text"):
        tr_html = f"<div class='transcript'>{E(tr['text'])}</div>"

    facts = _facts(ev)
    facts_html = "".join(f"<div class='fact'><span>{E(k)}</span><b>{E(str(v))}</b></div>" for k, v in facts)
    facts_html += (f"<div class='fact wide'><span>SHA-256</span><b class='mono hash'>{E(ev['sha256'])}"
                   f"<button class='copy' onclick=\"cp('{E(ev['sha256'])}',this)\">copy</button></b></div>")
    n_man = sum(1 for i in ev["indicators"] if i["supports"] == "manipulation")
    n_aut = len(ev["indicators"]) - n_man
    inds = _ind_cards(ev["indicators"], top_ids) or "<p class='empty'>No indicators raised for this file.</p>"
    tabs = [("vis", f"Visual evidence <span class='cnt'>{len(vis)}</span>", (_timeline(ev) + f"<div class='gallery'>{''.join(vis)}</div>") if vis or _timeline(ev) else "<p class='empty'>No visual artifacts for this file.</p>"),
            ("ind", f"Indicators <span class='cnt'>{n_man}</span>" + (f" <span class='cnt ok'>+{n_aut}</span>" if n_aut else ""), inds),
            ("facts", "File facts &amp; metadata", f"<div class='facts'>{facts_html}</div>")]
    if tr_html:
        tabs.append(("tr", "Transcript", tr_html))
    default = "ind" if not vis else "vis"
    tabbar = "".join(f"<button class='tab {'on' if k == default else ''}' data-t='{k}'>{lab}</button>" for k, lab, _ in tabs)
    panels = "".join(f"<div class='panel {'on' if k == default else ''}' data-p='{k}'><h3 class='ponly'>{lab}</h3>{body}</div>" for k, lab, body in tabs)
    a = ev["assessment"]
    col = SEV.get(a["color"], "#a3a8af")
    size = ev.get("size")
    size_s = f"{size / 1e6:.1f} MB" if isinstance(size, (int, float)) and size > 1e6 else (f"{size / 1e3:.0f} KB" if isinstance(size, (int, float)) else "")
    return f"""<section class="ev card" id="ev-{E(eid)}" data-type="{E(ev['modality'])}">
<div class="evhead" style="--c:{col}">{_ico(ev['modality'])}<div class="evtitle"><div class="eyebrow">{E(eid)} · {E(ev['modality'])}{' · ' + size_s if size_s else ''}</div>
<h3>{E(ev['name'])}</h3></div>
<div class="evscore"><div class="big" style="color:{col}">{a['likelihood']}%</div><div class="small">{E(a['band'])}<br>confidence <b>{E(a['confidence'])}</b> · range {a['range'][0]}–{a['range'][1]}%</div></div></div>
{'<div class="banner warn">Only ML classifiers flagged this file — no independent forensic or metadata evidence. Treat as a weak lead.</div>' if a.get('ml_only') else ''}
<div class="tabs" role="tablist">{tabbar}</div>{panels}</section>"""


KIND_META = {"red": ("Red flag", "#e5484d"), "check": ("Worth checking", "#d99a00"), "info": ("Normal / not suspicious", "#a3a8af")}
PCOL = {"red": "#e5484d", "orange": "#f76b15", "amber": "#d99a00", "green": "#2a9d63"}
ORIGIN_ICON = {"ai": "✦", "screenshot": "▭", "camera": "◉", "edited": "✎", "unknown": "?"}


def _thumb(ev: dict, case_dir: Path) -> str:
    vis = ev.get("visuals", {})
    for want in ("Flagged regions", "Flagged / key frames", "Original", "spectrogram"):
        for k, p in vis.items():
            if want.lower() in k.lower():
                src = _img(case_dir, p)
                if src:
                    return f"<img src='{src}' alt='{E(k)}' onclick='lb(this)'>"
    return _ico(ev["modality"])


def _simple_section(P: dict, r: dict, case_dir: Path) -> str:
    col = PCOL.get(P["color"], "#a3a8af")
    evs = {e["id"]: e for e in r["evidence"]}
    cards = ""
    for f in P["files"]:
        fc = PCOL.get(f["color"], "#a3a8af")
        reasons = "".join(
            f"<li style='--k:{KIND_META[x['kind']][1]}'><span class='kd' title='{KIND_META[x['kind']][0]}'></span>"
            f"<span>{x['text']}</span><a class='tech' href='#{E(x['id'])}' title='Technical details'>{E(x['id'])}</a></li>"
            for x in f["reasons"]) or "<li style='--k:#a3a8af'><span class='kd'></span><span>No warning signs were raised for this file.</span></li>"
        facts = "".join(f"<div><span>{E(k)}</span><b>{E(str(v))}</b></div>" for k, v in f["facts"].items())
        o = f["origin"]
        cards += f"""<article class="pf" id="pf-{E(f['id'])}" style="--c:{fc}">
<div class="pfmedia">{_thumb(evs[f['id']], case_dir)}</div>
<div class="pfbody">
<div class="eyebrow">{E(f['id'])} · {E(f['noun'])} · {E(f['name'])}</div>
<h3 style="color:{fc}">{E(f['headline'])}</h3><p class="pfsub">{E(f['sub'])}</p>
<div class="qa">
<div class="q"><span class="ql">What is this file?</span><b><span class="oi">{ORIGIN_ICON.get(o['kind'], '?')}</span>{E(o['label'])}</b><small>{E(o['why'])}</small></div>
<div class="q"><span class="ql">How sure are we?</span><b>{E(f['confidence'])} confidence</b><small>{E(f['confidence_words'])}</small></div>
</div>
<div class="qa3">
<div><span class="ql">Is it AI-made?</span><div>{f['answers']['ai']}</div></div>
<div><span class="ql">Is it a screenshot?</span><div>{f['answers']['screenshot']}</div></div>
<div><span class="ql">Was it edited?</span><div>{f['answers']['edited']}</div></div>
</div>
<h4>What we found</h4><ul class="reasons">{reasons}</ul>
<div class="legend"><span style="--k:#e5484d"><i></i>Red flag</span><span style="--k:#d99a00"><i></i>Worth checking</span><span style="--k:#a3a8af"><i></i>Normal / not suspicious</span></div>
{('<details class="pfacts"><summary>File details</summary><div class="fgrid">' + facts + '</div></details>') if facts else ''}
<div class="pfact"><button class="ask" data-q="Is the {E(f['noun'])} {E(f['name'])} AI-generated?">Ask about this {E(f['noun'])}</button>
<a href="#ev-{E(f['id'])}" class="tech2">Technical details →</a></div>
</div></article>"""
    cross = ""
    if P["cross"]:
        cross = "<div class='card'><h3>Across the files</h3><ul class='reasons'>" + "".join(
            f"<li style='--k:{KIND_META[x['kind']][1]}'><span class='kd'></span><span>{x['text']}</span>"
            f"<a class='tech' href='#{E(x['id'])}'>{E(x['id'])}</a></li>" for x in P["cross"]) + "</ul></div>"
    claim = f"<div class='claimq'><span class='ql'>Claim you gave us</span>“{E(P['claim'])}”</div>" if P["claim"] else ""
    steps = "".join(f"<label class='chk'><input type='checkbox'><span>{E(s)}</span></label>" for s in P["next_steps"])
    meter_pos = max(2, min(98, P["likelihood"]))
    return f"""<section id="simple" class="s">
<div class="card answer" style="--c:{col}">
<div class="ansL"><div class="eyebrow">Our answer</div><h2>{E(P['headline'])}</h2><p>{E(P['sub'])}</p>
<div class="sus"><div class="sustrack"><i style="left:{meter_pos}%"></i></div>
<div class="suslab"><span>Looks normal</span><span>Unclear</span><span>Suspicious</span></div></div>
<p class="sure"><b>{E(P['confidence'])} confidence.</b> {E(P['confidence_words'])}</p>
{'<p class="mlnote">⚠ Only AI detectors raised a flag here. They are easily fooled by filters, compression and captions, so this is a hint, not proof.</p>' if P['ml_only'] else ''}
{claim}</div>
<div class="ansR"><div class="ql">Ask a question</div>
<button class="ask" data-q="Is this AI-generated?">Is this AI-generated?</button>
<button class="ask" data-q="Is this a screenshot?">Is this a screenshot?</button>
<button class="ask" data-q="Was it edited or photoshopped?">Was it edited?</button>
<button class="ask" data-q="What should I do next?">What should I do?</button></div>
</div>
{cards}
{cross}
<div class="card" id="todo"><h3>What you can do next</h3><div class="checks">{steps}</div>
<p class="small muted">TruthLens gives leads, not final proof. Switch to <a href="#" class="tomode" data-m="expert">Expert view</a> for every measurement, location and detector score.</p></div>
</section>"""


def _chat_html() -> str:
    return """<button id="chatbtn" class="noprint" aria-label="Ask about this evidence">""" + ICON_CHAT + """<span>Ask about this evidence</span></button>
<div id="chat" class="noprint" role="dialog" aria-label="TruthLens assistant">
<div class="chead"><div><b>TruthLens assistant</b><span>Answers only from this case's evidence · works offline</span></div><button id="chatx" aria-label="Close">×</button></div>
<div id="cmsgs"></div>
<div id="csug"></div>
<form id="cform"><input id="cin" autocomplete="off" placeholder="e.g. Is this AI-generated? Is it a screenshot?"><button aria-label="Send">➤</button></form>
</div>"""


ICON_CHAT = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v11H9l-5 4z"/><path d="M8 9h8M8 12h5"/></svg>'


def render_html(r: dict, case_dir: Path) -> str:
    P = build_plain(r)
    ov = r["overall"]
    col = SEV.get(ov["color"], "#a3a8af")
    inds_by_id = {i["id"]: i for i in r["indicators"]}
    top = [inds_by_id[t] for t in ov["top"] if t in inds_by_id]
    ev_names = {e["id"]: e["name"] for e in r["evidence"]}
    by_type: dict[str, list] = {}
    for ev in r["evidence"]:
        by_type.setdefault(ev["modality"], []).append(ev)

    # ---------- key findings
    keys = "".join(
        f"""<a class="kf" href="#{E(i['id'])}" style="--sev:{_sev_color(i['score'] * i['reliability'])}">
<div class="kfh"><span class="iid">{E(i['id'])}</span>{_ico(i['modality'] if i['modality'] in ICON else 'cross')}<b>{E(i['title'])}</b></div>
<div class="kfm">{E(ev_names.get(i['evidence_id'], 'Cross-modal'))} · {E(fmt_loc(i['location']))}</div>
<div class="kfw"><i style="width:{min(100, i['score'] * i['reliability'] * 100):.0f}%"></i></div></a>""" for i in top) \
        or "<p class='empty'>No manipulation indicator exceeded reporting thresholds.</p>"

    # ---------- score composition by family
    fs = ov.get("family_scores") or {}
    fam_rows = ""
    for k, name, desc in FAMILIES:
        v = fs.get(k)
        on = v is not None and v > 0
        fam_rows += (f"<div class='fam {'off' if not on else ''}'><div class='fn'><b>{E(name)}</b><span>{E(desc)}</span></div>"
                     f"<div class='fbar'><i style='width:{(v or 0) * 100:.0f}%;background:{_sev_color(v or 0) if on else 'transparent'}'></i></div>"
                     f"<div class='fv'>{f'{v * 100:.0f}%' if on else 'no signal'}</div></div>")

    banners = ""
    if ov.get("ml_only") and ov.get("validated_ml"):
        banners += ("<div class='banner warn'><b>ML-only finding from a validated detector.</b> A frame/image classifier that passed our "
                    "real-vs-AI evaluation is confident and consistent, so the score may reach <i>Moderate</i>. Without independent "
                    "forensic, metadata or temporal evidence it is never reported as <i>Strong</i>.</div>")
    elif ov.get("ml_only"):
        banners += ("<div class='banner warn'><b>ML-only finding.</b> Only machine-learning classifiers flagged this evidence. "
                    "No independent forensic, metadata or temporal evidence supports it, so the likelihood is capped and confidence is Low. "
                    "Classifiers often react to compression, beauty filters or captions.</div>")
    if ov.get("conflicting_signals"):
        banners += "<div class='banner info'><b>Conflicting signals.</b> Some detectors support authenticity while others flag manipulation — weigh both.</div>"
    if ov.get("corroborated"):
        banners += "<div class='banner bad'><b>Cross-modal corroboration.</b> Independent channels (e.g. picture and sound) point to the same edit or synthesis.</div>"

    # ---------- evidence filter + cards
    filt = f"<button class='fchip on' data-f='all'>All <span class='cnt'>{len(r['evidence'])}</span></button>"
    for t in ("image", "video", "audio", "text"):
        n = len(by_type.get(t, []))
        filt += f"<button class='fchip' data-f='{t}' {'disabled' if not n else ''}>{_ico(t)}{TYPE_LABEL[t]} <span class='cnt'>{n}</span></button>"
    top_ids = set(ov["top"])
    cards = "".join(_evidence_card(ev, case_dir, top_ids) for ev in r["evidence"])

    # ---------- sidebar navigation
    nav_ev = ""
    for t in ("image", "video", "audio", "text"):
        items = by_type.get(t, [])
        if not items:
            continue
        nav_ev += f"<div class='navgrp'>{_ico(t)}{TYPE_LABEL[t]}</div>"
        for ev in items:
            a = ev["assessment"]
            nav_ev += (f"<a href='#ev-{E(ev['id'])}' class='navev'><span class='dot' style='background:{SEV.get(a['color'], '#a3a8af')}'></span>"
                       f"<span class='nm'>{E(ev['name'])}</span><span class='pc'>{a['likelihood']}%</span></a>")

    # ---------- cross-modal
    xcards = ""
    for f in r["cross_modal"]:
        c, lab = REL_BADGE.get(f["relation"], ("#a3a8af", f["relation"].title()))
        links = "".join(f"<a class='loc' href='#{E(l)}'>{E(l)}</a>" for l in f["linked_indicators"])
        mods = " ⇄ ".join(E(m) for m in f["modalities"])
        xcards += (f"<article class='xm' id='{E(f['id'])}' style='--c:{c}'><div class='xmh'><span class='rel'>{E(lab)}</span>"
                   f"<span class='iid'>{E(f['id'])}</span><b>{E(f['title'])}</b></div>"
                   f"<div class='muted small'>{mods} · {E(', '.join(ev_names.get(x, x) for x in f['evidence_ids']))}</div>"
                   f"<p>{E(f['detail'])}</p>{('<div class=locs><span class=lk>Linked indicators</span>' + links + '</div>') if links else ''}</article>")
    xhtml = xcards or "<p class='empty'>No cross-modal relationships met reporting thresholds. Add a caption/claim or several related files to enable these checks.</p>"

    # ---------- claim
    ctx = r.get("context_analysis")
    ctx_html = ""
    if ctx:
        cl = ctx["claims"]

        def chips(vals, empty="—"):
            return "".join(f"<span class='loc'>{E(str(v))}</span>" for v in vals) or f"<span class='muted'>{empty}</span>"
        ctx_html = f"""<section class="card" id="claim"><div class="sh"><span class="num">05</span><h2>Claim &amp; context analysis</h2></div>
<blockquote>“{E(r['case']['context_text'] or '')}”</blockquote>
<div class="facts">
<div class="fact"><span>Dates mentioned</span><b>{chips([d['text'] + ' → ' + d['date'] for d in cl['dates']])}</b></div>
<div class="fact"><span>Years</span><b>{chips(cl['years'])}</b></div>
<div class="fact"><span>Recency terms</span><b>{chips(cl['recency_terms'])}</b></div>
<div class="fact"><span>Named entities</span><b>{chips(cl['entities'])}</b></div>
<div class="fact"><span>Numbers</span><b>{chips(cl['numbers'])}</b></div>
<div class="fact"><span>Persuasion markers</span><b>{chips(cl['sensational'], 'none')}</b></div></div>
{('<h3>Text indicators</h3>' + _ind_cards(ctx['indicators'])) if ctx['indicators'] else ''}</section>"""

    checks = "".join(f"<label class='chk'><input type='checkbox'><span>{E(c)}</span></label>" for c in r["recommended_checks"])
    lims = "".join(f"<li>{E(l)}</li>" for l in r["limitations"])
    inv = "".join(
        f"<tr><td class='mono'>{E(x['evidence_id'])}</td><td>{E(x['name'])}</td>"
        f"<td>{E(next((e['modality'] for e in r['evidence'] if e['id'] == x['evidence_id']), 'text'))}</td>"
        f"<td class='mono hash'>{E(x['sha256'])}</td>"
        f"<td>{'<span class=ok>✓ unchanged</span>' if x['unchanged'] else '<span class=bad>✗ MODIFIED</span>'}</td></tr>"
        for x in r["integrity"])
    all_ok = all(x["unchanged"] for x in r["integrity"]) if r["integrity"] else True
    logs = "".join(f"<li><span class='mono small muted'>{E(l['time'][11:19] if len(l['time']) > 19 else l['time'])}</span>"
                   f"<b>{E(l['action'].replace('_', ' '))}</b><span class='small'>{E(l['detail'])}</span></li>" for l in r["log"])
    mdl = "".join(f"<li class='mono small'>{E(m)}</li>" for m in r["tool"]["models"]) or "<li>No ML models were loaded.</li>"
    json_b64 = base64.b64encode(json.dumps(r, default=str).encode()).decode()
    n_xm = len(r["cross_modal"])

    rubric = [
        ("#evidence", "Flagged evidence", "Regions, frames, timestamps, audio segments"),
        ("#evidence", "Indicators", "What was observed and why it was flagged"),
        ("#crossmodal", "Cross-modal context", f"{n_xm} finding(s) across media and the claim"),
        ("#overview", "Confidence", f"{ov['confidence']} · range {ov['range'][0]}–{ov['range'][1]}%"),
        ("#custody", "Source trace", "SHA-256 hashes, read-only copies, re-verified"),
        ("#checks", "Summary & checks", "Caveats and recommended human checks"),
    ]
    rub = "".join(f"<a class='rub' href='{h}'><span class='tick'>✓</span><b>{E(t)}</b><span>{E(d)}</span></a>" for h, t, d in rubric)

    body = f"""
<aside class="side noprint">
<a class="logo" href="/">{_ico('shield')}<span>Truth<b>Lens</b></span></a>
<div class="sidecase"><div class="eyebrow">Case</div><div class="mono">{E(r['case']['id'])}</div>
<div class="sideverdict" style="--c:{col}"><b>{ov['likelihood']}%</b><span>{E(ov['band'].split(' —')[0])}</span></div></div>
<div class="mode" role="tablist" aria-label="View"><button data-m="simple">Simple</button><button data-m="expert">Expert</button></div>
<nav class="s"><a href="#simple"><span class="n">→</span>Our answer</a>
<div class="navsub">{''.join(f"<a href='#pf-{E(f['id'])}' class='navev'><span class='dot' style='background:{PCOL.get(f['color'], '#a3a8af')}'></span><span class='nm'>{E(f['name'])}</span></a>" for f in P['files'])}</div>
<a href="#todo"><span class="n">→</span>What to do next</a>
<a href="#" class="ask" data-q="help"><span class="n">?</span>Ask a question</a></nav>
<nav class="x">
<a href="#overview"><span class="n">01</span>Overview</a>
<a href="#findings"><span class="n">02</span>Key findings</a>
<a href="#evidence"><span class="n">03</span>Evidence <span class="cnt">{len(r['evidence'])}</span></a>
<div class="navsub">{nav_ev}</div>
<a href="#crossmodal"><span class="n">04</span>Cross-modal <span class="cnt">{n_xm}</span></a>
{'<a href="#claim"><span class="n">05</span>Claim analysis</a>' if ctx else ''}
<a href="#checks"><span class="n">06</span>Checks &amp; limits</a>
<a href="#custody"><span class="n">07</span>Chain of custody</a>
<a href="#method"><span class="n">08</span>Method &amp; history</a>
</nav>
<div class="sideactions"><button class="btn" onclick="window.print()">Download PDF</button>
<a class="btn ghost" download="{E(r['case']['id'])}_report.json" href="data:application/json;base64,{json_b64}">Download JSON</a>
<a class="btn ghost" href="/">+ New analysis</a></div>
</aside>

<main>
<header class="top">
<div><div class="eyebrow gold">TruthLens · Multimodal digital forensics</div><h1>Forensic report <span class="lite">that explains itself</span></h1>
<div class="meta">Case <b class="mono">{E(r['case']['id'])}</b> · {E(r['case']['created_at'][:19].replace('T', ' '))} · analysed in {r['case']['duration_s']}s · v{E(r['tool']['version'])}</div></div>
<div class="printonly mono">{E(r['case']['id'])}</div>
</header>

<div class="disclaimer">{_ico('shield')}<div>A detection score is an <b>indicator, not a verdict</b>. Every finding shows what was flagged, where, why,
how reliable the detector is, and what else could explain it. A human investigator must review before drawing conclusions.</div></div>

{_simple_section(P, r, case_dir)}

<div class="x">
<section class="card hero" id="overview" style="--c:{col}">
<div class="heroL">{_gauge(ov['likelihood'], col)}
<div><div class="eyebrow">Overall assessment</div><div class="band" style="color:{col}">{E(ov['band'])}</div>
<div class="stats"><div><span>Confidence</span><b>{E(ov['confidence'])}</b></div>
<div><span>Plausible range</span><b>{ov['range'][0]}–{ov['range'][1]}%</b></div>
<div><span>Indicators</span><b>{ov['n_manipulation_indicators']} <small>flag</small> · {ov['n_authenticity_signals']} <small>auth</small></b></div>
<div><span>Evidence</span><b>{len(r['evidence'])} file(s)</b></div></div>
{_range_bar(ov, col)}</div></div>
{banners}
<div class="rubric">{rub}</div>
</section>

<section class="card" id="findings"><div class="sh"><span class="num">02</span><h2>Key findings</h2><span class="muted small">ranked by weight (signal × reliability) · click to open the evidence</span></div>
<div class="kfgrid">{keys}</div>
<h3 class="mt">How the score was built</h3>
<p class="muted small">Indicators are grouped into detector families. Inside a family the strongest indicator dominates (correlated detectors don't stack);
families are combined as independent lines of evidence. ML-only findings are capped below "Moderate".</p>
<div class="fams">{fam_rows}</div>
</section>

<section id="evidence"><div class="sh pad"><span class="num">03</span><h2>Evidence</h2><div class="filters noprint">{filt}</div></div>
{cards}</section>

<section class="card" id="crossmodal"><div class="sh"><span class="num">04</span><h2>Cross-modal correlation</h2>
<span class="muted small">How signals from different media and the claim strengthen or weaken each other</span></div>
<div class="xmgrid">{xhtml}</div></section>

{ctx_html}

<section class="card two" id="checks"><div><div class="sh"><span class="num">06</span><h2>Recommended human checks</h2></div>
<div class="checks">{checks}</div></div>
<div><div class="sh"><h2>Limitations &amp; uncertainty</h2></div><ul class="lims">{lims}</ul></div></section>

<section class="card" id="custody"><div class="sh"><span class="num">07</span><h2>Chain of custody</h2>
<span class="{'ok' if all_ok else 'bad'}">{'✓ all evidence unchanged after analysis' if all_ok else '✗ integrity check failed'}</span></div>
<div class="tablewrap"><table class="inv"><thead><tr><th>ID</th><th>File</th><th>Type</th><th>SHA-256 (at intake)</th><th>Integrity</th></tr></thead><tbody>{inv}</tbody></table></div>
<p class="small muted">Originals were copied into the case folder as read-only files, hashed at intake and re-hashed after analysis. All visualizations are derived copies.</p></section>

<section class="card two" id="method"><div><div class="sh"><span class="num">08</span><h2>Analysis history</h2></div><ol class="log">{logs}</ol></div>
<div><div class="sh"><h2>Models used</h2></div><ul class="models">{mdl}</ul>
<p class="small muted">Signal = strength of the anomaly. Reliability = how far that detector family can be trusted (expert-set, not benchmark-calibrated).</p></div></section>

</div>
<footer>Generated by TruthLens · evidence-linked, reviewable output · <span style="color:{col}">■</span> {E(ov['band'])}</footer>
</main>
<div id="lb" class="noprint" onclick="this.classList.remove('on')"><img alt=""><div class="cap"></div></div>
{_chat_html()}
<script type="application/json" id="plain-data">{json.dumps(P).replace('</', '<\\/')}</script>"""

    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>TruthLens Report — {E(r["case"]["id"])}</title>'
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body class="simple">{body}<script>{JS}{JS_CHAT}</script></body></html>')
