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
                  ("Synthetic-speech mean", None if am.get("synthetic_mean_p") is None else f"{am['synthetic_mean_p'] * 100:.0f}% ({am.get('audio_model')})"),
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
            f'<style>{CSS}{CSS_SIMPLE}{CSS_LIGHT}{CSS_CRISP}</style></head><body class="simple">{body}<script>{JS}{JS_CHAT}</script></body></html>')


JS = r"""
document.querySelectorAll('.ev').forEach(card=>{
  card.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
    card.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
    card.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.dataset.p===t.dataset.t));
  });
});
document.querySelectorAll('.fchip').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.fchip').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('.ev').forEach(c=>c.style.display=(b.dataset.f==='all'||c.dataset.type===b.dataset.f)?'':'none');
});
function setMode(m){
  document.body.classList.toggle('simple',m==='simple');document.body.classList.toggle('expert',m==='expert');
  document.querySelectorAll('.mode button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  try{localStorage.setItem('tl-mode',m)}catch(e){}
}
(function(){let m='simple';try{m=new URLSearchParams(location.search).get('mode')||localStorage.getItem('tl-mode')||'simple'}catch(e){}setMode(m==='expert'?'expert':'simple')})();
document.querySelectorAll('.mode button,.tomode').forEach(b=>b.addEventListener('click',e=>{e.preventDefault();setMode(b.dataset.m);window.scrollTo(0,0)}));
function openFor(id){
  const el=document.getElementById(id);if(!el)return;
  if(el.closest('.x')&&document.body.classList.contains('simple'))setMode('expert');
  if(el.closest('.s')&&document.body.classList.contains('expert'))setMode('simple');
  const card=el.closest('.ev');
  if(card){card.style.display='';const p=el.closest('.panel');
    if(p&&!p.classList.contains('on')){card.querySelector('.tab[data-t="'+p.dataset.p+'"]').click()}}
  el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1600);
}
document.addEventListener('click',e=>{
  const a=e.target.closest('a[href^="#"]');if(!a)return;
  const id=a.getAttribute('href').slice(1);const el=document.getElementById(id);if(!el)return;
  e.preventDefault();openFor(id);el.scrollIntoView({behavior:'smooth',block:'start'});history.replaceState(null,'','#'+id);
});
function lb(img){const l=document.getElementById('lb');l.querySelector('img').src=img.src;l.querySelector('.cap').textContent=img.alt;l.classList.add('on')}
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lb').classList.remove('on')});
function cp(t,b){navigator.clipboard&&navigator.clipboard.writeText(t);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}
const io=new IntersectionObserver(es=>es.forEach(en=>{if(en.isIntersecting){
  document.querySelectorAll('.side nav a').forEach(a=>a.classList.toggle('cur',a.getAttribute('href')==='#'+en.target.id))}}),{rootMargin:'-30% 0px -60% 0px'});
document.querySelectorAll('main section[id]').forEach(s=>io.observe(s));
window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
"""

JS_CHAT = r"""
(function(){
const P=JSON.parse(document.getElementById('plain-data').textContent);
const box=document.getElementById('chat'),msgs=document.getElementById('cmsgs'),sug=document.getElementById('csug'),inp=document.getElementById('cin');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const KCOL={red:'#e5484d',check:'#d99a00',info:'#a3a8af'};
function open(){box.classList.add('on');document.getElementById('chatbtn').classList.add('hide');if(!msgs.children.length)greet();setTimeout(()=>inp.focus(),50)}
function close(){box.classList.remove('on');document.getElementById('chatbtn').classList.remove('hide')}
document.getElementById('chatbtn').onclick=open;document.getElementById('chatx').onclick=close;
function add(html,who){const d=document.createElement('div');d.className='cm '+who;d.innerHTML=html;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight}
function chips(list){sug.innerHTML=list.map(q=>`<button>${esc(q)}</button>`).join('');sug.querySelectorAll('button').forEach(b=>b.onclick=()=>ask(b.textContent))}
const DEFAULT=['Is this AI-generated?','Is it a screenshot?','Was it edited?','Is it real?','What should I do?'];
function greet(){
  const n=P.files.length;
  add(`Hi! I can answer questions about the ${n} file${n>1?'s':''} in this case, using only what TruthLens measured. `+
      `<br><br><b>Short answer:</b> ${esc(P.headline)}. ${esc(P.sub)}`,'bot');chips(DEFAULT);
}
const GLOSS={
 'error level|\\bela\\b':'<b>Error Level Analysis (ELA)</b> re-saves the picture and compares it with the original. Areas pasted in from another image often react differently, so they light up.',
 'noise':'<b>Sensor noise</b> is the fine grain every camera leaves. If one area has different grain, it may come from another photo.',
 'copy.?move|clone':'<b>Copy-move</b> means a piece of the picture was copied and pasted elsewhere in the same picture — e.g. to hide something or duplicate a crowd.',
 'deep ?fake':'A <b>deepfake</b> is video, audio or a picture where a face or voice was replaced or generated by AI to look like a real person.',
 'c2pa|content credential':'<b>Content Credentials (C2PA)</b> are a digital label some cameras, editors and AI tools attach to say how a file was made. They can be removed, so missing labels prove nothing.',
 'metadata|exif':'<b>Metadata (EXIF)</b> is hidden information stored inside a file: camera model, date, software, sometimes GPS location. It can be edited or stripped by apps like WhatsApp.',
 'hash|sha':'A <b>SHA-256 hash</b> is a fingerprint of the file. Change even one pixel and the fingerprint changes completely — that\'s how we prove the evidence wasn\'t altered during analysis.',
 'spectrogram':'A <b>spectrogram</b> is a picture of sound: time left-to-right, pitch bottom-to-top, loudness as brightness. Cuts and missing frequencies are easy to see on it.',
 'likelihood|percent|score':'The <b>score</b> combines how strong each warning sign is with how much we trust the check that found it. It is <i>not</i> "the chance it\'s fake" — it\'s an indicator for a human to review.',
 'confidence':'<b>Confidence</b> says how much the evidence agrees: High means several independent checks agree; Low means it rests on one weak or easily-fooled signal.',
 'reliab':'<b>Reliability</b> is how much we trust a type of check. Metadata tags from AI tools are very reliable; AI detectors are fooled more easily, so they count for less.',
 'fft|frequency':'<b>Frequency analysis (FFT)</b> looks for tiny repeating patterns. AI generators and heavy resizing often leave a grid-like fingerprint.',
 '\\bclip\\b':'<b>CLIP</b> is an AI model that compares pictures and text. We use it to check whether a caption actually describes what\'s in the image or video.',
 'whisper|transcri':'<b>Whisper</b> is a speech-to-text model. We use its transcript to compare what is said with the claim.'
};
const IN={
 help:/^(hi|hello|hey|help|what can you do|how does (this|it) work|who are you|\?)\b/,
 screenshot:/screen ?shot|screen ?grab|screen ?record|screen capture|\bsnip/,
 ai:/\bai\b|a\.i\.|artificial|generated|generat|midjourney|dall|stable diffusion|chat ?gpt|\bgpt|gemini|synthetic|computer.?(made|generated)|machine.?made|\bsora\b|\bveo\b|runway|\bbot\b/,
 face:/deep ?fake|face ?swap|\bface|swapped|lip.?sync/,
 voice:/voice|\baudio|speech|clon|\bsound|\btts\b|speaker/,
 edited:/edit|photo ?shop|manipulat|tamper|alter|modif|splic|paste|\bcut\b|cuts|crop|retouch|doctor|morph|fake parts|changed/,
 meta:/when was|what date|\bdate\b|taken|camera|phone|iphone|android|device|location|\bgps|where was|metadata|exif|created|who made|come from|source|origin/,
 where:/where|which part|which area|region|flagged|show me|timestamp|what time|which frame|when in|which second/,
 custody:/hash|sha|chain of custody|integrity|fingerprint|evidence (safe|changed|intact)/,
 claim:/caption|claim|headline|\bnews|story|\bpost\b|context|is it true|true or false|really happen/,
 conf:/confiden|how sure|certain|accura|reliab|how much (can i )?trust|wrong|mistake|false positive/,
 why:/\bwhy\b|reason|how did you|how do you know|explain|\bscore|percent|%|likelihood/,
 next:/what (should|can|do) i|next step|verify|check (it|this) myself|how (can|do) i (check|verify|confirm)|share it|advice|recommend/,
 limits:/limitation|can this tool|how accurate is (the )?tool|weakness|fooled|can you be wrong/,
 real:/\breal\b|genuine|authentic|legit|original|\btrue\b|\bfake\b|trust (it|this)|believe|hoax|safe to share|verdict|result|summary|overall|conclusion/
};
const ORDER=['help','screenshot','voice','face','ai','edited','meta','where','custody','claim','conf','why','next','limits','real'];
const TYPE={image:/photo|image|picture|\bpic\b|\bimg|png|jpe?g|screenshot/,video:/video|clip|reel|mp4|\bmov\b|footage/,audio:/voice note|recording|\baudio|wav|mp3|m4a|podcast|call/};
function targets(q){
  if(P.files.length<=1)return P.files;
  const byName=P.files.filter(f=>{const base=f.name.toLowerCase().replace(/\.[a-z0-9]+$/,'');
    return q.includes(base)||base.split(/[^a-z0-9]+/).some(w=>w.length>=4&&q.includes(w))||q.includes(f.id.toLowerCase())});
  if(byName.length)return byName;
  const byType=P.files.filter(f=>TYPE[f.modality]&&TYPE[f.modality].test(q));
  return byType.length?byType:P.files;
}
function per(fs,fn){return fs.map(f=>(fs.length>1?`<div class="cf">${esc(f.name)}</div>`:'')+fn(f)).join('')}
function rlist(rs){return rs.length?'<ul>'+rs.map(x=>`<li><span class="kd" style="background:${KCOL[x.kind]}"></span>${x.text} <a href="#${esc(x.id)}">${esc(x.id)}</a></li>`).join('')+'</ul>':''}
const A={
 help:fs=>`I answer questions about <b>this case only</b>, from the measurements TruthLens made — I don't guess or search the web.<br><br>Try asking: is it AI-made, is it a screenshot, was it edited, where exactly is the problem, how sure are you, or what should I do next. You can also ask what a term means, like "what is ELA?".`,
 ai:fs=>per(fs,f=>`<p>${f.answers.ai}</p>`),
 screenshot:fs=>per(fs,f=>`<p>${f.answers.screenshot}</p>`),
 edited:fs=>per(fs,f=>`<p>${f.answers.edited}</p>`),
 face:fs=>per(fs,f=>{if(f.modality==='audio')return '<p>This file is audio only — there\'s no face to check.</p>';
   const r=f.reasons.filter(x=>x.det==='ml_face_deepfake'||x.det==='xmodal_ml_agreement');
   return r.length?`<p><b>Our face detector raised a flag:</b></p>${rlist(r)}<p class="small">Face detectors are easily fooled by beauty filters and compression — look at the jawline, teeth and eyes yourself.</p>`
   :'<p><b>No face-swap signs</b> were raised (or no clear face was found).</p>'}),
 voice:fs=>per(fs,f=>{if(f.modality==='image'||f.modality==='text')return '<p>This file has no sound.</p>';
   const r=f.reasons.filter(x=>/^audio_|xmodal_ml/.test(x.det)||(x.det==='xmodal_corroboration'&&/sound|voice/.test(x.text)));
   return (r.length?`<p><b>Signs found in the sound:</b></p>${rlist(r)}`:'<p><b>No warning signs</b> were found in the sound.</p>')+
   '<p class="small">Honest note: the synthetic-voice detector is the weakest check in TruthLens. To be sure, compare with a recording you know is genuine.</p>'}),
 real:fs=>`<p><b>${esc(P.headline)}.</b> ${esc(P.sub)}</p>`+(P.cross.filter(x=>x.kind==='red').length?'<p>Strongest evidence:</p>'+rlist(P.cross.filter(x=>x.kind==='red')):'')+per(fs,f=>{const r=f.reasons.filter(x=>x.kind!=='info').slice(0,3);
   return (fs.length>1?`<p>${esc(f.headline)}.</p>`:'')+(r.length?'<p>Main reasons:</p>'+rlist(r):'')})+`<p class="small">${esc(P.confidence_words)} TruthLens gives leads, not proof.</p>`,
 where:fs=>per(fs,f=>{const r=f.reasons.filter(x=>x.kind!=='info');return r.length?rlist(r):'<p>Nothing was flagged in this file.</p>'})+(P.cross.filter(x=>x.kind!=='info').length?'<div class="cf">Across files</div>'+rlist(P.cross.filter(x=>x.kind!=='info')):'')+'<p class="small">Click an ID to jump to the exact spot in the technical view.</p>',
 why:fs=>`<p>The ${P.likelihood}% is <b>not</b> "the chance it's fake". It adds up how strong each warning sign is and how much we trust the check that found it. Signs from the same kind of check don't double-count.</p>`+per(fs,f=>{const r=f.reasons.filter(x=>x.kind!=='info').slice(0,3);return r.length?'<p>Biggest contributors:</p>'+rlist(r):'<p>No strong signs.</p>'}),
 conf:fs=>`<p><b>${esc(P.confidence)} confidence.</b> ${esc(P.confidence_words)}</p>`+(P.ml_only?'<p>⚠ Only AI detectors flagged this, and nothing else backs them up — they are often fooled by filters, compression and captions.</p>':'')+'<p class="small">All checks are rules and models tuned by hand, not certified tests. A person should always confirm before acting on the result.</p>',
 next:fs=>'<ol>'+P.next_steps.map(s=>`<li>${esc(s)}</li>`).join('')+'</ol>',
 claim:fs=>{if(!P.claim)return '<p>No caption or claim was given with this evidence. Next time, paste it on the upload page and TruthLens will check it against the media — dates, what\'s in the picture and what\'s said.</p>';
   const r=P.cross.filter(x=>/caption|claim|date|said/i.test(x.text));
   return `<p>You told us: <i>“${esc(P.claim)}”</i></p>`+(r.length?rlist(r):'<p>Nothing in the media clearly contradicts or confirms the claim.</p>')+'<p class="small">Check the claim itself on trusted news sites — real events are usually reported by several outlets.</p>'},
 meta:fs=>per(fs,f=>`<p><b>${esc(f.origin.label)}.</b> ${esc(f.origin.why)}</p>`+(Object.keys(f.facts).length?'<table>'+Object.entries(f.facts).map(([k,v])=>`<tr><td>${esc(k)}</td><td><b>${esc(v)}</b></td></tr>`).join('')+'</table>':'')),
 custody:fs=>`<p>Every file was fingerprinted (SHA-256) when it was uploaded and checked again after analysis: <b>${P.intact?'unchanged ✓':'CHANGED ✗'}</b>.</p>`+per(fs,f=>`<p class="small mono">${esc(f.sha256.slice(0,24))}…</p>`)+'<p class="small">If even one pixel changed, the fingerprint would be completely different.</p>',
 limits:fs=>'<ul><li>Detectors were trained on public data and can miss new AI tools.</li><li>Filters, compression and captions can trigger false alarms.</li><li>No warning signs does <b>not</b> prove a file is real.</li><li>Files forwarded through WhatsApp or social media lose much of their evidence.</li></ul>'
};
const FOLLOW={ai:['Why?','Where exactly?','How sure are you?'],screenshot:['Is it AI-generated?','When was it taken?'],edited:['Where exactly?','What should I do?'],
 real:['Is it AI-generated?','Was it edited?','What should I do?'],where:['Why?','What is ELA?'],why:['How sure are you?','What should I do?'],
 conf:['What should I do?','What are the limitations?'],help:DEFAULT,meta:['Is it a screenshot?','Is it AI-generated?'],face:['Is the voice cloned?','What should I do?'],
 voice:['Is the face a deepfake?','What should I do?'],next:['Is it real?'],claim:['Is it real?','What should I do?'],custody:['Is it real?'],limits:['Is it real?']};
function answer(raw){
  const q=raw.toLowerCase().trim();
  if(/what('s| is| are| does)|meaning|\bmean\b|define/.test(q)){
    for(const k in GLOSS){if(new RegExp(k).test(q))return {html:GLOSS[k],follow:['Is it real?','Where exactly?']}}
  }
  let hits=ORDER.filter(k=>IN[k].test(q));
  if((hits.includes('voice')||hits.includes('face'))&&hits.includes('ai'))hits=hits.filter(k=>k!=='ai');
  if(hits.length>1)hits=hits.filter(k=>k!=='real');
  hits=hits.slice(0,2);
  if(!hits.length)return {html:'<p>I can only answer questions about what TruthLens measured in this case — I can\'t look things up or guess. Try one of these:</p>',follow:DEFAULT};
  const fs=targets(q);
  const LAB={screenshot:'Is it a screenshot?',ai:'Is it AI-made?',edited:'Was it edited?',face:'Face',voice:'Voice',meta:'Where it came from',
    where:'Where the problems are',custody:'Evidence integrity',claim:'The claim',conf:'How sure we are',why:'Why this score',next:'What to do',limits:'Limitations',real:'Verdict'};
  const multi=hits.length>1;
  return {html:hits.map(k=>(multi?`<div class="ih">${LAB[k]||''}</div>`:'')+A[k](fs)).join('<hr>'),follow:FOLLOW[hits[0]]||DEFAULT};
}
function ask(q){if(!q||!q.trim())return;if(!box.classList.contains('on'))open();
  if(q==='help'){add(A.help(P.files),'bot');chips(DEFAULT);return}
  add(esc(q),'me');const a=answer(q);setTimeout(()=>{add(a.html,'bot');chips(a.follow)},180)}
document.getElementById('cform').onsubmit=e=>{e.preventDefault();const q=inp.value;inp.value='';ask(q)};
document.querySelectorAll('.ask').forEach(b=>b.addEventListener('click',e=>{e.preventDefault();ask(b.dataset.q)}));
try{const q=new URLSearchParams(location.search).get('ask');if(q)setTimeout(()=>{open();ask(q)},400)}catch(e){}
window.tlAsk=ask;
})();
"""

CSS_LIGHT = r"""
/* ---- light, soft (neumorphic) layer: monochrome ink, airy type, pill controls ---- */
:root{--r:24px;--neu-d:rgba(163,177,198,.38);--neu-l:rgba(255,255,255,.95);--lite:#c3c8ce}
body{background:#eef0f2;background-image:none;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
.lite{color:var(--lite)}
h1,h2,h3,h4{font-weight:500;letter-spacing:-.03em}
.top h1{font-size:46px;line-height:1.05;letter-spacing:-.045em;font-weight:500}
.card,.pf,.ev{background:#fbfbfc;border:1px solid rgba(255,255,255,.9);border-radius:var(--r);
  box-shadow:-10px -10px 24px var(--neu-l),12px 14px 30px var(--neu-d)}
.side{background:#f3f4f6;border-right:0;box-shadow:8px 0 30px rgba(163,177,198,.22)}
.sidecase,.stats div,.rub,.kf,.ind,.xm,.fact,.q,.qa3>div,.reasons li,.chk,.tline,.fgrid div{background:#f4f5f7;border:1px solid rgba(255,255,255,.9);
  box-shadow:inset 2px 2px 5px rgba(255,255,255,.9),3px 4px 10px rgba(163,177,198,.22);border-radius:16px}
.ind,.kf{border-left-width:3px;border-left-style:solid}.ind{border-left-color:var(--sev)}.kf{border-left-color:var(--sev)}
.btn{border-radius:999px;background:#0d0d0f;color:#fff;box-shadow:0 8px 18px rgba(13,13,15,.18)}
.btn.ghost{background:#f4f5f7;color:var(--text);border:0;box-shadow:-3px -3px 8px #fff,4px 4px 10px var(--neu-d)}
.mode{border-radius:999px;background:#e8eaed;border:0;padding:4px;box-shadow:inset 3px 3px 7px var(--neu-d),inset -3px -3px 7px #fff}
.mode button{border-radius:999px;display:flex;align-items:center;justify-content:center;gap:8px}
.mode button.on{background:#0d0d0f;color:#fff;box-shadow:0 6px 14px rgba(13,13,15,.22)}
.mode button.on:before{content:"";width:12px;height:12px;border-radius:50%;background:#fff}
.chip,.fchip,.loc,.tech,.rel,#csug button,.copy,.cnt{border-radius:999px}
.fchip{background:#f4f5f7;border:1px solid #e3e6ea}.fchip.on{background:#0d0d0f;color:#fff;border-color:#0d0d0f}.fchip.on .cnt{background:#fff;color:#0d0d0f}
button.ask{border-radius:999px;background:#f4f5f7;border:1px solid #e6e8eb;box-shadow:-3px -3px 8px #fff,4px 4px 10px rgba(163,177,198,.28)}
button.ask:hover{border-color:#0d0d0f}button.ask:before{content:"→ ";color:var(--muted)}
.loc{background:#fff;border:1px solid #e6e8eb}
.disclaimer{background:#f4f5f7;border:1px solid rgba(255,255,255,.9);border-radius:18px;box-shadow:inset 2px 2px 6px var(--neu-d),inset -2px -2px 6px #fff}
.disclaimer .ico{color:var(--text)}
.sh .num{border:0;background:#0d0d0f;color:#fff;border-radius:999px;padding:2px 9px}
.side nav .n{color:var(--muted)}
.side nav>a:hover,.side nav>a.cur{background:#fff;box-shadow:3px 3px 10px rgba(163,177,198,.25)}
/* answer hero: big two-tone type + embossed concentric rings like the reference */
.answer{position:relative;overflow:hidden;background:#fbfbfc}
.answer:before{content:"";position:absolute;right:-120px;top:50%;width:520px;height:520px;transform:translateY(-50%);border-radius:50%;pointer-events:none;
  background:radial-gradient(circle,transparent 0 92px,#fff 93px 96px,rgba(163,177,198,.22) 97px 104px,transparent 105px 150px,#fff 151px 154px,
  rgba(163,177,198,.2) 155px 162px,transparent 163px 208px,#fff 209px 212px,rgba(163,177,198,.17) 213px 220px,transparent 221px 258px)}
.answer>*{position:relative;z-index:1}
.answer h2{font-size:42px;font-weight:500;letter-spacing:-.045em;line-height:1.05}
.sustrack{height:16px;border-radius:999px;background:#e8eaed;box-shadow:inset 3px 3px 7px var(--neu-d),inset -3px -3px 7px #fff}
.sustrack:after{content:"";position:absolute;inset:4px;border-radius:999px;background:linear-gradient(90deg,rgba(42,157,99,.35),rgba(217,154,0,.35) 50%,rgba(229,72,77,.45));opacity:.8}
.sustrack i{top:-6px;width:28px;height:28px;margin-left:-14px;background:#fff;border:0;z-index:2;box-shadow:0 4px 12px rgba(16,24,40,.22),inset -2px -2px 4px rgba(163,177,198,.35)}
.mlnote{background:#fff7e6;border:1px solid #f5dfae;border-radius:14px}
.claimq{background:#f4f5f7;border-left:3px solid #0d0d0f;border-radius:12px}
.pf h3{font-size:28px;letter-spacing:-.04em}
.oi{background:#0d0d0f;color:#fff;border-radius:999px}
.pfmedia,figure{background:#f1f2f4;border:0;border-radius:18px;box-shadow:inset 2px 2px 8px var(--neu-d),inset -2px -2px 8px #fff}
.kd{box-shadow:0 0 0 3px color-mix(in srgb,var(--k) 18%,transparent)}
.hero{border-top:0}.gauge circle:first-child{stroke:#e8eaed}
.evhead{background:linear-gradient(90deg,color-mix(in srgb,var(--c) 7%,transparent),transparent 60%)}
.evhead>.ico{background:#fff;border:0;box-shadow:3px 3px 10px var(--neu-d),-3px -3px 8px #fff;border-radius:14px}
.tab.on{border-bottom-color:#0d0d0f}
.transcript{background:#f4f5f7;border:0;box-shadow:inset 2px 2px 6px var(--neu-d),inset -2px -2px 6px #fff}
blockquote{background:#f4f5f7;border-left:3px solid #0d0d0f;border-radius:12px}
/* chat */
#chatbtn{background:#0d0d0f;color:#fff;box-shadow:0 14px 30px rgba(13,13,15,.28)}
#chat{background:#fbfbfc;border:1px solid rgba(255,255,255,.9);border-radius:28px;box-shadow:0 30px 70px rgba(16,24,40,.22)}
.chead{background:transparent}.cm.bot{background:#f2f3f5;border:0}.cm.me{background:#0d0d0f}
#cin{background:#f2f3f5;border:0;border-radius:999px;box-shadow:inset 2px 2px 6px var(--neu-d),inset -2px -2px 6px #fff}
#cform button{background:#0d0d0f;color:#fff;border-radius:999px}
#csug button{background:#fff;border:1px solid #e3e6ea}
#lb{background:rgba(238,240,242,.94)}#lb .cap{color:#0d0d0f}
@media (max-width:900px){.top h1{font-size:32px}.answer h2{font-size:30px}.answer:before{display:none}}
@media print{.card,.pf,.ev,.sidecase,.ind,.kf,.xm,.fact{box-shadow:none!important}.answer:before{display:none}}
"""

CSS_CRISP = r"""
/* ===== crisp instrument layer: hairlines, small radii, mono labels, no soft embossing ===== */
:root{--r:10px;--line:#dcdfe3;--sig:#e5484d}
body{background:#f6f7f8;background-image:linear-gradient(#eceef1 1px,transparent 1px),linear-gradient(90deg,#eceef1 1px,transparent 1px);background-size:32px 32px}
h1,h2,h3,h4{font-weight:600}
.top h1{font-weight:600;letter-spacing:-.045em}
.card,.pf,.ev{background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:none}
.sidecase,.stats div,.rub,.kf,.ind,.xm,.fact,.q,.qa3>div,.reasons li,.chk,.tline,.fgrid div,.steps li{background:#fff;border:1px solid var(--line);box-shadow:none;border-radius:8px}
.ind,.kf{border-left:3px solid var(--sev)}
.side{background:#fbfbfc;border-right:1px solid var(--line);box-shadow:none}
.side nav>a:hover,.side nav>a.cur{background:#f1f2f4;box-shadow:none}
.logo{font-family:var(--mono);font-size:15px;font-weight:600}.logo .ico{color:#0d0d0f}.logo b{color:#0d0d0f}
.btn{border-radius:8px;box-shadow:none}.btn.ghost{background:#fff;border:1px solid var(--line);box-shadow:none}
.mode{border-radius:8px;background:#f1f2f4;border:1px solid var(--line);box-shadow:none}
.mode button{border-radius:6px;font-family:var(--mono);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.mode button.on:before{width:7px;height:7px;border-radius:1px;background:var(--sig)}
.chip,.fchip,.loc,.tech,.rel,#csug button,.copy,.cnt,button.ask,.tags span{border-radius:5px}
button.ask{background:#fff;border:1px solid var(--line);box-shadow:none;font-family:var(--mono);font-size:12px}
button.ask:before{content:"> ";color:var(--sig)}
.eyebrow,.ql,.lk,.mlab,.stats span,thead th,.rlabels,.suslab,.legend{font-family:var(--mono);letter-spacing:.08em}
.eyebrow{text-transform:uppercase}
.disclaimer{background:#fff;border:1px solid var(--line);border-left:3px solid #0d0d0f;border-radius:8px;box-shadow:none}
.sh .num{border-radius:4px;font-family:var(--mono)}
.answer{background:#fff}
.answer:before{background:none;border:1px solid rgba(13,13,15,.10);width:420px;height:420px;right:-90px;
  box-shadow:0 0 0 60px transparent,inset 0 0 0 60px #fff,inset 0 0 0 61px rgba(13,13,15,.08),inset 0 0 0 120px #fff,inset 0 0 0 121px rgba(13,13,15,.07)}
.answer h2{font-weight:600}
.sustrack{height:8px;border-radius:2px;background:#eceef1;box-shadow:none}
.sustrack:after{inset:0;border-radius:2px;opacity:1;background:linear-gradient(90deg,#2a9d63,#d99a00 50%,#e5484d)}
.sustrack i{top:-7px;width:4px;height:22px;margin-left:-2px;border-radius:1px;background:#0d0d0f;box-shadow:0 0 0 3px #fff}
.mlnote{border-radius:8px}.claimq,blockquote{border-radius:6px}
.pfmedia,figure{background:#f4f5f7;border:1px solid var(--line);border-radius:8px;box-shadow:none}
.oi{border-radius:4px}
.evhead>.ico{border:1px solid var(--line);box-shadow:none;border-radius:8px}
.transcript{background:#fff;border:1px solid var(--line);box-shadow:none;border-radius:8px}
.gauge .gnum{font-family:var(--mono);font-weight:600}
.evscore .big,.sideverdict b,.stats b,.fv,.iid{font-family:var(--mono)}
#chatbtn{border-radius:8px;font-family:var(--mono);font-size:12.5px;box-shadow:0 10px 24px rgba(13,13,15,.2)}
#chat{border-radius:12px;border:1px solid var(--line);box-shadow:0 24px 60px rgba(16,24,40,.18)}
.chead{border-bottom:1px solid var(--line)}.chead b{font-family:var(--mono)}
.cm{border-radius:8px}.cm.bot{background:#f4f5f7;border:1px solid var(--line)}
#cin{border-radius:8px;background:#fff;border:1px solid var(--line);box-shadow:none}#cform button{border-radius:8px}
"""

CSS_SIMPLE = r"""
body.simple .x,body.expert .s{display:none!important}
.mode{display:grid;grid-template-columns:1fr 1fr;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:3px}
.mode button{background:none;border:0;color:var(--muted);font:600 13px Inter,sans-serif;padding:7px;border-radius:7px;cursor:pointer}
.mode button.on{background:linear-gradient(135deg,#0d0d0f,#2b2d31);color:#ffffff}
.answer{display:grid;grid-template-columns:1fr 260px;gap:26px;border-top:4px solid var(--c);background:linear-gradient(135deg,color-mix(in srgb,var(--c) 10%,transparent),transparent 55%),var(--panel)}
.answer h2{font-size:30px;line-height:1.15;margin:6px 0 6px;color:var(--c);letter-spacing:-.02em}.answer p{margin:0 0 10px;font-size:15px}
.sus{margin:14px 0 12px;max-width:520px}.sustrack{position:relative;height:10px;border-radius:6px;background:linear-gradient(90deg,#2a9d63,#d99a00 45%,#f76b15 65%,#e5484d)}
.sustrack i{position:absolute;top:-5px;width:20px;height:20px;margin-left:-10px;border-radius:50%;background:#fff;border:4px solid var(--c);box-shadow:0 2px 8px rgba(16,24,40,.14)}
.suslab{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:6px}
.sure{color:#4a4f57}.mlnote{background:rgba(217,154,0,0.08);border:1px solid rgba(217,154,0,0.35);border-radius:10px;padding:10px 12px;font-size:13.5px!important}
.claimq{margin-top:10px;background:var(--panel2);border-left:3px solid var(--gold);border-radius:8px;padding:10px 12px;font-size:14px}
.ql{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-bottom:4px}
.ansR{display:flex;flex-direction:column;gap:8px;align-self:center}
button.ask{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:12px;padding:10px 14px;font:500 14px Inter,sans-serif;text-align:left;cursor:pointer}
button.ask:hover{border-color:var(--gold)}button.ask:before{content:"💬 "}
.pf{display:grid;grid-template-columns:280px 1fr;gap:22px;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:var(--r);padding:20px;margin-bottom:18px;scroll-margin-top:20px}
.pfmedia{background:#f1f2f4;border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center;min-height:180px;max-height:420px;color:var(--muted)}
.pfmedia img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}.pfmedia .ico{width:56px;height:56px}
.pf h3{font-size:22px;margin:4px 0 2px}.pfsub{margin:0 0 14px;color:#4a4f57}
.qa{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.q{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 12px}.q b{display:flex;align-items:center;gap:8px;font-size:15px}
.q small{display:block;color:var(--muted);font-size:12px;margin-top:3px}
.oi{display:inline-flex;width:22px;height:22px;border-radius:6px;background:var(--line);align-items:center;justify-content:center;font-size:12px;color:var(--gold)}
.qa3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.qa3>div{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:13px}
.qa3 ul{margin:4px 0 0;padding-left:16px}.qa3 li{margin-bottom:3px}
.pf h4{margin:16px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.reasons{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.reasons li{display:flex;gap:10px;align-items:flex-start;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 12px;font-size:14px}
.kd{width:10px;height:10px;border-radius:50%;background:var(--k);flex:none;margin-top:6px;box-shadow:0 0 0 3px color-mix(in srgb,var(--k) 20%,transparent)}
.reasons li>span:nth-child(2){flex:1}.tech{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:5px;padding:1px 6px;flex:none}
.tech:hover{color:var(--gold);border-color:var(--gold)}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:11.5px;color:var(--muted)}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--k);margin-right:5px}
.pfacts{margin-top:12px}.pfacts summary{cursor:pointer;color:var(--muted);font-weight:600;font-size:13px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:8px}.fgrid div{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
.fgrid span{display:block;font-size:11px;color:var(--muted)}.fgrid b{font-weight:500;font-size:13px}
.pfact{display:flex;gap:12px;align-items:center;margin-top:14px;flex-wrap:wrap}.tech2{font-weight:600;font-size:13px}
/* chat */
#chatbtn{position:fixed;right:24px;bottom:24px;z-index:40;display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#0d0d0f,#2b2d31);color:#ffffff;border:0;border-radius:30px;
padding:14px 20px;font:700 14px Inter,sans-serif;cursor:pointer;box-shadow:0 10px 30px rgba(16,24,40,.14)}#chatbtn.hide{display:none}
#chat{position:fixed;right:24px;bottom:24px;z-index:41;width:400px;max-width:calc(100vw - 32px);height:600px;max-height:calc(100vh - 48px);display:none;flex-direction:column;
background:#ffffff;border:1px solid var(--line);border-radius:18px;box-shadow:0 20px 60px rgba(16,24,40,.14);overflow:hidden}#chat.on{display:flex}
.chead{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid var(--line);background:linear-gradient(90deg,rgba(13,13,15,0.054),transparent)}
.chead b{display:block}.chead span{font-size:11.5px;color:var(--muted)}.chead button{background:none;border:0;color:var(--muted);font-size:24px;cursor:pointer;line-height:1}
#cmsgs{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
.cm{max-width:88%;padding:10px 13px;border-radius:14px;font-size:13.5px;line-height:1.5}.cm p{margin:0 0 6px}.cm p:last-child{margin:0}
.cm.me{align-self:flex-end;background:linear-gradient(135deg,#0d0d0f,#2b2d31);color:#ffffff;border-bottom-right-radius:4px;font-weight:500}
.cm.bot{align-self:flex-start;background:var(--panel2);border:1px solid var(--line);border-bottom-left-radius:4px}
.cm ul,.cm ol{margin:4px 0 6px;padding-left:18px}.cm li{margin-bottom:4px}.cm .kd{display:inline-block;margin:0 6px 1px 0;width:8px;height:8px;box-shadow:none}
.cm hr{border:0;border-top:1px solid var(--line);margin:10px 0}.cm table{font-size:12.5px}.cm td{padding:3px 8px 3px 0;border:0}
.cm .cf{font-weight:700;color:var(--gold);font-size:12px;margin:6px 0 2px;text-transform:none}.cm .small{font-size:12px;color:var(--muted)}
.cm a{font-family:var(--mono);font-size:11px}.cm .ih{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;margin-bottom:4px}
#csug{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}#csug button{background:none;border:1px solid var(--line);color:var(--text);border-radius:16px;padding:5px 11px;font:12.5px Inter,sans-serif;cursor:pointer}
#csug button:hover{border-color:var(--gold);color:var(--gold)}
#cform{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line)}#cin{flex:1;background:#eef0f2;border:1px solid var(--line);border-radius:12px;padding:11px 13px;color:var(--text);font:14px Inter,sans-serif}
#cin:focus{outline:none;border-color:var(--gold)}#cform button{background:var(--gold);border:0;border-radius:12px;width:44px;color:#ffffff;font-size:16px;cursor:pointer}
@media (max-width:900px){.answer,.pf{grid-template-columns:1fr}.qa,.qa3{grid-template-columns:1fr}.answer h2{font-size:24px}
#chat{right:0;bottom:0;width:100vw;max-width:none;height:100vh;max-height:none;border-radius:0}#chatbtn{right:16px;bottom:16px}#chatbtn span{display:none}}
@media print{body.simple .x,body.expert .s{display:block!important}.s .ask,.pfact,.ansR{display:none!important}.pf{break-inside:avoid}}
"""

CSS = r"""
:root{--bg:#eef0f2;--panel:#ffffff;--panel2:#f6f7f8;--line:#e7e9ec;--text:#0d0d0f;--muted:#8a9099;--gold:#0d0d0f;--accent:#2f6fed;
--ok:#2a9d63;--bad:#e5484d;--warn:#d99a00;--r:14px;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background-image:radial-gradient(1200px 500px at 80% -10%,rgba(47,111,237,0.08),transparent),radial-gradient(900px 400px at 0% 0%,rgba(13,13,15,0.027),transparent)}
a{color:var(--accent);text-decoration:none}.mono{font-family:var(--mono)}.small{font-size:12px}.muted{color:var(--muted)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}.eyebrow.gold{color:var(--gold)}
.ico{display:inline-flex;width:18px;height:18px;flex:none}.ico svg{width:100%;height:100%;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.cnt{display:inline-block;min-width:20px;padding:0 6px;border-radius:10px;background:var(--line);color:var(--muted);font-size:11px;font-weight:600;text-align:center;margin-left:4px}
.cnt.ok{background:rgba(42,157,99,0.15);color:var(--ok)}
/* sidebar */
.side{position:fixed;inset:0 auto 0 0;width:268px;background:rgba(255,255,255,.72);backdrop-filter:blur(10px);border-right:1px solid var(--line);
padding:20px 16px;display:flex;flex-direction:column;gap:16px;overflow:auto;z-index:10}
.logo{display:flex;align-items:center;gap:10px;color:var(--text);font-size:19px;letter-spacing:.02em}.logo .ico{width:26px;height:26px;color:var(--gold)}.logo b{color:var(--gold)}
.sidecase{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}.sidecase .mono{font-size:12px;margin:2px 0 8px}
.sideverdict{display:flex;align-items:baseline;gap:8px;border-left:3px solid var(--c);padding-left:10px}.sideverdict b{font-size:24px;color:var(--c)}.sideverdict span{font-size:12px;color:var(--muted)}
.side nav{display:flex;flex-direction:column;gap:2px}
.side nav>a{display:flex;align-items:center;gap:10px;color:var(--muted);padding:7px 10px;border-radius:8px;font-weight:500}
.side nav>a:hover,.side nav>a.cur{background:var(--panel2);color:var(--text)}.side nav .n{font-family:var(--mono);font-size:11px;color:var(--gold);opacity:.8}
.side nav>a .cnt{margin-left:auto}
.navsub{margin:2px 0 6px 18px;border-left:1px solid var(--line);padding-left:8px}
.navgrp{display:flex;align-items:center;gap:6px;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:8px 0 2px}.navgrp .ico{width:14px;height:14px}
.navev{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:6px;color:var(--text);font-size:12.5px}.navev:hover{background:var(--panel2)}
.navev .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}.navev .pc{font-family:var(--mono);font-size:11px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;flex:none}
.sideactions{margin-top:auto;display:flex;flex-direction:column;gap:8px}
.btn{display:block;text-align:center;background:linear-gradient(135deg,#0d0d0f,#2b2d31);color:#ffffff;border:0;border-radius:10px;padding:10px 14px;font:600 13px Inter,sans-serif;cursor:pointer}
.btn.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}.btn.ghost:hover{border-color:var(--muted)}
/* main */
main{margin-left:268px;padding:28px 36px 40px;max-width:1340px}
.top{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:16px}.top h1{margin:6px 0 4px;font-size:30px;letter-spacing:-.02em}
.meta{color:var(--muted);font-size:12.5px}.printonly{display:none}
.disclaimer{display:flex;gap:12px;align-items:flex-start;background:linear-gradient(90deg,rgba(13,13,15,0.054),rgba(13,13,15,0.013));border:1px solid rgba(13,13,15,0.135);
border-radius:12px;padding:12px 16px;margin-bottom:18px;font-size:13px}.disclaimer .ico{color:var(--gold);width:22px;height:22px;margin-top:1px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:22px 24px;margin-bottom:18px}
.sh{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}.sh h2{margin:0;font-size:18px;letter-spacing:-.01em}.sh.pad{margin:6px 0 14px}
.sh .num{font-family:var(--mono);color:var(--gold);font-size:12px;border:1px solid rgba(13,13,15,0.158);border-radius:6px;padding:1px 6px}
h3{font-size:15px;margin:0 0 10px}.mt{margin-top:22px}
.empty{color:var(--muted);font-style:italic;margin:6px 0}
/* hero */
.hero{border-top:3px solid var(--c);background:linear-gradient(180deg,rgba(0,0,0,0.012),transparent),var(--panel)}
.heroL{display:grid;grid-template-columns:170px 1fr;gap:26px;align-items:center}
.gauge .gnum{font:800 30px Inter,sans-serif}.gauge .gsub{font:500 11px Inter,sans-serif;fill:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.band{font-size:24px;font-weight:800;letter-spacing:-.01em;margin:4px 0 12px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:16px}
.stats div{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 12px}.stats span{display:block;font-size:11px;color:var(--muted)}
.stats b{font-size:16px}.stats small{font-weight:500;color:var(--muted)}
.rbar{position:relative;height:14px;margin-top:4px}.zones{display:flex;height:6px;margin-top:4px;border-radius:4px;overflow:hidden;opacity:.5}
.zones i{display:block;height:100%}.z1{width:20%;background:#2a9d63}.z2{width:25%;background:#d99a00}.z3{width:25%;background:#f76b15}.z4{width:30%;background:#e5484d}
.rng{position:absolute;top:0;height:14px;border:2px solid;border-radius:6px;background:rgba(0,0,0,0.036)}
.pin{position:absolute;top:-3px;width:4px;height:20px;border-radius:2px;margin-left:-2px;box-shadow:0 0 0 3px var(--panel)}
.ticks span{position:absolute;top:0;width:1px;height:14px;background:var(--line)}
.rlabels{display:grid;grid-template-columns:20% 25% 25% 30%;font-size:10.5px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.06em}
.banner{border-radius:10px;padding:10px 14px;margin-top:14px;font-size:13px;border:1px solid;color:var(--text);font-weight:400}
.banner b{font-weight:700}
.banner.warn{background:rgba(217,154,0,0.08);border-color:rgba(217,154,0,0.35)}.banner.bad{background:rgba(229,72,77,0.08);border-color:rgba(229,72,77,0.35)}
.banner.info{background:rgba(47,111,237,0.08);border-color:rgba(47,111,237,0.35)}
.rubric{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-top:18px}
.rub{display:flex;flex-direction:column;gap:2px;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px;color:var(--text);position:relative}
.rub:hover{border-color:var(--gold)}.rub b{font-size:12.5px}.rub span:last-child{font-size:11px;color:var(--muted)}
.rub .tick{position:absolute;top:8px;right:10px;color:var(--ok);font-size:12px}
/* key findings */
.kfgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
.kf{display:block;background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--sev);border-radius:10px;padding:12px 14px;color:var(--text)}
.kf:hover{border-color:var(--sev)}.kfh{display:flex;align-items:center;gap:8px}.kfh .ico{color:var(--muted);width:15px;height:15px}
.kfm{color:var(--muted);font-size:12px;margin:4px 0 8px}.kfw{height:4px;background:var(--line);border-radius:3px}.kfw i{display:block;height:100%;background:var(--sev);border-radius:3px}
.iid{font-family:var(--mono);font-size:11px;color:var(--gold);background:rgba(13,13,15,0.045);border-radius:5px;padding:1px 6px;flex:none}
.fams{display:flex;flex-direction:column;gap:6px}
.fam{display:grid;grid-template-columns:minmax(220px,300px) 1fr 70px;gap:14px;align-items:center;padding:6px 0;border-bottom:1px dashed var(--line)}
.fam .fn b{display:block;font-size:13px}.fam .fn span{font-size:11.5px;color:var(--muted)}.fam.off{opacity:.45}
.fbar{height:8px;background:var(--line);border-radius:5px;overflow:hidden}.fbar i{display:block;height:100%;border-radius:5px}
.fv{font-family:var(--mono);font-size:12px;text-align:right}
/* evidence */
.filters{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.fchip{display:inline-flex;align-items:center;gap:6px;background:var(--panel);border:1px solid var(--line);color:var(--muted);border-radius:20px;padding:6px 12px;font:500 12.5px Inter,sans-serif;cursor:pointer}
.fchip .ico{width:14px;height:14px}.fchip.on{background:var(--text);color:#ffffff;border-color:var(--text)}.fchip.on .cnt{background:#ffffff;color:#fff}
.fchip:disabled{opacity:.35;cursor:default}
.ev{padding:0;overflow:hidden}
.evhead{display:flex;align-items:center;gap:14px;padding:18px 22px;border-bottom:1px solid var(--line);background:linear-gradient(90deg,color-mix(in srgb,var(--c) 10%,transparent),transparent 60%)}
.evhead>.ico{width:40px;height:40px;padding:9px;border-radius:10px;background:var(--panel2);border:1px solid var(--line);color:var(--c)}
.evtitle{flex:1;min-width:0}.evtitle h3{margin:2px 0 0;font-size:17px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.evscore{text-align:right}.evscore .big{font-size:28px;font-weight:800;line-height:1}.evscore .small{color:var(--muted)}
.ev>.banner{margin:12px 22px 0}
.tabs{display:flex;gap:2px;padding:10px 18px 0;border-bottom:1px solid var(--line);overflow-x:auto}
.tab{background:none;border:0;color:var(--muted);font:600 13px Inter,sans-serif;padding:9px 12px;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap}
.tab.on{color:var(--text);border-bottom-color:var(--gold)}.tab:hover{color:var(--text)}
.panel{display:none;padding:18px 22px 22px}.panel.on{display:block}.ponly{display:none}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin-top:12px}
figure{margin:0;background:#f1f2f4;border:1px solid var(--line);border-radius:10px;overflow:hidden}
figure img{width:100%;display:block;max-height:380px;object-fit:contain;cursor:zoom-in;transition:transform .2s}figure img:hover{transform:scale(1.01)}
figcaption{padding:7px 11px;font-size:12px;font-weight:600;color:var(--muted);border-top:1px solid var(--line)}
/* indicators */
.ind{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--sev);border-radius:10px;padding:14px 16px;margin-bottom:10px;scroll-margin-top:20px}
.ind.key{box-shadow:0 0 0 1px color-mix(in srgb,var(--sev) 40%,transparent)}.ind.auth{background:rgba(42,157,99,0.05)}
.ind.flash,.xm.flash{animation:flash 1.6s}@keyframes flash{0%,40%{box-shadow:0 0 0 3px var(--gold)}100%{box-shadow:none}}
.ihead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.ihead h4{margin:0;font-size:14.5px}
.chip{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);border:1px solid var(--line);border-radius:12px;padding:1px 8px}
.chip.ok{color:var(--ok);border-color:rgba(42,157,99,0.4)}.chip.key{color:var(--gold);border-color:rgba(13,13,15,0.180)}
.locs{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0}
.lk{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-right:4px}
.loc{font-family:var(--mono);font-size:11.5px;background:#f3f4f6;border:1px solid var(--line);border-radius:6px;padding:2px 8px;color:var(--text)}
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.igrid p{margin:3px 0 0;font-size:13px}.igrid>div:last-child p{color:#4a4f57}
.meters{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:12px}
.mlab{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}.mlab b{color:var(--text);font-family:var(--mono)}
.mbar{height:6px;background:var(--line);border-radius:4px;margin-top:4px;overflow:hidden}.mbar i{display:block;height:100%;border-radius:4px}
.alts{margin-top:10px;font-size:12.5px}.alts summary{cursor:pointer;color:var(--muted);font-weight:600}.alts ul{margin:6px 0 0;padding-left:18px;color:#4a4f57}
/* timeline */
.tline{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tl-wrap{display:flex;gap:10px;margin-top:8px}.tl-lab{position:relative;width:60px;flex:none}.tlab{position:absolute;font-family:var(--mono);font-size:11px;color:var(--gold);line-height:22px}
.tl-track{position:relative;flex:1;background:repeating-linear-gradient(90deg,var(--line) 0 1px,transparent 1px 25%);border-radius:6px;background-color:#f3f4f6}
.tpt{position:absolute;width:3px;height:14px;margin-left:-1px;border-radius:2px}.tseg{position:absolute;height:14px;border-radius:3px;opacity:.85}
.tpt:hover,.tseg:hover{outline:2px solid #fff}
.tl-ticks{position:relative;height:16px;margin-left:70px;font-family:var(--mono);font-size:10px;color:var(--muted)}.tl-ticks span{position:absolute;transform:translateX(-50%)}
.tl-ticks span:first-child{transform:none}.tl-ticks span:last-child{transform:translateX(-100%)}
/* facts */
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.fact{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:0}.fact.wide{grid-column:1/-1}
.fact span{display:block;font-size:11px;color:var(--muted)}.fact b{font-weight:500;font-size:13px;word-break:break-word;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.hash{font-size:11.5px;word-break:break-all}.copy{margin-left:8px;background:var(--line);border:0;color:var(--text);border-radius:5px;font:600 10.5px Inter;padding:2px 7px;cursor:pointer}
.transcript{background:#f3f4f6;border:1px solid var(--line);border-radius:10px;padding:10px 12px;max-height:340px;overflow:auto;font-size:13px}
.tr{display:flex;gap:10px;padding:3px 6px;border-radius:5px}.tr .ts{font-family:var(--mono);color:var(--muted);font-size:11.5px;flex:none;width:52px}
.tr.hit{background:rgba(229,72,77,0.12);color:#b42318}
/* cross-modal */
.xmgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}
.xm{background:var(--panel2);border:1px solid var(--line);border-top:3px solid var(--c);border-radius:10px;padding:14px 16px;scroll-margin-top:20px}
.xmh{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.xm p{margin:8px 0 0;font-size:13px;color:#4a4f57}
.rel{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--c);border:1px solid var(--c);border-radius:12px;padding:1px 8px}
blockquote{margin:0 0 14px;padding:12px 16px;background:var(--panel2);border-left:3px solid var(--gold);border-radius:8px;font-size:15px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:28px}
.checks{display:flex;flex-direction:column;gap:6px}.chk{display:flex;gap:10px;align-items:flex-start;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;font-size:13px}
.chk input{margin-top:3px;accent-color:var(--gold)}.chk:has(input:checked) span{text-decoration:line-through;color:var(--muted)}
.lims{margin:0;padding-left:18px;color:#4a4f57}.lims li{margin-bottom:6px;font-size:13px}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid var(--line);font-size:12.5px}
thead th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.log{list-style:none;margin:0;padding:0;border-left:1px solid var(--line)}.log li{position:relative;padding:4px 0 8px 16px;display:flex;flex-direction:column}
.log li:before{content:"";position:absolute;left:-4px;top:10px;width:7px;height:7px;border-radius:50%;background:var(--gold)}.log b{font-size:12.5px;text-transform:capitalize}
.models{padding-left:18px;color:#4a4f57}
footer{text-align:center;color:var(--muted);font-size:12px;padding:20px 0}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:50;cursor:zoom-out;padding:30px}
#lb.on{display:flex}#lb img{max-width:95%;max-height:88vh;border-radius:8px}#lb .cap{color:#fff;margin-top:10px;font-weight:600}
@media (max-width:1100px){.rubric{grid-template-columns:repeat(3,1fr)}.stats{grid-template-columns:repeat(2,1fr)}}
@media (max-width:900px){.side{position:static;width:auto;border-right:0;border-bottom:1px solid var(--line)}.side nav,.navsub{display:none}
main{margin-left:0;padding:16px}.heroL{grid-template-columns:1fr;justify-items:center;text-align:center}.two,.igrid{grid-template-columns:1fr}
.meters{grid-template-columns:1fr}.fam{grid-template-columns:1fr 60px}.fam .fbar{grid-column:1/-1;grid-row:2}.rubric{grid-template-columns:repeat(2,1fr)}
.xmgrid,.gallery{grid-template-columns:1fr}.evhead{flex-wrap:wrap}.rlabels{font-size:8.5px;letter-spacing:0}}
@media print{
:root{--bg:#fff;--panel:#fff;--panel2:#f8f9fb;--line:#dfe3ea;--text:#111827;--muted:#5b6474}
body{background:#fff;font-size:11.5px}.noprint,.side{display:none!important}main{margin:0;padding:0;max-width:none}
.printonly{display:block}.card{break-inside:auto;margin-bottom:10px;border-radius:6px}
.panel{display:block!important;padding-top:6px}.ponly{display:block;margin-top:10px}.tabs{display:none}.ev{display:block!important}
.ind,.kf,.xm,figure,.fact{break-inside:avoid}.loc,.transcript,figure{background:#f3f4f6}.igrid>div:last-child p,.xm p,.lims,.alts ul{color:#374151}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}.transcript{max-height:none}.gallery{grid-template-columns:1fr 1fr}
.btn{display:none}}
"""
