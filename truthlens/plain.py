"""Plain-language layer: turns a technical report into answers a non-expert can follow.

Everything here is a pure function of the report dict, so it can also be applied to
old reports. It powers the report's "Simple view" and the offline case assistant.
Nothing is invented: every sentence is derived from a measured indicator or metadata field.
"""
from __future__ import annotations

import html
import re

E = html.escape

CONF_WORDS = {
    "High": "We're fairly sure about this result.",
    "Medium": "Reasonably sure, but a person should double-check.",
    "Low": "Not sure — treat this as a hint, not an answer.",
}
TYPE_NOUN = {"image": "picture", "video": "video", "audio": "recording", "text": "text"}


def _t(x: float) -> str:
    m, s = divmod(float(x), 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f}s"


def _where(loc: dict) -> str:
    t = (loc or {}).get("type")
    if t == "regions":
        n = len(loc["regions"])
        return f"in {n} marked area{'s' if n > 1 else ''} of the picture"
    if t == "frames":
        ts = [f["time"] for f in loc["frames"]]
        return "at " + ", ".join(_t(x) for x in ts[:4]) + (f" and {len(ts) - 4} more moments" if len(ts) > 4 else "")
    if t == "points":
        return "at " + ", ".join(_t(x) for x in loc["times"][:4])
    if t == "segments":
        segs = loc["segments"]
        return "during " + ", ".join(f"{_t(s[0])}–{_t(s[1])}" for s in segs[:3]) + (" and more" if len(segs) > 3 else "")
    return ""


# detector -> (kind, plain sentence). kind: red = red flag, check = worth checking, info = normal / not suspicious
def _sentence(i: dict) -> tuple[str, str]:
    d, s, obs = i["detector"], i["score"], i["observation"]
    w = _where(i.get("location"))
    w = f" ({w})" if w else ""
    auth = i["supports"] == "authenticity"
    if d == "metadata_ai_signature" and i["title"].startswith("Content Credentials"):
        acts = [a.strip().rstrip(".") for a in re.findall(r"“([^”]+)”", obs)]
        return "red", ("The file carries Content Credentials, a signed label from the tool that made it, saying it was "
                       "<b>made with generative AI</b>" + (f": <i>{E('; '.join(acts[:2]))}</i>" if acts else "") +
                       ". This is direct evidence, not a guess.")
    if d == "metadata_ai_signature":
        return "red", f"The file contains a tag written by an AI tool: <i>{E(obs.split(':', 1)[-1].strip(' .'))}</i>. This is direct evidence, not a guess."
    if d == "metadata_editor":
        if "C2PA" in i["title"]:
            return "info", "The file carries Content Credentials (a digital label some cameras, editors and AI tools add). On its own this is neutral."
        return "check", "The file was saved by editing software. That's common, but it means the picture may have been changed."
    if d == "metadata_missing":
        return "info", "There's no camera information inside the file. That's normal for screenshots and for pictures sent through WhatsApp or social media."
    if d == "metadata_dates":
        return "check", "The file was modified after the moment it was captured."
    if d == "ela":
        return ("red" if s >= 0.5 else "check"), f"Some areas were saved differently from the rest of the picture{w} — a common trace of pasted-in content."
    if d == "noise":
        return ("red" if s >= 0.5 else "check"), f"The natural grain of the picture doesn't match in some places{w}, as if parts came from another photo."
    if d == "copy_move":
        return "red", f"A piece of the picture appears to be copied and pasted somewhere else in the same picture{w}."
    if d == "fft":
        return "check", "The picture has tiny repeating patterns that AI generators and heavy resizing often leave."
    if d == "ml_image_ai":
        if auth:
            return "info", "Our AI-picture detector leans towards a real camera image."
        if i.get("validated"):
            return ("red" if s >= 0.85 else "check"), (f"Our AI-picture detector — one that passed our real-vs-AI test — says it looks AI-made "
                                                      f"({s * 100:.0f}%){w}. It was tested on a small sample, so a human should still confirm.")
        return "check", f"Our AI-picture detector says it looks AI-made ({s * 100:.0f}%). These detectors are often fooled by filters, compression and text overlays."
    if d == "ml_face_deepfake":
        return "check", f"Our face detector thinks a face may be swapped or computer-generated{w}. It can be fooled by beauty filters."
    if d == "video_temporal":
        return "check", f"The video jumps suddenly{w} — likely cuts. Normal for edited clips, suspicious if it's sold as one continuous recording."
    if d == "video_duplicate_frames":
        return "check", f"Some frames repeat while things should be moving{w}, which can hide removed footage."
    if d == "video_metadata":
        if "length" in i["title"].lower():
            return "check", "The sound and picture tracks have different lengths, a sign they may have been combined."
        return "info", "The video was re-saved by software. Almost every shared video is, so this is mostly neutral."
    if d == "video_watermark":
        return "check", ("A fixed logo, watermark or caption sits in the corner of the video. Zoom in and read it — if it names an AI tool "
                         "(Veo, Gemini, Kling, Meta AI, Sora), the video was made with AI. Captions and app logos are harmless.")
    if d == "audio_format":
        return "check", ("The audio is saved at a sample rate (22 or 24 kHz) that many AI voice tools use by default; phones and cameras "
                         "record at 44 or 48 kHz. This hints the voice may be generated — but apps can convert audio, so it isn't proof.")
    if d == "audio_ml":
        if auth:
            return "info", "Our voice detector leans towards a natural human voice."
        return "check", f"Our voice detector thinks the voice may be computer-generated{w}. This detector is unreliable on real-world recordings."
    if d == "audio_bandwidth":
        return "info", "The sound is missing very high tones. AI voices do this, but so do phone calls and WhatsApp voice notes."
    if d == "audio_digital_silence":
        return "red", f"There are moments of perfect digital silence inside speech{w} — a typical trace of cutting and joining audio."
    if d == "audio_splice":
        return "check", f"The background noise changes suddenly{w}, as if two recordings were joined."
    if d == "text_language":
        return "check", "The caption uses pressure language (\"share before it's deleted\", \"breaking\") common in misinformation."
    if d == "xmodal_caption_visual":
        return ("info", "The caption matches what's actually in the picture.") if auth else \
               ("red", "The caption doesn't match what's actually in the picture.")
    if d == "xmodal_caption_transcript":
        return ("info", "What is said in the recording matches the caption.") if auth else \
               ("check", "What is said in the recording doesn't match the caption.")
    if d == "xmodal_dates":
        return ("info", "The date in the caption matches the date stored in the file.") if auth else \
               ("red", "The date in the caption doesn't match the date stored in the file.")
    if d == "xmodal_corroboration":
        if "cut" in i["title"].lower():
            return "red", f"The picture and the sound both show an edit at the same moment{w}. Two independent checks agree."
        if "same visual" in i["title"].lower():
            return "info", "Two of the files show the same scene."
        return "check", i["title"] + "."
    if d == "xmodal_ml_agreement":
        return "check", "Both the face and the voice detectors flagged it — but both are AI detectors that can make the same mistake."
    return ("info" if auth else "check"), E(i["title"]) + "."


def origin(ev: dict) -> dict:
    """What kind of file is this? (camera photo, screenshot, AI-generated, edited, unknown)."""
    md = ev.get("metadata") or {}
    name = ev.get("name", "")
    ex = md.get("exif") or {}
    det = {i["detector"] for i in ev.get("indicators", []) if i["supports"] == "manipulation"}
    blob = " ".join(str(v) for v in ex.values()).lower() + " " + " ".join(str(v) for v in (md.get("png_text") or {}).values()).lower()
    mod = ev["modality"]
    camera = " ".join(x for x in [ex.get("Make"), ex.get("Model")] if x).strip() or (md.get("device") or "").strip()
    if "metadata_ai_signature" in det or md.get("ai_signatures"):
        return {"kind": "ai", "label": "AI-generated", "certain": True,
                "why": "The file itself carries a tag from an AI generation tool."}
    if mod == "image":
        tag = "screenshot" in blob
        fname = bool(re.search(r"screen ?shot|screen ?grab|scrn", name, re.I))
        if tag or fname:
            return {"kind": "screenshot", "label": "Screenshot", "certain": tag,
                    "why": ("The file's own data says “Screenshot” (written by the phone or computer that took it)."
                            if tag else "The file name says it's a screenshot (the file's data doesn't confirm it).")}
        if camera:
            return {"kind": "camera", "label": f"Camera photo ({camera})", "certain": True,
                    "why": f"The file records the camera that took it: {camera}."}
        if md.get("editors"):
            return {"kind": "edited", "label": "Edited image", "certain": False,
                    "why": "Editing software is recorded in the file: " + ", ".join(md["editors"]) + "."}
        return {"kind": "unknown", "label": "Unknown source", "certain": False,
                "why": "No camera or screenshot information. Pictures saved from WhatsApp, Instagram or the web usually lose it."}
    if mod == "video":
        if re.search(r"screen ?record", name, re.I):
            return {"kind": "screenshot", "label": "Screen recording", "certain": False, "why": "The file name says it's a screen recording."}
        if camera:
            return {"kind": "camera", "label": f"Recorded on {camera}", "certain": True, "why": f"The video records the device: {camera}."}
        return {"kind": "unknown", "label": "Unknown source", "certain": False,
                "why": "No recording-device information. Videos downloaded from social media usually lose it."}
    if mod == "audio":
        return {"kind": "unknown", "label": "Audio recording", "certain": False, "why": "Audio files rarely record where they came from."}
    return {"kind": "unknown", "label": "Text", "certain": False, "why": ""}


ML_DETECTORS = {"ml_image_ai", "ml_face_deepfake", "audio_ml", "xmodal_ml_agreement"}


def _independent(inds: list) -> list:
    """Non-ML manipulation indicators strong enough to count as independent evidence."""
    return [i for i in inds if i["supports"] == "manipulation" and i["detector"] not in ML_DETECTORS
            and i["score"] * i["reliability"] >= 0.1 and _sentence(i)[0] != "info"]


def _ml_only(a: dict, inds: list) -> bool:
    if "ml_only" in a:
        return bool(a["ml_only"])
    man = [i for i in inds if i["supports"] == "manipulation"]
    return bool(man) and not _independent(inds)


def _headline(a: dict, org: dict, inds: list) -> tuple[str, str, str]:
    """(headline, colour, one-line explanation) for one assessment."""
    p = a["likelihood"]
    if org["kind"] == "ai":
        return "Made with AI", "red", "The file contains a label from an AI tool."
    if _ml_only(a, inds) and a.get("validated_ml"):
        return ("Likely made with AI", "orange",
                "Our tested AI detector is confident across the whole file, but nothing else in the file confirms it yet.")
    if _ml_only(a, inds):
        return "Can't tell for sure", "amber", "Only AI detectors raised a flag, and nothing else in the file backs them up."
    if p >= 70:
        return "Strong signs it was manipulated or AI-made", "red", "Several independent checks found problems."
    if p >= 45:
        return "Some signs of manipulation", "orange", "A few checks found problems worth a closer look."
    if p >= 20:
        return "Can't tell for sure", "amber", "Only weak or easily-explained signs were found."
    return "No clear signs of editing or AI", "green", "Nothing suspicious stood out — but that doesn't prove it's real."


def _ai_answer(ev: dict, org: dict) -> str:
    mod = ev["modality"]
    inds = ev.get("indicators", [])
    ml = [i for i in inds if i["detector"] in ("ml_image_ai", "ml_face_deepfake", "audio_ml")]
    man_ml = [i for i in ml if i["supports"] == "manipulation"]
    m = ev.get("metrics") or {}
    am = m if mod == "audio" else (m.get("audio") or {})
    if mod in ("audio", "video") and am.get("audio_model_unreliable") and org["kind"] != "ai":
        fmt = any(i["detector"] == "audio_format" for i in inds)
        voice = ("<b>We can't tell yet from the voice.</b> Our AI-voice detector failed our tests and is switched off, "
                 "so “no warning signs” does <i>not</i> mean the voice is human.")
        if fmt:
            voice += " One clue points towards AI: the audio is saved at 22–24 kHz, the default of many AI voice tools."
        if mod == "audio":
            return voice
        extra = _ai_answer({**ev, "metrics": {**m, "audio": {}}}, org)   # still report the picture-based detectors
        return extra + " " + voice
    if org["kind"] == "ai":
        return "<b>Yes, very likely.</b> " + org["why"] + " That's direct evidence from the file, not a statistical guess."
    parts = []
    if man_ml:
        names = {"ml_image_ai": "AI-picture detector", "ml_face_deepfake": "face-swap detector", "audio_ml": "voice detector"}
        scores = "; ".join(f"our {names[i['detector']]}{' (tested)' if i.get('validated') else ''} scored it {i['score'] * 100:.0f}%"
                           for i in man_ml)
        strong_valid = any(i.get("validated") and i["detector"] == "ml_image_ai" and i["score"] >= 0.85 for i in man_ml)
        parts.append(("<b>Likely.</b> " if strong_valid else "<b>Possibly.</b> ") + scores[0].upper() + scores[1:] + ".")
        indep = _independent(inds)
        if not indep and strong_valid:
            parts.append("The AI-picture detector passed our real-vs-AI test and is confident across the whole file, "
                         "but it's still one model tested on a small sample, and nothing else in the file confirms it. "
                         "Check the SynthID watermark (see “What you can do next”) to be sure.")
        elif not indep:
            parts.append("But that's the <b>only</b> sign — nothing in the file's data, pixels or timing confirms it. "
                         "These detectors are often fooled by filters, compression or text on screen, so treat it as a hint.")
        else:
            parts.append(f"{len(indep)} other, independent check{'s' if len(indep) > 1 else ''} also found problems, which makes it more believable.")
    elif ml:
        parts.append("<b>Probably not.</b> Our AI detectors lean towards real for this file.")
    else:
        parts.append("<b>No sign of AI was found</b>" + (" — but no AI detector could run on this file." if not ml and mod != "text" else "."))
    if org["kind"] == "screenshot":
        parts.append("Note: this is a screenshot, so we can only judge the picture <i>inside</i> it — the original may be AI-made even if the screenshot is real.")
    return " ".join(parts)


def _screenshot_answer(ev: dict, org: dict) -> str:
    if ev["modality"] not in ("image", "video"):
        return "Screenshots only apply to pictures and videos — this file is " + ("a recording." if ev["modality"] == "audio" else "text.")
    if org["kind"] == "screenshot":
        return f"<b>Yes{'.' if org['certain'] else ', probably.'}</b> {org['why']}"
    if org["kind"] == "camera":
        return f"<b>No.</b> {org['why']} Screenshots don't contain camera details."
    if org["kind"] == "ai":
        return "<b>No sign of a screenshot.</b> The file looks like a direct export from an AI tool."
    if ev["modality"] == "video":
        return ("<b>No sign of a screen recording.</b> There's no recording-device information either — "
                "videos downloaded from social media usually lose it.")
    return ("<b>We can't tell.</b> There's no screenshot tag and no camera information. "
            "Pictures saved from WhatsApp, Instagram or the web lose this information, so either is possible.")


def _edited_answer(ev: dict) -> str:
    fam = {"ela", "noise", "copy_move", "video_temporal", "video_duplicate_frames", "audio_digital_silence", "audio_splice", "metadata_editor", "metadata_dates"}
    hits = [i for i in ev.get("indicators", []) if i["detector"] in fam and i["supports"] == "manipulation"
            and not ("C2PA" in i["title"])]
    if not hits:
        return "<b>No signs of cut-and-paste editing</b> were found in this file."
    lines = [f"<li>{_sentence(i)[1]} <a href='#{E(i['id'])}'>{E(i['id'])}</a></li>" for i in
             sorted(hits, key=lambda i: -i["score"] * i["reliability"])[:4]]
    return f"<b>There are signs it was edited.</b><ul>{''.join(lines)}</ul>"


def _facts(ev: dict) -> dict:
    md = ev.get("metadata") or {}
    m = ev.get("metrics") or {}
    ex = md.get("exif") or {}
    f = {}
    if ev["modality"] == "image":
        cam = " ".join(x for x in [ex.get("Make"), ex.get("Model")] if x)
        f["Camera"] = cam or "not recorded"
        if ex.get("DateTimeOriginal"):
            f["Taken"] = ex["DateTimeOriginal"]
        if ex.get("Software"):
            f["Software"] = ex["Software"]
        f["Location (GPS)"] = "stored in file" if md.get("has_gps") else "not stored"
        if md.get("size"):
            f["Size"] = "×".join(map(str, md["size"])) + " px"
    elif ev["modality"] == "video":
        f["Recording device"] = (md.get("device") or "").strip() or "not recorded"
        f["Created"] = md.get("creation_time") or "not recorded"
        if m.get("duration_s"):
            f["Length"] = f"{m['duration_s']} s"
        if m.get("resolution"):
            f["Resolution"] = m["resolution"]
    elif ev["modality"] == "audio":
        if m.get("duration_s"):
            f["Length"] = f"{m['duration_s']} s"
        if m.get("codec"):
            f["Format"] = m["codec"]
    return f


def file_plain(ev: dict) -> dict:
    org = origin(ev)
    a = ev["assessment"]
    inds = ev.get("indicators", [])
    reasons = []
    for i in sorted(inds, key=lambda i: (i["supports"] != "manipulation", -i["score"] * i["reliability"])):
        kind, text = _sentence(i)
        reasons.append({"kind": kind, "text": text, "id": i["id"], "det": i["detector"]})
    order = {"red": 0, "check": 1, "info": 2}
    reasons.sort(key=lambda r: order[r["kind"]])
    head, color, sub = _headline(a, org, inds)
    noun = TYPE_NOUN.get(ev["modality"], "file")
    return {"id": ev["id"], "name": ev["name"], "modality": ev["modality"], "noun": noun,
            "origin": org, "headline": head, "color": color, "sub": sub,
            "likelihood": a["likelihood"], "confidence": a["confidence"],
            "confidence_words": CONF_WORDS.get(a["confidence"], ""), "ml_only": _ml_only(a, inds),
            "reasons": reasons[:10], "facts": _facts(ev), "sha256": ev.get("sha256", ""),
            "answers": {"ai": _ai_answer(ev, org), "screenshot": _screenshot_answer(ev, org), "edited": _edited_answer(ev)}}


def next_steps(r: dict) -> list[str]:
    det = {i["detector"] for i in r["indicators"] if i["supports"] == "manipulation"}
    mods = {e["modality"] for e in r["evidence"]}
    steps = []
    if mods & {"image", "video"}:
        steps.append("Check for Google's invisible SynthID watermark: upload the file to the Gemini app and ask "
                     "“Was this made with Google AI?” (or use Google's SynthID Detector). It catches Veo and Gemini "
                     "content even after Instagram or WhatsApp re-compression; other AI tools need other checks.")
    if "image" in mods:
        steps.append("Do a reverse image search (Google Lens or TinEye) to find where the picture first appeared.")
    if "video" in mods:
        steps.append("Search for the same video from another angle or another source.")
    if "audio" in mods or "audio_ml" in det:
        steps.append("Compare the voice with a recording you know is genuine, or ask the person directly.")
    if r.get("context_analysis"):
        steps.append("Check the claim on trusted news sites — real events are usually reported by more than one outlet.")
    steps.append("Ask whoever shared it for the original file — forwarded copies lose evidence.")
    steps.append("Don't forward it until you're sure. This tool gives leads, not final proof.")
    return steps


def build(r: dict) -> dict:
    files = [file_plain(ev) for ev in r["evidence"]]
    ov = r["overall"]
    any_ai = any(f["origin"]["kind"] == "ai" for f in files)
    org = {"kind": "ai" if any_ai else "mixed"}
    head, color, sub = _headline(ov, org, r["indicators"])
    if any_ai and len(files) > 1:
        head, sub = "At least one file was made with AI", "One of the files carries a label from an AI tool."
    xm = []
    for f in r.get("cross_modal", []):
        ind = next((i for i in r["indicators"] if i["id"] == f["id"]), None)
        if ind:
            kind, text = _sentence(ind)
            xm.append({"kind": kind, "text": text, "id": f["id"]})
    integ = r.get("integrity") or []
    return {"headline": head, "color": color, "sub": sub, "likelihood": ov["likelihood"],
            "confidence": ov["confidence"], "confidence_words": CONF_WORDS.get(ov["confidence"], ""),
            "ml_only": _ml_only(ov, r["indicators"]), "files": files, "cross": xm, "next_steps": next_steps(r),
            "claim": (r.get("case") or {}).get("context_text") or "",
            "intact": all(x.get("unchanged") for x in integ) if integ else True,
            "case_id": (r.get("case") or {}).get("id", "")}
