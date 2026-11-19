"""Cross-modal correlation: does the evidence agree with itself and with the claim?"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
from PIL import Image

from . import models
from .config import RELIABILITY as R
from .evidence import Indicator
from .text_forensics import extract_claims, parse_media_date

# Claims about what someone SAID — only these are checked against the transcript.
SPEECH_CLAIM = re.compile(r"\b(said|says|saying|say|announc\w*|admit\w*|confess\w*|declar\w*|promis\w*|threaten\w*|"
                          r"statement|speech|interview|tells?|told|talk(s|ing)? about|claims? that|voice note|audio of|"
                          r"recording of|quote[sd]?|warns?|warned)\b|[\"“”]", re.I)

DISTRACTORS = ["a photo of an animal", "a screenshot of text on a phone", "a landscape photo of nature", "a photo of food",
               "a cartoon illustration", "a photo of a building", "a vehicle on a road", "a person at a desk with a laptop",
               "a sports match", "a product on a white background"]
SCENES = ["a crowd or protest", "a politician giving a speech", "a TV news broadcast", "a flood", "a fire or explosion",
          "a road accident", "a portrait of a person", "a selfie", "a document or certificate", "a social media screenshot",
          "soldiers or military vehicles", "a sports event", "a natural landscape", "a city street", "an indoor office or room",
          "a stage or concert", "an AI-generated digital artwork", "a cartoon or animation", "a classroom or lecture",
          "a religious gathering", "a hospital", "money or banknotes"]


def dhash(img: Image.Image, n=8) -> int:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _clip_scores(images: list[Image.Image], texts: list[str]):
    c = models.clip()
    if c is None or not images or not texts:
        return None
    model, proc, torch = c
    with torch.no_grad():
        inp = proc(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
        out = model(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (ie @ te.T).cpu().numpy()  # [n_img, n_txt]


def _clip_text_sim(a: str, b: str):
    c = models.clip()
    if c is None:
        return None
    model, proc, torch = c
    with torch.no_grad():
        inp = proc(text=[a, b], return_tensors="pt", padding=True, truncation=True, max_length=77)
        out = model.text_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        e = model.text_projection(out.pooler_output)
        e = e / e.norm(dim=-1, keepdim=True)
        return float((e[0] @ e[1]).item())


def correlate(case, results: dict, context_text: str, nid) -> dict:
    findings, inds = [], []
    claims = extract_claims(context_text) if context_text else None

    def add(title, relation, modalities, eids, detail, score, rel_key, linked=(), alternatives=(), location=None):
        f = {"id": nid(), "title": title, "relation": relation, "modalities": modalities, "evidence_ids": list(eids),
             "detail": detail, "linked_indicators": list(linked)}
        findings.append(f)
        supports = "authenticity" if relation == "consistent" else "manipulation"
        inds.append(Indicator(f["id"], "+".join(eids) or "X", "cross-modal", rel_key, title,
                              "consistency", detail,
                              "Cross-modal check: " + ("signals agree, which weakens a manipulation hypothesis."
                                                       if supports == "authenticity" else
                                                       "independent signals point the same way or contradict the claim."),
                              score, R[rel_key], location or {"type": "cross"}, list(alternatives), supports=supports))

    # --- collect visuals for CLIP
    visual_items = []  # (eid, label, PIL)
    for eid, r in results.items():
        ev = r["evidence"]
        if ev.modality == "image":
            from .image_forensics import load_rgb
            visual_items.append((eid, ev.name, load_rgb(ev.path)))
        elif ev.modality == "video":
            for k, rgb in list(r.get("keyframes", {}).items())[:6]:
                visual_items.append((eid, f"{ev.name}@frame{k}", Image.fromarray(rgb)))

    scene_desc = {}
    if visual_items:
        imgs = [v[2] for v in visual_items]
        texts = ([context_text] if context_text else []) + DISTRACTORS + SCENES
        sims = _clip_scores(imgs, texts)
        if sims is not None:
            off = 1 if context_text else 0
            for eid in {v[0] for v in visual_items}:
                rows = [i for i, v in enumerate(visual_items) if v[0] == eid]
                sc = sims[rows][:, off + len(DISTRACTORS):].mean(0)
                top = np.argsort(-sc)[:3]
                scene_desc[eid] = [SCENES[i] for i in top]
                results[eid]["scene"] = scene_desc[eid]
            if context_text:
                for eid in {v[0] for v in visual_items}:
                    rows = [i for i, v in enumerate(visual_items) if v[0] == eid]
                    cap = float(sims[rows, 0].max())
                    dis = sims[rows][:, 1:1 + len(DISTRACTORS)].max(1)
                    rank_ok = float((sims[rows, 0] > dis).mean())
                    results[eid]["caption_similarity"] = round(cap, 3)
                    seen = ", ".join(scene_desc.get(eid, []))
                    if cap < 0.19 and rank_ok < 0.5:
                        add("Caption does not match visual content", "inconsistent", ["text", results[eid]["evidence"].modality], [eid],
                            f"Image–text similarity between the claim and {results[eid]['evidence'].name} is low ({cap:.2f}); "
                            f"generic unrelated captions score higher. The visual most resembles: {seen}.",
                            0.65, "xmodal_caption_visual",
                            alternatives=["CLIP misses specifics (names, places, dates) and abstract claims — low similarity can be benign."])
                    elif cap >= 0.24 and rank_ok >= 0.5:
                        add("Caption consistent with visual content", "consistent", ["text", results[eid]["evidence"].modality], [eid],
                            f"Claim and visual content align (similarity {cap:.2f}). Visual resembles: {seen}.",
                            0.5, "xmodal_caption_visual",
                            alternatives=["Matching the topic does not verify the specific event, place or date."])

    # --- dates: claim vs capture metadata
    if claims:
        claim_years = set(claims["years"]) | {int(d["date"][:4]) for d in claims["dates"]}
        for eid, r in results.items():
            ev = r["evidence"]
            md = None
            if ev.modality == "image":
                ex = (r.get("metadata") or {}).get("exif", {})
                md = parse_media_date(ex.get("DateTimeOriginal") or ex.get("DateTime"))
            elif ev.modality == "video":
                md = parse_media_date((r.get("metadata") or {}).get("creation_time"))
            if not md:
                continue
            r["capture_date"] = md.isoformat()
            if claim_years and md.year not in claim_years:
                add("Claimed date contradicts capture metadata", "inconsistent", ["text", ev.modality], [eid],
                    f"Text refers to {', '.join(map(str, sorted(claim_years)))} but {ev.name} metadata says it was captured on {md.isoformat()}.",
                    0.7, "xmodal_dates", alternatives=["Camera clock may be wrong; metadata can be edited.",
                                                       "Re-encoding can reset creation_time to the upload date."])
            elif claims["recency_terms"] and (dt.date.today() - md).days > 30:
                add("'Recent' claim but old capture date", "inconsistent", ["text", ev.modality], [eid],
                    f"Text says \"{claims['recency_terms'][0]}\" but {ev.name} metadata dates it to {md.isoformat()} "
                    f"({(dt.date.today() - md).days} days ago) — consistent with old media recirculated as new.",
                    0.65, "xmodal_dates", alternatives=["Metadata may reflect an earlier camera clock or file copy date."])
            elif claim_years and md.year in claim_years:
                add("Claimed date consistent with metadata", "consistent", ["text", ev.modality], [eid],
                    f"Capture date {md.isoformat()} matches the year in the claim.", 0.35, "xmodal_dates",
                    alternatives=["Metadata can be forged to match."])

    # --- claim vs transcript
    for eid, r in results.items():
        tr = (r.get("transcript") or {}).get("text", "")
        if not (context_text and tr and len(tr.split()) >= 8):
            continue
        tclaims = extract_claims(tr)
        ck, tk = set(claims["keywords"]), set(tclaims["keywords"])
        overlap = len(ck & tk) / max(1, len(ck))
        sim = _clip_text_sim(context_text, tr[:300])
        miss_ent = [e for e in claims["entities"] if e.lower() not in tr.lower()]
        miss_num = [n for n in claims["numbers"] if n.split()[0] not in tr]
        about_speech = bool(SPEECH_CLAIM.search(context_text))
        r["claim_transcript"] = {"keyword_overlap": round(overlap, 2), "text_similarity": None if sim is None else round(sim, 2),
                                 "claim_entities_not_spoken": miss_ent, "claim_numbers_not_spoken": miss_num,
                                 "claim_is_about_speech": about_speech}
        # A caption that only describes the scene ("a man in a white t-shirt…") is not expected to be spoken,
        # so a missing overlap is only evidence when the claim is about what someone SAID.
        if not about_speech:
            continue
        if overlap < 0.12 and (sim is None or sim < 0.75):
            add("Claim not supported by what is said", "inconsistent", ["text", "audio"], [eid],
                f"Only {overlap * 100:.0f}% of the claim's key terms occur in the transcript"
                + (f"; names not spoken: {', '.join(miss_ent[:5])}" if miss_ent else "") + ".",
                0.5, "xmodal_caption_transcript",
                alternatives=["Speech may be in another language than the caption.", "ASR errors on noisy audio."])
        elif overlap >= 0.3:
            add("Claim consistent with spoken content", "consistent", ["text", "audio"], [eid],
                f"{overlap * 100:.0f}% of claim key terms appear in the transcript.", 0.35, "xmodal_caption_transcript")

    # --- temporal co-location within a video: visual cut + audio anomaly at the same time
    for eid, r in results.items():
        if r["evidence"].modality != "video":
            continue
        cuts = [c["time"] for c in r["metrics"].get("discontinuities", [])]
        a_times, a_ids = [], []
        dur = float(r["metrics"].get("duration_s") or 0) or 1e9
        for ind in r["indicators"]:
            # only localized audio EDIT evidence; a classifier's "synthetic" segment spanning the clip matches any cut
            if ind.modality.startswith("audio") and ind.detector in ("audio_digital_silence", "audio_splice"):
                loc = ind.location
                if loc.get("type") == "points":
                    a_times += [(t, t) for t in loc["times"]]; a_ids.append(ind.id)
                elif loc.get("type") == "segments":
                    segs = [(s[0], s[1]) for s in loc["segments"] if (s[1] - s[0]) < 0.4 * dur]
                    if segs:
                        a_times += segs; a_ids.append(ind.id)
        hits = [c for c in cuts if any(s - 0.6 <= c <= e + 0.6 for s, e in a_times)]
        vis_ids = [i.id for i in r["indicators"] if i.detector == "video_temporal"]
        if hits:
            add("Visual cut and audio anomaly at the same moment", "corroborates", ["video", "audio"], [eid],
                f"At {', '.join(f'{h:.1f}s' for h in hits[:6])} a picture discontinuity coincides with an audio splice/silence/"
                "synthetic segment. Two independent channels agree on an edit point.",
                min(1.0, 0.55 + 0.1 * len(hits)), "xmodal_corroboration", linked=vis_ids + a_ids,
                location={"type": "points", "times": hits})
        # synthetic voice while visuals look clean -> dubbing hypothesis
        audio_syn = [i for i in r["indicators"] if i.detector == "audio_ml" and i.supports == "manipulation"]
        vis_syn = [i for i in r["indicators"] if i.detector in ("ml_image_ai", "ml_face_deepfake") and i.supports == "manipulation"]
        if audio_syn and not vis_syn:
            add("Synthetic-sounding voice over natural-looking video", "inconsistent", ["video", "audio"], [eid],
                "The audio track scores as synthetic while frames do not — consistent with a cloned/dubbed voice laid over real footage.",
                0.55, "xmodal_corroboration", linked=[i.id for i in audio_syn],
                alternatives=["The audio detector may be a false positive on compressed speech."])
        if audio_syn and vis_syn:
            add("Both face and voice flagged as synthetic", "corroborates", ["video", "audio"], [eid],
                "Independent visual and audio detectors both flag synthesis — typical of talking-head deepfakes.",
                0.75, "xmodal_ml_agreement", linked=[i.id for i in audio_syn + vis_syn])

    # --- cross-file visual provenance (same picture appears in two files)
    hashes = []
    for eid, name, img in visual_items:
        hashes.append((eid, name, dhash(img)))
    seen = set()
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            a, b = hashes[i], hashes[j]
            if a[0] == b[0] or (a[0], b[0]) in seen:
                continue
            dist = bin(a[2] ^ b[2]).count("1")
            if dist <= 10:
                seen.add((a[0], b[0]))
                ra, rb = results[a[0]], results[b[0]]
                fa = [x for x in ra["indicators"] if x.supports == "manipulation" and x.score * x.reliability > 0.25]
                fb = [x for x in rb["indicators"] if x.supports == "manipulation" and x.score * x.reliability > 0.25]
                detail = f"{a[1]} and {b[1]} show the same scene (perceptual-hash distance {dist}/64)."
                if bool(fa) != bool(fb):
                    detail += " One copy carries manipulation indicators and the other does not — compare them side by side to find the altered version."
                add("Same visual content across files", "links", [ra["evidence"].modality, rb["evidence"].modality],
                    [a[0], b[0]], detail, 0.3 if bool(fa) == bool(fb) else 0.6, "xmodal_corroboration")
    return {"findings": findings, "indicators": inds, "claims": claims, "scenes": scene_desc}
