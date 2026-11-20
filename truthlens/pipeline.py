"""End-to-end orchestration: ingest -> analyze -> correlate -> score -> report."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from .acquisition import limitation as acquisition_limitation, skipped_image_checks
from . import models
from .ai_text import check_ai_writing
from .audio_forensics import analyze_audio
from .config import VERSION
from .correlate import correlate
from .document_forensics import analyze_document
from .evidence import Case, Findings, now
from .image_forensics import analyze_image
from .report import render_html
from .scoring import aggregate, recommended_checks
from .text_forensics import analyze_text
from .video_forensics import analyze_video


def _nid_factory(prefix):
    n = [0]

    def nid():
        n[0] += 1
        return f"{prefix}-{n[0]:02d}"
    return nid


def run_case(files: list[tuple[Path, str]], context_text: str = "", progress=None, case: Case | None = None,
             acquisition: dict | None = None):
    """Analyze uploaded files. `acquisition` (truthlens/acquisition.py) says how the browser extension obtained
    them; omit it for ordinary uploads."""
    progress = progress or (lambda pct, msg: None)
    t0 = time.time()
    case = case or Case()
    progress(3, "Securing evidence (hashing, read-only copies)")
    for path, name in files:
        case.add_file(Path(path), name, acquisition=acquisition)
    extra_text = []
    for ev in case.evidence:
        if ev.modality == "text":
            try:
                extra_text.append(ev.path.read_text(encoding="utf-8", errors="ignore")[:5000])
            except Exception:
                pass
    full_context = "\n".join([t for t in [context_text] + extra_text if t and t.strip()])
    case.set_context(context_text)

    results, families, limitations = {}, set(), []
    media = [e for e in case.evidence if e.modality in ("image", "video", "audio", "document", "text")]
    for k, ev in enumerate(media):
        base = 8 + int(70 * k / max(1, len(media)))
        progress(base, f"Analyzing {ev.name} ({ev.modality})")
        case.add_log("analysis_started", f"{ev.id} {ev.name} — {ev.modality} pipeline")
        nid = _nid_factory(ev.id)
        ts = time.time()
        try:
            if ev.modality == "image":
                r = analyze_image(ev, case, nid)
                families |= {"metadata", "ela", "noise", "copy_move", "fft"} - skipped_image_checks(ev)
                if r["metrics"]["ml"].get("image_ai_p") is not None:
                    families.add("ml_image_ai")
                if (r["metadata"].get("format") or "").upper() not in ("JPEG", "MPO"):
                    limitations.append(f"{ev.name}: not a JPEG, so Error Level Analysis is less informative.")
            elif ev.modality == "video":
                r = analyze_video(ev, case, nid)
                families |= {"video_metadata", "video_temporal", "video_duplicate_frames"} - set(r["metrics"].get("skipped_checks", []))
                if any(f.get("p_ai") is not None for f in r["metrics"].get("frame_scores", [])):
                    families.add("ml_image_ai")
                if any(f.get("p_face_fake") is not None for f in r["metrics"].get("frame_scores", [])) \
                        and not r["metrics"].get("face_model_unreliable"):
                    families.add("ml_face_deepfake")
                if r["metrics"]["audio"]:
                    families |= {"audio_splice", "audio_digital_silence", "audio_bandwidth"}
                    if r["metrics"]["audio"].get("segment_scores"):
                        families.add("audio_ml")
                if r["metrics"]["frames_total"] > r["metrics"]["frames_scanned"]:
                    limitations.append(f"{ev.name}: only the first {r['metrics']['frames_scanned']} of "
                                       f"{r['metrics']['frames_total']} frames were scanned for temporal checks.")
                limitations.append(f"{ev.name}: ML detectors ran on {len(r['metrics'].get('frame_scores', []))} sampled frames, not every frame.")
            elif ev.modality == "document":
                r = analyze_document(ev, case, nid)
                families |= {"document_generator", "document_history", "text_ai_vocabulary"}
                if ev.path.suffix.lower() == ".pdf":
                    limitations.append(f"{ev.name}: the text inside PDF files is not read yet, so only the file's "
                                       "properties were checked; upload the .docx for a check of the wording too.")
            elif ev.modality == "text":
                out = Findings(ev, nid, "text")
                w = check_ai_writing(out, ev.path.read_text(encoding="utf-8", errors="ignore"))
                r = {"indicators": out, "metrics": {"writing": w}, "metadata": {}, "visuals": {}}
                families.add("text_ai_vocabulary")
            else:
                span = 70 / max(1, len(media))
                r = analyze_audio(ev, case, nid, step=lambda f, m, b=base, sp=span, n=ev.name:
                                  progress(b + int(sp * f), f"{n}: {m}"))
                families |= {"audio_splice", "audio_digital_silence", "audio_bandwidth"}
                if r["metrics"].get("segment_scores"):
                    families.add("audio_ml")
            r["evidence"] = ev
            results[ev.id] = r
            case.add_log("analysis_completed", f"{ev.id}: {len(r['indicators'])} indicator(s) in {time.time() - ts:.1f}s")
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            case.add_log("analysis_error", f"{ev.id}: {type(e).__name__}: {e}")
            limitations.append(f"{ev.name}: analysis failed ({type(e).__name__}: {e}); results for this file are incomplete.")
            results[ev.id] = {"evidence": ev, "indicators": [], "visuals": {}, "metrics": {"error": str(e), "trace": tb}}

    progress(80, "Analyzing accompanying text")
    text_res = None
    if full_context.strip():
        text_res = analyze_text("CTX", full_context, _nid_factory("CTX"))
        case.add_log("analysis_completed", f"Context text: {len(text_res['indicators'])} indicator(s)")
    progress(84, "Cross-modal correlation (CLIP, transcript, timestamps, dates)")
    x = correlate(case, results, full_context, _nid_factory("X"))
    case.add_log("correlation_completed", f"{len(x['findings'])} cross-modal finding(s)")
    if x["findings"]:
        families.add("cross_modal")

    all_inds = [i for r in results.values() for i in r["indicators"]] + x["indicators"] + (text_res["indicators"] if text_res else [])
    missing = [k for k, v in models.STATUS.items() if not v["model"]]
    for k in missing:
        limitations.append(f"Model '{k}' unavailable ({models.STATUS[k]['error']}); its indicators were skipped.")
    if not full_context.strip():
        limitations.append("No caption/claim text was provided, so text-vs-media consistency checks were not run.")
    limitations += [
        "Pretrained detectors were trained on public datasets and generalize imperfectly to new generators, filters and heavy compression.",
        "Indicator weights and score thresholds are expert-set heuristics, not calibrated on a labelled benchmark for this case type.",
        "Absence of indicators does not prove authenticity; skilled manipulation can evade every automated check.",
        "Social-media re-encoding strips metadata and erases fine traces; analyze the earliest available original when possible.",
    ]

    progress(90, "Scoring and building report")
    per_ev = {}
    for eid, r in results.items():
        fam = {i.detector for i in r["indicators"]} | (({"metadata", "ela", "noise"} - skipped_image_checks(r["evidence"]))
                                                       if r["evidence"].modality == "image" else set())
        per_ev[eid] = aggregate(r["indicators"] + [i for i in x["indicators"] if eid in i.evidence_id.split("+")], fam)
    overall = aggregate(all_inds, families, len(missing))
    unchecked = [e for e in case.evidence if e.id not in results]
    for e in unchecked:
        limitations.insert(0, f"{e.name}: TruthLens can't analyse this file type yet ({e.path.suffix or 'no extension'}), "
                              "so no checks ran on it. Supported: photos, videos, audio, documents (.docx, .pptx, .xlsx, .pdf) and text.")
        per_ev[e.id] = {**aggregate([], set()), "band": "Not analysed — file type not supported", "color": "amber",
                        "likelihood": 0, "range": [0, 0], "not_analysed": True}
    if unchecked and not results and not full_context.strip():
        overall.update(band="Not analysed — file type not supported", color="amber", likelihood=0, range=[0, 0], not_analysed=True)
    if overall.get("ml_only"):
        limitations.insert(0, "Finding rests on ML classifiers only — no independent forensic, metadata or temporal evidence. "
                              + ("A validated detector was confident, so the likelihood was capped at Moderate (not Strong)."
                                 if overall.get("validated_ml") else "Likelihood was capped and confidence set to Low."))
    from .config import UNRELIABLE_AUDIO_MODELS
    if any((r.get("metrics") or {}).get("audio_model_unreliable") or ((r.get("metrics") or {}).get("audio") or {}).get("audio_model_unreliable")
           for r in results.values()):
        mid = next((m for r in results.values() for m in [(r.get("metrics") or {}).get("audio_model")
                    or ((r.get("metrics") or {}).get("audio") or {}).get("audio_model")] if m), "")
        limitations.insert(0, f"Synthetic-voice classifier {mid} failed local evaluation ({UNRELIABLE_AUDIO_MODELS.get(mid, '')}), "
                              "so it is not run. AI-generated voices cannot currently be ruled out by ear-level analysis; "
                              "file labels (C2PA) and signal checks still run.")
    from .config import UNRELIABLE_FACE_MODELS
    def _face_off(r):
        m = r.get("metrics") or {}
        return m.get("face_model_unreliable") or (m.get("ml") or {}).get("face_model_unreliable")
    if any(_face_off(r) for r in results.values()):
        fid = models.STATUS.get("face_deepfake", {}).get("model") or next(iter(UNRELIABLE_FACE_MODELS))
        limitations.insert(0, f"Face-deepfake classifier {fid} did not separate real from AI faces in our evaluation "
                              f"({UNRELIABLE_FACE_MODELS.get(fid, '')}); its scores are shown but not counted.")
    for r in results.values():
        note = acquisition_limitation(r["evidence"].name, r["evidence"].acquisition)
        if note:
            limitations.insert(0, note)
    for r in results.values():
        tr = r.get("transcript") or {}
        if tr.get("truncated_at_s"):
            limitations.append(f"{r['evidence'].name}: speech was transcribed for the first {tr['truncated_at_s']:.0f} s only.")
    if models.image_ai_validated():
        from .config import VALIDATED_MODELS
        mid = models.STATUS["image_ai"]["model"]
        limitations.append(f"AI-image detector {mid} was validated on a small local set ({VALIDATED_MODELS[mid]}). "
                           "Its accuracy on other generators and real footage is not yet established.")
    integrity = case.verify_integrity()

    report = {
        "tool": {"name": "TruthLens", "version": VERSION, "models": models.status_lines()},
        "case": {"id": case.id, "created_at": case.created_at, "completed_at": now(),
                 "duration_s": round(time.time() - t0, 1), "context_text": context_text},
        "overall": overall,
        "evidence": [],
        "cross_modal": x["findings"],
        "context_analysis": None if not text_res else {"claims": text_res["claims"],
                                                        "indicators": [i.to_dict() for i in text_res["indicators"]]},
        "recommended_checks": recommended_checks(all_inds, {e.modality for e in case.evidence}),
        "limitations": list(dict.fromkeys(limitations)),
        "integrity": integrity,
        "log": case.log,
        "indicators": [i.to_dict() for i in all_inds],
    }
    for e in unchecked:
        results[e.id] = {"evidence": e, "indicators": [], "visuals": {}, "metrics": {"not_analysed": True}}
    for eid, r in results.items():
        ev = r["evidence"]
        report["evidence"].append({
            "id": ev.id, "name": ev.name, "modality": ev.modality, "sha256": ev.sha256, "size": ev.size,
            "received_at": ev.received_at, "stored_as": case.rel(ev.path), "acquisition": ev.acquisition,
            "assessment": per_ev[eid], "visuals": r.get("visuals", {}),
            "indicators": [i.to_dict() for i in r["indicators"]],
            "metrics": r.get("metrics", {}), "metadata": _clean_meta(r.get("metadata")),
            "transcript": r.get("transcript"), "scene": r.get("scene"),
            "caption_similarity": r.get("caption_similarity"), "capture_date": r.get("capture_date"),
            "claim_transcript": r.get("claim_transcript"),
        })
    case.add_log("report_generated", "report.json and report.html written")
    report["log"] = case.log
    case.save_json("report.json", report)
    (case.dir / "report.html").write_text(render_html(report, case.dir), encoding="utf-8")
    progress(100, "Done")
    return case, report


def _clean_meta(m):
    if not m:
        return None
    m = dict(m)
    m.pop("tags", None) if isinstance(m.get("tags"), dict) and len(m["tags"]) > 40 else None
    return m
