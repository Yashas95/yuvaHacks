"""Video forensics: container metadata, temporal consistency, per-frame ML, faces, audio track."""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import models
from .audio_forensics import analyze_audio
from .config import RELIABILITY as R, VALIDATED_RELIABILITY, VIDEO_MAX_SAMPLED_FRAMES, VIDEO_TEMPORAL_MAX_FRAMES
from .evidence import Indicator
from .provenance import indicators as provenance_indicators, scan as scan_provenance
from .image_forensics import AI_SIGNATURES, EDITORS, _draw_boxes, detect_faces, ml_face_scores, ml_image_score
from .media_io import has_audio, probe

VIDEO_AI = ["sora", "runway", "kling", "pika", "veo", "heygen", "synthesia", "d-id", "hedra", "luma", "dream machine",
            "hailuo", "minimax", "invideo", "deepfacelab", "faceswap", "roop", "facefusion"]
VIDEO_EDITORS = ["capcut", "premiere", "davinci", "resolve", "final cut", "imovie", "inshot", "vn video", "kinemaster",
                 "filmora", "after effects", "handbrake", "lavf", "lavc"]


def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def container_indicators(ev, info, nid):
    out, meta = [], {}
    fmt = info.get("format", {})
    tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
    for s in info.get("streams", []):
        for k, v in (s.get("tags") or {}).items():
            tags.setdefault(f"{s.get('codec_type')}.{k.lower()}", str(v))
    meta["tags"] = tags
    raw = ev.path.read_bytes()
    rawtxt = (raw[:8 << 20] + raw[-(2 << 20):]).lower().decode("latin-1", "ignore")
    tagtxt = " ".join(tags.values()).lower()
    # In compressed bytes short words occur by chance, so only long, unambiguous strings are searched there;
    # short names are only trusted inside readable metadata tags, as whole words.
    ai = {s for s in VIDEO_AI + AI_SIGNATURES if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", tagtxt)}
    ai |= {s for s in VIDEO_AI + AI_SIGNATURES if len(s) >= 9 and s in rawtxt}
    ai = sorted(ai)
    ed = sorted({s for s in VIDEO_EDITORS + EDITORS if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", tagtxt)})
    blob = rawtxt
    meta["ai_signatures"], meta["editors"] = ai, ed
    prov = scan_provenance(ev.path)
    meta["provenance"] = prov
    meta["c2pa_present"] = prov["c2pa_present"] or "c2pa" in blob
    out += provenance_indicators(ev, prov, nid, "video", R)        # Content Credentials: Veo, Sora, Firefly…
    meta["ai_signatures"] = ai = ai + ["C2PA digitalSourceType: " + t for t in prov["ai_source_types"]]
    if ai and not prov["ai_source_types"]:
        out.append(Indicator(nid(), ev.id, "video", "metadata_ai_signature", "Generator / face-swap tool signature",
                             "synthetic", f"Container metadata mentions: {', '.join(ai)}.",
                             "AI video tools and face-swap software often stamp their name in the file.",
                             0.9, R["metadata_ai_signature"], {"type": "file"},
                             ["A keyword can appear by coincidence in a title/comment; verify the field it came from."]))
    if ed:
        out.append(Indicator(nid(), ev.id, "video", "video_metadata", "Re-encoded / edited container", "editing",
                             f"Encoder/handler tags: {', '.join(ed)}.",
                             "The video was processed by an editor or re-encoder after recording.",
                             0.4, R["video_metadata"], {"type": "file"},
                             ["Almost every shared video is re-encoded by the platform (WhatsApp, Instagram use ffmpeg/Lavf)."]))
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    try:
        dv, da = float(v.get("duration", 0)), float(a.get("duration", 0))
        meta["video_duration"], meta["audio_duration"] = dv, da
        if dv and da and abs(dv - da) > 0.75:
            out.append(Indicator(nid(), ev.id, "video", "video_metadata", "Audio/video track length mismatch", "splice",
                                 f"Video track {dv:.2f}s vs audio track {da:.2f}s ({abs(dv - da):.2f}s difference).",
                                 "When audio is replaced or dubbed onto a video, track lengths often stop matching.",
                                 0.45, R["video_metadata"], {"type": "file"},
                                 ["Some cameras and screen recorders legitimately pad or trim one track."]))
    except Exception:
        pass
    meta["creation_time"] = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
    meta["device"] = " ".join(x for x in [tags.get("com.apple.quicktime.make"), tags.get("com.apple.quicktime.model")] if x)
    return out, meta


def find_static_overlay(frames: dict):
    """Find a logo / watermark that stays fixed in a corner while the rest of the picture changes.

    Gemini (Veo), Kling, Hailuo and Meta AI stamp a visible mark in a corner; so do TV channels and apps.
    Returns (bbox, corner_name, median_rgb_crop) or None. Needs a moving picture: in a fully static shot
    every pixel is 'static', so nothing is reported.
    """
    ks = sorted(frames)
    if len(ks) < 8:
        return None
    shapes = {frames[k].shape for k in ks}
    if len(shapes) != 1:
        return None
    rgb = np.stack([frames[k] for k in ks])
    g = rgb.mean(axis=3).astype(np.float32)
    H, W = g.shape[1:]
    std = g.std(0)
    med = np.median(g, 0)
    # Key cue: in the temporal MEDIAN a moving background blurs away, while an overlay (even a semi-transparent
    # one) stays sharp. Static scenery also stays sharp, so the area around the mark must be blurred/moving.
    edge = cv2.magnitude(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    moving = std > 6.0
    if moving.mean() < 0.25:          # mostly static footage: cannot separate an overlay from the scene
        return None
    bh, bw = int(H * 0.22), int(W * 0.32)
    zones = {"top-left": (0, 0), "top-right": (0, W - bw), "bottom-left": (H - bh, 0), "bottom-right": (H - bh, W - bw)}
    best = None
    for name, (y0, x0) in zones.items():
        ez = edge[y0:y0 + bh, x0:x0 + bw]
        cand = (ez > 120).astype(np.uint8)
        if cand.sum() < 30:
            continue
        cand = cv2.dilate(cand, np.ones((5, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
        for j in range(1, n):
            x, y, w, h, area = stats[j]
            if not (0.001 * H * W <= w * h <= 0.04 * H * W) or w < 14 or h < 8:
                continue
            X, Y = x0 + x, y0 + y
            inner = edge[Y:Y + h, X:X + w]
            pad = max(10, h)
            ys, ye, xs, xe = max(0, Y - pad), min(H, Y + h + pad), max(0, X - pad), min(W, X + w + pad)
            ring_mask = np.ones((ye - ys, xe - xs), bool)
            ring_mask[Y - ys:Y - ys + h, X - xs:X - xs + w] = False
            if ring_mask.sum() < 50:
                continue
            ring_edge = edge[ys:ye, xs:xe][ring_mask]
            ring_moving = moving[ys:ye, xs:xe][ring_mask].mean()
            sharp_in = float(np.percentile(inner, 90))
            sharp_out = float(np.percentile(ring_edge, 90)) + 1.0
            # the mark must be much sharper than its blurred surroundings, and the surroundings must be moving
            if ring_moving < 0.6 or sharp_in / sharp_out < 4.0:
                continue
            strength = sharp_in / sharp_out
            if best is None or strength > best[0]:
                best = (strength, (int(X), int(Y), int(w), int(h)), name)
    if not best:
        return None
    X, Y, w, h = best[1]
    crop = np.median(rgb[:, Y:Y + h, X:X + w], axis=0).astype(np.uint8)
    return best[1], best[2], crop


def analyze_video(ev, case, nid) -> dict:
    info = probe(ev.path)
    inds, cmeta = container_indicators(ev, info, nid)
    cap = cv2.VideoCapture(str(ev.path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scan_n = min(total, VIDEO_TEMPORAL_MAX_FRAMES) if total > 0 else VIDEO_TEMPORAL_MAX_FRAMES
    sample_idx = set(np.linspace(0, max(scan_n - 1, 0), min(VIDEO_MAX_SAMPLED_FRAMES, max(scan_n, 1))).astype(int).tolist())
    diffs, frames, prev, i = [], {}, None, 0
    while i < scan_n:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(fr, (160, max(1, int(160 * fr.shape[0] / fr.shape[1])))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
        if i in sample_idx or (len(diffs) > 2 and diffs[-1] > 25):
            s = min(1.0, 720 / max(fr.shape[:2]))
            frames[i] = cv2.cvtColor(cv2.resize(fr, None, fx=s, fy=s) if s < 1 else fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    n = len(diffs)
    d = np.array(diffs)
    times = np.arange(n) / fps
    metrics = {"fps": round(fps, 2), "frames_scanned": n, "frames_total": total, "resolution": f"{W}x{H}",
               "duration_s": round(n / fps, 2), **{k: v for k, v in cmeta.items() if k != "tags"}}

    # temporal discontinuities
    cuts = []
    if n > 10:
        med = np.median(d[1:]) + 1e-3
        mad = np.median(np.abs(d[1:] - med)) + 0.2
        z = (d - med) / (1.4826 * mad)
        cuts = [int(k) for k in np.where((z > 10) & (d > 4 * med) & (d > 6))[0] if k > 0]
        # collapse neighbours
        cuts = [c for j, c in enumerate(cuts) if j == 0 or c - cuts[j - 1] > fps * 0.3]
        edge = fps * 0.25   # recorder start-up / stop and exposure settling are not edits
        cuts = [c for c in cuts if edge <= c <= len(d) - edge]
    metrics["discontinuities"] = [{"frame": c, "time": round(c / fps, 2)} for c in cuts]
    if cuts:
        inds.append(Indicator(nid(), ev.id, "video", "video_temporal", "Abrupt temporal discontinuities", "splice",
                              f"{len(cuts)} abrupt frame-to-frame change(s) at " + ", ".join(_fmt(c / fps) for c in cuts[:8]) + ".",
                              "A sudden jump in picture content marks a cut. In footage presented as one continuous take, "
                              "cuts indicate removed or inserted material.",
                              min(1.0, 0.25 + 0.1 * len(cuts)), R["video_temporal"],
                              {"type": "frames", "frames": [{"frame": c, "time": round(c / fps, 2)} for c in cuts[:20]]},
                              ["Ordinary edited videos (news packages, reels) contain intentional cuts.",
                               "Camera flashes and fast pans produce spikes."]))
    # duplicated / frozen frames inside motion
    if n > 30:
        moving = np.convolve(d, np.ones(15) / 15, mode="same") > 1.5
        dup = (d < 0.02) & moving
        dup[0] = False
        runs, start = [], None
        for k, v in enumerate(np.append(dup, False)):
            if v and start is None:
                start = k
            elif not v and start is not None:
                runs.append((start, k)); start = None
        long_runs = [r for r in runs if r[1] - r[0] >= 2]
        metrics["frozen_runs"] = [{"from": round(a / fps, 2), "to": round(b / fps, 2)} for a, b in long_runs]
        if len(long_runs) >= 1 or dup.sum() > max(5, 0.05 * n):
            inds.append(Indicator(nid(), ev.id, "video", "video_duplicate_frames", "Duplicated / frozen frames during motion",
                                  "splice", f"{int(dup.sum())} identical consecutive frames while the scene is moving"
                                  + (f"; frozen runs at {', '.join(_fmt(a / fps) for a, _ in long_runs[:6])}." if long_runs else "."),
                                  "Frame duplication inside motion can indicate frame-rate padding after removing or inserting frames.",
                                  min(1.0, 0.3 + 0.1 * len(long_runs)), R["video_duplicate_frames"],
                                  {"type": "frames", "frames": [{"frame": a, "time": round(a / fps, 2)} for a, _ in long_runs[:20]]},
                                  ["Variable-frame-rate phone video and screen recordings duplicate frames normally."]))

    # per-frame ML
    rows = []
    for k in sorted(frames):
        rgb = frames[k]
        p, _ = ml_image_score(Image.fromarray(rgb))
        faces = detect_faces(rgb, 2)
        fs = ml_face_scores(rgb, faces)
        fp = max([f["p_fake"] for f in fs if f["p_fake"] is not None], default=None)
        rows.append({"frame": k, "time": round(k / fps, 2), "p_ai": None if p is None else round(p, 3),
                     "faces": len(faces), "p_face_fake": fp, "face_boxes": fs})
    metrics["frame_scores"] = [{k: v for k, v in r.items() if k != "face_boxes"} for r in rows]
    metrics["image_ai_model"] = models.STATUS.get("image_ai", {}).get("model")
    metrics["face_model"] = models.STATUS.get("face_deepfake", {}).get("model")

    pa = [r["p_ai"] for r in rows if r["p_ai"] is not None]
    pf = [r["p_face_fake"] for r in rows if r["p_face_fake"] is not None]
    flagged_rows = []
    validated = models.image_ai_validated()
    rel_ai = VALIDATED_RELIABILITY if validated else R["ml_image_ai"]
    metrics["image_ai_validated"] = validated
    if pa:
        hi = [r for r in rows if r["p_ai"] is not None and r["p_ai"] >= 0.6]
        frac = len(hi) / len(pa)
        med = float(np.median(pa))
        # a video counts as flagged only when the signal is consistent over time, not from isolated spikes
        if hi and (frac >= 0.25 or med >= 0.5):
            flagged_rows += hi
            inds.append(Indicator(nid(), ev.id, "video", "ml_image_ai", "Frames resemble AI-generated imagery", "synthetic",
                                  f"{len(hi)}/{len(pa)} sampled frames ({frac * 100:.0f}%) scored ≥60% synthetic "
                                  f"(median {med * 100:.0f}%) by {metrics['image_ai_model']}"
                                  + (" — a model validated on real vs AI samples." if validated else "."),
                                  "Frame-level generator fingerprints, consistent across time, point to synthetic video.",
                                  float(0.5 * med + 0.5 * max(pa)) * (0.6 + 0.4 * frac), rel_ai,
                                  {"type": "frames", "frames": [{"frame": r["frame"], "time": r["time"], "p": r["p_ai"]} for r in hi[:20]]},
                                  ["Image detectors are not trained on video compression; isolated spikes are weak evidence.",
                                   "Heavy filters, CGI, animation and game footage can look 'generated' to a classifier."],
                                  validated=validated))
        elif med < 0.3:
            inds.append(Indicator(nid(), ev.id, "video", "ml_image_ai", "Frame classifier: leans authentic", "synthetic",
                                  f"Median synthetic probability {med * 100:.0f}% over {len(pa)} frames"
                                  + (f" ({len(hi)} isolated frame spike(s) ignored)." if hi else "."),
                                  "No consistent generator signature across frames.", 1 - med,
                                  rel_ai * 0.5, {"type": "file"}, ["Not proof of authenticity; new generators can evade detectors."],
                                  supports="authenticity", validated=validated))
    if pf:
        hi = [r for r in rows if r["p_face_fake"] is not None and r["p_face_fake"] >= 0.6]
        if hi:
            flagged_rows += [r for r in hi if r not in flagged_rows]
            inds.append(Indicator(nid(), ev.id, "video", "ml_face_deepfake", "Face-swap / deepfake face detected", "synthetic",
                                  f"Faces in {len(hi)}/{len(pf)} frames scored ≥60% deepfake (max {max(pf) * 100:.0f}%).",
                                  "Face-region texture/blending statistics match face-swap or talking-head synthesis.",
                                  float(0.5 * max(pf) + 0.5 * np.mean([r['p_face_fake'] for r in hi])), R["ml_face_deepfake"],
                                  {"type": "frames", "frames": [{"frame": r["frame"], "time": r["time"], "p": r["p_face_fake"],
                                                                 "boxes": [f["bbox"] for f in r["face_boxes"]]} for r in hi[:20]]},
                                  ["Small faces, motion blur and beauty filters cause false positives."]))

    # visible watermark / logo overlay (provenance lead for a human to read)
    visuals = {}
    try:
        ov = find_static_overlay(frames)
    except Exception:
        ov = None
    metrics["overlay"] = None
    if ov:
        (X, Y, w, h), corner, crop = ov
        metrics["overlay"] = {"bbox": [X, Y, w, h], "corner": corner}
        big = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), None, fx=max(1, 240 // max(w, 1)), fy=max(1, 240 // max(w, 1)),
                         interpolation=cv2.INTER_NEAREST)
        op = case.artifacts_dir / f"{ev.id}_overlay.png"
        cv2.imwrite(str(op), big)
        visuals["Corner overlay (zoomed) — read it"] = case.rel(op)
        inds.append(Indicator(nid(), ev.id, "video", "video_watermark", "Static logo / watermark in corner", "metadata",
                              f"A fixed overlay sits in the {corner} corner (x={X}, y={Y}, {w}×{h}px in sampled frames) "
                              "for the whole video while the picture around it moves.",
                              "AI video tools such as Gemini/Veo, Kling, Hailuo and Meta AI stamp a visible mark in a corner. "
                              "If the zoomed overlay names an AI tool, that is strong provenance evidence; a human must read it.",
                              0.4, R["video_watermark"],
                              {"type": "regions", "regions": [{"label": f"overlay ({corner})", "bbox": [X, Y, w, h]}]},
                              ["TV channel logos, app watermarks (TikTok, CapCut, Instagram) and static captions look the same.",
                               "Generator watermarks are often cropped or blurred out before sharing, so absence proves nothing."]))

    # visuals: timeline + flagged frame strip
    tl = case.artifacts_dir / f"{ev.id}_timeline.png"
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    ax[0].plot(times, d, lw=0.8, color="#334")
    for c in cuts:
        ax[0].axvline(c / fps, color="#ff8a00", lw=1.5)
    ax[0].set_ylabel("frame Δ")
    ax[0].set_title(f"{ev.name} — temporal consistency (orange = discontinuity) and per-frame detector scores", fontsize=10)
    if pa:
        ax[1].plot([r["time"] for r in rows if r["p_ai"] is not None], pa, "o-", ms=3, label="AI-generated (frame)", color="#1c64f2")
    if pf:
        ax[1].plot([r["time"] for r in rows if r["p_face_fake"] is not None], pf, "s-", ms=3, label="Deepfake (face)", color="#e02424")
    ax[1].axhline(0.6, ls="--", color="gray", lw=0.8)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("P(synthetic)")
    ax[1].set_xlabel("time (s)")
    if pa or pf:
        ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(tl, dpi=110)
    plt.close(fig)
    visuals["Temporal timeline"] = case.rel(tl)

    show = flagged_rows[:6] or [r for r in rows][:: max(1, len(rows) // 6)][:6]
    cut_frames = [c for c in cuts if c in frames][:3]
    tiles = []
    for r in show:
        bgr = cv2.cvtColor(frames[r["frame"]], cv2.COLOR_RGB2BGR).copy()
        _draw_boxes(bgr, [{"bbox": f["bbox"], "label": "" if f["p_fake"] is None else f"{f['p_fake'] * 100:.0f}%"} for f in r["face_boxes"]], (0, 0, 255))
        ai_txt = "" if r["p_ai"] is None else "{:.0f}%".format(r["p_ai"] * 100)
        lab = f"{_fmt(r['time'])}  AI {ai_txt}"
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(bgr, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(bgr)
    for c in cut_frames:
        bgr = cv2.cvtColor(frames[c], cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), (0, 120, 255), -1)
        cv2.putText(bgr, f"{_fmt(c / fps)}  DISCONTINUITY", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(bgr)
    if tiles:
        th = 240
        tiles = [cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th)) for t in tiles]
        rows_img, row, wsum = [], [], 0
        for t in tiles:
            if wsum + t.shape[1] > 1400 and row:
                rows_img.append(row); row, wsum = [], 0
            row.append(t); wsum += t.shape[1]
        rows_img.append(row)
        maxw = max(sum(t.shape[1] for t in r) for r in rows_img)
        canvas = np.full((th * len(rows_img), maxw, 3), 255, np.uint8)
        for ri, r in enumerate(rows_img):
            x = 0
            for t in r:
                canvas[ri * th:(ri + 1) * th, x:x + t.shape[1]] = t
                x += t.shape[1]
        strip = case.artifacts_dir / f"{ev.id}_frames.jpg"
        cv2.imwrite(str(strip), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        visuals["Flagged / key frames"] = case.rel(strip)

    transcript, audio_metrics = {"text": "", "chunks": []}, {}
    if has_audio(info):
        ar = analyze_audio(ev, case, nid, source_path=ev.path, label="audio track")
        inds += ar["indicators"]
        visuals.update(ar["visuals"])
        transcript, audio_metrics = ar["transcript"], ar["metrics"]
    metrics["audio"] = audio_metrics
    return {"indicators": inds, "visuals": visuals, "metrics": metrics, "metadata": cmeta,
            "transcript": transcript, "keyframes": {k: frames[k] for k in list(sorted(frames))[:: max(1, len(frames) // 8)]}}
