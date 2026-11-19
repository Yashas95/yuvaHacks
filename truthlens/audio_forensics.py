"""Audio / voice analysis: synthetic-speech classifier, splice & silence checks, bandwidth, transcript."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from . import models
from .config import AUDIO_CHUNK_SECONDS, RELIABILITY as R
from .evidence import Indicator
from .media_io import load_audio, probe


def _frames_db(y, sr, win=0.025, hop=0.010):
    n, h = int(sr * win), int(sr * hop)
    if len(y) < n:
        return np.array([]), h
    idx = np.arange(0, len(y) - n, h)
    fr = np.stack([y[i:i + n] for i in idx])
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    peak = np.abs(fr).max(1)
    return 20 * np.log10(rms + 1e-12), h, peak


def _runs(mask, min_len):
    runs, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def _merge(segs, gap=0.5):
    segs = sorted(segs)
    out = []
    for s in segs:
        if out and s[0] <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], s[1]), max(out[-1][2], s[2]))
        else:
            out.append(tuple(s))
    return out


def _refine_change(db, peak, t_rough, hop, span=3.0, side=1.5):
    """Pin the exact moment the background noise level changes (±0.1 s)."""
    fps = 16000 / hop
    best, best_t = 0.0, t_rough
    lo, hi = int((t_rough - span) * fps), int((t_rough + span) * fps)
    w = int(side * fps)
    for c in range(max(w, lo), min(len(db) - w, hi), 2):
        L, Rr = db[c - w:c][peak[c - w:c] >= 1e-4], db[c:c + w][peak[c:c + w] >= 1e-4]
        if len(L) < 20 or len(Rr) < 20:
            continue
        d = abs(np.percentile(Rr, 10) - np.percentile(L, 10))
        if d > best:
            best, best_t = d, c / fps
    return round(best_t, 2)


def transcribe(y16: np.ndarray) -> dict:
    pipe = models.asr()
    if pipe is None or len(y16) < 16000 * 0.5:
        return {"text": "", "chunks": [], "available": pipe is not None}
    inp = {"raw": y16, "sampling_rate": 16000}
    for kwargs in ({"return_timestamps": True, "chunk_length_s": 30}, {"return_timestamps": True}, {}):
        try:
            res = pipe(dict(inp), **kwargs)
            chunks = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"].strip()}
                      for c in res.get("chunks", []) if c.get("timestamp")]
            return {"text": res.get("text", "").strip(), "chunks": chunks, "available": True}
        except Exception as e:
            err = str(e)
    return {"text": "", "chunks": [], "available": True, "error": err}


def analyze_audio(ev, case, nid, source_path=None, label="audio", t_offset=0.0) -> dict:
    path = source_path or ev.path
    info = probe(path)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    y, sr = load_audio(path)
    y16, _ = load_audio(path, sr=16000)
    dur = len(y16) / 16000 if len(y16) else 0.0
    metrics = {"duration_s": round(dur, 2), "sample_rate": sr, "codec": a.get("codec_name"),
               "bit_rate": a.get("bit_rate") or info.get("format", {}).get("bit_rate"),
               "encoder": (info.get("format", {}).get("tags", {}) or {}).get("encoder")}
    inds, marks = [], []
    if dur < 0.5:
        return {"indicators": [], "visuals": {}, "metrics": metrics, "transcript": {"text": "", "chunks": []}}

    db, hop, peak = _frames_db(y16, 16000)
    t_of = lambda i: i * hop / 16000
    speech = db > (np.percentile(db, 90) - 25)
    metrics["speech_ratio"] = round(float(speech.mean()), 3)

    # 1) Digital silence: exact-zero gaps inside the recording (real microphones always have a noise floor)
    zero = peak < 1e-4
    inner = np.zeros_like(zero)
    sp_idx = np.where(speech)[0]
    if len(sp_idx):
        inner[sp_idx[0]:sp_idx[-1]] = True
    zruns = [(t_of(s), t_of(e)) for s, e in _runs(zero & inner, 5)]
    metrics["digital_silence_gaps"] = len(zruns)
    if zruns:
        tot = sum(e - s for s, e in zruns)
        inds.append(Indicator(nid(), ev.id, label, "audio_digital_silence", "Digital-silence gaps inside speech", "splice",
                              f"{len(zruns)} gap(s) of absolute digital silence ({tot:.2f}s total) between speech.",
                              "A live microphone recording always carries background noise. Perfectly silent gaps appear when "
                              "clips are joined in an editor or when speech is synthesized segment by segment.",
                              min(1.0, 0.45 + 0.1 * len(zruns)), R["audio_digital_silence"],
                              {"type": "segments", "segments": [[round(s + t_offset, 2), round(e + t_offset, 2)] for s, e in zruns[:12]]},
                              ["Noise gates / noise suppression (Zoom, Krisp, phone calls) produce digital silence legitimately."]))
        marks += [("silence", s, e) for s, e in zruns]

    # 2) Background-noise-floor jumps (possible splice points)
    win, hopw = 250, 50  # 2.5 s windows, 0.5 s hop -> each window normally contains a pause
    floors, times = [], []
    for i in range(0, max(0, len(db) - win), hopw):
        seg = db[i:i + win][peak[i:i + win] >= 1e-4]   # ignore digital silence (handled above)
        floors.append(np.percentile(seg, 5) if len(seg) > 20 else np.nan)
        times.append(t_of(i + win // 2))
    floors = np.array(floors)
    splices = []
    K = 4
    if np.isfinite(floors).sum() > 2 * K:
        cand = []
        for k in range(K, len(floors) - K + 1):
            L, Rr = floors[k - K:k], floors[k:k + K]
            L, Rr = L[np.isfinite(L)], Rr[np.isfinite(Rr)]
            if len(L) < 2 or len(Rr) < 2:
                continue
            jump = float(np.median(Rr) - np.median(L))
            if abs(jump) > 10 and np.std(L) < 5 and np.std(Rr) < 5:
                cand.append((k, jump))
        # keep local maxima, one per change point
        for k, jump in cand:
            if all(abs(jump) >= abs(j2) for k2, j2 in cand if abs(k2 - k) <= K):
                # 5th-percentile floor: an upward jump is first seen when the window START passes it,
                # a downward jump when the window END passes it.
                t_change = times[k] - 1.5 if jump > 0 else times[k] + 1.0
                t_change = _refine_change(db, peak, t_change, hop)
                if not splices or t_change - splices[-1][0] > 2:
                    splices.append((round(t_change, 2), round(jump, 1)))
    metrics["noise_floor_jumps"] = len(splices)
    if splices:
        inds.append(Indicator(nid(), ev.id, label, "audio_splice", "Abrupt background-noise changes", "splice",
                              f"{len(splices)} point(s) where the background noise level jumps by >10 dB: "
                              + ", ".join(f"{t + t_offset:.1f}s ({j:+.0f} dB)" for t, j in splices[:6]) + ".",
                              "The room tone of a single continuous recording is stable. A sudden change suggests audio from "
                              "different recordings was joined at that moment.",
                              min(1.0, 0.4 + 0.15 * len(splices)), R["audio_splice"],
                              {"type": "points", "times": [round(t + t_offset, 2) for t, _ in splices[:12]]},
                              ["Moving between rooms, a door/AC switching on, or the mic being moved also changes background noise."]))
        marks += [("splice", t, t) for t, _ in splices]

    # 3) Bandwidth
    f, psd = signal.welch(y, sr, nperseg=4096)
    pdb = 10 * np.log10(psd + 1e-20)
    pdb_s = np.convolve(pdb, np.ones(9) / 9, mode="same")
    keep = (f > 20) & (f < 0.97 * sr / 2)  # drop DC and the Nyquist edge bins (Welch/smoothing artifacts)
    f, pdb_s = f[keep], pdb_s[keep]
    ref = float(np.median(pdb_s[(f > 300) & (f < 3000)]))
    above = np.where(pdb_s > ref - 20)[0]
    cutoff = float(f[above[-1]]) if len(above) else 0.0
    tail = pdb_s[f > cutoff + 1000]
    sharp = len(tail) > 5 and float(np.median(tail)) < ref - 40
    metrics["effective_bandwidth_hz"] = int(cutoff)
    metrics["sharp_cutoff"] = bool(sharp)
    if sr >= 32000 and sharp and cutoff < 0.6 * (sr / 2) and any(lo <= cutoff <= hi for lo, hi in [(7000, 8800), (10000, 11600), (11600, 12600)]):
        inds.append(Indicator(nid(), ev.id, label, "audio_bandwidth", "Band-limited audio", "synthetic",
                              f"Energy stops sharply at ~{cutoff / 1000:.1f} kHz although the file is sampled at {sr / 1000:.1f} kHz.",
                              "Many TTS / voice-cloning vocoders generate audio at 16–24 kHz and are later up-sampled, "
                              "leaving an empty band above the original Nyquist frequency.",
                              0.45, R["audio_bandwidth"], {"type": "file"},
                              ["Phone calls, WhatsApp voice notes and low-bitrate MP3/AAC are band-limited too."]))

    # 4) Synthetic speech classifier, per chunk
    pipe = models.audio_deepfake()
    metrics["audio_model"] = models.STATUS.get("audio_deepfake", {}).get("model")
    seg_scores = []
    if pipe is not None:
        n = int(AUDIO_CHUNK_SECONDS * 16000)
        for s in range(0, max(1, len(y16) - n // 3), n):
            chunk = y16[s:s + n]
            if len(chunk) < 16000:
                continue
            fi, fj = s // hop, (s + len(chunk)) // hop
            if speech[fi:fj].mean() < 0.2:
                continue
            try:
                p = models.fake_probability(pipe({"raw": chunk, "sampling_rate": 16000}, top_k=5))
            except Exception:
                p = None
            if p is not None:
                seg_scores.append((s / 16000, (s + len(chunk)) / 16000, round(p, 3)))
    metrics["segment_scores"] = [[round(a + t_offset, 2), round(b + t_offset, 2), p] for a, b, p in seg_scores]
    if seg_scores:
        ps = np.array([p for *_, p in seg_scores])
        metrics["synthetic_mean_p"] = round(float(ps.mean()), 3)
        flagged = _merge([(a, b, p) for a, b, p in seg_scores if p >= 0.6])
        if flagged:
            mx = max(p for *_, p in flagged)
            inds.append(Indicator(nid(), ev.id, label, "audio_ml", "Synthetic / cloned speech classifier", "synthetic",
                                  f"{len(flagged)} segment(s) scored as synthetic (max {mx * 100:.0f}%, mean over all speech "
                                  f"{ps.mean() * 100:.0f}%) by {metrics['audio_model']}.",
                                  "The learned acoustic fingerprint of these segments resembles TTS / voice-conversion output.",
                                  float(0.6 * mx + 0.4 * ps.mean()), R["audio_ml"],
                                  {"type": "segments", "segments": [[round(a + t_offset, 2), round(b + t_offset, 2), p] for a, b, p in flagged]},
                                  ["Classifiers trained on public datasets generalize poorly to new voice-cloning tools and noisy audio.",
                                   "Heavy compression, music or overlapping speakers can produce false positives."]))
            marks += [("ml", a, b) for a, b, _ in flagged]
        elif ps.mean() < 0.35:
            inds.append(Indicator(nid(), ev.id, label, "audio_ml", "Speech classifier: leans natural", "synthetic",
                                  f"Mean synthetic probability {ps.mean() * 100:.0f}% across {len(ps)} speech segments.",
                                  "No strong synthetic-speech signature.", 1 - float(ps.mean()), R["audio_ml"] * 0.6,
                                  {"type": "file"}, ["High-quality modern voice clones can evade detection."], supports="authenticity"))

    tr = transcribe(y16)
    if t_offset:
        for c in tr["chunks"]:
            c["start"] = None if c["start"] is None else c["start"] + t_offset
            c["end"] = None if c["end"] is None else c["end"] + t_offset

    # visualization
    vis = case.artifacts_dir / f"{ev.id}_{label}_spectrogram.png"
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
    t = np.arange(len(y16)) / 16000 + t_offset
    step = max(1, len(y16) // 20000)
    ax0.plot(t[::step], y16[::step], lw=0.5, color="#334")
    ax0.set_ylabel("amp")
    ax0.set_title(f"{ev.name} — {label} waveform & spectrogram (shaded = flagged)", fontsize=10)
    ff, tt, S = signal.spectrogram(y16, 16000, nperseg=512, noverlap=384)
    ax1.pcolormesh(tt + t_offset, ff / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma")
    ax1.set_ylabel("kHz")
    ax1.set_xlabel("time (s)")
    colors = {"ml": "#e02424", "silence": "#7e3af2", "splice": "#ff8a00"}
    for kind, s, e in marks:
        for ax in (ax0, ax1):
            if s == e:
                ax.axvline(s + t_offset, color=colors[kind], lw=2)
            else:
                ax.axvspan(s + t_offset, e + t_offset, color=colors[kind], alpha=0.28)
    fig.tight_layout()
    fig.savefig(vis, dpi=110)
    plt.close(fig)
    return {"indicators": inds, "visuals": {f"{label.capitalize()} spectrogram": case.rel(vis)}, "metrics": metrics,
            "transcript": tr}
