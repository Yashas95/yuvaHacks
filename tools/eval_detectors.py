"""Measure which AI-image/frame detectors actually separate REAL from AI-GENERATED media.

Use this to choose models with evidence instead of tuning to one file.

    .venv/bin/python tools/eval_detectors.py \
        --real  real1.mp4 real2.jpg ...  \
        --ai    veo.mp4 sora.mp4 gen.png ... \
        [--models Organika/sdxl-detector haywoodsloan/ai-image-detector-deploy] [--frames 12] [--allow-download]

Or put files in folders and pass --real-dir / --ai-dir.

For every model it prints each file's median P(AI) over sampled frames, then:
  * AUC           — probability a random AI file scores above a random real file (0.5 = coin flip, 1.0 = perfect)
  * FPR@0.5       — share of REAL files scored ≥ 0.5  (false alarms)
  * TPR@0.5       — share of AI files scored ≥ 0.5    (detections)
  * saturated     — the model outputs ≥ 0.99 on every real file (it is not discriminating at all)
Runs offline (local cache only) unless --allow-download is given.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def frames_of(path: Path, n: int):
    import cv2
    from PIL import Image
    if path.suffix.lower() in IMG:
        return [Image.open(path).convert("RGB")]
    cap = cv2.VideoCapture(str(path))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if tot <= 0:
        return []
    want = sorted({int(i * (tot - 1) / max(1, n - 1)) for i in range(n)})
    out = []
    for k in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if ok:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    return out


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--ai", nargs="*", default=[])
    ap.add_argument("--real-dir")
    ap.add_argument("--ai-dir")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()
    if not a.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from truthlens.config import MODELS
    from truthlens.models import _device, fake_probability
    from transformers import pipeline

    def collect(files, d):
        out = [Path(f) for f in files]
        if d:
            out += sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG | VID)
        return out
    real, ai = collect(a.real, a.real_dir), collect(a.ai, a.ai_dir)
    if not real or not ai:
        sys.exit("Give at least one --real and one --ai file (more is much better: aim for 10+ of each).")
    models = a.models or MODELS["image_ai"]
    cache = {p: frames_of(p, a.frames) for p in real + ai}

    summary = []
    for mid in models:
        try:
            pipe = pipeline("image-classification", model=mid, device=_device())
        except Exception as e:
            print(f"\n== {mid}: NOT AVAILABLE ({type(e).__name__}: {str(e)[:120]})")
            continue
        print(f"\n== {mid}   labels={list(pipe.model.config.id2label.values())}")
        scores = {}
        for p in real + ai:
            ps = [fake_probability(pipe(f, top_k=10)) for f in cache[p]]
            ps = [x for x in ps if x is not None]
            scores[p] = st.median(ps) if ps else None
            tag = "REAL" if p in real else "AI  "
            print(f"   {tag}  {scores[p] if scores[p] is not None else float('nan'):6.3f}   {p.name}")
        rs = [scores[p] for p in real if scores[p] is not None]
        as_ = [scores[p] for p in ai if scores[p] is not None]
        res = {"model": mid, "auc": auc(as_, rs), "fpr": sum(x >= 0.5 for x in rs) / len(rs) if rs else None,
               "tpr": sum(x >= 0.5 for x in as_) / len(as_) if as_ else None,
               "saturated": bool(rs) and all(x >= 0.99 for x in rs)}
        summary.append(res)

    print("\nModel".ljust(48) + "AUC    FPR@0.5  TPR@0.5  note")
    for s in sorted(summary, key=lambda s: -(s["auc"] or 0)):
        note = "SATURATED on real media — not usable" if s["saturated"] else ""
        f = lambda v: "  -  " if v is None else f"{v:.2f}"
        print(f"{s['model']:47s} {f(s['auc'])}   {f(s['fpr'])}     {f(s['tpr'])}     {note}")
    print(f"\n{len(real)} real / {len(ai)} AI files. With fewer than ~10 of each, treat these numbers as anecdotes.")


if __name__ == "__main__":
    main()
