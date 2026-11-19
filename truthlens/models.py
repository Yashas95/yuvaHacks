"""Lazy, fault-tolerant loading of pretrained Hugging Face models."""
from __future__ import annotations

import logging
import os
import re

from .config import MODELS

log = logging.getLogger("truthlens.models")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_CACHE: dict = {}
STATUS: dict = {}  # key -> {"model": id | None, "error": str | None}

FAKE_WORDS = {"fake", "artificial", "ai", "synthetic", "generated", "spoof", "deepfake", "aigenerated", "sdxl", "machine"}
REAL_WORDS = {"real", "human", "bonafide", "authentic", "genuine", "natural", "realism", "original"}


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load(key: str, task: str, device=None):
    if key in _CACHE:
        return _CACHE[key]
    pipe, used, err = None, None, None
    try:
        from transformers import pipeline
    except Exception as e:  # transformers missing entirely
        STATUS[key] = {"model": None, "error": f"transformers not installed: {e}"}
        _CACHE[key] = None
        return None
    for mid in MODELS.get(key, []):
        for dev in ([device] if device else [_device(), "cpu"]):
            try:
                pipe = pipeline(task, model=mid, device=dev)
                used = mid
                break
            except Exception as e:
                err = f"{mid}: {type(e).__name__}: {str(e)[:160]}"
                log.warning("model load failed %s", err)
        if pipe is not None:
            break
    STATUS[key] = {"model": used, "error": None if pipe else err}
    _CACHE[key] = pipe
    return pipe


def image_ai():
    return _load("image_ai", "image-classification")


def image_ai_validated() -> bool:
    """True when the loaded AI-image model passed tools/eval_detectors.py (config.VALIDATED_MODELS)."""
    from .config import VALIDATED_MODELS
    return STATUS.get("image_ai", {}).get("model") in VALIDATED_MODELS


def face_deepfake():
    return _load("face_deepfake", "image-classification")


def audio_deepfake():
    return _load("audio_deepfake", "audio-classification")


def asr():
    return _load("asr", "automatic-speech-recognition", device="cpu")


def clip():
    """Returns (model, processor) or None."""
    if "clip" in _CACHE:
        return _CACHE["clip"]
    out = None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        mid = MODELS["clip"][0]
        model = CLIPModel.from_pretrained(mid).eval()
        proc = CLIPProcessor.from_pretrained(mid)
        out = (model, proc, torch)
        STATUS["clip"] = {"model": mid, "error": None}
    except Exception as e:
        STATUS["clip"] = {"model": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    _CACHE["clip"] = out
    return out


def _words(label: str) -> set:
    return set(re.split(r"[^a-z]+", label.lower())) - {""}


def fake_probability(preds) -> float | None:
    """Map a classifier's label/score list to P(synthetic). None if labels are unmappable."""
    if not preds:
        return None
    if isinstance(preds, dict):
        preds = [preds]
    fake, real = None, None
    for p in preds:
        w = _words(str(p.get("label", "")))
        if w & FAKE_WORDS and not (w & REAL_WORDS):
            fake = (fake or 0.0) + float(p["score"])
        elif w & REAL_WORDS:
            real = (real or 0.0) + float(p["score"])
    if fake is not None:
        return max(0.0, min(1.0, fake))
    if real is not None:
        return max(0.0, min(1.0, 1.0 - real))
    return None


def status_lines() -> list[str]:
    lines = []
    for k, v in STATUS.items():
        if v["model"]:
            lines.append(f"{k}: {v['model']}")
        else:
            lines.append(f"{k}: UNAVAILABLE ({v['error']})")
    return lines
