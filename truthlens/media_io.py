"""ffmpeg / ffprobe helpers."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


def probe(path: Path) -> dict:
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                             capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}")
    except Exception as e:
        return {"error": str(e)}


def has_audio(info: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def load_audio(path: Path, sr: int | None = None, max_seconds: float = 600) -> tuple[np.ndarray, int]:
    info = probe(path)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if a is None:
        return np.zeros(0, np.float32), sr or 16000
    native = int(a.get("sample_rate", 16000))
    target = sr or min(native, 48000)
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", str(target),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), target
