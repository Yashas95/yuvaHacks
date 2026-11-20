"""How a piece of evidence was obtained, and which checks are valid on it.

Uploaded files are originals as far as TruthLens can tell. The browser extension, however, often cannot get the
original: Instagram, YouTube and X stream video in pieces (a `blob:` address), so it records a few seconds of the
playing video, grabs a frame, or, as a last resort, screenshots the visible area. Those are *derived copies*:

- File-level checks describe the copy, not the media. A frame drawn to a canvas has no camera metadata or AI label;
  its compression (ELA), sensor-noise and frequency patterns come from the video codec and the re-render; a
  screenshot adds scaling and the site's own icons. Running them would produce findings about the capture.
- Content-level checks stay valid: the AI-image model on the pixels, faces, cuts and frozen-frame checks inside a
  recording (except frozen frames: the recorder repeats frames by design), and the audio of a recording.

Recording the acquisition is also standard chain-of-custody practice: the report states where the media was
captured, how, and when, next to its SHA-256.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

METHODS = {
    "original": "original file downloaded from the page",
    "recording": "short recording of the playing video",
    "video-frame": "single frame taken from the playing video",
    "screen-capture": "screenshot of the visible area",
    "image-pixels": "copy of the image as the page shows it",
}
CAPTURED = {"recording", "video-frame", "screen-capture", "image-pixels"}   # derived copies, not the original file

# Image checks that describe the file itself; skipped on captured frames and screenshots.
FILE_LEVEL_IMAGE_CHECKS = {"metadata", "ela", "noise", "fft"}
# Screenshots also contain the site's own icons and text, which repeat inside the image.
SCREENSHOT_EXTRA_SKIPS = {"copy_move"}


def parse(raw: str | None) -> dict:
    """Validate the extension's JSON description. Unknown keys are dropped; bad values are ignored, never fatal."""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(d, dict) or d.get("method") not in METHODS:
        return {}
    out = {"method": d["method"]}
    for key in ("page_url", "source_url"):
        v = str(d.get(key) or "")[:2000]
        if re.match(r"https?://", v):
            out[key] = v
    for key, n in (("page_title", 200), ("tool", 80), ("captured_at", 40)):
        v = str(d.get(key) or "").strip()[:n]
        if v:
            out[key] = v
    for key, lo, hi in (("seconds", 0, 600), ("width", 1, 20000), ("height", 1, 20000)):
        try:
            v = float(d.get(key))
            if lo <= v <= hi:
                out[key] = round(v, 2) if key == "seconds" else int(v)
        except (TypeError, ValueError):
            pass
    return out


def method(ev) -> str:
    acq = getattr(ev, "acquisition", None) or (ev.get("acquisition") if isinstance(ev, dict) else None) or {}
    return acq.get("method", "")


def is_captured(ev) -> bool:
    return method(ev) in CAPTURED


def host(acq: dict) -> str:
    return (urlparse(acq.get("page_url", "")).hostname or "").removeprefix("www.")


def skipped_image_checks(ev) -> set[str]:
    m = method(ev)
    if m == "screen-capture":
        return FILE_LEVEL_IMAGE_CHECKS | SCREENSHOT_EXTRA_SKIPS
    if m in CAPTURED:
        return set(FILE_LEVEL_IMAGE_CHECKS)
    return set()


def describe(acq: dict) -> str:
    """One line for the log and the report, e.g. 'short recording of the playing video (6.0 s) on instagram.com'."""
    if not acq:
        return ""
    s = METHODS.get(acq.get("method"), "unknown method")
    if acq.get("seconds"):
        s += f" ({acq['seconds']:.1f} s)"
    if acq.get("width") and acq.get("height"):
        s += f", {acq['width']}×{acq['height']} px"
    h = host(acq)
    if h:
        s += f" on {h}"
    if acq.get("captured_at"):
        s += f", {acq['captured_at']}"
    if acq.get("tool"):
        s += f", by {acq['tool']}"
    return s


def limitation(name: str, acq: dict) -> str:
    m = acq.get("method")
    where = host(acq) or "the web page"
    if m == "recording":
        return (f"{name} is a {acq.get('seconds', 0):.0f}-second recording of the video playing on {where}, made by the "
                "TruthLens extension, not the original file. The site's file labels and container data are not in it, "
                "and frozen-frame checks were skipped (the recorder repeats frames). Cuts, frames and audio were checked. "
                "For a full check, get the original file.")
    if m == "video-frame":
        return (f"{name} is one frame taken from the video playing on {where}, not the original file. Only the AI-image "
                "and face models and the copy-paste check ran; metadata, compression, sensor-noise and frequency checks "
                "don't apply to a captured frame, and there is no audio or timing to check.")
    if m == "image-pixels":
        return (f"{name} is a copy of the picture as {where} shows it, made by the TruthLens extension because the original "
                "file couldn't be downloaded. Camera data and AI labels aren't in the copy, and the compression, sensor-noise "
                "and frequency checks were skipped (they would describe the copy). The AI-image model, faces and the "
                "copy-paste check ran.")
    if m == "screen-capture":
        return (f"{name} is a screenshot of what was visible on {where}, not the original file: it may include the site's "
                "own buttons and text. Only the AI-image and face models ran. For a real check, get the original file.")
    return ""
