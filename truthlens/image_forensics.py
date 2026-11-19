"""Image authenticity analysis: metadata, ELA, noise, copy-move, spectrum, ML detectors."""
from __future__ import annotations

import io
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from . import models
from .config import RELIABILITY as R, VALIDATED_RELIABILITY
from .evidence import Indicator

AI_SIGNATURES = ["stable diffusion", "stable-diffusion", "midjourney", "dall-e", "dall·e", "novelai", "comfyui",
                 "automatic1111", "invokeai", "firefly", "imagen", "flux.1", "sdxl", "trainedalgorithmicmedia",
                 "openai", "chatgpt", "gemini", "nano banana", "leonardo.ai", "ideogram", "dreamstudio"]
EDITORS = ["photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "pixelmator", "affinity", "facetune",
           "meitu", "remini", "capcut", "illustrator", "photopea", "luminar", "faceapp"]


# ----------------------------------------------------------------------------- helpers
def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _robust_z(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = x[valid] if valid is not None and valid.any() else x.ravel()
    med = np.median(v)
    mad = max(float(np.median(np.abs(v - med))), 0.08)  # floor: avoids absurd z on flat/synthetic images
    return np.clip((x - med) / (1.4826 * mad), -50, 50)


def _blocks(a: np.ndarray, b: int, fn=np.mean) -> np.ndarray:
    h, w = a.shape[:2]
    h2, w2 = h - h % b, w - w % b
    a = a[:h2, :w2]
    return fn(fn(a.reshape(h2 // b, b, w2 // b, b), axis=3), axis=1) if fn is np.mean else \
        a.reshape(h2 // b, b, w2 // b, b).swapaxes(1, 2).reshape(h2 // b, w2 // b, b * b).std(axis=2)


def _components(mask: np.ndarray, block: int, zmap: np.ndarray, min_blocks=4, top=5):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_blocks:
            continue
        strength = float(zmap[lab == i].mean())
        comps.append({"bbox": [int(x * block), int(y * block), int(w * block), int(h * block)],
                      "blocks": int(area), "mean_z": round(strength, 2)})
    comps.sort(key=lambda c: c["mean_z"] * np.sqrt(c["blocks"]), reverse=True)
    return comps[:top]


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray, boxes, out: Path, max_side=1280):
    heat = cv2.resize(heat.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    heat = np.clip(heat / (np.percentile(heat, 99.5) + 1e-6), 0, 1)
    cm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(bgr, 0.55, cm, 0.45, 0)
    _draw_boxes(blend, boxes)
    _save(blend, out, max_side)


def _draw_boxes(bgr, boxes, color=(0, 0, 255)):
    t = max(2, int(max(bgr.shape[:2]) / 400))
    for i, b in enumerate(boxes):
        x, y, w, h = b["bbox"] if isinstance(b, dict) else b
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, t)
        label = b.get("label", f"R{i + 1}") if isinstance(b, dict) else f"R{i + 1}"
        cv2.putText(bgr, label, (x + 3, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, color, t)


def _save(bgr, out: Path, max_side=1280):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------------------- metadata
def read_metadata(path: Path) -> dict:
    meta = {"format": None, "exif": {}, "png_text": {}, "xmp_tools": [], "c2pa_present": False,
            "ai_signatures": [], "editors": [], "has_gps": False}
    raw = path.read_bytes()
    low = raw[: 8 << 20].lower() if len(raw) > (8 << 20) else raw.lower()
    try:
        img = Image.open(path)
        meta["format"] = img.format
        meta["size"] = list(img.size)
        for k, v in (img.info or {}).items():
            if isinstance(v, (str, bytes)) and k.lower() not in {"icc_profile", "exif"}:
                s = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
                meta["png_text"][k] = s[:600]
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        try:
            sub = exif.get_ifd(0x8769)
            tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in sub.items()})
            meta["has_gps"] = bool(exif.get_ifd(0x8825))
        except Exception:
            pass
        for k in ["Make", "Model", "Software", "DateTime", "DateTimeOriginal", "DateTimeDigitized",
                  "LensModel", "Artist", "ImageDescription", "UserComment", "OffsetTimeOriginal"]:
            if k in tags:
                v = tags[k]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "ignore")
                meta["exif"][k] = str(v).strip("\x00 ")[:200]
    except Exception as e:
        meta["error"] = str(e)
    if b"<x:xmpmeta" in low:
        meta["xmp_tools"] = list({m.decode("utf-8", "ignore") for m in re.findall(rb"creatortool[=\">]+([^\"<]{2,80})", low)})
    meta["c2pa_present"] = (b"c2pa" in low) or (b"jumb" in low and b"jumd" in low)
    rawtxt = low.decode("latin-1", "ignore")
    readable = " ".join(meta["png_text"].values()).lower() + " " + " ".join(str(v) for v in meta["exif"].values()).lower() \
        + " " + " ".join(meta["xmp_tools"]).lower()
    # XMP / C2PA blocks are plain text inside the file: extract them so whole-word matching is reliable there
    for blk in re.findall(r"<x:xmpmeta.*?</x:xmpmeta>", rawtxt, flags=re.S)[:3]:
        readable += " " + blk[:20000]
    if meta["c2pa_present"]:
        readable += " " + " ".join(re.findall(r"[a-z0-9 .:_/\-]{6,}", rawtxt[:2_000_000]))[:20000]
    wb = lambda s, t: re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", t) is not None
    meta["ai_signatures"] = sorted({s for s in AI_SIGNATURES if wb(s, readable) or (len(s) >= 9 and s in rawtxt)})
    blob = readable
    if any(k in meta["png_text"] for k in ("parameters", "prompt", "workflow")):
        meta["ai_signatures"].append("generation-parameters chunk (" + ",".join(
            k for k in ("parameters", "prompt", "workflow") if k in meta["png_text"]) + ")")
    meta["editors"] = sorted({e for e in EDITORS if e in blob})
    return meta


def metadata_indicators(ev, meta, nid) -> list[Indicator]:
    out = []
    ex = meta.get("exif", {})
    if meta["ai_signatures"]:
        out.append(Indicator(nid(), ev.id, "image", "metadata_ai_signature", "Generator signature in metadata",
                             "synthetic", f"Embedded metadata contains: {', '.join(meta['ai_signatures'])}.",
                             "AI image tools frequently write their name, prompt or workflow into the file. "
                             "This is direct provenance evidence, not a statistical guess.",
                             0.95, R["metadata_ai_signature"], {"type": "file"},
                             ["Metadata can be copied or forged; confirm with the original source.",
                              "Some editors (e.g. Photoshop generative fill) add the tag for partial AI edits only."]))
    if meta["c2pa_present"]:
        out.append(Indicator(nid(), ev.id, "image", "metadata_editor", "Content Credentials (C2PA) manifest present",
                             "metadata", "A C2PA/JUMBF provenance manifest was found in the file.",
                             "Content Credentials record which tools created or edited the file. Inspect the manifest "
                             "(e.g. contentcredentials.org/verify) for AI-generation or edit actions.",
                             0.35, R["metadata_editor"], {"type": "file"},
                             ["Cameras and phones also sign genuine photos with C2PA; presence alone is neutral."]))
    tools = meta["editors"] + [t for t in meta["xmp_tools"] if t not in meta["editors"]]
    if ex.get("Software"):
        tools.append(ex["Software"])
    tools = sorted(set(t for t in tools if t))
    if meta["editors"]:
        out.append(Indicator(nid(), ev.id, "image", "metadata_editor", "Editing software recorded",
                             "editing", f"Software traces: {', '.join(tools)}.",
                             "The file passed through image-editing software after capture.",
                             0.6, R["metadata_editor"], {"type": "file"},
                             ["Routine edits (crop, exposure, resize) also leave this trace; it does not prove deceptive edits."]))
    dto, dtm = ex.get("DateTimeOriginal"), ex.get("DateTime")
    if dto and dtm and dto[:19] != dtm[:19]:
        out.append(Indicator(nid(), ev.id, "image", "metadata_dates", "Modified after capture",
                             "metadata", f"Captured {dto} but last modified {dtm}.",
                             "The file was re-saved after the original capture time.",
                             0.45, R["metadata_dates"], {"type": "file"},
                             ["Phone gallery apps and cloud sync re-save files without changing content."]))
    if not ex.get("Make") and not ex.get("Model") and not meta["ai_signatures"]:
        out.append(Indicator(nid(), ev.id, "image", "metadata_missing", "No camera metadata",
                             "metadata", "No camera make/model in EXIF" + (" (no EXIF at all)." if not ex else "."),
                             "Original camera photos normally carry make/model. Missing data removes a provenance anchor.",
                             0.35, R["metadata_missing"], {"type": "file"},
                             ["WhatsApp, Instagram, X and most messengers strip EXIF on upload — very common for genuine images.",
                              "Screenshots never have camera EXIF."]))
    return out


# ----------------------------------------------------------------------------- pixel forensics
def ela(rgb: np.ndarray, ev, case, nid, is_jpeg: bool):
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    re_ = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(rgb.astype(np.float32) - re_).max(axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = 16
    e = _blocks(diff, B)
    tex = _blocks(gray, B, fn=None) + 2.0
    ratio = e / tex
    valid = (tex > 3.0)
    z = _robust_z(np.log(ratio + 1e-3), valid)
    hot = (z > 3.5) & valid & (e > 1.0)
    comps = _components(hot, B, z)
    frac = hot.mean()
    out_path = case.artifacts_dir / f"{ev.id}_ela.jpg"
    _heat_overlay(rgb, np.where(valid, np.clip(z, 0, None), 0), comps, out_path)
    metrics = {"hot_fraction": round(float(frac), 4), "regions": comps}
    if not comps:
        return [], metrics, out_path
    top = comps[0]["mean_z"]
    loc_factor = 1.0 if 0.001 < frac < 0.2 else 0.45
    score = float(np.clip((top - 3.5) / 5.0, 0, 1)) * loc_factor
    rel = R["ela"] * (1.0 if is_jpeg else 0.6)
    ind = Indicator(nid(), ev.id, "image", "ela", "Compression-level inconsistency (ELA)", "editing",
                    f"{len(comps)} region(s) re-compress differently from the rest of the image "
                    f"(strongest z={top}, {frac * 100:.1f}% of image).",
                    "Pasted or retouched areas often carry a different JPEG compression history than their surroundings."
                    + ("" if is_jpeg else " (Source is not JPEG, so ELA is weaker here.)"),
                    score, rel, {"type": "regions", "regions": comps},
                    ["High-contrast edges, text and fine textures can light up in ELA without editing.",
                     "Multiple re-saves by social platforms flatten ELA differences."],
                    case.rel(out_path))
    return [ind], metrics, out_path


def noise_map(rgb: np.ndarray, ev, case, nid):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    B = 32
    s = _blocks(resid, B, fn=None)
    brightness = _blocks(gray.astype(np.float32), B)
    tex = _blocks(cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32), B, fn=None)
    valid = (brightness > 12) & (brightness < 243) & (tex < np.percentile(tex, 85))
    z = _robust_z(np.log(s + 0.05), valid)
    hot = (np.abs(z) > 3.5) & valid
    comps = _components(hot, B, np.abs(z), min_blocks=3)
    out_path = case.artifacts_dir / f"{ev.id}_noise.jpg"
    _heat_overlay(rgb, np.where(valid, np.abs(z), 0), comps, out_path)
    metrics = {"inconsistent_fraction": round(float(hot.mean()), 4), "regions": comps}
    if not comps:
        return [], metrics, out_path
    top = comps[0]["mean_z"]
    score = float(np.clip((top - 3.5) / 5.0, 0, 1)) * (1.0 if hot.mean() < 0.2 else 0.4)
    return [Indicator(nid(), ev.id, "image", "noise", "Sensor-noise inconsistency", "editing",
                      f"{len(comps)} region(s) have a noise level that differs from the rest of the frame (strongest |z|={top}).",
                      "Every camera leaves a fairly uniform noise pattern. A region with a different noise level is consistent "
                      "with content pasted from another source, AI inpainting, or local smoothing/beautify filters.",
                      score, R["noise"], {"type": "regions", "regions": comps},
                      ["Depth-of-field blur, strong denoise in night mode, and HDR merges change local noise legitimately."],
                      case.rel(out_path))], metrics, out_path


def copy_move(rgb: np.ndarray, ev, case, nid):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    sc = min(1.0, 1024 / max(h, w))
    g = cv2.resize(gray, (int(w * sc), int(h * sc))) if sc < 1 else gray
    try:
        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015)
    except Exception:
        return [], {"error": "SIFT unavailable"}
    kps, des = sift.detectAndCompute(g, None)
    if des is None or len(kps) < 20:
        return [], {"keypoints": 0 if des is None else len(kps)}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, des, k=3)
    pairs = []
    for m in matches:
        if len(m) < 3:
            continue
        a, b = m[1], m[2]
        if a.distance < 0.5 * b.distance:
            p1 = np.array(kps[a.queryIdx].pt)
            p2 = np.array(kps[a.trainIdx].pt)
            if np.linalg.norm(p1 - p2) > 40:
                if (p2[0], p2[1]) < (p1[0], p1[1]):
                    p1, p2 = p2, p1
                pairs.append((p1, p2))
    if not pairs:
        return [], {"keypoints": len(kps), "matches": 0}
    offs = np.round(np.array([p2 - p1 for p1, p2 in pairs]) / 12).astype(int)
    keys, counts = np.unique(offs, axis=0, return_counts=True)
    best = keys[np.argmax(counts)]
    cluster = [pairs[i] for i in range(len(pairs)) if np.all(np.abs(offs[i] - best) <= 1)]
    n = len(cluster)
    metrics = {"keypoints": len(kps), "matches": len(pairs), "largest_cluster": n}
    if n < 8:
        return [], metrics
    src = np.array([c[0] for c in cluster]) / sc
    dst = np.array([c[1] for c in cluster]) / sc

    def bb(pts):
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        return [int(x0) - 8, int(y0) - 8, int(x1 - x0) + 16, int(y1 - y0) + 16]

    regions = [{"bbox": bb(src), "label": "source"}, {"bbox": bb(dst), "label": "copy"}]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for p1, p2 in zip(src, dst):
        cv2.line(bgr, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (0, 255, 255), 1)
    _draw_boxes(bgr, regions)
    out_path = case.artifacts_dir / f"{ev.id}_copymove.jpg"
    _save(bgr, out_path)
    score = float(np.clip((n - 6) / 25, 0.2, 1))
    return [Indicator(nid(), ev.id, "image", "copy_move", "Duplicated region (copy-move)", "editing",
                      f"{n} matching keypoints share the same displacement "
                      f"({int(best[0] * 12 / sc)}, {int(best[1] * 12 / sc)}) px between two areas.",
                      "A consistent displacement between many identical features is the signature of a region copied and "
                      "pasted within the same image (cloning objects or hiding something).",
                      score, R["copy_move"], {"type": "regions", "regions": regions},
                      ["Genuinely repetitive scenes (tiles, windows, fences, crowds in uniform) can match.",
                       "Panorama stitching can produce limited duplication."],
                      case.rel(out_path))], metrics


def spectrum(rgb: np.ndarray, ev, case, nid):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    N = 512
    if min(h, w) < N:
        N = 256 if min(h, w) >= 256 else 0
    if not N:
        return [], {}
    y0, x0 = (h - N) // 2, (w - N) // 2
    g = gray[y0:y0 + N, x0:x0 + N]
    g = (g - g.mean()) * np.outer(np.hanning(N), np.hanning(N))
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - N / 2, xx - N / 2).astype(int)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    resid = mag - radial[r]
    mask = (r > N * 0.12) & (r < N * 0.48)
    jpeg_grid = np.zeros_like(mask)
    step = N // 8
    for k in range(-4, 5):
        c = N // 2 + k * step
        jpeg_grid[:, max(c - 1, 0):c + 2] = True
        jpeg_grid[max(c - 1, 0):c + 2, :] = True
    mask &= ~jpeg_grid
    vals = resid[mask]
    peak_z = float((vals.max() - np.median(vals)) / (vals.std() + 1e-6))
    n_peaks = int((resid[mask] > np.median(vals) + 6 * vals.std()).sum())
    vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out_path = case.artifacts_dir / f"{ev.id}_spectrum.jpg"
    cv2.imwrite(str(out_path), cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO))
    metrics = {"peak_z": round(peak_z, 2), "n_peaks": n_peaks}
    if peak_z < 8 or n_peaks < 4:
        return [], metrics
    score = float(np.clip((peak_z - 8) / 12, 0, 1))
    return [Indicator(nid(), ev.id, "image", "fft", "Periodic frequency artifacts", "synthetic",
                      f"{n_peaks} isolated peaks in the frequency spectrum (peak z={peak_z:.1f}), excluding JPEG 8x8 grid frequencies.",
                      "Up-sampling layers in GAN/diffusion generators often leave a regular grid pattern visible as spectral peaks.",
                      score, R["fft"], {"type": "file"},
                      ["Resizing, screen photos (moiré), halftone prints and sharpening also create periodic peaks."],
                      case.rel(out_path))], metrics


# ----------------------------------------------------------------------------- ML detectors
_FACE = None


def detect_faces(rgb: np.ndarray, max_faces=4):
    """YuNet face detector (OpenCV DNN). Returns [(x, y, w, h), ...]; [] if unavailable."""
    global _FACE
    from .config import ROOT
    try:
        if _FACE is None:
            onnx = ROOT / "weights" / "face_detection_yunet_2023mar.onnx"
            if onnx.exists() and hasattr(cv2, "FaceDetectorYN"):
                _FACE = cv2.FaceDetectorYN.create(str(onnx), "", (320, 320), 0.7, 0.3, 50)
            elif hasattr(cv2, "CascadeClassifier"):
                _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            else:
                _FACE = False
        if _FACE is False:
            return []
        h, w = rgb.shape[:2]
        if hasattr(_FACE, "setInputSize"):
            s = min(1.0, 1280 / max(h, w))
            img = cv2.resize(rgb, (int(w * s), int(h * s))) if s < 1 else rgb
            _FACE.setInputSize((img.shape[1], img.shape[0]))
            _, det = _FACE.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            faces = [] if det is None else [tuple(int(v / s) for v in d[:4]) for d in det]
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            m = max(48, min(gray.shape) // 12)
            faces = [tuple(map(int, f)) for f in _FACE.detectMultiScale(gray, 1.1, 6, minSize=(m, m))]
        faces = [f for f in faces if f[2] >= 32 and f[3] >= 32]
        return sorted(faces, key=lambda f: -f[2] * f[3])[:max_faces]
    except Exception:
        return []


def crop_face(rgb, f, margin=0.25):
    x, y, w, h = f
    H, W = rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return Image.fromarray(rgb[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)])


def ml_image_score(pil: Image.Image):
    pipe = models.image_ai()
    if pipe is None:
        return None, None
    try:
        preds = pipe(pil, top_k=10)
        return models.fake_probability(preds), preds
    except Exception as e:
        return None, str(e)


def ml_face_scores(rgb: np.ndarray, faces):
    pipe = models.face_deepfake()
    if pipe is None or not faces:
        return []
    out = []
    for f in faces:
        try:
            p = models.fake_probability(pipe(crop_face(rgb, f), top_k=10))
        except Exception:
            p = None
        out.append({"bbox": list(f), "p_fake": None if p is None else round(p, 3)})
    return out


def ml_indicators(rgb, ev, case, nid):
    out, metrics = [], {}
    p, raw = ml_image_score(Image.fromarray(rgb))
    metrics["image_ai_p"] = None if p is None else round(p, 3)
    metrics["image_ai_model"] = models.STATUS.get("image_ai", {}).get("model")
    validated = models.image_ai_validated()
    metrics["image_ai_validated"] = validated
    rel_ai = VALIDATED_RELIABILITY if validated else R["ml_image_ai"]
    if p is not None:
        if p >= 0.5:
            out.append(Indicator(nid(), ev.id, "image", "ml_image_ai", "AI-generation classifier", "synthetic",
                                 f"Pretrained detector ({metrics['image_ai_model']}) estimates {p * 100:.0f}% probability of AI generation"
                                 + (" — a model validated on real vs AI samples." if validated else "."),
                                 "Learned statistical patterns resemble images produced by diffusion/GAN generators.",
                                 p, rel_ai, {"type": "file"},
                                 ["Detectors are weaker on newer generators and on heavily compressed/resized images.",
                                  "Heavy beauty filters, HDR and CGI can trigger false positives."], validated=validated))
        else:
            out.append(Indicator(nid(), ev.id, "image", "ml_image_ai", "AI-generation classifier: leans authentic",
                                 "synthetic",
                                 f"Pretrained detector ({metrics['image_ai_model']}) estimates {p * 100:.0f}% probability of AI generation.",
                                 "No strong learned generator signature.", 1 - p, rel_ai * 0.6, {"type": "file"},
                                 ["A low score is not proof of authenticity; novel generators can evade detectors."],
                                 supports="authenticity", validated=validated))
    faces = detect_faces(rgb)
    metrics["faces"] = len(faces)
    fs = ml_face_scores(rgb, faces)
    metrics["face_scores"] = fs
    if faces:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = [{"bbox": list(f), "label": f"F{i + 1}" + ("" if not fs or fs[i]["p_fake"] is None else f" {fs[i]['p_fake'] * 100:.0f}%")}
                 for i, f in enumerate(faces)]
        _draw_boxes(bgr, boxes, (0, 165, 255))
        fpath = case.artifacts_dir / f"{ev.id}_faces.jpg"
        _save(bgr, fpath)
        for i, s in enumerate(fs):
            if s["p_fake"] is not None and s["p_fake"] >= 0.5:
                out.append(Indicator(nid(), ev.id, "image", "ml_face_deepfake", f"Face F{i + 1}: deepfake classifier",
                                     "synthetic", f"Face-swap/deepfake detector estimates {s['p_fake'] * 100:.0f}% for face F{i + 1}.",
                                     "Facial texture and blending statistics resemble face-swapped or synthesized faces.",
                                     s["p_fake"], R["ml_face_deepfake"], {"type": "regions", "regions": [{"bbox": s["bbox"], "label": f"F{i + 1}"}]},
                                     ["Low resolution, motion blur, makeup filters and unusual lighting raise false positives."],
                                     case.rel(fpath)))
    return out, metrics


# ----------------------------------------------------------------------------- entry
def analyze_image(ev, case, nid) -> dict:
    meta = read_metadata(ev.path)
    ev.meta = meta
    rgb = np.asarray(load_rgb(ev.path))
    is_jpeg = (meta.get("format") or "").upper() in {"JPEG", "MPO"}
    inds = metadata_indicators(ev, meta, nid)
    visuals, metrics, errors = {}, {}, []

    def safe(name, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # one failing detector must never sink the whole file
            errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    r = safe("ela", ela, rgb, ev, case, nid, is_jpeg)
    if r:
        inds += r[0]; metrics["ela"] = r[1]; visuals["Error Level Analysis"] = case.rel(r[2])
    r = safe("noise", noise_map, rgb, ev, case, nid)
    if r:
        inds += r[0]; metrics["noise"] = r[1]; visuals["Noise consistency"] = case.rel(r[2])
    r = safe("copy_move", copy_move, rgb, ev, case, nid)
    if r:
        inds += r[0]; metrics["copy_move"] = r[1]
    r = safe("spectrum", spectrum, rgb, ev, case, nid)
    if r:
        inds += r[0]; metrics["spectrum"] = r[1]
    r = safe("ml", ml_indicators, rgb, ev, case, nid)
    inds += r[0] if r else []
    metrics["ml"] = r[1] if r else {}
    metrics["errors"] = errors
    for ind in inds:
        if ind.artifact and ind.title not in visuals:
            visuals[ind.title] = ind.artifact
    # annotated master view: all flagged regions on the original
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    allb = []
    for ind in inds:
        if ind.location.get("type") == "regions" and ind.supports == "manipulation":
            for r in ind.location["regions"]:
                allb.append({"bbox": r["bbox"], "label": f"{ind.id}"})
    _draw_boxes(bgr, allb)
    ann = case.artifacts_dir / f"{ev.id}_annotated.jpg"
    _save(bgr, ann)
    orig = case.artifacts_dir / f"{ev.id}_original_view.jpg"
    _save(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), orig)
    visuals = {"Original (view copy)": case.rel(orig), "Flagged regions": case.rel(ann), **visuals}
    return {"indicators": inds, "visuals": visuals, "metrics": metrics, "metadata": meta}
