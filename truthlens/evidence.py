"""Evidence intake, chain of custody, and the shared Indicator structure."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import mimetypes
import os
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
TEXT_EXT = {".txt", ".md", ".srt", ".vtt", ".json"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def modality_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in TEXT_EXT:
        return "text"
    mt = mimetypes.guess_type(str(path))[0] or ""
    return mt.split("/")[0] if mt.split("/")[0] in {"image", "video", "audio", "text"} else "unknown"


@dataclass
class Indicator:
    id: str
    evidence_id: str
    modality: str
    detector: str            # key into config.RELIABILITY
    title: str
    category: str            # editing | synthetic | splice | metadata | context | consistency
    observation: str         # what was measured
    why: str                 # why it is suspicious
    score: float             # 0..1 strength of the anomaly
    reliability: float       # 0..1 how much we trust this detector family
    location: dict = field(default_factory=dict)   # {"type": "region"|"frame"|"segment"|"file", ...}
    alternatives: list = field(default_factory=list)
    artifact: str | None = None   # relative path of visualization
    supports: str = "manipulation"  # "manipulation" | "authenticity"
    validated: bool = False         # produced by a model that passed tools/eval_detectors.py (config.VALIDATED_MODELS)

    def to_dict(self):
        d = asdict(self)
        d["score"] = round(float(self.score), 3)
        d["reliability"] = round(float(self.reliability), 2)
        return d


class Findings(list):
    """Indicators for one evidence item. Checks call `add(detector, title, ...)` instead of repeating the
    id / evidence / modality arguments; the order of `add` calls keeps indicator ids stable."""

    def __init__(self, ev, nid, modality: str):
        super().__init__()
        self.ev, self.nid, self.modality = ev, nid, modality

    def add(self, *args, **kwargs) -> Indicator:
        ind = Indicator(self.nid(), self.ev.id, self.modality, *args, **kwargs)
        self.append(ind)
        return ind


@dataclass
class Evidence:
    id: str
    name: str
    modality: str
    path: Path          # read-only working copy inside the case folder
    sha256: str
    size: int
    received_at: str
    meta: dict = field(default_factory=dict)
    acquisition: dict = field(default_factory=dict)   # how it was obtained (truthlens/acquisition.py); {} = uploaded file


class Case:
    def __init__(self, case_id: str | None = None, root: Path | None = None):
        from .config import CASES_DIR
        # hosted demo: long random suffix so other visitors cannot guess a report URL
        n = 16 if os.environ.get("TRUTHLENS_HOSTED") else 4
        self.id = case_id or dt.datetime.now().strftime("TL-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:n].upper()
        self.dir = (root or CASES_DIR) / self.id
        self.evidence_dir = self.dir / "evidence"
        self.artifacts_dir = self.dir / "artifacts"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.evidence: list[Evidence] = []
        self.context_text: str = ""
        self.log: list[dict] = []
        self.created_at = now()
        self.add_log("case_created", f"Case {self.id} opened")

    def add_log(self, action: str, detail: str):
        self.log.append({"time": now(), "action": action, "detail": detail})

    def add_file(self, src: Path, display_name: str | None = None, acquisition: dict | None = None) -> Evidence:
        src = Path(src)
        name = display_name or src.name
        eid = f"E{len(self.evidence) + 1:02d}"
        h_src = sha256(src)
        dst = self.evidence_dir / f"{eid}_{Path(name).name}"
        shutil.copy2(src, dst)
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only working copy
        h_dst = sha256(dst)
        if h_src != h_dst:
            raise RuntimeError(f"Hash mismatch while copying {name}")
        ev = Evidence(eid, name, modality_of(dst), dst, h_dst, dst.stat().st_size, now(), acquisition=dict(acquisition or {}))
        self.evidence.append(ev)
        self.add_log("evidence_ingested", f"{eid} {name} ({ev.modality}, {ev.size} bytes) sha256={h_dst}")
        if ev.acquisition:
            from .acquisition import describe
            self.add_log("evidence_acquired", f"{eid} acquired as {describe(ev.acquisition)}"
                         + (f"; page {ev.acquisition['page_url']}" if ev.acquisition.get("page_url") else ""))
        return ev

    def set_context(self, text: str):
        self.context_text = (text or "").strip()
        if self.context_text:
            p = self.evidence_dir / "context_statement.txt"
            p.write_text(self.context_text, encoding="utf-8")
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            self.add_log("context_recorded", f"Context statement recorded ({len(self.context_text)} chars) sha256={sha256(p)}")

    def verify_integrity(self) -> list[dict]:
        out = []
        for ev in self.evidence:
            ok = sha256(ev.path) == ev.sha256
            out.append({"evidence_id": ev.id, "name": ev.name, "sha256": ev.sha256, "unchanged": ok})
        self.add_log("integrity_verified", "All evidence hashes re-verified after analysis: "
                     + ("UNCHANGED" if all(o["unchanged"] for o in out) else "MISMATCH DETECTED"))
        return out

    def rel(self, p: Path) -> str:
        return str(Path(p).relative_to(self.dir))

    def save_json(self, name: str, data):
        (self.dir / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
