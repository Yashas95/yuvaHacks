"""TruthLens web app.  Run:  .venv/bin/python app.py   ->  http://localhost:8000"""
from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from truthlens.config import CASES_DIR, ROOT
from truthlens.evidence import Case, modality_of
from truthlens.pipeline import run_case

app = FastAPI(title="TruthLens")
JOBS: dict = {}
LOCK = threading.Lock()  # one analysis at a time keeps the MacBook Air responsive

# Sample folders offered for one-click demo analysis (never arbitrary paths).
SAMPLE_DIRS = {"demo": ROOT / "demo_samples", "planted": ROOT / "test_assets"}


def _work(job_id, files, text, case):
    def prog(p, m):
        JOBS[job_id].update(progress=p, message=m)
    try:
        with LOCK:
            run_case(files, text, prog, case=case)
        JOBS[job_id].update(status="done", progress=100, message="Report ready")
    except Exception as e:
        JOBS[job_id].update(status="error", message=f"{type(e).__name__}: {e}")
    finally:
        for p, _ in files:
            Path(p).unlink(missing_ok=True)


def _start(saved, context):
    case = Case()
    job = uuid.uuid4().hex[:10]
    JOBS[job] = {"status": "running", "progress": 1, "message": "Queued", "case_id": case.id}
    threading.Thread(target=_work, args=(job, saved, context, case), daemon=True).start()
    return {"job": job, "case_id": case.id}


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(default=[]), context: str = Form(default="")):
    files = [f for f in files if f.filename]
    if not files and not context.strip():
        raise HTTPException(400, "Upload at least one file or enter a claim.")
    tmp = Path(tempfile.mkdtemp(prefix="tl_"))
    saved = []
    for f in files:
        p = tmp / Path(f.filename).name
        with open(p, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append((p, Path(f.filename).name))
    return _start(saved, context)


def _sample_path(ref: str) -> Path:
    group, _, name = ref.partition("/")
    base = SAMPLE_DIRS.get(group)
    if not base or not name or Path(name).name != name:
        raise HTTPException(400, f"Unknown sample {ref!r}")
    p = base / name
    if not p.is_file():
        raise HTTPException(404, f"Sample {ref!r} not found")
    return p


@app.get("/api/samples")
def samples():
    out = []
    for group, base in SAMPLE_DIRS.items():
        if base.is_dir():
            for p in sorted(base.iterdir()):
                if p.is_file() and not p.name.startswith("."):
                    out.append({"ref": f"{group}/{p.name}", "group": group, "name": p.name,
                                "modality": modality_of(p), "size": p.stat().st_size})
    return out


@app.get("/api/sample/{group}/{name}")
def sample_file(group: str, name: str):
    return FileResponse(_sample_path(f"{group}/{name}"))


@app.post("/api/analyze_samples")
async def analyze_samples(refs: list[str] = Form(default=[]), context: str = Form(default="")):
    if not refs:
        raise HTTPException(400, "Pick at least one sample.")
    tmp = Path(tempfile.mkdtemp(prefix="tl_"))
    saved = []
    for ref in refs:
        src = _sample_path(ref)
        dst = tmp / src.name
        shutil.copy2(src, dst)
        saved.append((dst, src.name))
    return _start(saved, context)


@app.get("/api/status/{job}")
def status(job: str):
    if job not in JOBS:
        raise HTTPException(404)
    return JOBS[job]


@app.get("/case/{case_id}", response_class=HTMLResponse)
def report(case_id: str):
    p = (CASES_DIR / Path(case_id).name / "report.html")
    if not p.exists():
        raise HTTPException(404, "Report not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/case/{case_id}/report.json")
def report_json(case_id: str):
    p = CASES_DIR / Path(case_id).name / "report.json"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json", filename=f"{case_id}_report.json")


@app.get("/api/cases")
def cases():
    out = []
    for d in sorted(CASES_DIR.glob("TL-*"), reverse=True)[:30]:
        rp = d / "report.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text())
            except ValueError:
                continue
            out.append({"id": d.name, "band": r["overall"]["band"], "likelihood": r["overall"]["likelihood"],
                        "color": r["overall"]["color"], "confidence": r["overall"]["confidence"],
                        "created": r["case"]["created_at"], "context": r["case"].get("context_text") or "",
                        "files": [{"name": e["name"], "modality": e["modality"]} for e in r["evidence"]]})
    return out


HANDS_FILE = Path(__file__).resolve().parent / "truthlens" / "hand_field.json"
HANDS = HANDS_FILE.read_text() if HANDS_FILE.exists() else "null"   # built by tools/make_hand_field.py


@app.get("/", response_class=HTMLResponse)
def home():
    url = os.environ.get("TRUTHLENS_SPLINE_URL", "").strip()
    if url.startswith("https://prod.spline.design/") and url.endswith(".splinecode"):
        tag = ('<script type="module" src="https://unpkg.com/@splinetool/viewer@2.0.126/build/spline-viewer.js"></script>'
               f'<spline-viewer url="{html.escape(url)}" loading-anim-type="none"></spline-viewer>')
    else:
        tag = ""   # offline-safe default: the CSS blob + embossed rings
    return INDEX.replace("__SPLINE__", tag).replace("__HANDS__", HANDS)


# The upload page lives in web/index.html; __SPLINE__ and __HANDS__ are filled in by home().
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    print("TruthLens running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
