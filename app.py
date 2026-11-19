"""TruthLens web app.  Run:  .venv/bin/python app.py   ->  http://localhost:8000

Hosted mode (Hugging Face Space, see Dockerfile) is switched on by environment variables:
  TRUTHLENS_HOSTED=1        honest "hosted" wording, visitors only see their own cases, unguessable case ids
  TRUTHLENS_PASSWORD=...    optional: require this password before anything is shown (set as a Space secret)
  PORT=7860                 listening port
"""
from __future__ import annotations

import hashlib
import hmac
import html
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from truthlens.config import CASES_DIR, ROOT
from truthlens.evidence import Case, modality_of
from truthlens.pipeline import run_case
from truthlens.plain import build as build_plain

app = FastAPI(title="TruthLens")
# The TruthLens browser extension (extension/) calls the API from its own chrome-extension:// origin.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://.*", allow_methods=["GET", "POST"],
                   allow_headers=["*"], allow_credentials=True)
HOSTED = bool(os.environ.get("TRUTHLENS_HOSTED"))
PASSWORD = os.environ.get("TRUTHLENS_PASSWORD", "")
_TOKEN = hashlib.sha256(("truthlens:" + PASSWORD).encode()).hexdigest() if PASSWORD else ""


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """Optional shared password for the public demo (cookie set by /login)."""
    if not PASSWORD or request.url.path in ("/login", "/favicon.ico") or \
            hmac.compare_digest(request.cookies.get("tl_auth", ""), _TOKEN):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return HTMLResponse("Password required", status_code=401)
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(error: int = 0):
    return LOGIN.replace("__ERR__", "Wrong password — try again." if error else "")


@app.post("/login")
def login(request: Request, password: str = Form(default="")):
    if not PASSWORD or not hmac.compare_digest(password, PASSWORD):
        return RedirectResponse("/login?error=1", status_code=303)
    r = RedirectResponse("/", status_code=303)
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    # SameSite=None lets the cookie work inside the Hugging Face page iframe (needs HTTPS)
    r.set_cookie("tl_auth", _TOKEN, httponly=True, secure=https, samesite="none" if https else "lax", max_age=7 * 86400)
    return r


LOGIN = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TruthLens — sign in</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f6f6f3;
font:15px -apple-system,Inter,sans-serif;color:#0d0d0f}form{background:#fff;border:1px solid #dcdfe3;border-radius:10px;padding:28px;width:min(340px,90vw)}
h1{font-size:22px;margin:0 0 4px}p{color:#6b7179;margin:0 0 16px;font-size:13px}input{width:100%;box-sizing:border-box;padding:11px 12px;
border:1px solid #dcdfe3;border-radius:8px;font:inherit}button{margin-top:12px;width:100%;padding:11px;border:0;border-radius:8px;background:#0d0d0f;
color:#fff;font:600 14px inherit;cursor:pointer}.e{color:#e5484d;font-size:13px;margin-top:10px}</style></head><body>
<form method="post" action="/login"><h1>TruthLens</h1><p>Hackathon demo — enter the access password.</p>
<input type="password" name="password" autofocus required aria-label="Password"><button>Enter</button><div class="e">__ERR__</div></form></body></html>"""
JOBS: dict = {}
LOCK = threading.Lock()  # one analysis at a time keeps the MacBook Air responsive

# Sample folders offered for one-click demo analysis (never arbitrary paths).
SAMPLE_DIRS = {"demo": ROOT / "demo_samples", "planted": ROOT / "test_assets"}
if os.environ.get("TRUTHLENS_HOSTED") and not os.environ.get("TRUTHLENS_PUBLIC_DEMO_SAMPLES"):
    # demo_samples/ holds real people's media: never serve it to the public unless explicitly opted in
    SAMPLE_DIRS = {"planted": ROOT / "test_assets"}


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


@app.get("/api/summary/{case_id}")
def summary(case_id: str):
    """Plain-English verdict for the inline result card and the browser extension."""
    p = CASES_DIR / Path(case_id).name / "report.json"
    if not p.exists():
        raise HTTPException(404)
    r = json.loads(p.read_text())
    P, ov = build_plain(r), r["overall"]
    return {"case_id": Path(case_id).name, "headline": P["headline"], "sub": P["sub"], "color": ov["color"],
            "band": ov["band"], "likelihood": ov["likelihood"], "confidence": ov["confidence"],
            "files": [{"name": f.get("name"), "origin": (f.get("origin") or {}).get("label"),
                       "ai": re.sub(r"<[^>]+>", "", f["answers"]["ai"])} for f in P["files"]],
            "next_steps": P["next_steps"][:3], "report": f"/case/{Path(case_id).name}"}


@app.get("/api/cases")
def cases(ids: str = ""):
    out = []
    dirs = sorted(CASES_DIR.glob("TL-*"), reverse=True)
    if HOSTED:   # a public server must not list other visitors' evidence: only cases this browser created
        mine = {Path(x).name for x in ids.split(",") if x}
        dirs = [d for d in dirs if d.name in mine]
    for d in dirs[:30]:
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


EXT_DIR = ROOT / "extension"
EXT_DEFAULT = "http://localhost:8000"


@app.get("/extension.zip")
def extension_zip(request: Request):
    """The Chrome extension as a ready-to-load zip, pre-set to the server it was downloaded from."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    base = f"{proto}://{host}"
    if not re.fullmatch(r"https?://[A-Za-z0-9.\-]+(:\d+)?", base):
        base = EXT_DEFAULT
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(EXT_DIR.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                data = f.read_bytes()
                if f.suffix in (".js", ".html"):
                    data = data.replace(EXT_DEFAULT.encode(), base.encode())
                z.writestr(f"truthlens-extension/{f.relative_to(EXT_DIR)}", data)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="truthlens-extension.zip"'})


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
    page = INDEX.replace("__SPLINE__", tag).replace("__HANDS__", HANDS)
    if HOSTED:   # keep the privacy wording honest when not running on the user's own machine
        for a, b in HOSTED_WORDING:
            page = page.replace(a, b)
    return page


HOSTED_WORDING = [
    ("Runs locally · nothing uploaded", "Hosted demo · no third-party APIs"),
    ("<i>DATA UPLOADED</i><b>0 B</b><small>runs fully on this machine</small>",
     "<i>THIRD-PARTY APIS</i><b>0</b><small>analysis runs on this server only</small>"),
    ("Models run locally on Apple Silicon. Large videos can take a minute.",
     "Analysis runs on the TruthLens server — large videos can take a minute or two."),
]


# The upload page lives in web/index.html; __SPLINE__ and __HANDS__ are filled in by home().
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"TruthLens running at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
