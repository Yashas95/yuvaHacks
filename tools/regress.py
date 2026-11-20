"""Golden regression check: prove a refactor or rule change didn't silently change any result.

  record:  .venv/bin/python tools/regress.py record  [golden.json] [files...]
  check:   .venv/bin/python tools/regress.py check   [golden.json] [files...]

For every file it stores the verdict (likelihood, band, confidence) and the list of indicators
(detector, supports, score rounded to 2 decimals). `check` re-runs the same files and prints every
difference; exit code 1 if anything changed. With no files given it uses test_assets/ (synthetic, safe
to commit) plus demo_samples/ if present. Golden files for personal media belong outside the repo.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from truthlens.evidence import Case  # noqa: E402
from truthlens.pipeline import run_case  # noqa: E402

MEDIA = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".wav", ".mp3", ".m4a"}


def default_files() -> list[Path]:
    out = []
    for d in (ROOT / "test_assets", ROOT / "demo_samples"):
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.suffix.lower() in MEDIA)
    return out


def snapshot(path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:   # throwaway case folder: a run must not leave copies of the media behind
        case = Case(root=Path(tmp))
        run_case([(path, path.name)], "", None, case=case)
        r = json.loads((case.dir / "report.json").read_text())
    o = r["overall"]
    inds = sorted((i["detector"], i["supports"], round(i["score"], 2)) for i in r["indicators"])
    return {"likelihood": o["likelihood"], "band": o["band"], "confidence": o["confidence"],
            "indicators": [list(x) for x in inds]}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "check"):
        sys.exit(__doc__)
    mode = sys.argv[1]
    golden = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "tests" / "golden.json"
    files = [Path(f) for f in sys.argv[3:]] or default_files()
    snaps = {}
    for f in files:
        snaps[f.name] = snapshot(f)
        s = snaps[f.name]
        print(f"{f.name[:40]:42s}{s['likelihood']:>3}%  {s['band'][:34]:36s}{s['confidence']}")
    if mode == "record":
        golden.write_text(json.dumps(snaps, indent=1, sort_keys=True))
        print(f"recorded {len(snaps)} file(s) → {golden}")
        return
    want = json.loads(golden.read_text())
    diffs = 0
    for name, s in snaps.items():
        w = want.get(name)
        if w is None:
            print(f"  NEW   {name}: not in golden file")
            continue
        for k in ("likelihood", "band", "confidence"):
            if s[k] != w[k]:
                diffs += 1
                print(f"  DIFF  {name}: {k} {w[k]} → {s[k]}")
        a, b = {tuple(x) for x in w["indicators"]}, {tuple(x) for x in s["indicators"]}
        for x in sorted(a - b):
            diffs += 1
            print(f"  GONE  {name}: {x}")
        for x in sorted(b - a):
            diffs += 1
            print(f"  ADDED {name}: {x}")
    print("no differences" if not diffs else f"{diffs} difference(s)")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
