"""Command-line run:  .venv/bin/python cli.py file1.jpg clip.mp4 --context "claimed caption" """
import argparse
import time
from pathlib import Path

from truthlens.pipeline import run_case

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*")
ap.add_argument("--context", default="")
a = ap.parse_args()
t = time.time()
case, rep = run_case([(Path(f), Path(f).name) for f in a.files], a.context,
                     lambda p, m: print(f"[{p:3d}%] {m}", flush=True))
o = rep["overall"]
print(f"\n{o['band']} — likelihood {o['likelihood']}% ({o['range'][0]}-{o['range'][1]}), confidence {o['confidence']}")
for i in rep["indicators"]:
    if i["supports"] == "manipulation":
        print(f"  {i['id']:8s} {i['score'] * i['reliability']:.2f}  {i['title']}")
print(f"\nReport: {case.dir / 'report.html'}   ({time.time() - t:.1f}s)")
