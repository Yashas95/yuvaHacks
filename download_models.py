"""Pre-download every model so the demo works fully offline.

Usage:  .venv/bin/python download_models.py
"""
import sys

from huggingface_hub import list_repo_files, snapshot_download

sys.path.insert(0, ".")
from truthlens.config import MODELS  # noqa: E402

SKIP = (".onnx", ".h5", ".msgpack", ".ot", ".tflite", ".mlmodel", ".pb")

for key, candidates in MODELS.items():
    ok = False
    for mid in candidates:
        try:
            files = list_repo_files(mid)
            has_st = any(f.endswith(".safetensors") for f in files)
            allow = [f for f in files if not f.endswith(SKIP) and not f.startswith(("onnx/", "tf_", "flax_"))
                     and not (has_st and f.endswith((".bin", ".pt", ".pth")))]
            snapshot_download(mid, allow_patterns=allow)
            print(f"[OK]   {key:15s} {mid}", flush=True)
            ok = True
            break
        except Exception as e:
            print(f"[FAIL] {key:15s} {mid}: {type(e).__name__}: {str(e)[:150]}", flush=True)
    if not ok:
        print(f"[MISS] {key}: no candidate downloaded — that detector will be skipped and reported as a limitation.")
print("DONE")
