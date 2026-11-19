#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, face-detector weights, and all Hugging Face models (for offline demos).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg/ffprobe not found (macOS: brew install ffmpeg) — video/audio analysis needs them."

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

mkdir -p weights
[ -f weights/face_detection_yunet_2023mar.onnx ] || curl -L -o weights/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

.venv/bin/python download_models.py
.venv/bin/python tests/test_core.py
echo "Done. Start the web app:  .venv/bin/python app.py   ->  http://localhost:8000"
