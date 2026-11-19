#!/usr/bin/env bash
# Started by Codespaces on every start: run TruthLens (hosted mode) in the background, detached from this shell.
cd "$(dirname "$0")/.."
mkdir -p cases
pkill -f "python app.py" 2>/dev/null || true
export TRUTHLENS_HOSTED=1 PORT=7860
setsid nohup python app.py > /tmp/truthlens.log 2>&1 < /dev/null &
echo "TruthLens starting on port 7860 (log: /tmp/truthlens.log)"
