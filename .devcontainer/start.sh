#!/usr/bin/env bash
# Started by Codespaces on every start: run TruthLens (hosted mode) in the background, detached from this shell.
cd "$(dirname "$0")/.."
mkdir -p cases
# The container already runs TruthLens as its main process; only start a copy if nothing is serving 7860.
if curl -s -o /dev/null --max-time 5 http://localhost:7860/; then echo "TruthLens already running on 7860"; exit 0; fi
export TRUTHLENS_HOSTED=1 PORT=7860
setsid nohup python app.py > /tmp/truthlens.log 2>&1 < /dev/null &
echo "TruthLens starting on port 7860 (log: /tmp/truthlens.log)"
