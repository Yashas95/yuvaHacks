#!/bin/zsh
# Runs each demo sample through TruthLens and prints a one-line verdict per file.
cd "$(dirname "$0")/.."
for f in demo_samples/*; do
  echo "=== $(basename "$f")"
  .venv/bin/python cli.py "$f" 2>&1 | grep -E "likelihood|^  [A-Z0-9]+-|Report:|Traceback|Error"
done
echo ALLDONE
