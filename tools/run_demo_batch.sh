#!/bin/zsh
# Runs every file in a folder through TruthLens and prints a one-line verdict per file.
# Default: demo_samples/ (local, git-ignored); in a fresh clone it falls back to the planted test_assets/.
cd "$(dirname "$0")/.."
dir=${1:-demo_samples}
[[ -d $dir ]] || dir=test_assets
for f in $dir/*; do
  echo "=== $(basename "$f")"
  .venv/bin/python cli.py "$f" 2>&1 | grep -E "likelihood|^  [A-Z0-9]+-|Report:|Traceback|Error"
done
echo ALLDONE
