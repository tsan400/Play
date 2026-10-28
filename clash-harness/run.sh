#!/usr/bin/env bash
# Usage: ./run.sh "<project folder>" ["<project folder>" ...]
# Analyzes each folder of IFC exports, then serves the viewer on :8765.
set -euo pipefail
cd "$(dirname "$0")"
python3 analyze.py "$@" --out viewer/data
echo "Open http://localhost:8765"
cd viewer && python3 -m http.server 8765
