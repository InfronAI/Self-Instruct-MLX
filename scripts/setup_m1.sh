#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Use native arm64 macOS on Apple Silicon (not Rosetta)." >&2
  exit 1
fi
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null; then
  echo "Install Python 3.11: brew install python@3.11" >&2
  exit 1
fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[mlx,viz,dev]'
.venv/bin/python -m self_instruct doctor
