#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -u -m self_instruct run --config configs/m1_demo.json --output "${1:-runs/m1-demo}"
