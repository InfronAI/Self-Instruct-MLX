#!/usr/bin/env bash
# Optional cloud upload + paid fine-tuning job. Local training: run_m1_demo.sh.
set -euo pipefail
exec python -m self_instruct.finetune_api "$@"
