#!/usr/bin/env bash
# API workflow: set OPENAI_API_KEY and OPENAI_MODEL; use run_m1_demo.sh for local MLX.
set -euo pipefail
batch_dir=${BATCH_DIR:-runs/api}
model=${OPENAI_MODEL:?Set OPENAI_MODEL to a supported Chat Completions model}

python -m self_instruct.identify_clf_or_not \
    --batch_dir "${batch_dir}" \
    --engine "$model" \
    --request_batch_size 5
