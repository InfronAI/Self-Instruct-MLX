#!/usr/bin/env bash
# API workflow: set OPENAI_API_KEY and OPENAI_MODEL; use run_m1_demo.sh for local MLX.
set -euo pipefail
batch_dir=${BATCH_DIR:-runs/api}
model=${OPENAI_MODEL:?Set OPENAI_MODEL to a supported Chat Completions model}

python -m self_instruct.bootstrap_instructions \
    --batch_dir "${batch_dir}" \
    --num_instructions_to_generate 50000 \
    --seed_tasks_path data/seed_tasks.jsonl \
    --engine "$model"
