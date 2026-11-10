#!/usr/bin/env bash
# API workflow: set OPENAI_API_KEY and OPENAI_MODEL; use run_m1_demo.sh for local MLX.
set -euo pipefail
batch_dir=${BATCH_DIR:-runs/api}
model=${OPENAI_MODEL:?Set OPENAI_MODEL to a supported Chat Completions model}

python -m self_instruct.generate_instances \
    --batch_dir "${batch_dir}" \
    --input_file machine_generated_instructions.jsonl \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_generate 5 \
    --engine "$model" \
    --request_batch_size 5
