#!/usr/bin/env bash
# API workflow: set OPENAI_API_KEY and OPENAI_MODEL; use run_m1_demo.sh for local MLX.
set -euo pipefail
batch_dir=${BATCH_DIR:-runs/api}
model=${OPENAI_MODEL:?Set OPENAI_MODEL to a supported Chat Completions model}

python -m self_instruct.prepare_for_finetuning \
    --instance_files "${batch_dir}/machine_generated_instances.jsonl" \
    --classification_type_files "${batch_dir}/is_clf_or_not_${model}_template_1.jsonl" \
    --output_dir "${batch_dir}/finetuning_data" \
    --include_seed_tasks \
    --seed_tasks_path data/seed_tasks.jsonl
