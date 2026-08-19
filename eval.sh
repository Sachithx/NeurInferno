#!/usr/bin/env bash
# Evaluate trained checkpoints and compare against the reference tables.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash eval.sh
#   bash eval.sh --ckpt_root checkpoints/seed789

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-.}"

CKPT_ROOT="checkpoints/seed789"
TIER2="data/protocols"
RESULTS_DIR="results/seed789"
THRESHOLD=0.75
MAX_MSGS=1000

if [[ $# -gt 0 ]]; then
    # allow --ckpt_root PATH and other run_eval flags
    python -m src.evaluation.run_eval \
        --ckpt_root   "$CKPT_ROOT" \
        --tier2_dir   "$TIER2" \
        --results_dir "$RESULTS_DIR" \
        --threshold   "$THRESHOLD" \
        --max_msgs    "$MAX_MSGS" \
        "$@"
else
    python -m src.evaluation.run_eval \
        --ckpt_root   "$CKPT_ROOT" \
        --tier2_dir   "$TIER2" \
        --results_dir "$RESULTS_DIR" \
        --threshold   "$THRESHOLD" \
        --max_msgs    "$MAX_MSGS"
fi

python -m src.evaluation.compare_results \
    --got "$RESULTS_DIR" \
    --ref results/reference
