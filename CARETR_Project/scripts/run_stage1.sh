#!/usr/bin/env bash
# Train Stage 1 (Binary CARE Transformer) with the paper configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
LOG_DIR="$ROOT/results/stage1"
mkdir -p "$LOG_DIR" "$ROOT/checkpoints"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" "$ROOT/stage1_care_transformer/train_stage1.py" \
  --data-root "$ROOT/data/grids_448" \
  --save-dir "$ROOT/checkpoints" \
  --results-dir "$ROOT/results/stage1" \
  --tag caret_s0_448_bal \
  "$@" 2>&1 | tee "$LOG_DIR/train_stage1.log"
