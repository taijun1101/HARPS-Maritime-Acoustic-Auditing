#!/usr/bin/env bash
set -euo pipefail
: "${HARPS_DATA_ROOT:?Set HARPS_DATA_ROOT to the original experiment-ready shipsear_5s_16k folder}"
: "${HARPS_AST_PATH:?Set HARPS_AST_PATH to the original AST checkpoint folder}"
HARPS_OUTPUT_DIR="${HARPS_OUTPUT_DIR:-./results_reproduction}"
python run_experiments.py \
  --data_root "$HARPS_DATA_ROOT" \
  --ast_model_path "$HARPS_AST_PATH" \
  --output_dir "$HARPS_OUTPUT_DIR" \
  --run_all \
  --run_asymmetric \
  --run_mfcc_svdd \
  --run_pink \
  --run_sensitivity \
  --run_eigen \
  --benchmark_cost \
  --batch_size 16 \
  --run_geometry_comparison
