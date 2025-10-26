#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-artifacts/baseline.joblib}
PAIRS_PATH=${2:-data/manifests/sample_pairs.csv}

python -m src.train.baseline \
  --pairs "${PAIRS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --test-size 0.3 \
  --seed 13 \
  --evaluate-only
