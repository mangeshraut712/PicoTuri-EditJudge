#!/usr/bin/env bash
set -euo pipefail

PAIRS_PATH=${1:-data/manifests/sample_pairs.csv}
MODEL_PATH=${2:-artifacts/baseline.joblib}

python -m src.train.baseline \
  --pairs "${PAIRS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --test-size 0.3 \
  --seed 13
