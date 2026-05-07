#!/usr/bin/env bash
# Run main_experiment.py over the 5 canonical seeds, each with its own cache
# slot. Writes results to results/metrics/comparison_seed{N}.csv. Aggregation
# is done by scripts/aggregate_multi_seed.py afterwards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(3 7 21 42 99)
DEVICE="${DEVICE:-cuda:0}"

for seed in "${SEEDS[@]}"; do
  echo "==== seed=${seed} ===="
  .venv/bin/python scripts/main_experiment.py \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --raw-dir "${REPO_ROOT}/data/raw/pokec-z" \
    --cache \
    --out-csv "${REPO_ROOT}/results/metrics/comparison_seed${seed}.csv"
done

echo "==== aggregating ===="
.venv/bin/python scripts/aggregate_multi_seed.py
