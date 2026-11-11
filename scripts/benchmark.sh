#!/usr/bin/env bash
# Reproduce the week 4 baseline results: 512 GPUs, 60 days, three independent fleets.
# Takes about 2 minutes per seed on a 4-core laptop. Results land in reports/baselines/.
set -euo pipefail

SEEDS="${SEEDS:-1 2 3}"
for seed in $SEEDS; do
  data="data/bench-seed${seed}"
  gpufleet simulate --nodes 64 --days 60 --failure-rate 0.004 --seed "$seed" --out "$data"
  gpufleet features "$data"
  gpufleet baselines "$data/features.parquet" --out "reports/baselines/seed${seed}"
done
python scripts/summarize_benchmark.py reports/baselines
