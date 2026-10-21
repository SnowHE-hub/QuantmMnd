#!/bin/bash
cd /home/lenovo/projects/qm-t035
R=/home/lenovo/projects/quantmind/data
echo "=== x14 $(date +%T)"
PYTHONPATH=. python -u scripts/line_a/run_x14_injection.py --out $R/line_a_v2/eval/x14_20261021 --budget 2
echo "=== heldout v2 $(date +%T)"
PYTHONPATH=. python -u scripts/line_a/run_eval_a1.py --lake $R/lake --out $R/line_a_v2/eval/a1_v2_heldout_20261021 --budget 5 \
  --seed 20261022 --as-ofs 2024-05-10,2024-09-10,2024-11-11,2025-05-12
echo "=== dev v2 $(date +%T)"
PYTHONPATH=. python -u scripts/line_a/run_eval_a1.py --lake $R/lake --out $R/line_a_v2/eval/a1_v2_dev_20261021 --budget 5
