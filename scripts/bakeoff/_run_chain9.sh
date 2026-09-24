#!/usr/bin/env bash
# 单一 9-桶链：3 模型 × 3 流动性桶（各 ~458 票，单跑峰值~6.5GB<12GB）。
# 用脚本文件而非内联 -lic 串：可按文件名干净 kill（pkill -f _run_chain9.sh），无 tty 乱码。
set -uo pipefail
cd /home/lenovo/projects/quantmind
: > scripts/bakeoff/_memmon.log
bash scripts/bakeoff/_memmon.sh &
MEMPID=$!
PY=/home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python
export QM_N_BUCKETS=3
for M in gru lstm transformer; do
  for B in 0 1 2; do
    echo "=== START ${M} bucket${B} $(date +%T) ==="
    QM_BUCKET=$B "$PY" -u scripts/bakeoff/p3c_sequence.py \
      --model "$M" --period 12d --refit quarterly --seed 0
    echo "=== END ${M} bucket${B} rc=$? $(date +%T) ==="
  done
done
kill "$MEMPID" 2>/dev/null
echo "CHAIN9_ALL_DONE $(date +%T)"
