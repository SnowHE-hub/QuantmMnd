#!/usr/bin/env bash
# 单桶单任务：bash _run_one.sh <model> <bucket>  （~84min < harness 2h 上限，可完整落盘）
# 停：pkill -f _run_one.sh ; pkill -f p3c_sequence
set -uo pipefail
cd /home/lenovo/projects/quantmind
M="$1"; B="$2"
: > scripts/bakeoff/_memmon.log
bash scripts/bakeoff/_memmon.sh & MP=$!
echo "=== START ${M} bucket${B} $(date +%T) ==="
QM_N_BUCKETS=3 QM_BUCKET="$B" /home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python -u \
  scripts/bakeoff/p3c_sequence.py --model "$M" --period 12d --refit quarterly --seed 0
RC=$?
kill "$MP" 2>/dev/null
pkill -9 -f _memmon.sh 2>/dev/null
# 主 python 已退出；杀残留 loky fork worker（同 cmdline），否则它们挂住 wsl.exe → 任务延迟数小时完成 → 接力停滞
pkill -9 -f "p3c_sequence.py --model ${M}" 2>/dev/null
sleep 1
echo "=== END ${M} bucket${B} rc=${RC} $(date +%T) ==="
exit 0
