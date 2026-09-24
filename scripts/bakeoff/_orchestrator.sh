#!/usr/bin/env bash
# 全自动编排器：顺序跑完所有未完成的桶（可断点续跑、loky 每桶清理、每模型出 p3f）。
# 脱离运行：setsid nohup bash _orchestrator.sh。Agent 只快速轮询 _orchestrator_status.txt + 内存；
# 死了就重启（它会跳过已完成的桶）。一个进程跑完全链，不依赖 harness 任务寿命/通知。
set -uo pipefail
cd /home/lenovo/projects/quantmind
PY=/home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python
PREDS=data/bakeoff/preds
ST=scripts/bakeoff/_orchestrator_status.txt
echo "ORCH_START $(date +%F_%H:%M:%S) pid=$$" >> "$ST"
pkill -9 -f _memmon.sh 2>/dev/null || true
bash scripts/bakeoff/_memmon.sh >/dev/null 2>&1 &
for mb in gru:0 gru:1 gru:2 lstm:0 lstm:1 lstm:2 transformer:0 transformer:1 transformer:2; do
  M="${mb%:*}"; B="${mb#*:}"
  f="$PREDS/${M}_alpha360_12d_quarterly_s0_b${B}of3.parquet"
  if [ -f "$f" ]; then echo "SKIP ${M}_b${B} (done) $(date +%H:%M)" >> "$ST"; continue; fi
  echo "START ${M}_b${B} $(date +%H:%M:%S)" >> "$ST"
  QM_N_BUCKETS=3 QM_BUCKET="$B" "$PY" -u scripts/bakeoff/p3c_sequence.py \
    --model "$M" --period 12d --refit quarterly --seed 0 \
    > "scripts/bakeoff/_orch_${M}_b${B}.log" 2>&1
  rc=$?
  echo "END ${M}_b${B} rc=${rc} $(date +%H:%M:%S)" >> "$ST"
  pkill -9 -f "p3c_sequence.py --model ${M}" 2>/dev/null || true   # 杀残留 loky worker
  sleep 3
  if [ "$B" = "2" ]; then
    echo "P3F after ${M} $(date +%H:%M)" >> "$ST"
    "$PY" scripts/bakeoff/p3f_bucket_eval.py > "scripts/bakeoff/_p3f_after_${M}.log" 2>&1 || true
  fi
done
"$PY" scripts/bakeoff/p3f_bucket_eval.py > scripts/bakeoff/_p3f_final.log 2>&1 || true
pkill -9 -f _memmon.sh 2>/dev/null || true
echo "ALL_DONE $(date +%F_%H:%M:%S)" >> "$ST"
