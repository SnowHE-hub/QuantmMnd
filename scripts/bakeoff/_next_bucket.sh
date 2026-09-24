#!/usr/bin/env bash
# 自驱动接力判定器：输出 RUNNING / NEXT <model> <bucket> / ALL_DONE
# 顺序：gru 0/1/2 → lstm 0/1/2 → transformer 0/1/2（gru:0 已完成）
cd /home/lenovo/projects/quantmind
if pgrep -f "p3c_sequence.py --model" >/dev/null 2>&1; then echo "RUNNING"; exit 0; fi
PREDS=data/bakeoff/preds
for mb in gru:0 gru:1 gru:2 lstm:0 lstm:1 lstm:2 transformer:0 transformer:1 transformer:2; do
  M="${mb%:*}"; B="${mb#*:}"
  f="$PREDS/${M}_alpha360_12d_quarterly_s0_b${B}of3.parquet"
  if [ ! -f "$f" ]; then echo "NEXT $M $B"; exit 0; fi
done
echo "ALL_DONE"
