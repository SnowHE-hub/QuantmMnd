#!/usr/bin/env bash
# 编排器状态速查（供 agent 轮询）：输出 ALL_DONE / ALIVE / DEAD + 最近状态 + 内存 + 已完成桶数。
cd /home/lenovo/projects/quantmind
ST=scripts/bakeoff/_orchestrator_status.txt
if grep -q ALL_DONE "$ST" 2>/dev/null; then echo "STATE=ALL_DONE"
elif pgrep -f "_orchestrator.sh" >/dev/null 2>&1; then echo "STATE=ALIVE"
else echo "STATE=DEAD"; fi
echo "last: $(tail -1 "$ST" 2>/dev/null)"
echo "mem_used_MB=$(free -m | awk '/^Mem/{print $3}')  avail_MB=$(free -m | awk '/^Mem/{print $7}')"
echo "buckets_done=$(ls data/bakeoff/preds/*alpha360*_b?of3.parquet 2>/dev/null | wc -l)/9"
echo "p3c_running=$(pgrep -cf 'p3c_sequence.py --model')"
