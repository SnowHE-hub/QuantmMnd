#!/usr/bin/env bash
# 每 5s 采样 WSL 可用内存 + 最大 python RSS，写 _memmon.log，跑 ~25min。
OUT=/home/lenovo/projects/quantmind/scripts/bakeoff/_memmon.log
: > "$OUT"
for i in $(seq 1 300); do
  avail=$(free -m | awk '/^Mem:/{print $7}')
  used=$(free -m | awk '/^Mem:/{print $3}')
  rss=$(ps -eo rss,comm | awk '$2 ~ /python/{print $1}' | sort -n | tail -1)
  echo "$(date +%T) used_MB=${used} avail_MB=${avail} max_py_rss_KB=${rss:-0}" >> "$OUT"
  sleep 5
done
