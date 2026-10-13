#!/bin/bash
# T-030 任务队列：按行顺序执行 <queue_file> 里的命令（相对仓库根目录；空行与 # 开头的行跳过）。
# - 每个任务自带断点续跑（台账 / checkpoint / 已写出的折文件），所以整条队列可以随时停、随时重启。
# - 退出码 75 = 内存守卫让路给同机生产任务 → 等 5 分钟重跑同一任务（最多 100 次）；其他非零退出码 → 记 FAILED，继续下一个。
# - 状态：<log_dir>/queue_<name>.status（每个任务一行：开始 / 结束时间、退出码）；每个任务的输出：<log_dir>/<序号>_<名>.log
# 用法：setsid nohup bash run_queue.sh <queue_file> <log_dir> > <log_dir>/queue_<name>.out 2>&1 < /dev/null &
set -u
QUEUE=$1
LOGDIR=$2
NAME=$(basename "$QUEUE" .txt)
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PY=${PY:-/home/lenovo/miniforge3/envs/qm-gpu/bin/python}
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/queue_${NAME}.status"
cd "$REPO" || exit 1
export PYTHONPATH=. WRITE_MODE=parquet_only
i=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  i=$((i + 1))
  tag=$(printf '%02d_%s' "$i" "$(echo "$line" | sed -E 's#[^ ]*/##g; s/\$PY|env|nice -n [0-9]+|[A-Z_]+=[^ ]*//g; s/[^A-Za-z0-9,]+/_/g; s/^_+|_+$//g' | cut -c1-70)")
  if grep -q "^DONE $tag " "$STATUS" 2>/dev/null; then
    continue
  fi
  tries=0
  while :; do
    echo "START $tag $(date '+%F %T')" >> "$STATUS"
    eval "$line" >> "$LOGDIR/$tag.log" 2>&1 < /dev/null
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "DONE $tag $(date '+%F %T') rc=0" >> "$STATUS"
      break
    elif [ $rc -eq 75 ] && [ $tries -lt 100 ]; then
      tries=$((tries + 1))
      echo "RAMGUARD $tag $(date '+%F %T') retry=$tries" >> "$STATUS"
      sleep 300
    else
      echo "FAILED $tag $(date '+%F %T') rc=$rc" >> "$STATUS"
      break
    fi
  done
done < "$QUEUE"
echo "QUEUE_END $NAME $(date '+%F %T')" >> "$STATUS"
