#!/bin/bash
# T-031：巨潮公告元数据回补（2019-01-01 ~ 2026-10-16）。由定时任务每小时调用一次：每次最多跑 55 分钟后在两天之间退出，
# 文件锁防止重入；断点续跑，WSL 重启或进程被杀后下一个整点自动接上。请求间隔 1.5 秒、单线程。
# 退出码：0 = 全部日期完成；2 = 有 failed / truncated（下次运行重试）。日志：<data>/data/lake/text/cninfo/_logs/cninfo_meta.log
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${QM_DATA_ROOT:-/home/lenovo/projects/quantmind}
PY=${PY:-/home/lenovo/miniforge3/envs/quantmind/bin/python}
LOG_DIR=$DATA/data/lake/text/cninfo/_logs
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
export PYTHONPATH=. WRITE_MODE=parquet_only OMP_NUM_THREADS=1
{
  echo "=== $(date '+%F %T %z') start (repo $(git rev-parse --short HEAD))"
  flock -n /tmp/qm_cninfo_meta.lock nice -n 10 "$PY" -u scripts/data_pipeline/download_cninfo.py \
    --data-root "$DATA" --start 2019-01-01 --end 2026-10-16 --interval 1.5 --max-seconds 3300
  rc=$?
  echo "=== $(date '+%F %T %z') exit $rc"
} >> "$LOG_DIR/cninfo_meta.log" 2>&1
