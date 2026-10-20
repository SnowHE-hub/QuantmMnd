#!/bin/bash
# T-031 L4：前瞻新闻采集，由定时任务每 10 分钟调用一次（来源每次只返回最近约 20 条）。文件锁防重入。
# 日志：<data>/data/lake/text/news_fwd/_logs/news_fwd_<YYYYMM>.log
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${QM_DATA_ROOT:-/home/lenovo/projects/quantmind}
PY=${PY:-/home/lenovo/miniforge3/envs/quantmind/bin/python}
LOG_DIR=$DATA/data/lake/text/news_fwd/_logs
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
export PYTHONPATH=. WRITE_MODE=parquet_only OMP_NUM_THREADS=1
flock -n /tmp/qm_news_fwd.lock timeout 300 nice -n 10 "$PY" -u scripts/data_pipeline/collect_news_fwd.py \
  --data-root "$DATA" >> "$LOG_DIR/news_fwd_$(date +%Y%m).log" 2>&1
