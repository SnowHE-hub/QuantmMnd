#!/bin/bash
# T-031：report_rc（卖方盈利预测）每日慢速回补。2000 积分档每天只能调 10 次，由定时任务每天跑一次，断点续跑。
# 回补区间固定为 2018-10-01 ~ 2026-10-16（196 个 15 天窗口；季报季一个窗口最多约 5 次调用，按前几个窗口推算约 30–60 天下完）；之后的增量另行配置。
# 退出码：0 = 全部窗口完成；3 = 今日额度或调用预算用完（正常，明天继续）；2 = 有 failed / truncated（明天重试）。
# 日志：<data>/data/lake/events/_logs/report_rc_daily.log
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${QM_DATA_ROOT:-/home/lenovo/projects/quantmind}
PY=${PY:-/home/lenovo/miniforge3/envs/quantmind/bin/python}
LOG_DIR=$DATA/data/lake/events/_logs
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
export PYTHONPATH=. WRITE_MODE=parquet_only OMP_NUM_THREADS=1
{
  echo "=== $(date '+%F %T %z') start (repo $(git rev-parse --short HEAD))"
  flock -n /tmp/qm_report_rc_daily.lock "$PY" -u scripts/data_pipeline/download_events.py \
    --data-root "$DATA" --apis report_rc --start 2018-10-01 --end 2026-10-16 --max-calls 10 --rolling-limit 10 --rolling-hours 24 --rolling-max-wait-min 40
  rc=$?
  echo "=== $(date '+%F %T %z') exit $rc"
} >> "$LOG_DIR/report_rc_daily.log" 2>&1
