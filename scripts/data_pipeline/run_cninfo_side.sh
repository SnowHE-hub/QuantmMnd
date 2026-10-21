#!/bin/bash
# T-038：巨潮公告元数据的旁路下载（2026-01-01～2026-10-16），用 T-031 的同一份下载代码、同样的 1.5 秒间隔，
# 写到单独的 data-root；下完后拿住定时任务的同一把锁，合并进主目录（cninfo_side_merge.py）。
set -u
SIDE=${SIDE:-/home/lenovo/t030_scratch/cninfo_side}
MAIN=${QM_DATA_ROOT:-/home/lenovo/projects/quantmind}
PY=/home/lenovo/miniforge3/envs/quantmind/bin/python
T031=/home/lenovo/projects/qm-t031
HERE=$(cd "$(dirname "$0")/../.." && pwd)
echo "=== $(date '+%F %T %z') side download (t031 repo $(git -C $T031 rev-parse --short HEAD))"
for i in 1 2 3 4 5; do
  (cd "$T031" && PYTHONPATH=. WRITE_MODE=parquet_only OMP_NUM_THREADS=1 nice -n 10 "$PY" -u scripts/data_pipeline/download_cninfo.py \
    --data-root "$SIDE" --start 2026-01-01 --end 2026-10-16 --interval 1.5) && break
  echo "pass $i ended with failed / truncated days; retrying"
done
exec 9>/tmp/qm_cninfo_meta.lock
flock 9
echo "=== $(date '+%F %T %z') merge"
"$PY" "$HERE/scripts/data_pipeline/cninfo_side_merge.py" "$SIDE" "$MAIN"
echo "=== $(date '+%F %T %z') exit $?"
