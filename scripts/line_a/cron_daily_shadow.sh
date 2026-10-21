#!/bin/bash
# 主线 A v2 每日影子运行的 cron 入口（用户 2026-10-21 批准：每天 00:30 悉尼时间运行；影子数据根只保留最近 3 个）。
# - END = 北京时间的今天（00:30 悉尼 = 前一日 21:30 北京，当天行情已经入库）；不是交易日就只记录、不运行；
# - LAST = 上一次 ALL DONE 的 END（~/t030_scratch/logs/daily/last_end）；
# - 整个流程在一个 MemoryMax=6G 的 systemd 用户单元里运行；flock 防止重叠；
# - ALL DONE 之后更新 last_end，并按保留策略删除最旧的影子数据根（~/qm_fwd_root/<日期>，可由数据湖与冻结脚本重新生成）。
#   价格面板、qlib 前瞻目录、事件表等数据产品不在删除范围内。
set -u
W=/home/lenovo/projects/qm-t035
D=/home/lenovo/t030_scratch/logs/daily; mkdir -p $D
P=/home/lenovo/miniforge3/envs/quantmind/bin/python
KEEP=3
exec 9> /tmp/qm_daily_shadow.lock
flock -n 9 || { echo "$(date '+%F %T') already running" >> $D/cron.log; exit 0; }
END=${END_OVERRIDE:-$(TZ=Asia/Shanghai date +%F)}   # END_OVERRIDE：手动补跑某一天（例如被 OOM 杀掉之后）
LAST=$(cat $D/last_end 2>/dev/null || echo 2026-10-20)
cd $W || exit 1
OPEN=$(PYTHONPATH=. $P - <<PY 2>>$D/cron.log
import sys
sys.path.insert(0, ".")
import scripts.backfill_tushare as bf
bf.REPO_ROOT = "/home/lenovo/projects/quantmind"
bf._load_token_into_env()
from quantmind.data import tushare_provider as tp
try:
    from loguru import logger as _lg
    _lg.remove()
except Exception:
    pass
d = "$END".replace("-", "")
df = tp._call("trade_cal", exchange="SSE", start_date=d, end_date=d)
print(int(df["is_open"].iloc[0]) if df is not None and len(df) else -1)
PY
)
echo "$(date '+%F %T') END=$END LAST=$LAST open=$OPEN" >> $D/cron.log
[ "$OPEN" = "1" ] || exit 0
[ "$END" \> "$LAST" ] || exit 0
TAG=${END//-/}
# cron 环境里没有用户会话的总线变量：补上；用户管理器不在时退回直接运行（WSL 总内存已限 8 GB）
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
if [ -S "$XDG_RUNTIME_DIR/bus" ]; then
  systemctl --user reset-failed "qm-daily-$TAG" 2>/dev/null
  systemd-run --user --wait --collect --unit="qm-daily-$TAG" -q -p MemoryMax=6G -p MemorySwapMax=0 \
    --working-directory=$W /bin/bash $W/scripts/line_a/daily_shadow.sh "$END" "$LAST" >> $D/cron.log 2>&1
else
  echo "$(date '+%F %T') no user bus; running without cgroup cap" >> $D/cron.log
  nice -n 5 /bin/bash $W/scripts/line_a/daily_shadow.sh "$END" "$LAST" >> $D/cron.log 2>&1
fi
if grep -q "^ALL DONE" "$D/$TAG/status" 2>/dev/null; then
  echo "$END" > $D/last_end
  ls -d /home/lenovo/qm_fwd_root/2[0-9]* 2>/dev/null | sort | head -n -$KEEP | while read -r old; do
    echo "$(date '+%F %T') retention: remove $old" >> $D/cron.log
    rm -rf -- "$old"
  done
  echo "$(date '+%F %T') done $END" >> $D/cron.log
else
  echo "$(date '+%F %T') FAILED $END (see $D/$TAG/status)" >> $D/cron.log
fi
