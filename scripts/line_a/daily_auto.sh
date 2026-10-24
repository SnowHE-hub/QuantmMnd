#!/bin/bash
# 主线 A v2 每日运行的自动启动（用户 2026-10-24 批准恢复正式荐股；cron 在悉尼时间周二至周六 00:30，即北京时间周一至周五 21:30）
# - END = 北京时间今天（必须是上交所交易日，见 data/lake/trade_cal_sse_future.parquet）；
# - LAST = 最近一次 ALL DONE 的 END；
# - END 已经跑完（ALL DONE 且 formal 成功）就退出；正在跑就退出；
# - 在 MemoryMax 受限的 systemd 用户单元 qm-daily-<YYYYMMDD> 里运行 daily_shadow.sh（含研究、正式日报、Agent 大脑）。
set -u
R=/home/lenovo/projects/quantmind; W=/home/lenovo/projects/qm-t035; LOGS=/home/lenovo/t030_scratch/logs/daily
P=/home/lenovo/miniforge3/envs/qm-gpu/bin/python
END=$(TZ=Asia/Shanghai date +%F); TAG=${END//-/}
echo "$(date '+%F %T') daily_auto: END=$END"
is_open=$($P -c "
import pandas as pd
c = pd.read_parquet('$R/data/lake/trade_cal_sse_future.parquet')['cal_date']
print(int(pd.Timestamp('$END') in set(pd.to_datetime(c))))")
[ "$is_open" = 1 ] || { echo "not a trading day, skip"; exit 0; }
S=$LOGS/$TAG/status
if [ -f "$S" ] && grep -q '^ALL DONE' "$S" && grep -q '^formal exit 0' "$S"; then echo "already done"; exit 0; fi
systemctl --user is-active -q "qm-daily-$TAG" && { echo "already running"; exit 0; }
LAST=$(for d in $(ls -d $LOGS/2*/ 2>/dev/null | sort -r); do t=$(basename $d); [ "$t" \< "$TAG" ] && grep -q '^ALL DONE' $d/status 2>/dev/null && { echo ${t:0:4}-${t:4:2}-${t:6:2}; break; }; done)
[ -n "$LAST" ] || { echo "no previous ALL DONE run found"; exit 1; }
echo "LAST=$LAST"
cd $W && bash /home/lenovo/c3_scratch/capped.sh "daily-$TAG" 7G "bash $W/scripts/line_a/daily_shadow.sh $END $LAST"
