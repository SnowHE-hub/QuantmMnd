#!/bin/bash
# 主线 A v2 每日影子运行（T-036；03_final_plan_v2 第 13 节 P1）。只读研究与影子，不发正式日报，不形成交易指令。
# 用法：bash scripts/line_a/daily_shadow.sh <END YYYY-MM-DD> <LAST_END YYYY-MM-DD>
#   END：这次数据补到的交易日；LAST_END：上一次成功运行的 END（决定补跑哪些交易日、财报触发窗口的起点）。
# 步骤（任何一步失败就停，状态写在 ~/t030_scratch/logs/daily/<END>/status）：
#   1 数据：行情序列 → 事件表（最近 2 个分块强制重拉，处理迟到与修订）→ 合并 → 股票基础表 → 财报三队列 → 指数补充 → F0 快照
#   2 影子数据根：价格面板 → 股票基础表合并 → qlib 前瞻目录 → 链接冻结月份 → 原样运行冻结的构建器
#   3 日任务：(LAST_END, END] 内每个交易日跑 M、F0–F5c、F9（已成功的任务键直接复用）
#   4 账本：截至 END 的前瞻账本与容量情景
#   5 影子研究（T-040）：END 当天的 F6 研究、F7 验收、F9 研究报告
#   6 正式荐股日报（T-057）：三档纸面账户的持仓与明日委托、Agent 大脑的提示（T-056）
# 巨潮公告（第 1 步里）和影子研究（第 5 步）是软步骤：失败只记录，不阻断量化部分，也不影响 ALL DONE。
# 不删除任何旧的影子数据根（保留策略需要用户批准）；每个重任务都在 MemoryMax 受限的 systemd 用户单元里由调用方运行。
set -u
END=$1; LAST=$2; TAG=${END//-/}
R=/home/lenovo/projects/quantmind; W=/home/lenovo/projects/qm-t035; FR=/home/lenovo/qm_fwd_root/$TAG
L=/home/lenovo/t030_scratch/logs/daily/$TAG; mkdir -p "$L"; S=$L/status
P=/home/lenovo/miniforge3/envs/quantmind/bin/python
G=/home/lenovo/miniforge3/envs/qm-gpu/bin/python
Q=/home/lenovo/miniforge3/envs/qlib_bakeoff/bin/python
EV=stk_limit,suspend_d,namechange,forecast,express,disclosure_date,stk_holdernumber,stk_holdertrade,repurchase,share_float,moneyflow,top_list,top_inst,block_trade,index_weight,pledge_stat
cd "$W" || exit 1
step() {  # name, command...
  local n=$1; shift
  if grep -q "^$n exit 0" "$S" 2>/dev/null; then echo "$n skip (done)" >> "$S"; return 0; fi
  echo "$n start $(date '+%F %T')" >> "$S"
  PYTHONPATH=. WRITE_MODE=parquet_only "$@" > "$L/$n.log" 2>&1
  local rc=$?
  echo "$n exit $rc $(date '+%F %T')" >> "$S"
  [ $rc -eq 0 ] || { echo "STOP at $n" >> "$S"; exit $rc; }
}
soft_step() {  # 同 step，但失败只记录、继续往下走
  local n=$1; shift
  if grep -q "^$n exit 0" "$S" 2>/dev/null; then echo "$n skip (done)" >> "$S"; return 0; fi
  echo "$n start $(date '+%F %T')" >> "$S"
  PYTHONPATH=. WRITE_MODE=parquet_only "$@" > "$L/$n.log" 2>&1
  echo "$n exit $? $(date '+%F %T') (soft)" >> "$S"
}
CN0=$(date -d "$END -3 day" +%F)
# 本机 Agent 后端（llama-server，T-048）常驻时占约 6 GB 显存；每日运行的 ENS 打分要用显卡，开始前先让它退出
systemctl --user stop qm-llama9b 2>/dev/null && echo "stopped qm-llama9b $(date '+%F %T')" >> "$S"
# 1 数据
step lake        $P -u scripts/data_pipeline/update_lake_latest.py --end "$END" --data-root "$R"
step events      $P -u scripts/data_pipeline/download_events.py --data-root "$R" --apis $EV --start 2018-01-01 --end "$END" --refresh-recent 2
step events_cmp  $P -u scripts/data_pipeline/download_events.py --data-root "$R" --apis $EV --end "$END" --compact
step stock_basic $P -u scripts/data_pipeline/update_fundamentals_latest.py --data-root "$R" --end "$END" --stock-basic-only
step fin_queues  $P -u scripts/data_pipeline/update_fundamentals_latest.py --data-root "$R" --end "$END" --queues-since "$LAST" --workers 2
step index_extra $P -u scripts/data_pipeline/download_index_extra.py --data-root "$R" --end "$END"
# 巨潮公告元数据：最近 4 天强制重抓并合并（补迟到的公告）；拿巨潮下载的文件锁，最多等 30 分钟
soft_step cninfo flock -w 1800 /tmp/qm_cninfo_meta.lock $P -u scripts/data_pipeline/download_cninfo.py --data-root "$R" \
     --start "$CN0" --end "$END" --refresh-recent 4 --interval 1.5
# 卖方盈利预测（report_rc）每日增量：一个 [END − 3, END] 窗口，最多 2 次调用（2000 积分档每 24 小时 10 次；T-040）
soft_step report_rc $P -u scripts/data_pipeline/download_events.py --data-root "$R" --apis report_rc \
     --start "$CN0" --end "$END" --max-calls 2
step snapshot    $P -m quantmind.line_a.snapshot --data-root "$R" --as-of "$END"
# 2 影子数据根
step prices      $P -u scripts/data_pipeline/build_price_panels.py --end "$END" --data-root "$R"
step sb_merge    $P docs/work/evidence/T-036/merge_stock_basic_fwd.py "$R" "$TAG" "$TAG"
step qlib_dump   $Q docs/work/evidence/T-036/dump_qlib_fwd.py "$W" "$R" "$TAG"
step fwd_root    $P docs/work/evidence/T-036/make_fwd_root.py "$R" "$TAG" "$FR"
step builders    bash docs/work/evidence/T-036/run_builders_fwd.sh "$FR"
# 3 日任务（交易日历来自湖里的 trade_cal_sse）
DAYS=$($P -c "
import pandas as pd
c = pd.to_datetime(pd.read_parquet('$R/data/lake/trade_cal_sse.parquet')['cal_date'].astype(str))
print(' '.join(str(d.date()) for d in sorted(c) if pd.Timestamp('$LAST') < d <= pd.Timestamp('$END')))")
for d in $DAYS; do
  step "day_$d" $G -u -m quantmind.line_a.pipeline day --as-of "$d" --data-root "$R" --model-root "$FR/main" \
       --bundle20 "$R/data/gpu_v1/forward_models/20d_fold18" --bundle63 "$R/data/gpu_v1/forward_models/63d_fold8"
done
# 4 账本
step books $G -u -m quantmind.line_a.pipeline books --as-of "$END" --data-root "$R" --model-root "$FR/main" \
     --prices "$R/data/raw/alpha_prices_panel_v6_$TAG.parquet"
echo "ALL DONE $(date '+%F %T')" >> "$S"
# 5 影子研究（软步骤，放在 ALL DONE 之后：研究失败不会让量化部分重跑）
# 研究要用 DeepSeek：网络在解密 HTTPS（校园网）时证书校验会失败。先等网络，最多 8 小时，每 10 分钟查一次
for i in $(seq 1 48); do
  curl -s -o /dev/null --max-time 20 https://api.deepseek.com/ && break
  [ "$i" = 1 ] && echo "research waiting for network (TLS) $(date '+%F %T')" >> "$S"
  sleep 600
done
soft_step research $G -u -m quantmind.line_a.pipeline research --as-of "$END" --data-root "$R"
# 6 正式荐股日报（用户 2026-10-24 批准）：正式计划 → Agent 大脑 → 日报（复制到 Windows 的 formal_reports 目录）
soft_step formal $G -u -m quantmind.line_a.formal --as-of "$END" --data-root "$R" --with-brain
# 7 模拟账户账本（正式三档 + L9 影子账本，给前端；T-052 的账户是固定历史，不每天重跑）
soft_step ledger $G -u -m quantmind.line_a.ledger build --formal --shadow --as-of "$END"
