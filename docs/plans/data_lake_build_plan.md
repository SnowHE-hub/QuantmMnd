# 执行计划：连续日频数据湖 + Tushare 回补

> 依据 `docs/plans/tushare_probe_report.md`（§6 策略已评审）与 `data_sufficiency_audit.md`（§2/§4）。
> 本 session 独占 `data/lake/`。策略已评审，直接执行。

## 目标与交付
- `quantmind/data/lake.py`：共享读写契约。
- `scripts/build_data_lake.py`：从 `data/snapshots/*` 去重并集建湖表。
- `scripts/backfill_tushare.py`：按探针 §6 回补 daily_basic（全程）+ margin/hk_hold（2019 头部）。
- `tests/test_lake.py`、`data/lake/*.parquet`、`docs/plans/data_lake_completion_report.md`。

## 设计要点
- **日历**：以 `alpha_prices_panel`（1762 日，2019-01-02→2026-05-11）为准；`read_lake_window` 用它算 lookback 窗口，支持 ≥252 交易日（beta_252d）。
- **read_lake_window(series, as_of, lookback_trading_days)**：严格 `trade_date ≤ as_of`（PIT），返回最近 lookback 个交易日窗口，保留原生列。
- **write_lake(series, df)**：幂等 merge；key 默认 `(ts_code, trade_date)`，`north_bound=(trade_date)`，`stock_basic=(ts_code)`；`drop_duplicates(keep="last")` 让回补覆盖快照。
- **建湖（5 表）**：index_daily / north_bound / margin / hk_hold / stock_basic，快照升序并集 → write_lake；打印每表覆盖&缺口自检。daily_basic 无快照，纯回补。
- **回补（3 项）**：
  - daily_basic：全 1762 交易日，`_call("daily_basic", trade_date=d)` 全市场一调。
  - margin_detail / hk_hold：仅 2019-01→2019-11 头部（约 220 日各）。**不碰** hk_hold 2024-09→2026（源端已停）。
  - 断点续传：`data/lake/_checkpoints/<series>.json` 记已完成 trade_date；幂等 (ts_code,trade_date)；按月增量落盘；指数退避（复用 provider）；限速 ≤120 调/分钟（运行期设 `_MIN_INTERVAL=0.5`）。

## 执行顺序
1. lake.py → 2. build_data_lake（快，本地）+ 自检 → 3. test_lake 通过 → 4. backfill_tushare → 5. 后台挂跑 daily_basic（长任务）→ 6. 验证 + 完成报告。

## 安全
token 在 `api_key.txt`（标签 `TUSHARE_TOKEN`），读取/使用不回显/不落盘/不入日志（`logging.disable(INFO)` 屏蔽 provider 的 `token[:8]` 行）。
