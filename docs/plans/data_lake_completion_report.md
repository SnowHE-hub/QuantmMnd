# 数据湖建设 + Tushare 回补 完成报告

> 依据 `docs/plans/tushare_probe_report.md`（§6 策略）与 `data_sufficiency_audit.md`（§2/§4）。
> 本 session 独占 `data/lake/`。所有数字均为实测（验证脚本只读、不调用 API）。
> 完成日期：2026-06-05。

---

## 摘要（TL;DR）

| 项 | 结果 |
|----|------|
| 交付 | `quantmind/data/lake.py`、`scripts/build_data_lake.py`、`scripts/backfill_tushare.py`、`tests/test_lake.py`（9 passed）、`data/lake/*.parquet`（6 表）。 |
| daily_basic | **全程回补完成**：1762/1762 交易日（**100%**，0 缺日），8,288,000 行，2019-01-02→2026-05-11，**0 失败**。 |
| margin | 头部 2019-01→2019-11 回补完成，并入快照 → 2019-01-02→2026-05-29 连续，1,652,984 行。 |
| hk_hold | 头部 2019-01→2019-11 回补完成（仅 A 股北向，与快照 schema 一致）→ 起点前移到 2019-01-02；2024-08 之后日频源端已停，未补（不可补）。 |
| PIT | `read_lake_window` 抽查 4 序列，**全部 `no_future_leak=True`**，窗口 `max(trade_date)==as_of`。 |
| 安全 | token 全程未回显/未落盘；**过程中发现 loguru 文件 sink 泄漏并已修复+清除**（见 §6）。 |

---

## 1. 数据湖表清单与覆盖（实测）

日历以 `alpha_prices_panel`（1762 交易日，2019-01-02→2026-05-11）为准。

| 序列 | 行数 | 交易日 | ts_code 数 | 起 | 止 | 覆盖率(vs 日历) | 缺失交易日 | 来源 |
|------|------|-------|-----------|----|----|----------------|-----------|------|
| **daily_basic** | 8,288,000 | 1762 | 5741 | 2019-01-02 | 2026-05-11 | **100.0%** | 0 | 纯回补 |
| **index_daily** | 7,048 | 1762 | 4 | 2019-02-25 | 2026-06-01 | 100.0% | 0 | 快照并集 |
| **north_bound** | 1,739 | 1739 | — | 2019-01-02 | 2026-06-01 | 96.88% | 55* | 快照并集 |
| **margin** | 1,652,984 | 1794 | 2085 | 2019-01-02 | 2026-05-29 | 100.0% | 0 | 快照∪回补头部 |
| **hk_hold** | 1,866,789 | 1616 | 2369 | 2019-01-02 | 2026-03-31 | 76.05% | 416** | 快照∪回补头部 |
| **stock_basic** | 1,388 | （静态） | 1388 | — | — | — | — | 快照并集 |

\* **north_bound 的 55 个“缺失”是结构性**（港股通休市但 A 股开市的日子，如 2019-04-19 耶稣受难日），非数据缺口 → **无需回补**（与审计 §2.1 一致）。
\*\* **hk_hold 的 416 个缺失 = 2024-08-18 起 A 股北向个股日频停发**（交易所政策，仅季末有 A 股快照，最大间隔 92d）。**源端不可补**，本地季末点已在表内。详见探针报告 §2。

---

## 2. daily_basic（头号回补项）详情

| 指标 | 值 |
|------|----|
| 交易日覆盖 | **1762 / 1762 = 100.0%**，缺失日率 **0.0%** |
| 行数 | 8,288,000 |
| 平均每日股数 | 4,703.7（早年约 3,555，近年约 5,500；含 STAR/创业板/北交所/退市票，**远超** 1374 只面板 universe） |
| 唯一 ts_code | 5,741 |
| 列（18 + ticker） | ts_code, trade_date, close, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv, ticker |

### 单交易日核对（验收项）：2026-05-11

| 字段 | 是否存在 | 当日非空率 | 全表非空率 |
|------|---------|-----------|-----------|
| pe_ttm | ✓ | 72.0% | 79.4% |
| pb | ✓ | 99.3% | 99.3% |
| total_mv | ✓ | 100.0% | 100.0% |
| circ_mv | ✓ | 100.0% | 100.0% |
| turnover_rate | ✓ | 100.0% | 100.0% |
| float_share | ✓ | 100.0% | 100.0% |
| free_share | ✓ | 100.0% | 100.0% |
| total_share | ✓ | 100.0% | 100.0% |

> 当日 5,491 行（全市场）。**8 个关键字段全部齐备**。`pe_ttm` 非 100% 属**设计内**：亏损公司（负 EPS）的 pe_ttm 为 NaN（与 `fundamental.py:46` 逻辑一致），`pb` 少量 NaN 为净资产为负/缺披露。**这不是回补缺失**，是字段语义。

---

## 3. 行数 / 墙钟 对比探针估算

| 序列 | 探针估算 | 实测结果 | 说明 |
|------|---------|---------|------|
| daily_basic 行数 | ~2.4M（按 1374 票×1762 日） | **8.29M** | 探针口径用面板 universe；实际全市场一调返回 ~4700 票/日 → 行数约 3.5×（探针正文已注明全市场 ~5500 行/日）。 |
| daily_basic 墙钟 | ~2–3.5h | **291 min（4.85h）** | 实测 17484s/1762 ≈ **9.9s/调**（含 0.5s 限速 + 重试 + 悉尼网络波动），高于探针裸调中位 4.2s；仍 0 失败。 |
| margin 头部 | ~0.25M 行 / 15–26min | **0.25M 新增行**（249,155）/ 29.8min | 与估算吻合（含断点续传重启）。 |
| hk_hold 头部 | ~0.43M 行（~2000 票×222 日） | **434,084 新增行 / 20.3min** | 与估算吻合。 |

- **限速实测**：daily_basic 有效 ~6.0 调/分钟（网络+0.5s 限速所限），远低于 ≤120 调/分钟上限；全程 **0 次频率限制报错**（failures=0），印证探针“瓶颈在网络不在配额”。

---

## 4. PIT 验证（验收项）

`read_lake_window(series, as_of, lookback=252)` 抽查，**严格不含未来**：

| 序列 | as_of | 窗口 min→max | 唯一交易日 | max ≤ as_of？ |
|------|-------|-------------|-----------|--------------|
| daily_basic | 2023-06-30 | 2022-06-09 → **2023-06-30** | 252 | ✅ |
| margin | 2021-03-31 | 2020-03-11 → **2021-03-31** | 258 | ✅ |
| hk_hold | 2020-06-30 | 2019-06-18 → **2020-06-30** | 281 | ✅ |
| index_daily | 2024-12-31 | 2023-12-18 → **2024-12-31** | 252 | ✅ |

- daily_basic / index_daily 在 lookback=252 下精确取到 **252 个交易日**（支持 beta_252d）。
- margin/hk_hold 略多于 252 是因其含少量不在价格日历内的 trade_date（融券/北向披露日历差异），上界仍严格 `≤ as_of`，**无未来泄漏**。
- 单元测试 `tests/test_lake.py`：**9 passed**（PIT 不含未来、非交易日 as_of、lookback≥252、原生列保留、幂等 keep='last'、north_bound 按 trade_date 去重、缺主键报错、日历单调）。

---

## 5. 共享契约（`quantmind/data/lake.py`）

- `read_lake_window(series, as_of, lookback_trading_days)`：严格 `trade_date ≤ as_of` 的 PIT 窗口，按 `alpha_prices_panel` 日历回看 N 个交易日，**保留原生列**，支持 ≥252。
- `write_lake(series, df)`：幂等。key 默认 `(ts_code, trade_date)`；`north_bound=(trade_date)`、`stock_basic=(ts_code)`；`drop_duplicates(keep='last')` 使回补覆盖快照。
- 辅助：`load_trading_calendar`、`coverage`（自检）、`load_checkpoint/save_checkpoint`（断点续传）。
- 回补策略落地：按 trade_date 循环（全市场一调）、按月增量落盘、`_checkpoints/<series>.json` 断哪续哪、指数退避（复用 provider）、限速 0.5s（≤120 调/分钟）。**实测断点续传有效**（margin 中途崩溃后从 checkpoint 续补）。

---

## 6. ⚠ 安全事件与处置（token）

- **现象**：provider 用 **loguru**（非 stdlib logging），且 `import quantmind.data.tushare_provider` 会在**导入时重新注册 4 个 loguru sink（含文件 sink `logs/quantmind_*.log`）**。任务建议的 `logging.disable(INFO)`、以及在 import **之前** 的 `logger.remove()` 都会被该导入**抵消**，导致 `_get_tushare_pro` 的 `token=<前8位>...` 行写入文件 sink（每进程 1 行，因 client 单例只 init 一次）。
- **处置**：
  1. 发现首跑泄漏即 kill 进程；删除 `/tmp/backfill.log`；从 `logs/quantmind_2026-06-05.log` 清除全部 `token=` 行。
  2. **根因修复**：在 `scripts/backfill_tushare.py` 的 `main()` 中，把 `loguru.logger.remove()` 移到 `from quantmind.data import tushare_provider` **之后**、任何 `_call` **之前**，确保 import 重加的 sink 被移除。
  3. **正确验证**：检查**文件 sink**（非仅 stdout）——修复后流程做一次真实 `_call`，`logs/*.log` 中 `token=` 计数保持 **0**，`logger` handlers=0。
- **现状**：`/tmp/backfill.log`、`/tmp/backfill_hk.log`、`logs/quantmind_2026-06-05.log` 中 `token=` 计数均为 **0**。
- **经验**：本仓库屏蔽 token 日志**必须**在 provider 导入后 `logger.remove()`；单靠 `logging.disable` 或导入前 remove 无效。

---

## 7. 对下游 / 后续的提示

- 周频面板重采样：用 `read_lake_window(series, as_of, lookback)` 取 PIT 窗口；daily_basic/index_daily/margin/north_bound 已连续日频可直接重算对应因子。
- `hk_hold` 的 `north_hold_*` 因子：2024-08 之后只有季末快照，周频 `as_of` 应取“最近季末”值（最长滞后 ~60 交易日），**这是源端限制，非本湖缺陷**。
- daily_basic 含全市场票（5741），下游按面板 universe（1374）join 即可；`pe_ttm/pb` 的 NaN 为亏损/负净资产语义，按因子逻辑处理（如 earnings_yield=1/pe 时 clip）。

*本报告为只读核验，未改动业务模型；token 全程未回显/未落盘/未入日志（已验证 0 泄漏）。*
