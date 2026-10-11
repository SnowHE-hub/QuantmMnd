# T-015 · 数据补齐到 2026-10-09：执行与核验报告

- 执行：Claude Code 桌面版 · Opus 5.5，2026-10-10（用户指令："可以下载需要的数据，推荐补充到最新的时间节点"）
- 证据：`docs/work/evidence/T-015/`（核验脚本、校验 JSON、更新前 md5）；运行报告 `data/lake/_checkpoints/update_latest_*.json`

## 1. 结论

| 项 | 结果 |
|---|---|
| 交易日历 | Tushare `trade_cal`（SSE）2019-01-02 → 2026-10-09 共 **1,882** 个交易日，存 `data/lake/trade_cal_sse.parquet` |
| 行情类 lake 表 | daily / adj_factor / daily_basic / margin / index_daily **零缺日**，止于 2026-10-09；north_bound 只缺港股通休市日 |
| 18 个历史空洞 | 已补回（2021-03-05～15、2021-06-10、2023 年 9 天、2026-03-03、03-04） |
| 旧数据 | 每张表的旧行**逐值未变**（`verify_lake_update.py` 输出 OK）；原价格文件 md5 不变 |
| 新价格文件 | `alpha_prices_panel_v6_20261009.parquet`、`alpha_prices_panel_20261009.parquet`，与旧文件重叠行逐值相同 |
| 财报类 | 补到 2026-10-09：目标 5,243 只全部到终态（done 5,238 / partial 2 / no_data 3 / failed 0）；在市公司 2026 半年报覆盖率 100%；旧行逐值未变（§4） |

## 2. 行情类 lake 表 `[实测]`

`python scripts/data_pipeline/update_lake_latest.py --end 2026-10-09`（第一次运行在 daily_basic 中途随会话中断，续跑时脚本按"lake 已有日期"自动接续）。

| 表 | 旧行数 | 新增行 | 补入交易日 | 空表 | 失败 | 区间 |
|---|---:|---:|---:|---:|---:|---|
| daily | 8,345,172 | 651,518 | 120 | 0 | 0 | 2019-01-02 → 2026-10-09 |
| adj_factor | 8,522,667 | 655,148 | 120 | 0 | 0 | 同上 |
| daily_basic | 8,288,000 | 650,860 | 120 | 0 | 0 | 同上 |
| margin | 1,652,984 | 385,816 | 88 | 0 | 0 | 同上 |
| index_daily（4 个指数） | 7,048 | 480 | 120 | 0 | 0 | 同上（含原先缺的 2019-01～02） |
| north_bound | 1,739 | 86 | 86 | 57 | 0 | 57 个空表均为港股通休市日，未写入 |

核验 `verify_lake_update.py --backup data/lake/_backup_20261010`：
- 每张表 `missing_in_new = 0`、`value_mismatch = {}`、`duplicate_keys = 0`
- daily / adj_factor / daily_basic / margin / index_daily 与 `trade_cal_sse` 相比 `missing_trading_days = []`、`non_trading_days = []`
- 备份：`data/lake/_backup_20261010/`（daily、adj_factor、daily_basic、margin、north_bound、index_daily 的更新前版本）

**新 verifier 全量复跑（2026-10-11，3c3517f 之后的版本）**：
`PANDAS_COPY_ON_WRITE=1 MALLOC_ARENA_MAX=2 python scripts/data_pipeline/verify_lake_update.py --backup data/lake/_backup_20261010 --end 2026-10-09 --fund-checkpoint data/lake/_checkpoints/fundamentals_update_20261009.v2.json`
→ 12 个表（daily、adj_factor、daily_basic、margin、index_daily、north_bound、五张财报表）与财报检查点全部 OK，**退出码 0**；
每张表 `missing_in_new = 0`、`value_mismatch = {}`、无 dtype / 列变化、无字面空值、无晚于截止日的公告；行情表无缺日 / 非交易日。
耗时 2 分 12 秒，峰值内存 8.2 GB。日志：`docs/work/evidence/T-015/verify_full_20261011.log`。

## 3. 新价格文件 `[实测]`

`python scripts/data_pipeline/build_price_panels.py --end 2026-10-09`，校验 `data/raw/price_panels_20261009.validation.json`：

| 文件 | 口径 | 行 / 票 / 交易日 | 与旧文件重叠行 | pct_chg 自洽 |
|---|---|---|---|---|
| `alpha_prices_panel_v6_20261009.parquet` | 全市场，adj_close = close×adj_factor（后复权，同 v6） | 8,996,690 / 5,813 / 1,882 | 8,345,172 行全部逐值相同（含 adj_close） | 99.99998% |
| `alpha_prices_panel_20261009.parquet` | 1374 票池，adj_close 前复权（同原 v5，基准为 2026-10-09） | 2,436,872 / 1,374 / 1,882 | 2,273,529 行原始列逐值相同、前复权日收益相同 | 100% |

- 原 v5 价格表来自另一条下载流水线（`step2_download_prices.py`），与 lake 逐值一致，相当于两个来源互证。
- md5（更新前后不变）：`alpha_prices_panel.parquet` 884261fb…（= `p2_build_v6` 的 v5 md5 守卫值）、`alpha_prices_panel_v6.parquet` 75d0951e…、
  `alpha_panel_v4.parquet` d98146c2…、`alpha_panel_v4_1.parquet` 9c407358…
- 新文件 md5：`alpha_prices_panel_20261009` c9a758ae…、`alpha_prices_panel_v6_20261009` 6401678b…

### 3.1 数据质量说明：Tushare 复权因子的微小漂移 `[实测]`

`evidence/T-015/check_af_drift.py`：没有除权（pre_close = 前收）却出现的复权因子变化约 19,722 行，幅度中位 0.01%，>1% 的 103 行；
集中在少数日期（如 2024-06-26 4,431 只、2021-01-07 3,439 只、2020-01-02 3,216 只——**均为原有数据**），补回的日期（2023-03-16、05-31、06-01）表现相同。
另有 4,909 行"当天变动、次日回落"的单日跳动。推测为 Tushare 不同时间点对复权因子的精度/口径修订 `[推断]`。
影响：后复权价的日收益有极小误差（`check_adj_seam.py`：与 pct_chg 不一致的行占 0.012%）。标签用 pct_change 连乘（T-003），不受影响。

## 4. 财报类 lake 表 `[实测]`

股票列表刷新为 5,557 只（原 5,529，新增 28 只 6 月后上市），存 `data/lake/stock_basic_full_20261009.parquet`；拉取在市或 2026 年后退市的 5,243 只，
每只 4 个接口（fina_indicator、income、balancesheet、cashflow），公告期 2026-01-01～2026-10-09。

**过程中的缺陷与处置（Codex 第一轮 REJECT）**：第一次运行的合并把旧表 null 主键改成字面 "None"（fina 两张表各 8 行），
且空返回会被记为完成。2026-10-11 发现时下载进行到 1,000 / 5,243，立即停止；修代码、用 `_backup_20261010` 修复两张表
（受损版本存 `data/lake/_quarantine_20261011/`），旧版检查点不采信，**全部 5,243 只重新拉取**。详见卡片"重验修复"。

**结果**（`python scripts/data_pipeline/update_fundamentals_latest.py --end 2026-10-09 --since 2026-01-01 --workers 3 --empty-retries 2 --retry-wait 300`，
检查点 `data/lake/_checkpoints/fundamentals_update_20261009.v2.json`）：

| 终态 | 只数 | 说明 |
|---|---:|---|
| done | 5,238 | 四个接口都有 2026 年以来的公告 |
| partial | 2 | 002231.SZ（2026-03-27 退市）、600355.SH（2026-04-27 退市）：fina_indicator 为空，三张报表各 1 期 |
| no_data | 3 | 300379.SZ（2026-01-22 退市）、300391.SZ（2026-04-13 退市）；002731.SZ *ST萃华——在市但最后一期报告是 2025 三季报、2026-08-31 后无交易、收盘 0.77 元（未披露 2025 年报，处于退市流程）。均为两次尝试（间隔约 3 小时）一致为空 |
| failed / pending | 0 | |

相对 T-015 之前（`_backup_20261010`）的新增行与核验：

| 表 | 新增行 | 其中 2026 半年报 | 在市公司半年报覆盖率¹ | 旧行逐值 / 主键 / 字面空值 / 晚于截止日 |
|---|---:|---:|---:|---|
| fina_indicator_full | 5,268 | 5,239 | 100% | OK |
| fina_indicator | 5,268 | 同上（子集表） | — | OK |
| income | 5,344 | 5,235 | 100% | OK |
| balancesheet | 5,320 | 5,239 | 100% | OK |
| cashflow | 5,334 | 5,234 | 100% | OK |

¹ 分母 = 2026-06-30 前上市、状态 L、2026-09 仍有交易的 5,197 家公司。新增行公告月份集中在 2026-08（约 5,170 行，半年报法定截止 8 月 31 日）；
其余为 2022～2026 年报告期的更正或补发。脚本 `evidence/T-015/fundamentals_coverage.py`、结果 `fundamentals_coverage.json`。

verifier（`--backup data/lake/_backup_20261010 --fund-checkpoint ...v2.json`）：五张财报表、margin、north_bound、index_daily、财报检查点全部 OK，退出码 0
（日志 `~/c3_scratch/t015_verify_part1.log`）。三张大行情表也已用新 verifier 复跑通过，见 §2 末尾。

## 5. 用新数据完成的下游工作

| 产物 | 说明 | 关键数字 |
|---|---|---|
| `data/panel/alpha_panel_v4_2.parquet` | 决定 3：不保留 v4 掩码、价格到 2026-10-09（T-003 代码，含第 3 个提交的覆盖修复） | 21d/63d 标签 27,942 / 27,938（v4 为 25,513 / 24,168）；独立核验偏差 >0.5%：2 / 4 个 |
| T-010 延长 | NAV 到 2026-10-09（T-010 报告 §3.5） | 真实成交净超额 −5.0%，IR −0.48，结论不变 |
| `data/feedback_20261009/` | T-013 代码（复权价）重算已实现收益，`WRITE_MODE=parquet_only`，未写数据库 | 150 条 / 15 个推荐日（原 80 / 8）；整体 realized IC 0.079，命中率 0.56 |
| `data/paper_trading_20261009/` | T-014 代码（复权价）重算模拟盘，未写数据库 | 22 期；3m 均值 +0.43%、中位 −3.11%、胜率 43.3%、IC −0.16；在原 90 条相同持仓上复权使 3m 均值从 +5.28% 升到 +6.85% |

独立核验 v4_2 时发现了 T-003 的"缺数据当停牌"漏洞，见 T-003 卡片第 3 个提交与 `evidence/T-015/crosscheck_v42.py`。

## 6. 未做与后续

1. **默认路径未切换**：所有代码仍默认读原文件（`alpha_prices_panel.parquet` 等）。验收后建议由 C1 开卡：
   让 `lake.load_trading_calendar()` 改读 `trade_cal_sse.parquet`，各脚本默认价格文件指向新文件（或把新文件改名为正式名、旧文件归档）。
2. **v6 周频面板未延长**：`alpha_panel_weekly_v6.parquet` 止于 2026-04-20（amihud 等因子依赖它）。
   （2026-10-11 更正）v6 周频面板本身只用价格类因子（technical / sentiment / expansion），**不需要财报**；需要财报的是主线 B 的基本面因子面板
   `fundamental_factors_v6.parquet`（`p63_2_build_factors.py`）。另发现 v6 标签同样因缺 18 个交易日而 horizon 错位，重建与延长已另开 **T-017**。
3. **v4 季频面板未新增截面**：2026-09-30 截面需要用快照重算因子（`build_full_panel.py`），本次只重算了标签。
4. 定时任务（daily_update）不读这些新文件，行为不变。
