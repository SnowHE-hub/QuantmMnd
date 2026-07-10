# 数据充分性审计报告：周频重建训练面板

> **只读审计**，未修改任何代码/数据/模型。所有数字均为实测。
> 审计日期：2026-06-05　|　面板：`data/panel/alpha_panel_v4.parquet`

---

## 摘要（TL;DR）

| 结论 | 内容 |
|------|------|
| 面板现状 | 75 列 = **73 因子 + 2 标签**；**30 个季度截面**（2019-03-31 → 2026-06-30）。实测 `as_of` 数 = 30。 |
| 行情主源 | `data/raw/alpha_prices_panel.parquet`：2019-01-02 → **2026-05-11**，1762 个交易日，1374 只票。这是周频重采样的价格主源。 |
| 可任意频率重算 | **(a) 纯价格派生 17 个** + **静态标签 4 个** —— 仅靠 `alpha_prices_panel` / `stock_basic` 即可。 |
| 须拼接但已有日频 | **指数（index_daily）、市场北向（moneyflow_hsgt）** —— 快照窗口拼接后已是连续日频，**无需回补**。 |
| **必须回补（关键）** | **daily_basic（估值/市值/换手/流通股）全程缺日频**、**margin_detail 缺 2019 年头部**、**hk_hold 缺 2024-09 之后日频**。 |
| 基本面 PIT | 财报带 `ann_date`/`f_ann_date` → **可正确 PIT 对齐**；daily_basic 按 `trade_date` 天然 PIT。 |
| 标签边界 | 受行情止于 2026-05-11 限制：`fwd_12d` 末截面 ≈ **2026-04-20**，`fwd_21d` ≈ **2026-04-07**，`fwd_63d` ≈ **2026-01-26**。 |

---

## 1. 因子分类（73 个）

数据来源核验：面板列实测 75 列，去掉 `forward_return_21d`、`forward_return_63d` 两个标签列后为 73 因子。计算函数位置通过 grep `quantmind/features/*.py` 的 `def <factor>` 定义确认。

### 1.1 (a) 纯价格派生 —— 可从 `alpha_prices_panel.parquet` 任意频率重算（17 个）

依赖列：`close/adj_close/high/low/open/pre_close/vol/amount/pct_chg`，均在 `alpha_prices_panel`。

| 因子 | 计算函数（文件:行） | 依赖字段 |
|------|------|------|
| momentum_1m | `quantmind/features/technical.py:46` | close |
| momentum_3m | `technical.py:52` | close |
| momentum_6m | `technical.py:57` | close |
| momentum_12m_skip_1m | `technical.py:62` | close |
| reversal_1w | `technical.py:80` | close |
| volatility_3m | `technical.py:95` | close→returns |
| volatility_1y | `technical.py:104` | close→returns |
| downside_volatility_3m | `technical.py:113` | close→returns |
| max_drawdown_3m | `technical.py:123` | close |
| amihud_illiquidity | `technical.py:139` | close, **amount** |
| volume_spike_5_30 | `technical.py:165` | vol（value_col=volume） |
| rsi_14 | `technical.py:183` | close |
| bollinger_position | `technical.py:196` | close |
| distance_to_52w_high | `technical.py:214` | close（252d max） |
| price_to_52w_low | `technical.py:227` | close（252d min） |
| amplitude_quantile | `technical.py`（technical 注册表） | high/low/pre_close |
| volume_price_corr_20d | `quantmind/features/expansion.py:573` | close, vol（注：函数在 expansion，但只用 prices） |

> 表达式等价实现亦见 `quantmind/features/expr_factors.py`（momentum_21d/reversal_5d/volatility_63d/amihud_63d/volume_ratio_5_20/rsi_14_expr/bollinger_pos_20），按 CLAUDE.md「方式 A」可在任意 `as_of` 重算。

**⚠ 字段命名差异**：`alpha_prices_panel` 用 `vol`，函数按 `volume` 取值；用 `pct_chg`，快照用 `pct_change`。快照 `prices` 表是经过**重命名+富化**后的版本（见 1.4）。周频重采样脚本须做同样的列名映射，但**数据本身齐备**，不需回补。

### 1.2 (b) 依赖外部日频序列（22 个）

| 因子 | 计算函数（文件:行） | 底层序列 | 字段 |
|------|------|------|------|
| north_bound_30d_net_inflow | `quantmind/features/sentiment.py:39`（取数 `sentiment.py:30`） | **north_bound**（市场北向） | north_money（近30日和） |
| north_hold_ratio | `expansion.py:192` | **hk_hold** | hold_ratio |
| north_hold_amount | `expansion.py:201` | **hk_hold** | hold_vol×close |
| north_hold_ratio_change_20d | `expansion.py:230` | **hk_hold** | hold_ratio（lag 20） |
| north_hold_ratio_change_60d | `expansion.py:240` | **hk_hold** | hold_ratio（lag 60） |
| north_hold_amount_change_20d | `expansion.py:250` | **hk_hold** | hold_vol（lag 20） |
| north_hold_trend_60d | `expansion.py:270` | **hk_hold** | hold_ratio（60d 斜率） |
| margin_balance | `expansion.py:295` | **margin** | rzye |
| margin_balance_change_20d | `expansion.py:316` | **margin** | rzye（lag 20） |
| margin_buy_amount_20d | `expansion.py:326` | **margin** | rzmre（近20日和） |
| margin_buy_intensity | `expansion.py:339` | **margin** + **daily_basic** | rzmre / circ_mv |
| short_balance_change_20d | `expansion.py:350` | **margin** | rqye（lag 20） |
| short_sell_pressure | `expansion.py:361` | **margin** + **daily_basic** | Δrqye / circ_mv |
| margin_short_ratio | `expansion.py:375` | **margin** | rzye/rqye |
| beta_252d | `expansion.py:444`（`beta_n` 408） | **index_daily**(000300.SH) + prices | 252d OLS β |
| beta_60d | `expansion.py:448` | **index_daily** | 60d OLS β |
| relative_strength_vs_csi300_60d | `expansion.py:452` | **index_daily**(000300) | 60d 相对收益 |
| relative_strength_vs_csi300_120d | `expansion.py:457` | **index_daily**(000300) | 120d |
| relative_strength_vs_csi500_60d | `expansion.py:543` | **index_daily**(000905.SH) | 60d |
| market_momentum_60d | `expansion.py:485` | **index_daily** | 指数 60d 动量 |
| market_volatility_60d | `expansion.py:499` | **index_daily** | 指数 60d 波动 |
| market_drawdown_60d | `expansion.py:513` | **index_daily** | 指数 60d 回撤 |

> 取数路径：均通过 `snapshot[<key>]` + `_filter_pit(df, as_of, 'trade_date')`（`expansion.py:67`）取截至 `as_of` 的日频窗口，再做 20/60/120/252 日滚动。**因此在任意周频 `as_of` 重算的前提，是底层序列有"截至该 as_of、回看至多 252 交易日"的连续日频数据。**

### 1.3 (c) 依赖低频基本面（26 个）

**c1 估值/市值（来自 `daily_basic`，实为日频接口，但本地只物化了季度单日）**：

| 因子 | 计算函数 | daily_basic 字段 |
|------|------|------|
| pe_ttm | `fundamental.py:43` | pe_ttm |
| pb | `fundamental.py:55` | pb |
| ps_ttm | `fundamental.py:66` | ps_ttm |
| book_to_market | `fundamental.py`（1/pb） | pb |
| earnings_yield | `fundamental.py`（1/pe） | pe_ttm |
| dividend_yield_ttm | `fundamental.py:85` | dv_ttm |
| log_market_cap | `fundamental.py:98` | total_mv |
| log_circ_market_cap | `fundamental.py:106` | circ_mv |
| size_rank | `fundamental.py`（total_mv 排名） | total_mv |
| fcf_yield | `fundamental.py` | cashflow + total_mv |

**c2 财报指标（来自 `fina_indicator` / 三大表，季度天然低频）**：

| 因子 | 计算函数 | 字段 / 来源 |
|------|------|------|
| roe_ttm | `fundamental.py:124` 起 / `em_fundamental.py` | fina_indicator |
| roa_ttm | `fundamental.py:132` | fina_indicator |
| gross_margin | `fundamental.py` | grossprofit_margin |
| net_margin | `fundamental.py:152` | netprofit_margin |
| debt_to_assets | `fundamental.py:161` | debt_to_assets |
| current_ratio | `fundamental.py:170` | current_ratio |
| asset_turnover | `fundamental.py:179` | assets_turn |
| equity_multiplier | `fundamental.py:188` | assets_to_eqt |
| revenue_yoy | `fundamental.py` / `em_fundamental.py` | fina_indicator |
| operating_profit_yoy | `fundamental.py:212` | op_yoy |
| net_profit_yoy | `fundamental.py:221` | fina_indicator |
| quarterly_revenue_yoy | `fundamental.py:231` | q_sales_yoy |
| accruals | `fundamental.py` | income/cashflow/balance |
| ocf_to_revenue_ttm | `fundamental.py:270` | cashflow/income |
| earnings_accel_q | `fundamental.py:318` | fina_indicator（环比加速） |
| revenue_accel_q | `fundamental.py:335` | fina_indicator |

### 1.4 额外发现：4 个"技术因子"实则依赖 `daily_basic`（不是纯价格）

这些列名像技术面，但取数依赖 `turnover_rate` / `float_share` / `free_share` —— 这些字段**不在 `alpha_prices_panel`**（其列仅 OHLCV+adj），而在 `daily_basic`：

| 因子 | 计算函数 | 缺失字段 |
|------|------|------|
| turnover_3m_avg | `technical.py:156`（`if 'turnover_rate' not in px` 兜底） | turnover_rate |
| turnover_acceleration | `technical.py:244` | turnover_rate |
| turnover_rate_quantile | `technical.py`（technical 注册表） | turnover_rate |
| free_float_ratio | `sentiment.py:93` | free_share / total_share |

> 证据：`alpha_prices_panel` 实测列 = `['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount','adj_factor','adj_close']` —— **无 turnover_rate**。
> 而快照 `data/snapshots/2024-09-30/prices.parquet` 实测列含 `turnover_rate, turnover_rate_f, pe, pe_ttm, pb, ps_ttm, total_mv, circ_mv` —— 这是 OHLCV **merge daily_basic** 后的富化版本。
> **结论**：换手率/自由流通因子在任意周频 `as_of` 重算，需要 `daily_basic` 日频（与 c1 同一回补项即可一并解决）。

### 1.5 静态标签（4 个，来自 `stock_basic`，无频率问题）

| 因子 | 计算函数 | 来源 |
|------|------|------|
| exposure_industry | `expansion.py:117` | stock_basic.industry（静态） |
| exposure_area | `expansion.py:134` | stock_basic.area（静态） |
| list_age_years | `expansion.py:147` | stock_basic.list_date（与 as_of 之差） |
| is_recent_ipo | `expansion.py:169` | 由 list_age_years 派生 |

**分类合计**：(a)17 + (b)22 + (c)26 + 1.4 中 4 个 daily_basic 技术因子 + 静态 4 = **73**。✓

---

## 2. 外部序列的本地存量（关键）

> 本地外部序列有两种物化形态：
> 1. **季度快照目录** `data/snapshots/<as_of>/{north_bound,margin,index_daily,hk_hold,daily_basic}.parquet` —— 每个快照含一段**截至该 as_of 的日频回看窗口**（不是单点！）。
> 2. **零散日频文件** `data/text/north_flow.parquet`、`data/text/market_north_flow.parquet` —— 覆盖很短，不堪用（见下）。

实测各快照内单文件窗口（以 `2024-09-30` 为例）：north_bound 61 个交易日、margin 83 日、index_daily 265 日、hk_hold 65 日、**daily_basic 仅 1 日**。

**跨全部快照拼接后的并集覆盖**（实测）：

| 序列 | 快照文件数 | 并集交易日 | 起 | 止 | 最大日历缺口 | 连续？ | 判定 |
|------|----------|-----------|----|----|------------|-------|------|
| **index_daily** | 32 | 1762 | 2019-02-25 | 2026-06-01 | 11d（节假日） | ✅ 连续 | **无需回补**（仅 2019-01-02→02-22 头部约 33 日缺失，可选补） |
| **north_bound**（市场北向） | 40 | 1739 | 2019-01-02 | 2026-06-01 | 11d | ✅ 连续 | **无需回补** |
| **margin** | 32 | 1572 | **2019-12-02** | 2026-05-29 | 11d | ⚠ 中段连续但头部缺 | **须回补 2019-01→2019-11** |
| **hk_hold** | 32 | 1398 | 2019-12-02 | 2026-03-31 | **92d** | ❌ 2024-08-17 起仅剩季度点 | **须回补 2024-09→2026-05 日频 + 2019 头部** |
| **daily_basic** | 41 | **34** | 2019-03-29 | 2026-06-02 | 94d | ❌ 基本是每快照单日 | **须全程回补日频 2019-2026** |

### 2.1 北向资金（分两层）

- **市场层 north_bound**（驱动 `north_bound_30d_net_inflow`，字段 `north_money`）：拼接并集 **2019-01-02 → 2026-06-01 连续日频**（最大缺口 11 日=节假日）。**已在本地、可直接周频重采样，无需回补。**
- **个股层 hk_hold**（驱动全部 `north_hold_*` 6 因子，字段 `hold_vol`/`hold_ratio`）：实测 hk_hold 大缺口起于 **2024-08-17 → 2024-09-30（44d）**，其后每点间隔 90–92 日 —— 即 **2024-09 之后只在季度末单日物化**，日频断裂。**2019-12 之前亦缺**。**须回补。**
- `data/text/north_flow.parquet`：实测仅 4 个交易日（2024-12-31 → 2025-09-30，13879 行），且字段为 `vol/ratio` 而非 hk_hold 的 `hold_vol/hold_ratio`，**不可替代**。
- `data/text/market_north_flow.parquet`：实测 209 行，2025-01-02 → 2025-11-19，**覆盖太短**，不如快照拼接的 north_bound。

### 2.2 融资融券余额 margin

- 驱动全部 `margin_*`/`short_*` 7 因子（字段 rzye/rqye/rzmre）。
- 拼接并集 **2019-12-02 → 2026-05-29 连续日频**（最大缺口 11 日）。
- 缺口：**2019-01-02 → 2019-11-29 整段无数据**（约 220 个交易日）。仅此头部须回补。

### 2.3 指数 index_daily

- 驱动 beta/相对强弱/市场状态 8 因子（000300.SH / 000905.SH 等）。
- 快照拼接 **2019-02-25 → 2026-06-01 连续日频，无需回补**。
- 注意：`data/raw/index_daily_panel.parquet` 实测只到 **2024-12-31**（且仅 4 个指数码），已**陈旧**，不要用它做主源；应优先用快照 `index_daily` 拼接（更新到 2026-06）。

### 2.4 daily_basic（估值/市值/换手/流通股）—— 缺口最严重

- 驱动 c1 估值 10 因子 + 1.4 换手/自由流通 4 因子 + (b) 类中 `margin_buy_intensity`/`short_sell_pressure` 的 `circ_mv` 分母。
- 实测并集仅 **34 个唯一交易日**（41 文件×约 1 日），完全不连续（94d 缺口）。
- **必须全程回补 2019-2026 日频。** 这是周频重建的**头号阻塞项**（影响约 16 个因子）。

---

## 3. 基本面 PIT（Point-In-Time）

| 数据 | 文件 | PIT 时间戳 | 能否周频 PIT 对齐 |
|------|------|-----------|------------------|
| 财报指标 financial_indicators | `data/snapshots/*/financial_indicators.parquet`（108 列） | 含 **`ann_date`、`report_date`** | ✅ 现有 `latest_report_per_ticker(ind, date_col='ann_date')`（`fundamental.py:127` 等）已按公告日取数 |
| 三大表 financials_income/balance/cashflow | `data/snapshots/*/financials_income.parquet` 等 | 含 **`ann_date`、`f_ann_date`、`report_date`、`report_type`、`update_flag`** | ✅ 有实际公告日 `f_ann_date`，可严格 PIT |
| 估值 daily_basic | 快照 `daily_basic.parquet` | 仅 `trade_date` | ✅ trade_date 即观测日，按 `trade_date ≤ as_of` 天然 PIT（**但日频缺失，须先回补，见 §2.4**） |

**结论**：
- 财报具备 `ann_date`/`f_ann_date` → **周频重采样可正确做 PIT 对齐**。做法：把各季度快照的 `financial_indicators`/`financials_*` **按 (ts_code, report_date) 去重合并成 PIT 主表**，对每个周频 `as_of` 取 `f_ann_date ≤ as_of` 的最新一期。
- **PIT 陷阱**：务必用 `f_ann_date`（实际公告日）而非 `report_date`（报告期）过滤——否则会用到未来信息（如 Q3 报表 report_date=09-30 但公告于 10 月底，在 10-15 的周频 as_of 上不可见）。
- 跨快照拼接的必要性：单个季度快照只含"截至该季度末已公告"的财报；介于两季度之间的周频 `as_of` 若要拿到最新已公告季报，需用**下一个**快照的财报并按 `f_ann_date` 过滤。故须建 PIT 并集主表，而非逐快照独立取数。

---

## 4. 回补清单（产出）

> 估算口径：2019-01-02 → 2026-05-11 约 1762 个交易日；全市场约 1374 只票；融资融券/北向约 1100–1400 只/日（按快照实测密度）。

### 4.1 必须回补

| 序列 | Tushare 接口 | 缺口区间 | 估算量 | PIT 注意 |
|------|------------|---------|-------|---------|
| **daily_basic**（估值/市值/换手/流通股）| `daily_basic`（`tushare_provider.py:201` 按 ts_code 全history / `:254` 按 trade_date） | **全程 2019-01→2026-05**（本地仅 34 单日） | ≈1374×1762 ≈ **2.4M 行**，~18 列，约 150–250MB parquet | trade_date=观测日，按 `≤as_of` 过滤即可；注意 pe_ttm 负值→NaN（`fundamental.py:46` 逻辑） |
| **hk_hold**（个股北向持股）| `hk_hold`（`tushare_provider.py:349`） | **2024-09→2026-05 日频** + 2019-01→2019-11 头部 | ≈(410+230)×1183 ≈ **0.76M 行**，约 10–20MB | trade_date 为披露日；北向 T+1，重采样时 `≤as_of` 取最后值 |
| **margin_detail**（融资融券）| `margin_detail`（`tushare_provider.py:355`） | **2019-01→2019-11 头部**（约 220 交易日） | ≈220×1120 ≈ **0.25M 行**，约 5–10MB | trade_date 为交易日，按 `≤as_of` |

### 4.2 已有日频、无需回补（仅靠现有快照拼接即可周频重采样）

| 序列 | 现有覆盖（拼接并集实测） | 说明 |
|------|----------------------|------|
| **index_daily** | 2019-02-25→2026-06-01 连续（缺口≤11d） | beta/相对强弱/市场状态 8 因子可重算；可选补 2019-01-02→02-22 头部 ~33 日 |
| **north_bound**（市场北向，moneyflow_hsgt）| 2019-01-02→2026-06-01 连续 | `north_bound_30d_net_inflow` 可重算 |
| **alpha_prices_panel**（价格主源）| 2019-01-02→2026-05-11，1762 日 ×1374 票 | (a) 类 17 因子 + 静态 4 因子全部可任意频率重算 |
| **financial_indicators / financials_***（财报）| 各季度快照并集，带 ann_date/f_ann_date | (c2) 16 财报因子可 PIT 重算（需建去重 PIT 主表，见 §3） |

> 备注：若希望一次性补齐头部使所有序列从 2019-01-02 起完全对齐，可顺带补 index_daily 2019-01 头部（4 指数×~33 日 ≈ 132 行，量极小）。

### 4.3 回补优先级

1. **P0 — daily_basic 全程**：阻塞约 16 个因子（10 估值 + 4 换手/流通 + 2 个 circ_mv 分母），缺口最大。
2. **P1 — hk_hold 2024-09 之后**：阻塞 6 个 north_hold_* 因子的近两年截面。
3. **P2 — margin 2019 头部 + hk_hold 2019 头部 + index 2019 头部**：仅影响 2019 上半年的早期周频截面，可后补。

---

## 5. 标签可计算边界

行情主源 `alpha_prices_panel` 末日 = **2026-05-11**（实测 1762 个交易日）。`forward_return_Nd` 需要 `as_of` 之后第 N 个交易日的收盘价，故末个可训练 `as_of` = 倒数第 (N+1) 个交易日（实测按交易日历回数）：

| 标签 | 最后可训练 `as_of` | 标签取价日 | 说明 |
|------|------------------|-----------|------|
| **forward_return_12d**（新增目标）| **2026-04-20** | 2026-05-11 | 周频新列，支撑 12 日前瞻 |
| forward_return_21d（现有）| **2026-04-07** | 2026-05-11 | |
| forward_return_63d（现有）| **2026-01-26** | 2026-05-11 | 前瞻最长，边界最早 |

**含义**：
- 周采样（每 5 交易日一截面）下，加 `forward_return_12d` 后，最后一个带完整标签的截面落在 **2026-04-20 附近**（若周频对齐到周五等具体锚点，取 ≤2026-04-20 的最近周频锚点）。
- 2026-04-20 → 2026-05-11 之间的周频 `as_of` 可生成特征但 `fwd_12d` 标签缺失（需未来行情），应作为 **预测集 / 半成品**，不进训练。
- 面板现有 `as_of` 最末为 2026-06-30，已**超出**行情末日 2026-05-11 —— 说明季度面板末几个截面的 `forward_return_*` 本就是 NaN/外推；周频重建时应以行情边界为准裁剪。

---

## 附：核验方法与证据来源

- 面板列/截面：`pandas.read_parquet('data/panel/alpha_panel_v4.parquet')` → shape (29406, 75)，index=(as_of,ticker)，as_of 唯一值=30。
- 行情边界：`alpha_prices_panel` trade_date min/max = 2019-01-02 / 2026-05-11，nunique=1762。
- 外部序列覆盖：对 `data/snapshots/*/<file>.parquet` 逐文件读 `trade_date` 求并集、算 `diff().dt.days` 缺口（见 §2 表，全部实测）。
- 因子→函数：grep `quantmind/features/*.py` 的定义 + 读 `expansion.py`/`fundamental.py`/`technical.py`/`sentiment.py` 函数体（行号见 §1）。
- Tushare 接口名：`quantmind/data/tushare_provider.py` 的 `_call("daily_basic")`(201/254)、`_call("index_daily")`(337)、`_call("moneyflow_hsgt")`(343)、`_call("hk_hold")`(349)、`_call("margin_detail")`(355)。
- PIT 字段：快照 `financial_indicators.parquet` 含 `ann_date/report_date`；`financials_income.parquet` 含 `ann_date/f_ann_date/report_date/report_type/update_flag`。

*本报告为只读审计，未改动任何业务代码 / 数据 / 模型。*
