# 实现计划：周频重建训练面板 v5（第一版 · 仅免回补因子子集）

> **本文件仅为实现计划，不含任何实现代码。** 获批前不写实现。
> 依据：`docs/plans/data_sufficiency_audit.md`（数据充分性审计，已完成）。
> 产出物（本版）：`data/panel/alpha_panel_weekly_v5.parquet`（**新文件，绝不覆盖 v4**）。

---

## 0. 目标与边界

| 项 | 内容 |
|----|------|
| 目标 | 把季度面板（30 截面）重建为**周频面板**（每 5 交易日一个 `as_of`），新增 `forward_return_12d`，本版只纳入**全部免回补、免 PIT 主表**的因子子集。 |
| 显式排除 | 不调用 Tushare、不回补 daily_basic/hk_hold/financials、不构建 PIT 财报主表、不改 `alpha_panel_v4.parquet`、不改季度 `build_panel` 生产路径。 |
| 复用 | `compute_forward_returns`（`quantmind/features/panel.py:164`）、`FeaturePipeline.run_single_from_snapshot`（`quantmind/features/pipeline.py:260`）、各因子函数（签名 `fn(snapshot: dict, as_of) -> pd.Series`）。 |
| 架构原则 | 未纳入因子日后可**按 `(as_of, ticker)` 列对齐 join 增量补入**，不重建整表。 |

---

## 0.5 幸存者偏差审计（评审纠偏后置于实现第一步 · 只读调查已完成）

> 评审意见：**静态 universe 是判定"有没有真 alpha"时最危险的默认**。已先行执行退市审计，结论直接决定下游 P1 alpha 结论可不可信。

**实测结论（只读，已跑）：数据在源头即为幸存者池。**

| 证据 | 实测 |
|------|------|
| `alpha_prices_panel` 1374 票的最后交易日 | **全部落在 2026 年**；`last_trade_date < END−30d` 的票 = **0**；99.8%（1371/1374）交易到末日附近 |
| `stock_basic`（全 33 快照并集，37818 行） | `list_status` = **{'L': 37818}** —— 100% 在市，**0 个 'D'(退市) / 'P'(暂停)** |
| `delist_date` | 非空 **0/1374** |
| 快照并集唯一 ts_code | **1388**（仅比 v4 的 1374 多 14） |

**判定**：A 股 2019–2026 实际退市数以百计（注册制下每年 ~20–46 家），其在价格与 `stock_basic` 中**完全缺席**，证明该池按"构建时仍挂牌(L)的票"回填 → **源头幸存者偏差，任何 universe 选择都无法消除**（数据里没有退市票可纳入）。

**对计划的三点影响**：
1. **universe 仍改为非静态**（决策 3）：用**快照并集 + 按 `list_date` 做 PIT 入池**（`list_date` 实测 1374/1374 非空）。这修不了退市幸存，但能修**新股前视**（`as_of < list_date` 的票不应入池）并多纳入 14 个历史出现过的名字——是严格的改进，且免费。
2. **结论必须如实标注为乐观上界**：v5 面板与基于它的 P1 alpha 判定，**只能当作 optimistic upper bound 读**。短周期反转/动量在"只剩赢家"池里系统性虚高（退市票=已实现暴跌被剔除）。此局限写入 v5 `meta.json` 与下游回测报告，**不可省略**。
3. **新增回补项（超出原审计 §4 清单）**：真正修复需**回补退市票的行情 + stock_basic(list_status=D, delist_date)**（Tushare `stock_basic(list_status='D')` + 退市票 `daily`）。登记为数据湖的后续回补项，本版不做，但必须在清单中显式列出，否则幸存者偏差会被默默继承。

---

## 1. 采样方案（周频锚定）

**锚定方式：以 `alpha_prices_panel` 的 SSE 交易日历为基准，每 5 个交易日取一个 `as_of`，相位锚定到 fwd_12d 边界。**

- 取价主源 `data/raw/alpha_prices_panel.parquet`，唯一 `trade_date` 排序得 `td[0..1761]`（实测 1762 个交易日，末日 `td[1761]=2026-05-11`，审计 §摘要）。
- **fwd_12d 最后可训练锚点**：`td[1761-12] = td[1749] = 2026-04-20`（审计 §5 实测一致）。
- **相位锚定**：以 `td[1749]` 为锚点向前每 5 日回退生成训练区 `as_of`：
  `train_idx = range(1749, -1, -5)` → `[1749, 1744, 1739, …, 4]`，再升序。
  这样**保证最后一个带 fwd_12d 标签的 `as_of` 恰为 2026-04-20**，与审计 §5 对齐（验收点）。
- **holdout/预测区（可选）**：`td[1750..1761]`（2026-04-21 → 2026-05-11）内的 5 日锚点也可写入面板，但其 `forward_return_12d` 天然为 NaN（`compute_forward_returns` 边界返回 NaN），**不进训练**。本版默认只生成训练区（到 2026-04-20），是否带 holdout 行作为一个开关 `include_holdout=False`。
- 周频截面数估算：`1749 // 5 + 1 ≈ 350` 个 `as_of`（对比 v4 的 30）。
- 起点裁剪：早期 `as_of`（如 2019-01 头部）若某些外部序列缺数据，因子按 NaN 处理（见 §3 margin 头部）。

**为什么用交易日步进而非自然周**：A 股节假日多，自然周（每周五）会因停市漂移；按交易日 `[::5]` 步进可严格保证"每 5 个交易日一截面"且 fwd_Nd 用交易日位移（与 `compute_forward_returns` 的位置语义一致）。

**验收**：生成的 `as_of` 列表 `max == 2026-04-20`；相邻 `as_of` 在交易日历上间隔恰为 5；`len ≈ 350`。

---

## 2. 标签方案（PIT 前瞻收益）

- **复用** `compute_forward_returns(prices_pivot, as_of, horizons_days)`（`panel.py:164`），**不改该函数**。
  - 语义（已核验 `panel.py:182-196`）：以 `as_of` 当日（或之前最近交易日）close 为基准 `base`，取其后第 `h` 个交易日 close，`ret = target/base - 1`；越界返回 NaN。这是严格 PIT（只用未来价做标签，不泄露进特征）。
- **horizons**：`(12, 21, 63)`。`forward_return_12d` 为新增，`21d/63d` 保留（与 v4 列名一致，便于下游 `factor_model`/`meta_learner` 复用，见 `models/factor_model.py:208`、`models/meta_learner.py:282`）。
- **取价 pivot**：用 `alpha_prices_panel` 的复权收盘 `adj_close` 透视成 `index=trade_date, columns=ticker`（**注意用 `adj_close` 而非 `close`**，与 v4 forward return 用 qfq 复权一致，见 `panel.py:35`）。覆盖到 `td[1761]=2026-05-11` 即可（fwd_12d 锚点 2026-04-20 + 12 = 2026-05-11 正好落在边界内）。
- **边界自检**：对每个 horizon，最后一个非 NaN 标签的 `as_of` 应等于 `td[1761-h]`：
  fwd_12d→2026-04-20、fwd_21d→2026-04-07、fwd_63d→2026-01-26（审计 §5）。

**验收**：三档标签的"最后非 NaN as_of"与审计 §5 完全吻合（自动断言）。

---

## 3. 本版纳入的因子子集（全部免回补、免 PIT 主表）

> 来源依据：审计 §1.1（纯价格）、§1.2（指数/市场北向/margin）、§1.5（静态）、§2（拼接覆盖）。
> 计算函数位置见审计 §1 各表行号；本版**只启用这些因子**，其余列不写入（留给增量 join）。

### 3.1 纳入清单

**(a) 纯价格派生 17**（`technical.py` / `expansion.py:573`，仅依赖 `alpha_prices_panel`）：
`momentum_1m, momentum_3m, momentum_6m, momentum_12m_skip_1m, reversal_1w, volatility_3m, volatility_1y, downside_volatility_3m, max_drawdown_3m, amihud_illiquidity, volume_spike_5_30, rsi_14, bollinger_position, distance_to_52w_high, price_to_52w_low, amplitude_quantile, volume_price_corr_20d`

**静态标签 4**（`expansion.py:117/134/147/169`，依赖 `stock_basic`）：
`exposure_industry, exposure_area, list_age_years, is_recent_ipo`

**指数类 8**（`expansion.py`，依赖拼接后的 `index_daily`）：
`beta_252d, beta_60d, relative_strength_vs_csi300_60d, relative_strength_vs_csi300_120d, relative_strength_vs_csi500_60d, market_momentum_60d, market_volatility_60d, market_drawdown_60d`

**市场北向 1**（`sentiment.py:39`，依赖拼接后的 `north_bound.north_money`）：
`north_bound_30d_net_inflow`

**margin 纯量 5**（`expansion.py`，依赖拼接后的 `margin`，字段 rzye/rqye/rzmre）：
`margin_balance, margin_balance_change_20d, margin_buy_amount_20d, short_balance_change_20d, margin_short_ratio`

→ **v5 默认纳入 = 17 + 4 + 8 + 1 + 5 = 35 个因子。**

### 3.2 ⚠ 需评审决策：margin 的"第 7、第 6 个"因子

任务 §3 写"margin 7"，但审计 §1.2 显示其中 **`margin_buy_intensity`（`expansion.py:339`）与 `short_sell_pressure`（`expansion.py:361`）的分母用 `daily_basic.circ_mv`**——属审计 §4 的"daily_basic 依赖 16"集合。无 daily_basic 时这两列只能是 NaN。

**✅ 评审已定：v5 = 35（纯量 5）。** 不加 NaN 占位列——对早读无信息，且全 NaN 列可能让模型出问题。`margin_buy_intensity` / `short_sell_pressure` 这 2 个**等 daily_basic 回补后，随 daily_basic-16 那批一起 join 进来**（§4）。

### 3.3 margin 2019 头部缺口

审计 §2.2：margin 拼接覆盖始于 **2019-12-02**。本版**接受 2019-12 之前的 margin 因子为 NaN**（不回补）。即 `as_of < 2019-12-02` 的 margin 5 列为 NaN，其余因子正常。下游训练用 `dropna`/掩码处理（不在本版引入）。

---

## 4. 本版暂不纳入、但须预留增量接口的因子

> 共 3 组（审计 §1.3 / §1.4 / §3）：**daily_basic 依赖 16**、**hk_hold 依赖 6**、**财报 PIT 16**。
> （注：daily_basic-16 = 10 估值 + 4 换手/自由流通 + 2 margin 归一化；与 §3.2 的 2 个重叠。）

### 4.1 增量补入的架构契约（不重建整表）

v5 面板 index = `MultiIndex(as_of, ticker)`，与未来增量列**同键对齐**。增量流程：

```
新因子增量表  inc_<group>.parquet : MultiIndex(as_of, ticker) → [新因子列...]
        │  （用相同的 weekly as_of 网格 + 相同 universe 生成）
        ▼
alpha_panel_weekly_v5.parquet  ── join(how="left", on=[as_of,ticker]) ──▶  v5_1 / v6
```

- **接口函数（计划新增）**：`merge_increment(base_panel_path, inc_path, out_path)` —— 纯列对齐 join，校验 index 对齐率（见 §8 验收）。
- **约束**：增量表必须用**同一份 weekly `as_of` 列表**与**同一 universe**生成（把 as_of 网格落盘为 `data/panel/weekly_asof_grid.parquet` 供增量复用），否则 join 错位。
- **PIT 责任在增量侧**：daily_basic 增量按 `trade_date ≤ as_of` 取值；财报增量按 `f_ann_date ≤ as_of` 取最新一期（审计 §3）。base v5 不承担这部分 PIT。
- **列命名稳定**：增量列名与 v4 同名，避免下游改代码。

### 4.2 三组增量的未来数据来源（仅登记，不在本版实现）

| 组 | 因子数 | 依赖序列 | 回补接口（审计 §4） | PIT 规则 |
|----|-------|---------|--------------------|---------|
| daily_basic | 16 | daily_basic 日频 | Tushare `daily_basic` | `trade_date ≤ as_of` |
| hk_hold | 6 | hk_hold 日频 | Tushare `hk_hold` | `trade_date ≤ as_of` |
| 财报 PIT | 16 | fina_indicator / 三大表 | 快照拼接成 PIT 主表 | `f_ann_date ≤ as_of` 取最新期 |

---

## 5. 外部日频序列读取契约 + 数据湖落盘格式

> 本版不回补，但**用快照拼接得到连续日频**用于 index_daily / north_bound / margin（审计 §2 已验证可连续拼接）。同时**定义数据湖表格式**，作为产品级数据湖的落地起点。

### 5.1 数据湖表：`data/lake/<series>.parquet`

落盘格式（统一契约）：

| series 文件 | 主键 | 关键列 | 拼接来源 | 本版覆盖（审计 §2 实测） |
|------------|------|-------|---------|------------------------|
| `data/lake/index_daily.parquet` | (trade_date, ts_code) | close,open,high,low,pct_chg,vol,amount | `data/snapshots/*/index_daily.parquet` 去重并集 | 2019-02-25→2026-06-01 连续 |
| `data/lake/north_bound.parquet` | (trade_date) | north_money,south_money,hgt,sgt,ggt_ss,ggt_sz | `data/snapshots/*/north_bound.parquet` 去重并集 | 2019-01-02→2026-06-01 连续 |
| `data/lake/margin.parquet` | (trade_date, ticker) | rzye,rqye,rzmre,rqyl,rzche,… | `data/snapshots/*/margin.parquet` 去重并集 | 2019-12-02→2026-05-29 连续（头部缺） |

- **去重规则**：按主键 `drop_duplicates(keep="last")`，跨快照窗口重叠部分取任一（值应一致；不一致则记日志，取 `update_flag`/最新快照优先）。
- **元数据**：每张表配 `data/lake/<series>.meta.json`，记 `min_date/max_date/n_rows/source_snapshots/build_ts`。
- **数据湖是只读派生物**：由拼接脚本生成，不手改；未来回补数据也写入同一张表（append + 去重），实现"数据湖随回补增长"。

### 5.2 PIT 过滤读取接口（计划新增 `quantmind/data/lake.py`）

```
read_lake_window(series, as_of, lookback_calendar_days) -> DataFrame
    # 读 data/lake/<series>.parquet，过滤 trade_date ≤ as_of 且 ≥ as_of-lookback
    # 返回与"快照子表"同 schema 的 DataFrame，可直接塞进 snapshot dict
```

- **PIT 核心**：`trade_date ≤ as_of`（严格不含未来）。
- **🔒 硬条件（lookback 必须覆盖最长窗口因子）**：`beta_252d` / `relative_strength_vs_csi300_120d` 回看 252/120 **交易日**。`read_lake_window` 的 `lookback_calendar_days` 是**日历日**，252 交易日 ≈ 365 日历日，故 index_daily 必须给 **≥ 400 日历日**（留春节/国庆缓冲）；margin/north_bound ≥ 90。**给短了会静默算出错值、不报错**——这是高危静默错误，必须在验证清单中专门拦（§10-D/§10 硬条件）。
- **复用因子代码**：因子函数内部已用 `_filter_pit(df, as_of, "trade_date")`（`expansion.py:67`）再次裁剪，故 `read_lake_window` 给"足够长的窗口"即可，PIT 由因子层与读取层双重保证。
- **早期 as_of 数据不足**：2019 年初的 as_of 前置交易日 < 252，`beta_252d` 等长窗因子**应输出 NaN（而非用不足窗口硬算的垃圾值）**——验证清单专门核查（§10 硬条件 1）。

### 5.3 weekly snapshot dict 组装（不调 Tushare）

为任意 weekly `as_of` 组装 `FeaturePipeline.run_single_from_snapshot` 所需 dict：

| dict key | 来源 | 备注 |
|----------|------|------|
| `prices` | `alpha_prices_panel` 过滤 `trade_date ≤ as_of` 的 lookback 窗口（≥ 252 日）+ §6 列名映射 | 纯价格 17 + 指数类用 |
| `universe` | **非静态**（见 §7）：快照并集 + `list_date ≤ as_of` PIT 入池 → DataFrame[ticker] | |
| `stock_basic` | `data/lake/stock_basic.parquet`（快照并集去重，含 list_date/industry/area，静态字段）| 静态 4 + universe PIT 入池 |
| `index_daily` | `read_lake_window("index_daily", as_of, 400)`（≥400 日历日，覆盖 252 交易日，见 §5.2 硬条件） | 指数类 8 |
| `north_bound` | `read_lake_window("north_bound", as_of, 90)` | 市场北向 1 |
| `margin` | `read_lake_window("margin", as_of, 90)` | margin 5 |
| `daily_basic`/`hk_hold`/`financial_*` | **不放**（缺键 → 相关因子函数走 NaN 兜底） | 增量补入 |

- 缺键安全性已核验：如 `turnover_3m_avg`（`technical.py:159`）`if 'turnover_rate' not in px → NaN`；`short_sell_pressure`（`expansion.py:369`）`if circ_mv not in db → NaN`。故缺 daily_basic/hk_hold 不报错，只产 NaN——但本版**只取 §3 的 35 列输出**，NaN 列直接不写。

---

## 6. 字段映射（审计 §1.1 列名差异）

`alpha_prices_panel` 与因子函数期望的列名不一致，须在组装 `prices` 时重命名：

| 因子函数期望 | alpha_prices_panel 实际列 | 处理 |
|-------------|--------------------------|------|
| `volume` | `vol` | rename `vol→volume` |
| `pct_change` | `pct_chg` | rename `pct_chg→pct_change` |
| `close`（复权） | `adj_close` | label 用 `adj_close`；因子 close 按现有 v4 口径（核对 `pivot_prices` 默认列）|
| `ticker` | `ts_code` | rename `ts_code→ticker`（快照 prices 用 `ticker`，见 `data/snapshots/*/prices.parquet`）|
| `turnover_rate` 等 | （无） | 不提供 → 相关因子不在 v5 |

- **实现位置**：集中在"组装 prices 子表"的一个函数里（`weekly_panel.py` 内），**单点映射**，避免散落。
- **校验**：映射后 `prices` 的列集合 ⊇ 因子所需最小列集（`{close, high, low, open, pre_close, volume, amount}`），缺列即 assert 失败。
- **风险**：`pivot_prices`（`features/utils.py`）默认 `value_col` 与列名耦合——实现前先读 `utils.pivot_prices` 确认默认取列名，必要时显式传 `value_col`。

---

## 7. 输出与 universe

- **输出新文件**：`data/panel/alpha_panel_weekly_v5.parquet`（+ `alpha_panel_weekly_v5.meta.json`）。
  - index = `MultiIndex(as_of, ticker)`；columns = §3 的 **35** 因子 + `forward_return_12d/21d/63d`。
  - **绝不写 `alpha_panel_v4.parquet`**（季度 63d 生产路径不受影响）。
  - **meta.json 必须含**：survivorship 局限标注（§0.5，P1 结论=乐观上界）、`standardized=False`、margin 有效起始 2026 起的 NaN 语义、universe 构造方式。
- **as_of 网格副本**：`data/panel/weekly_asof_grid.parquet`（供 §4 增量复用，保证同键）。
- **✅ universe = 非静态（决策 3，评审纠偏）**：
  - **快照 `universe.parquet` / `stock_basic` 并集**（1388）为候选母集；每个 `as_of` 按 **`list_date ≤ as_of` PIT 入池**（剔除尚未上市的票，修新股前视）。
  - **退市票仍无法纳入**（§0.5：源头数据无退市行情）。非静态只修了"新股前视"+多 14 个名字，**不能消除幸存者偏差**。
  - **局限如实标注**：v5 及其 P1 结论为 optimistic upper bound；真正修复需回补退市票行情（§0.5 第 3 点，单列回补清单）。
- **✅ 标准化 = False（决策 2，已核实口径，无双重标准化）**：存原始因子值 + 真实 NaN。
  - 核实结论：**LGBM ranker** 直接用原始特征 `fillna(0.0)`、无 zscore（`models/lgbm_ranker.py:389`，树模型不需要）；**FactorCNN** 的 `preprocess_panel`（`models/factor_cnn.py:549`）内部 winsorize+截面 zscore，**自己标准化**。两条管线都期望**原始面板** → 面板存原始值不会双重标准化，且便于手算自验。

---

## 8. 分步实现顺序 + 每步验收 + 工作量

| 步 | 内容 | 触及/新增文件 | 验收方式 | 工作量 |
|----|------|--------------|---------|-------|
| **S0** | **退市/幸存者审计（实现第一步，已完成只读部分）**：确认源头幸存者偏差，决定 P1 可信度 | 只读，结论入 §0.5 + v5 meta | ✅ 已跑：1374 票全交易到 2026、stock_basic 100% list_status=L、delist_date 0/1374 → 源头幸存者池，P1=乐观上界 | 已完成 |
| **S1** | 数据湖拼接脚本：快照→`data/lake/{index_daily,north_bound,margin,stock_basic}.parquet`（+meta） | 新增 `quantmind/data/lake.py`、`scripts/build_data_lake.py`；新增 `data/lake/*` | 三表 min/max/n_rows 与审计 §2 表一致；stock_basic 含 list_date/industry/area；去重后主键唯一 | 0.5–1d |
| **S2** | `read_lake_window` + PIT 过滤 | `quantmind/data/lake.py` | 单测：给定 as_of，返回窗口 `max(trade_date) ≤ as_of`，长度≥阈值 | 0.5d |
| **S3** | weekly as_of 网格生成（§1 锚定） | 新增 `quantmind/features/weekly_panel.py`；`data/panel/weekly_asof_grid.parquet` | `max==2026-04-20`、相邻间隔=5 交易日、len≈350 | 0.5d |
| **S4** | weekly snapshot dict 组装 + §6 列名映射 | `weekly_panel.py` | 对某 as_of 组装 dict，列集合校验通过；不调用 Tushare（断网可跑） | 1d |
| **S5** | 跑因子子集（复用 `run_single_from_snapshot`，groups 限定 + 列白名单 35/37） | `weekly_panel.py`（复用 `pipeline.py`，**不改** pipeline） | 单 as_of 输出列 == §3 白名单；无报错；缺键因子被正确排除 | 1d |
| **S6** | 标签：`adj_close` pivot + 复用 `compute_forward_returns`(12,21,63) | `weekly_panel.py`（复用 `panel.py:164`） | 三档"最后非 NaN as_of"==审计 §5；抽查单票手算 | 0.5d |
| **S7** | 拼装全表 + 落盘 v5（+meta），写 as_of 网格 | `weekly_panel.py`；`data/panel/alpha_panel_weekly_v5.parquet` | 文件生成；`v4` 未被改（mtime/哈希不变）；index 有序唯一 | 0.5d |
| **S8** | 增量 join 接口（仅骨架 + 测试，不接真数据） | `weekly_panel.py::merge_increment` | 用假增量表测 join 对齐率=100%、列不重名 | 0.5d |
| **S9** | 自验证检查清单脚本（§10） | 新增 `scripts/verify_weekly_panel.py`；`tests/test_weekly_panel.py`、`tests/test_lake.py` | 全部检查项 PASS；`pytest` 既有测试不回归 | 1d |

**总估**：约 6–7 人日。**关键路径**：S1→S4→S5→S6。

> 复用而非新写的部分（**不修改**）：`compute_forward_returns`、`FeaturePipeline.run_single_from_snapshot`、各因子函数、`sse_calendar`。新增代码集中在 `lake.py` + `weekly_panel.py` + 两个脚本 + 两个测试文件，互不污染既有模块。

---

## 9. 风险点

| 级别 | 风险 | 缓解 |
|------|------|------|
| **高** | **PIT 泄露**：weekly as_of 比季度密，任何"用了 > as_of 的行情/序列"都会污染特征。 | 读取层 `trade_date ≤ as_of` + 因子层 `_filter_pit` 双保险；S9 写专门 PIT 断言（构造一个 as_of，篡改未来行情，确认因子值不变）。 |
| **高** | **列名映射错位**（§6）：`vol/volume`、`pct_chg/pct_change`、`ts_code/ticker`、`adj_close` 用错 → 因子静默算错或全 NaN。 | 单点映射函数 + 列集合 assert；S5 抽查某因子值与 v4 同 as_of 同票比对（季度 as_of 在 weekly 网格上若命中可直接对照）。 |
| **高** | **幸存者偏差（源头级，§0.5）**：数据无任何退市票，短周期反转/动量回测系统性虚高 → 误判出"假 alpha"。 | **无法在本版消除**（数据缺退市行情）。缓解=如实标注 P1 为乐观上界（meta + 报告）+ 登记退市票回补为后续项；非静态 universe 仅修新股前视。 |
| **高** | **lookback 给短致长窗因子静默出错（§5.2）**：beta_252d 给 < 252 交易日会算出垃圾值不报错。 | index_daily lookback ≥ 400 日历日；验证清单硬条件 1：早期 as_of 长窗因子须为 NaN 而非垃圾值。 |
| 中 | **拼接重叠值不一致**：同一 trade_date 在两快照窗口值不同。 | 去重 `keep=last` + 不一致计数日志；阈值超标则告警。 |
| 中 | **adj_close 复权口径**：v4 当初可能用 raw close；若口径不同 v5 的 63d 与 v4 数字不可直接比。 | **复核 63d 以 v5 新标签为基准手算**（不与 v4 旧数字对）；只在确认同口径时才做 v4 交叉比对。 |
| 低 | **2019/2026 头部 margin NaN** 被下游误当 0。 | meta 标注 margin 有效起始（拼接 2019-12-02 起）；不填充。 |

---

## 10. 自验证检查清单（"如何确认周频面板正确"）

实现后由 `scripts/verify_weekly_panel.py` 自动执行，全部须 PASS：

**🔒 评审纳入的 3 条硬条件（最高优先）**
- [ ] **硬条件 1 — 长窗 lookback**：2019 年初前置交易日 < 252 的 as_of，`beta_252d` / `relative_strength_vs_csi300_120d` 等长窗因子值为 **NaN**（不是用不足窗口硬算的垃圾值）；且 index_daily 的 lookback ≥ 400 日历日。（§5.2）
- [ ] **硬条件 2 — split-agnostic**：面板内**绝不做任何 train/test 切分**；只如实存 `as_of` 真实日期 + 三档标签。purge/embargo 属下一阶段 walk-forward，不在面板层。（确认 MultiIndex 带真实 as_of 即可）
- [ ] **硬条件 3 — 63d 复核基准**：复核 63d 标签**以 v5 自身新标签手算为准**（adj_close 口径），**不直接与 v4 旧 `forward_return_63d` 数字比**（v4 可能用 raw close，不可比）；仅在确认同复权口径时才做 v4 交叉验证。

**A. 结构**
- [ ] index 为 `MultiIndex(as_of, ticker)`，唯一、有序。
- [ ] 因子列集合 == §3 白名单（**35**）；标签列 == `forward_return_{12,21,63}d`。
- [ ] `alpha_panel_v4.parquet` 文件哈希/mtime **未变**（未被覆盖）。
- [ ] meta.json 含 survivorship 局限标注（§0.5）、`standardized=False`、universe 构造方式、margin NaN 语义。

**B. 采样与标签边界（对齐审计 §5）**
- [ ] `as_of.max() == 2026-04-20`；相邻 as_of 交易日间隔恒 = 5；截面数 ≈ 350。
- [ ] `forward_return_12d` 最后非 NaN as_of == **2026-04-20**。
- [ ] `forward_return_21d` 最后非 NaN == **2026-04-07**；`forward_return_63d` == **2026-01-26**。

**C. 标签数值正确性（手算抽查）**
- [ ] 随机抽 3 个 (as_of, ticker)：手动取 `adj_close[as_of]` 与其后第 12 个交易日 `adj_close`，`p1/p0-1` 与面板 `forward_return_12d` 相对误差 < 1e-9。
- [ ] `forward_return_63d` 复核**以 v5 自身手算为基准**（硬条件 3）：抽票手取 `adj_close[as_of]` 与其后第 63 个交易日 `adj_close`，比对面板值 < 1e-9；**仅当确认 v4 同为 adj_close 口径**时，才追加与 v4 的交叉比对。

**D. 因子数值正确性（手算/对照抽查）**
- [ ] `momentum_1m`：抽 1 票，手算 `adj_close[as_of]/adj_close[as_of-21bar]-1` ≈ 面板值（注意方向/窗口与 `technical.py:46` 一致）。
- [ ] `beta_252d`：抽 1 票，独立用 numpy OLS（票收益 vs 000300 收益，252 窗口）复算 ≈ 面板值（容差 1e-6）。
- [ ] `north_bound_30d_net_inflow`：某 as_of，`north_bound` 近 30 日 `north_money` 之和 ≈ 面板值。
- [ ] `margin_balance`：某 as_of 某票，`margin` 截至 as_of 最后一行 `rzye` == 面板值。
- [ ] `list_age_years`：某票 `(as_of - list_date)/365.25` ≈ 面板值。

**E. PIT 安全性**
- [ ] 构造篡改：把某票 `> as_of` 的行情/北向/margin 改为极端值，重算该 as_of 因子，**值不变**（证明无未来泄露）。
- [ ] `prices`/lake 窗口在任意 as_of 上 `max(trade_date) ≤ as_of`。

**F. margin 头部与缺值语义**
- [ ] `as_of < 2019-12-02`（margin 拼接起点前）的 5 个 margin 列全 NaN；其余因子非全 NaN。

**G. 增量接口**
- [ ] `merge_increment` 用假 daily_basic 增量表 join 后：行数不变、index 对齐率 100%、无重名列、原 35 列值不变。

**I. 幸存者偏差（§0.5）**
- [ ] universe 按 `list_date ≤ as_of` PIT 入池：抽查某早期 as_of，确认尚未上市的票不在该截面（无新股前视）。
- [ ] meta.json 明确写出 survivorship 局限与"P1=乐观上界"；退市票回补登记为后续项。

**H. 回归**
- [ ] `pytest`（既有用例）全绿；新增 `tests/test_weekly_panel.py`、`tests/test_lake.py` 通过。
- [ ] `git status`：仅新增 `quantmind/data/lake.py`、`quantmind/features/weekly_panel.py`、`scripts/build_data_lake.py`、`scripts/verify_weekly_panel.py`、`tests/test_*.py`、`data/lake/*`、`data/panel/alpha_panel_weekly_v5*.*`、`docs/`；**无既有业务文件被修改**。

---

## 11. 决策记录（评审已定 · 架构已批准）

| # | 决策 | 结论 | 依据 |
|---|------|------|------|
| 1 | margin 列数 | **35**（纯量 5，不加 NaN 占位列） | §3.2；2 个 circ_mv 列随 daily_basic-16 增量补入 |
| 2 | 是否标准化 | **False（存原始值）** | §7；已核实 LGBM(`lgbm_ranker.py:389`)/CNN(`factor_cnn.py:549`) 均期望原始面板，无双重标准化 |
| 3 | universe | **非静态**：快照并集 + `list_date` PIT 入池；**退市幸存无法消除，如实标注**；退市票回补登记为后续项 | §0.5 + §7；退市审计已坐实源头幸存者池 |
| 4 | holdout 行 | **不生成**（默认） | §1；上线推理另走最新网格路径 |

**三条硬条件**已纳入 §10 验证清单顶部（长窗 lookback / split-agnostic / 63d 以 v5 自身为基准复核）。

> **架构已批准，按上述调整进入实现。退市审计（S0）已作为第一步完成（只读，结论入 §0.5）。** 实现仍遵守：不改任何既有业务代码/数据/模型，新增代码集中于 `lake.py`+`weekly_panel.py`+脚本+测试；产出 `alpha_panel_weekly_v5.parquet` 新文件，绝不覆盖 v4。
