# 训练标签审计报告（LABEL_AUDIT）

> 目的：在把训练标签从 ~63 日前瞻收益改为 12 日**之前**，确认现状。
> 本报告**只读侦察**，未修改任何代码或数据。
> 审计日期：2026-06-04 ｜ 仓库：`/home/lenovo/projects/quantmind/`

---

## 0. 核心结论（TL;DR）

1. **训练标签**是 `forward_return_21d` 与 `forward_return_63d` 两列，由
   [`quantmind/features/panel.py:164`](../quantmind/features/panel.py) 的 `compute_forward_returns()`
   计算，物化进训练面板 `data/panel/alpha_panel_v4.parquet`（经实测含这两列）。
2. **`actual_return_63d` 不是训练标签**。它是 paper-trading 结算出的**已实现收益**
   （`data/feedback/realized_pnl.parquet`），仅供 meta-learner 训练目标、回测评估、
   实盘 PnL 跟踪使用。改 12 日标签时**不要**和它混淆。
3. 三个板块 LGBM 模型（main/gem/star）生产训练用 **`forward_return_63d`**（CLI 默认）。
4. FactorCNN 生产训练默认 **`forward_return_63d`**，与 LGBM **读同一个面板的同一标签体系**
   （都来自 `alpha_panel_v4.parquet`，非各自重算）。
5. meta-learner 用的是 **`actual_return_63d`**（已实现收益，来源不同）。
6. 标签计算**无前视风险**：标签是 `as_of` 之后 H 个交易日的前瞻收益，特征只用到 `as_of` 当日及之前数据。
7. 代码中大量 `63` 是**特征窗口**（动量/波动率 3 个月≈63 交易日），与**标签 horizon** 的 63 是两码事，改标签时切勿误改特征窗口。

---

## 1. 标签生成

### 1.1 标签计算的唯一核心函数

**文件**：[`quantmind/features/panel.py:164-197`](../quantmind/features/panel.py)

```python
def compute_forward_returns(prices_pivot, as_of, horizons_days=(21, 63)):
    # base = as_of 当日 close（非交易日回退到前一交易日）  -- 第 182-185 行
    on_or_before = prices_pivot.index[prices_pivot.index <= target]
    base_idx = on_or_before[-1] if len(on_or_before) > 0 else after[0]
    base = prices_pivot.loc[base_idx]
    for h in horizons_days:                              # 第 188 行
        col = f"forward_return_{h}d"                     # 第 189 行 → 列名在此生成
        future_idx = prices_pivot.index[prices_pivot.index > base_idx]
        target_close = prices_pivot.loc[future_idx[h - 1]]   # 第 195 行：base 之后第 h 个交易日 close
        out[col] = target_close / base.replace(0, np.nan) - 1.0   # 第 196 行：累计收益率
```

- **算法**：`forward_return_{h}d = close(as_of 后第 h 个交易日) / close(as_of) - 1`。
- **shift 等价值**：不是 pandas `shift`，而是按**交易日序号**取 `future_idx[h-1]`，等价 `shift(-h)` 的累计收益（h ∈ {21, 63}）。
- **前视风险**：**无**。标签本就是未来收益（预测目标），而特征在 `as_of` 截面计算，二者无泄漏。base 用 `as_of` 当日 close，target 严格取 `as_of` **之后**（`index > base_idx`）的交易日，不含当日，无 off-by-one 泄漏。

### 1.2 标签列名（事实）

| 列名 | 出现位置 | 性质 |
|------|---------|------|
| `forward_return_21d` | panel.py:189（动态生成）、build_full_panel.py:326 | **训练标签**（21 日） |
| `forward_return_63d` | panel.py:189（动态生成）、build_full_panel.py:326 | **训练标签**（63 日，生产默认） |
| `actual_return_63d` | realized_pnl 体系，见 §2.3 | **已实现收益**，非训练标签 |

> 当前训练标签列名是 **`forward_return_63d`**（不是 `actual_return_63d`）。

### 1.3 标签如何进入训练面板

**文件**：[`scripts/build_full_panel.py`](../scripts/build_full_panel.py)

- `scripts/build_full_panel.py:326` — `label_cols = ["forward_return_21d", "forward_return_63d"]`（写死的两列）。
- `scripts/build_full_panel.py:419` — 调用 `compute_forward_returns(pivot, as_of, (21, 63))`（horizon 元组写死）。
- `scripts/build_full_panel.py:404` — `fwd_end = latest + timedelta(days=int(63 * 1.6) + 10)`（为算 63 日标签预留的未来取价窗口，硬编码 `63`）。
- `scripts/build_full_panel.py:394` — `need_through = latest + timedelta(days=65)`（补价窗口，硬编码 65）。
- `scripts/build_full_panel.py:488` — `panel.to_parquet(out_path)`，默认输出 `data/features/csi300_full_panel.parquet`（build_full_panel.py:227）。
- `scripts/build_full_panel.py:500-501` — 日志统计 `forward_return_21d`/`forward_return_63d` 覆盖率。

> **注意**：build_full_panel 的 CLI 默认输出是 `data/features/csi300_full_panel.parquet`，
> 但**模型实际消费的训练面板是 `data/panel/alpha_panel_v4.parquet`**（见 §2）。
> 二者路径不同——生产 `alpha_panel_v4.parquet` 是用 `--output` 覆盖跑出来的；
> 实测该文件确含 `forward_return_21d` + `forward_return_63d` 两列（见 §3）。

---

## 2. 模型的标签来源

### 2.1 LGBM 三板块模型（main / gem / star）

**训练入口**：[`scripts/train_board_models.py`](../scripts/train_board_models.py)

- 标签列：CLI 参数 `--label`，**默认 `forward_return_63d`** → `scripts/train_board_models.py:433`。
- 读取面板：`--panel` 默认 `data/panel/alpha_panel_v4.parquet` → `scripts/train_board_models.py:429`。
- 板块切分逻辑（`get_board`）：`scripts/train_board_models.py:41-54`
  - `688xxx.SH → STAR`、`300xxx.SZ → GEM`、其余 → `MAIN`。
- 三套配置与输出模型：`scripts/train_board_models.py:71-108`
  - `models/lgbm_v6_main.pkl`、`models/lgbm_v6_gem.pkl`、`models/lgbm_v6_star.pkl`。
- 标签校验：`scripts/train_board_models.py:360-362`（标签列不在 panel 中即报错退出）。
- 底层 walk-forward：`quantmind/models/lgbm_ranker.py` 的 `walk_forward_evaluate()`。
  - **注意**：`lgbm_ranker.py:457` 该函数自身默认是 `forward_return_21d`，
    但被 `train_board_models.py` 用 `forward_return_63d` 显式覆盖（train_board_models.py:215/236）。
    → **生产板块模型实际用 63d。**

### 2.2 FactorCNN

**训练入口**：[`scripts/_save_cnn_v2.py`](../scripts/_save_cnn_v2.py)（生产模型保存脚本）

- 读取面板：`scripts/_save_cnn_v2.py:11` — `pd.read_parquet("data/panel/alpha_panel_v4.parquet")`（与 LGBM 同一面板）。
- 标签列：`scripts/_save_cnn_v2.py:15` 调用 `train_factor_cnn(panel=panel, ...)`，未传 `label_col`，
  → 用 `quantmind/models/factor_cnn.py` 的默认值 **`forward_return_63d`**：
  - `factor_cnn.py:227`（`preprocess_panel` 默认）、`:495`（`train_factor_cnn` 默认）、`:805`（另一入口默认）。
- 输出模型：`models/factor_cnn_v2_augmented.pkl`（_save_cnn_v2.py:26）。
- 推理消费：`scripts/daily_update.py:463` 加载 `factor_cnn_v2_augmented.pkl`，
  Step5c 与 LGBM 做 rank-based 6:4 融合（daily_update.py:543/1641）。

> **CNN 与 LGBM 是同一列标签**：都读 `alpha_panel_v4.parquet` 里的 `forward_return_63d`，
> **不是各自重算**。标签生成只发生在 `build_full_panel` 阶段（§1.3），训练脚本只读取。

### 2.3 meta-learner

**训练入口**：[`scripts/train_meta_learner.py`](../scripts/train_meta_learner.py)

- 标签：**`actual_return_63d`**（已实现收益），来源 `data/feedback/realized_pnl.parquet`：
  - `scripts/train_meta_learner.py:69` — `_PNL = data/feedback/realized_pnl.parquet`。
  - `scripts/train_meta_learner.py:121` — 对 `actual_return_63d` 做 per-quarter z-score → `ret_z`。
  - `scripts/train_meta_learner.py:126` — 最终训练目标含 `actual_return_63d`/`ret_z`/`hit`。
- 模块内默认路径：[`quantmind/models/meta_learner.py:353`](../quantmind/models/meta_learner.py)
  — `y = fb_day.loc[common, "actual_return_63d"]`（从 feedback 取标签）。
- **关键差异**：meta-learner 标签**不来自** `alpha_panel_v4.parquet` 的 `forward_return_*`，
  而来自 paper-trading 结算的 `realized_pnl`。`actual_return_63d` 由
  `scripts/track_realized_pnl.py:201` 与 `scripts/settle_forward_positions.py:121` 写出。
  → **把训练标签改 12d，不会自动改变 meta-learner 的标签口径**（除非另行处理 realized_pnl）。

### 2.4 各模型标签来源汇总

| 模型 | 训练入口 | 标签列 | 标签来源文件 | 与 LGBM 同源? |
|------|---------|--------|-------------|--------------|
| LGBM main/gem/star | `scripts/train_board_models.py` | `forward_return_63d` | `data/panel/alpha_panel_v4.parquet` | — |
| FactorCNN | `scripts/_save_cnn_v2.py` | `forward_return_63d` | `data/panel/alpha_panel_v4.parquet` | ✅ 同列 |
| meta-learner | `scripts/train_meta_learner.py` | `actual_return_63d` | `data/feedback/realized_pnl.parquet` | ❌ 不同源 |

---

## 3. 数据范围

实测（`pd.read_parquet` 直读，2026-06-04）：

### 3.1 `data/raw/alpha_prices_panel.parquet`（长表行情，用于算标签取价）

| 项 | 值 |
|----|----|
| shape | (2,273,529, 13) |
| 日期范围 | **2019-01-02 → 2026-05-11** |
| 交易日数 | 1762 |
| 股票数 | **1374** |
| 明显缺口 | 无（最大相邻交易日间隔 11 天，4 处，均为长假，正常） |
| 列 | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, adj_close |

### 3.2 `data/raw/daily_prices_panel.parquet`（旧长表行情）

| 项 | 值 |
|----|----|
| shape | (728,370, 11) |
| 日期范围 | **2018-10-08 → 2024-12-31**（已过期，停在 2024 年底） |
| 交易日数 | 1516 |
| 股票数 | **508** |
| 明显缺口 | 无（最大间隔 11 天，3 处，长假） |

### 3.3 `data/panel/alpha_panel_v4.parquet`（**模型实际训练面板**）

| 项 | 值 |
|----|----|
| shape | (29,406, 75) |
| index | MultiIndex(`as_of`, `ticker`) |
| 截面日范围 | **2019-03-31 → 2026-06-30**（30 个季度截面） |
| 股票数 | 1418 |
| 标签列 | `forward_return_21d`, `forward_return_63d`（**确含两列**） |
| 特征列 | 73 个因子（pe_ttm, pb, ... momentum_*, volatility_*, north_*, margin_*, beta_* ...） |

### 3.4 两个行情 parquet 现在被谁读

**`alpha_prices_panel.parquet`（新，到 2026-05，1374 票）——主用：**
- `quantmind/execution/price_source.py:24` — E3 执行层价格真源（`PRICE_PARQUET`）。
  注释（price_source.py:8-10）说明选它而非 daily：daily 只到 2024-12-31/508 票仅覆盖 10/80 笔，alpha 覆盖全部 80 笔。
- `scripts/train_momentum_patchtst.py:173`、`scripts/run_alpha_report.py:49`、`scripts/track_realized_pnl.py:358`、`scripts/compute_hold_baseline.py:50`、`scripts/run_full_system_demo.py:195`。
- `app/utils/data_loader.py:159/207`（优先级第一）、`scripts/db_migration/_restore_price_daily.py:20`、`scripts/db_migration/02_import_pg.py:304`。

**`daily_prices_panel.parquet`（旧，到 2024-12-31，508 票）——仍被读：**
- `quantmind/selection/lazy_data_engine.py:31/595`（`_LONG_PRICE_PANEL`，选股引擎价格回退源）。
- `quantmind/watchlist/manager.py:36/191`、`quantmind/watchlist/daily_scorer.py:107/361`（自选股最新价）。
- `quantmind/features/expr_factors.py:382`（表达式因子默认价格源）。
- `app/pages/9_持仓跟踪.py:39`、`app/pages/12_历史推荐.py:8`、`app/utils/rec_data.py:221/260`、`app/pages/7_系统控制台.py:657/781`、`app/utils/data_loader.py:160/208`（回退第二）。
- `quantmind/execution/manager.py:115`、`replay_engine.py:98`、`app/pages/14_执行管理.py:305` 注释标明「PG 表已空，改读 parquet」。

> **风险提示（事实陈述）**：`daily_prices_panel.parquet` 已停在 2024-12-31，
> 仍被自选股/选股引擎/表达式因子/多个 UI 页面读取——但这与标签 horizon 变更无直接关系。

---

## 4. 风险点（63d → 12d 改动影响面）

### 4.1 直接受影响文件清单（标签 horizon 相关）

| 文件:行 | 内容 | 影响 |
|---------|------|------|
| `quantmind/features/panel.py:167` | `compute_forward_returns(horizons_days=(21, 63))` 默认 | 改 12d 需在此加/换 horizon |
| `quantmind/features/panel.py:189` | `f"forward_return_{h}d"` 列名模板 | 新列名将是 `forward_return_12d` |
| `quantmind/features/panel.py:211` | `build_panel(forward_horizons_days=(21, 63))` 默认 | 同上 |
| `scripts/build_full_panel.py:326` | `label_cols = ["forward_return_21d","forward_return_63d"]` | 需加入 12d 列 |
| `scripts/build_full_panel.py:419` | `compute_forward_returns(pivot, as_of, (21, 63))` | horizon 元组写死 |
| `scripts/build_full_panel.py:404` | `fwd_end = latest + timedelta(days=int(63*1.6)+10)` | 取价窗口按 63 预留（12d 可缩短，但 63 这个值是硬编码） |
| `scripts/build_full_panel.py:500-501` | 覆盖率日志固定打印 21d/63d | 不影响逻辑，仅日志 |
| `scripts/train_board_models.py:433` | `--label` 默认 `forward_return_63d` | 改 12d 须改默认或传参 |
| `scripts/train_board_models.py:242` | 注释引用 `forward_return_63d` 数据特性 | 仅注释 |
| `quantmind/models/lgbm_ranker.py:457` | `walk_forward_evaluate(label_col="forward_return_21d")` 默认 | 默认已是 21d，被上层覆盖 |
| `quantmind/models/factor_cnn.py:227` | `preprocess_panel(label_col="forward_return_63d")` | CNN 默认标签 |
| `quantmind/models/factor_cnn.py:495` | `train_factor_cnn(label_col="forward_return_63d")` | CNN 训练默认 |
| `quantmind/models/factor_cnn.py:805` | 第三入口默认 `forward_return_63d` | CNN 默认 |
| `quantmind/risk/barra.py:128/136` | `label_col="forward_return_63d"` | Barra 归因前瞻收益列 |
| `data/panel/alpha_panel_v4.parquet` | 物化的训练面板（仅含 21d/63d） | **需重建**才能有 12d 列 |

### 4.2 标签来源不同、**不随面板标签自动变化**的部分（需单独决策）

| 文件:行 | 内容 |
|---------|------|
| `quantmind/models/meta_learner.py:353` | meta-learner 读 `actual_return_63d`（realized_pnl，非面板标签） |
| `scripts/train_meta_learner.py:121/126` | 同上，目标 z-score 基于 `actual_return_63d` |
| `scripts/track_realized_pnl.py:201` | 写出 `actual_return_63d`（结算口径，63 日持有） |
| `scripts/settle_forward_positions.py:121` | 写出 `actual_return_63d`（远期持仓结算） |

### 4.3 硬编码 `63` 的位置（区分「标签 horizon」与「特征窗口」）

**A. 标签 / 前瞻收益相关的 63（改标签时相关）：**
- `quantmind/features/panel.py:25`（docstring）、`:167`、`:211`（horizon 默认）。
- `scripts/build_full_panel.py:326/404/419`（label_cols、取价窗口、horizon 元组）。
- `scripts/train_board_models.py:433`（`--label` 默认 `forward_return_63d`）。
- `quantmind/models/factor_cnn.py:227/495/805`（CNN 标签默认）。
- `quantmind/risk/barra.py:128/136`（`forward_return_63d`）。

**B. 特征窗口的 63（≈3 个月，与标签无关，改标签时切勿误动）：**
- `quantmind/features/technical.py:54/100/118/128/150-151/162`（momentum_3m / volatility_3m / amihud / turnover_3m，均用 63 交易日窗口）。
- `quantmind/features/expr_factors.py:204-220`（`volatility_63d`、`amihud_63d` 表达式因子）。
- `quantmind/selection/funnel_selector.py:226-253`（趋势过滤 63 日价格窗口）。
- `quantmind/selection/lazy_data_engine.py:500`（`window_days=63`）。
- `quantmind/risk/drawdown.py:138`、`quantmind/risk/factor_risk.py:65`、`quantmind/portfolio/position_sizing.py:92`（`min_periods=63` 等统计窗口）。

**C. 执行 / 持有期的 63（实盘持有天数，独立于训练标签）：**
- `quantmind/execution/manager.py:34` — `DEFAULT_HOLDING_DAYS = 63`（默认持有 3 个月）。
- `quantmind/execution/manager.py:40-48` — `'3m' → 63` 解析规则。
- `quantmind/execution/optimizer.py:43/69` — `holding_days` 网格含 63、默认 63。

**D. 与 horizon 无关的巧合 63（不要碰）：**
- `quantmind/core/config.py:88`、`scripts/train_board_models.py:77` — `num_leaves=63`（LGBM 超参，纯巧合同值）。

---

## 5. 证据索引（关键文件一览）

| 主题 | 文件 |
|------|------|
| 标签计算核心 | [`quantmind/features/panel.py:164`](../quantmind/features/panel.py) |
| 标签物化进面板 | [`scripts/build_full_panel.py:326`](../scripts/build_full_panel.py) |
| LGBM 训练入口 | [`scripts/train_board_models.py:429`](../scripts/train_board_models.py) |
| LGBM walk-forward | [`quantmind/models/lgbm_ranker.py:457`](../quantmind/models/lgbm_ranker.py) |
| CNN 训练入口 | [`scripts/_save_cnn_v2.py:11`](../scripts/_save_cnn_v2.py) |
| CNN 标签默认 | [`quantmind/models/factor_cnn.py:495`](../quantmind/models/factor_cnn.py) |
| meta-learner 入口 | [`scripts/train_meta_learner.py:69`](../scripts/train_meta_learner.py) |
| meta-learner 标签 | [`quantmind/models/meta_learner.py:353`](../quantmind/models/meta_learner.py) |
| 训练面板 | `data/panel/alpha_panel_v4.parquet`（29406×75，含 forward_return_21d/63d） |
| 新行情源 | `data/raw/alpha_prices_panel.parquet`（2019-01-02→2026-05-11，1374 票） |
| 旧行情源 | `data/raw/daily_prices_panel.parquet`（2018-10-08→2024-12-31，508 票，已过期） |

---

*本报告仅陈述事实，未给出修改建议。审计过程未修改除本文件外的任何文件。*
