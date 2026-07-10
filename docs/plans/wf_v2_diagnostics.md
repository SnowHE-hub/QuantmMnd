# WF v2 诊断报告 —— 12d alpha 信号真伪判定

> 范围：**只做诊断，不建模、不接组合优化、不改 NAV 权重。**
> 目的：判断"要不要继续在这套 35 因子面板上追 12d alpha"。
> 数据：`alpha_panel_weekly_v5.parquet`（452,439 × 38，1374 票，幸存者池）
> 标签：`forward_return_12d` | 切分：H=12 / E=20 / rolling=756 / n_val=2，与 `scripts/run_wf_full_v1.py` 完全一致
> OOS：2022-01-01 → 2026-04-20，16 fold，142 个 OOS 截面
> 复现脚本：`scripts/_diag_wf_v2.py` ｜ 中间产物：`data/loss_signals_v4/wf_v2_*.{parquet,csv,json}`
> 生成：2026-06-06

⚠️ **所有指标均在幸存者池（乐观上界）上得到，真实含退市票后更低。**

---

## 0. 一句话结论

**没有**一个"稳定 + 中性后仍为正"的 12d 信号。
现有 12d 原始 IC（+0.0139）里约 **88% 是行业 / 规模 tilt**，行业+规模中性化后 IC 塌到 **+0.0017（IR 0.029，≈ 噪声）**；
方向在 16 fold 中翻转 8 次，且翻转集中在中性（震荡）regime。
**建议：停止在本因子集上追 12d alpha，回退到 63d 策略产品**（用同一 WF 框架换 `forward_return_63d` 复验）。

---

## 1. 基线复现（与 v2 报告对齐）

| 指标 | 本次复现 | v2 报告 | 说明 |
|---|---|---|---|
| OOS IC 均值 | **+0.0139** | +0.014 | 一致 |
| IC IR | **+0.133** | 0.13 | 一致 |
| 方向翻转 | **8 / 16 (50%)** | 7 / 16 (44%) | 差 1 fold，源自 val-IC 近 0 时的符号判定；不影响结论 |

逐 fold 方向与 IC（`fold_ic` / `fold_directions`）：

| fold | cutoff | dir | fold IC | | fold | cutoff | dir | fold IC |
|---|---|---|---|---|---|---|---|---|
| 0 | 2022-01-04 | −1 | +0.044 | | 8 | 2024-01-02 | +1 | +0.080 |
| 1 | 2022-04-01 | +1 | +0.013 | | 9 | 2024-04-01 | +1 | −0.024 |
| 2 | 2022-07-01 | +1 | −0.069 | | 10 | 2024-07-01 | −1 | −0.049 |
| 3 | 2022-10-10 | −1 | +0.063 | | 11 | 2024-10-08 | +1 | +0.130 |
| 4 | 2023-01-03 | −1 | +0.057 | | 12 | 2025-01-02 | −1 | −0.003 |
| 5 | 2023-04-03 | −1 | −0.020 | | 13 | 2025-04-01 | −1 | −0.030 |
| 6 | 2023-07-03 | +1 | −0.056 | | 14 | 2025-07-01 | +1 | −0.053 |
| 7 | 2023-10-09 | −1 | +0.043 | | 15 | 2025-10-09 | +1 | +0.058 |

方向（仅用 train/val 段决定，H-A 无 OOS 泄漏）正负各半 → 模型在样本内学不到稳定的因子方向，是 IC≈0 的直接表征。

---

## 2. 诊断一：Regime 条件 IC

### 2.1 Regime 标签可用性与 PIT

- `quantmind/models/hmm_regime.py` **不存在** → 采用面板内 `market_momentum_60d`（60 日市场动量）做三档切分。
- **PIT 安全性**：该列对每个 `as_of` 为常量（`nunique==1`，全市场同值），且输入是 **trailing 60 日**收益，仅用 as_of 当日及之前 → 满足"regime 只能用 as_of 当日及之前"。
- 三档切点为 OOS 期 tercile：`bear ≤ −0.0363 < neutral < +0.0225 ≤ bull`。
  ⚠️ tercile 切点用了 OOS 全期分布（描述性分桶，仅供诊断分组，非可交易决策）；regime **输入**本身 PIT 安全。
- 分布均衡：bear 69 / neutral 68 / bull 69 个 as_of。

### 2.2 IC × regime（总体）

| regime（市场动量档） | IC 均值 | IC std | OOS 截面数 | IC>0 占比 |
|---|---|---|---|---|
| **bear**（市场下行） | **+0.0280** | 0.093 | 45 | **55.6%** |
| **bull**（市场上行） | **+0.0241** | 0.136 | 44 | 52.3% |
| **neutral**（震荡） | **−0.0065** | 0.079 | 53 | 47.2% |

**结论**：
- IC 在 **方向性市场（bear / bull）为正**，在 **中性 / 震荡市场为负且 IC>0 占比 < 50%**。
- 44–50% 的方向翻转**主要来自 neutral regime**：震荡市里信号无方向。
- 但即便"最好"的 bear，IC>0 占比也仅 55.6%、bull 的 IC std 高达 0.136（均值正但极不稳）。**没有任何一个 regime 的 IC 达到稳定可用的强度。**

### 2.3 IC × regime × fold（完整表）

NaN = 该 fold 内该 regime 无 OOS 截面。

| fold | bear | bull | neutral |   | fold | bear | bull | neutral |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.044 | — | — | | 8 | +0.123 | +0.037 | +0.124 |
| 1 | +0.031 | +0.004 | −0.015 | | 9 | — | −0.008 | −0.033 |
| 2 | −0.007 | −0.161 | −0.055 | | 10 | −0.045 | −0.012 | −0.114 |
| 3 | +0.080 | — | +0.047 | | 11 | +0.098 | +0.135 | — |
| 4 | — | +0.076 | −0.061 | | 12 | +0.084 | −0.054 | −0.010 |
| 5 | +0.008 | — | −0.058 | | 13 | — | — | −0.030 |
| 6 | −0.053 | — | −0.057 | | 14 | — | −0.053 | — |
| 7 | +0.043 | — | — | | 15 | +0.152 | +0.069 | +0.044 |

bear 列 9 正 3 负、neutral 列 3 正 9 负 —— 与 2.2 一致：**neutral 是方向飘的来源**。

---

## 3. 诊断二：信号真伪（行业 + 规模中性化）—— 最关键

方法：对每个 OOS 截面，把**有效预测**（已乘方向）对 `exposure_industry`（行业哑变量）+ `log_market_cap`（标准化）做 OLS，取**残差**重算 rank IC（Barra 式）。规模取 `data/lake/daily_basic.parquet` 的 `circ_mv`（流通市值），按 `trade_date==as_of` PIT 对齐取 log（as_of 收盘已知，PIT 安全）。原始 IC 与中性 IC 在**同一只票子集**上计算，隔离样本差异。

### 3.1 总体（142 截面）

| | 原始 IC | 中性化后 IC | 保留比例 |
|---|---|---|---|
| IC 均值 | **+0.0139** | **+0.0017** | **12%** |
| IC IR | +0.133 | **+0.029** | — |

> 中性化后 IC 下降 **88%**，IR 从 0.133 塌到 0.029（≈ 噪声）。

### 3.2 分 regime

| regime | 原始 IC | 中性化后 IC |
|---|---|---|
| bear | +0.0280 | **−0.0066**（由正转负） |
| bull | +0.0241 | +0.0077（仅余 32%） |
| neutral | −0.0065 | +0.0037 |

**结论（信号真伪判定）**：
- 中性化后 IC **显著归零**（+0.0017，retention 12%）→ 这点 12d 信号**主要是行业 / 规模 tilt，不是选股 alpha**。
- 连"最稳"的 bear regime，中性化后都由 +0.028 **转负**（−0.007）；bull 仅余 1/3。
- 即"在方向性市场里 IC 为正"这件事，本质也是**行业 / 规模在不同市场状态下的 beta 漂移**，而非个股层面的选股能力。

---

## 4. 诊断三：特征 / IC 归因

### 4.1 特征重要性（gain，跨 16 fold 平均，归一化）

| 排名 | 特征 | gain 占比 | | 排名 | 特征 | gain 占比 |
|---|---|---|---|---|---|---|
| 1 | **exposure_industry** | **26.0%** | | 7 | momentum_6m | 3.5% |
| 2 | amihud_illiquidity | 7.7% | | 8 | distance_to_52w_high | 3.0% |
| 3 | max_drawdown_3m | 5.2% | | 9 | volume_price_corr_20d | 2.5% |
| 4 | list_age_years | 4.4% | | 10 | north_bound_30d_net_inflow | 2.4% |
| 5 | margin_balance | 3.9% | | 11 | market_drawdown_60d | 2.3% |
| 6 | market_volatility_60d | 3.8% | | 12 | momentum_12m_skip_1m | 2.2% |

- **`exposure_industry` 单列吃掉 26% 的 gain**，是第二名（amihud 7.7%）的 3.4 倍 —— 行业归属是模型最大的决策依据。
- 紧随其后的 `amihud_illiquidity` / `list_age_years` / `margin_balance` **均为规模 / 流动性代理**（小盘=高 amihud、低 list_age、低 margin_balance）。
- → gain 结构与第 3 节中性化结论互证：**行业 + 规模主导了模型，中性化把它们抽走后几乎不剩东西。**

### 4.2 单变量 IC（数值特征 vs forward_return_12d，OOS 截面 Spearman 均值）

| 特征 | 单变量 IC | | 特征 | 单变量 IC |
|---|---|---|---|---|
| volatility_3m | −0.063 | | margin_balance_change_20d | −0.044 |
| volatility_1y | −0.052 | | bollinger_position | −0.044 |
| volume_price_corr_20d | −0.049 | | margin_short_ratio | +0.044 |
| amplitude_quantile | −0.048 | | rsi_14 | −0.043 |
| momentum_1m | −0.047 | | price_to_52w_low | −0.041 |
| relative_strength_vs_csi500_60d | −0.046 | | reversal_1w | −0.040 |

- 单变量 IC 最强的全是 **低波动 / 反转 / 相对强弱**类**风格因子**（低波动、近期跌得多者反而强），是典型 A 股风格 tilt，**同样不是个股 idiosyncratic alpha**。
- `exposure_industry` / `exposure_area` 为类别变量，截面无 Spearman（故不在表内），其作用由 4.1 gain（26%）+ 第 3 节中性化体现。
- `market_*` 列截面内常量 → 单变量 IC 为 NaN（正常）。

---

## 5. 诊断结论 + 建议

### 5.1 有没有一个"稳定 + 中性后仍为正"的 12d 信号？

**没有。** 三条证据一致：

1. **不稳**：方向 16 fold 翻 8 次；IC 仅在方向性市场为正，中性（震荡）regime 翻负，IC>0 占比 47%。
2. **不真**：行业+规模中性化后 IC 从 +0.0139 → **+0.0017（retention 12%，IR 0.029）**；连 bear regime 中性化后都转负。
3. **来源是 tilt**：gain 第一名 `exposure_industry` 占 26%，2–5 名全是规模/流动性代理；单变量 IC 最强的是低波动/反转风格。
4. 以上还都在**幸存者乐观上界**上 —— 真实只会更差。

> 现有 12d "alpha" 本质 = 行业轮动 + 小盘/低波动风格 beta，**不是选股能力**。在本因子集上继续调参（学习率/树深/特征筛选）改变不了这一点，因为面板里就没有个股层面的 idiosyncratic 信息源。

### 5.2 建议

**回退到 63d 策略产品（首选）。**
- 用**同一 WF 框架**把标签换成 `forward_return_63d`（面板已有该列）复验：H/E 相应放大（H=63、E≥63），rolling 窗口不变。
- 理由：12d 短周期里量价信号被风格/行业 beta 主导、信噪比低；63d 长周期给基本面 / 价值 / 资金流因子更高信噪比，且交易成本占比下降。**先验证 63d 的中性化后 IC 是否为正**，再决定是否产品化。
- 验收沿用本套 gate（中性化后 IC>0 为新增硬条件）。

**不建议**继续追 12d，除非同时满足以下两点（属"换数据/换因子"而非"调参"，本 session 不展开）：
- **regime 条件化**：只在方向性（bear/bull）regime 持仓、震荡市空仓 —— 但本诊断显示即便如此，中性化后 bear 仍转负，单靠 regime 门控**救不回**选股 alpha。
- **引入催化 / 事件因子**：现面板缺个股 idiosyncratic 信息源（财报 surprise、事件、分析师修正等）。这是数据缺口，不是模型问题。

---

## 附：产物清单

| 文件 | 内容 |
|---|---|
| `scripts/_diag_wf_v2.py` | 复现脚本（捕获逐票 OOS 预测 + 三项诊断） |
| `data/loss_signals_v4/wf_v2_oos_predictions.parquet` | 191,744 行逐票 OOS 预测（pred/realized/industry/log_mktcap/regime/fold） |
| `data/loss_signals_v4/wf_v2_diag_summary.json` | 全部诊断数值 |
| `data/loss_signals_v4/wf_v2_per_date_ic.csv` | 逐 as_of IC + regime + fold |
| `data/loss_signals_v4/wf_v2_neutralization.csv` | 逐 as_of 原始 vs 中性化 IC |
| `data/loss_signals_v4/wf_v2_regime_fold_ic.csv` | IC × regime × fold 透视表 |
