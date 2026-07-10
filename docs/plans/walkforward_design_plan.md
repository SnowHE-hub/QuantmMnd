# 实现计划：Walk-Forward 验证框架（P1 决定性闸门 · 纯设计）

> **本文件仅为实现计划，不含任何实现代码、不改任何业务代码。**
> **实现前置条件：周频面板 `alpha_panel_weekly_v5.parquet` 通过 verify（`weekly_panel_build_plan.md` §10）+ 本计划通过 code-review。** 未满足不进入实现。
> 依据：`docs/plans/weekly_panel_build_plan.md`（面板 schema）、现有 `quantmind/models/lgbm_ranker.py::walk_forward_evaluate`（:454）、`quantmind/backtest/{execution,walk_forward,metrics,engine}.py`（已读，见 §9）。

---

## 0. 目标与边界

| 项 | 内容 |
|----|------|
| 目标 | 在周频面板 v5 上，用**带 purge/embargo 的 walk-forward** 在固定样本外区间判定 LGBM 截面信号是否构成可用 alpha——这是一道**事前定线、事后不挪门**的决定性闸门。 |
| 输入 | `data/panel/alpha_panel_weekly_v5.parquet`（MultiIndex(as_of,ticker)，35 因子 + `forward_return_{12,21,63}d`，标准化=False，幸存者池）；`data/lake/index_daily.parquet`（000300.SH 取基准与价格）；SSE 交易日历（`quantmind/data/sse_calendar.py`）。 |
| 本版判定标签 | **`forward_return_12d`**（12 交易日前瞻收益），与持有期 12 日一致。 |
| 显式排除 | 不上 Barra 约束优化器（P2）；不跑 FactorCNN（留作之后）；不做空、不杠杆、不择时；不调 Tushare；不改面板生产路径；不改任何既有业务/模型/回测代码（新增代码集中在新模块，见 §9）。 |
| 输出物（本版） | 新模块 `quantmind/backtest/wf_gate.py`（+ `wf_split.py` / `wf_costs.py` / `wf_metrics.py`，见 §9）+ 验收脚本 + 测试 + 一份报告 `docs/reports/p1_walkforward_<date>.md`（含幸存者上界标注）。**放 `quantmind/backtest/` 下复用 `ExecutionSimulator`/`PerformanceMetrics` 最自然，且绝不覆盖既有两套 walk-forward（§0.5）。** |
| 架构原则 | **复用既有回测/成本/指标设施**（§9），**仅新建** purge/embargo 切分器、成本扩展（流动性分层滑点【硬性】+ 时变印花税 + 板块/时变涨跌停）、IC/分位/净超额闸门编排与无泄漏自证。 |

---

## 0.5 与既有两套 walk-forward 的关系（先讲清，避免重复造轮子）

读码实测：仓库**已有两套** walk-forward，**都没有 purge/embargo**，且都不直接适配"周频 5 日采样 + 12 日重叠标签"的 PIT 要求。本闸门**不修改它们**，新建一套，并复用其下游设施。

| 既有实现 | 位置 | 它是什么 | 为何不能直接用 |
|---------|------|---------|---------------|
| `walk_forward_evaluate` + `WalkForwardSplit` | `models/lgbm_ranker.py:454`、`models/factor_model.py:113` | 截面 IC 用的**扩展窗口**切分：`train=dates[:test_idx]` 紧贴 `test=dates[test_idx]`（单测试截面），产 IC/分位 spread。 | ① **零 purge/零 embargo**：train 末尾紧贴 test，12 日标签必然跨界泄漏（§2）。② 测试是**单截面**逐点扩展，不是"固定样本外区段"。③ `auto_flip` 用**全体折（含 OOS）的 IC 均值**决定信号方向 = **用 OOS 定方向的数据窥探泄漏**（§12 重点拦截）。 |
| `WalkForwardValidator` | `backtest/walk_forward.py:118` | **日历滚动**窗口（train 756 / val 126 / test 63 / step 63 交易日），跑 Strategy 过 `BacktestEngine` 出 NAV。 | ① 同样**无 label-overlap purge、无 embargo**（val 夹在中间但不按标签 horizon 清洗）。② 按**日历日**而非**周频 as_of 网格**切，与 v5 的 5 日锚点不对齐。③ 面向日频 NAV，不产截面 IC 闸门指标。 |

**结论**：新建 `PurgedWalkForwardSplit`（交易日索引感知、purge+embargo），**编排**复用 `LGBMRankerModel`/`build_lgbm_arrays`（训练+IC）与 `BacktestEngine`/`ExecutionSimulator`/`PerformanceMetrics`（NAV+成本+风险指标）。详见 §9。

---

## 1. Fold 排期（滚动训练 → 固定样本外）

### 1.1 三段时间轴（固定切分）

| 段 | 区间 | 用途 | 备注 |
|----|------|------|------|
| **训练池** | 2019-01-02 → **2024-06-30** | 滚动训练来源 | 早期 2019 长窗因子（`beta_252d` 等）与 margin（2019-12-02 起）有 NaN，按面板 §3 语义处理（LGBM `fillna(0.0)`）。**有效满覆盖约 2020-01 起。** |
| **固定样本外（OOS）** | **2024-07-01 → 2025-12-31** | 决定性判定区 | 事前冻结；闸门只看这段。 |
| **实时留存** | 2026-01-01 → 2026-04-20（面板末锚） | 不参与判定 | 上线后真盘前向跟踪用；本闸门**完全不读**。 |

> 面板 as_of 网格步长 = **5 交易日**；总锚点 ≈ 350（面板 §1）。按交易日年化 ≈ 252/年 → 约 **50 as_of/年**。

**各段周频 as_of 数（估算，精确值由 `weekly_asof_grid.parquet` 落定）：**

| 段 | 交易日跨度（约） | 周频 as_of 数（约） |
|----|----------------|-------------------|
| 训练池 2019-01→2024-06 | ~1335 td | **~267** |
| OOS 2024-07→2025-12 | ~364 td | **~73** |
| 实时留存 2026 | ~70 td | ~14（不用） |

> 训练/测试边界的**精确 as_of** = 网格中 `≤ 2024-06-30` 的最后一个 as_of（训练侧）与 `> 2024-06-30 + embargo` 的第一个 as_of（测试侧）；由实现按日历索引计算，不手填。

### 1.2 训练窗口：扩展 vs 滚动

| 方案 | 定义 | 选用 |
|------|------|------|
| **扩展窗口（Expanding，主选）** | 训练起点固定 2019-01-02，截止线随 refit 前移；用尽全部历史 regime。 | ✅ **主判定**。A 股因子数据饥饿且 regime 敏感（参 `lgbm_ranker.py:551` 的 2021–2024 regime 反转记录），扩展窗口纳入更多 regime，结论更稳健、更难被单一切点 p-hack。 |
| **滚动窗口（Rolling-36mo，鲁棒性）** | 训练起点 = `refit_cutoff − 36 个月`，固定窗长。 | ⬜ **鲁棒性复核**。若扩展窗口过线但滚动窗口塌掉 → 信号依赖久远 regime，结论降级。两者**同时跑、并列报告**。 |

### 1.3 滚动 refit 步进与测试区段

- **Refit 步进**：每 **13 个周频 as_of ≈ 65 交易日 ≈ 1 季度**重训一次。OOS 约 73 as_of → 约 **5–6 次 refit**。
- **每次 refit k**：截止线 `C_k`（首个 `C_0 = 2024-06-30`，之后每季前移）。
  - 训练集 = `as_of ≤ C_k` 且通过 purge（§2）；val = 训练尾部最后 `n_val=2` 个 as_of（早停用，仍在 purge 内）。
  - 测试区段 = OOS 中落在 `(C_k + embargo, C_{k+1}]` 的 as_of（最后一段到 OOS 末）。
- **🔑 度量分两用（统计功效 vs 真实 PnL，刻意解耦）**：
  - **IC / 分位单调性 → 用全部周频截面**（每 5 交易日一个，OOS 共 ~73 个）。重叠不影响 Spearman 的单截面计算，**截面越多统计功效越大**——用满。
  - **组合 PnL → 用非重叠 12 交易日换仓**（§4），避免重叠收益在 NAV 里重复计数、虚高夏普/超额。
  - 两条线**同一模型、同一方向、不同采样**：IC 测信号强度，PnL 测产品真实可实现性。
- **另设静态单切基线**：单次 `C_0=2024-06-30` 训练、整段 OOS 评估，作为"无 refit 选择自由度"的最干净基线，与滚动结果对照（防 refit 引入隐性调参自由度）。

**验收（§10 S2）**：① 训练/测试 as_of 集合**零交集**且边界满足 `min(test_idx) − max(train_idx) ≥ H + E`；② OOS 测试 as_of 全部落在 `[2024-07-01+embargo, 2025-12-31]`；③ refit 次数与 as_of 计数与上表一致（±1）。

---

## 2. Purge / Embargo（核心正确性）

### 2.1 为什么必须有（5 日采样 × 12 日标签 = 必然重叠）

标签 `forward_return_12d` 的取值窗口 = `[as_of, as_of+12td]`（用未来 12 个交易日价格）。网格步长仅 5 日 → **相邻 as_of 的标签窗口重叠 7/12**。若训练集纳入"标签窗口跨越训练截止线 C"的样本，其标签就**用到了 C 之后（测试区）的价格** → 直接泄漏，IC 虚高。

### 2.2 精确逻辑（交易日索引空间）

设 `idx(a)` = as_of `a` 在 SSE 交易日历中的位置；`H = 12`（标签 horizon，交易日）；`E = embargo`（交易日，**≥12**，默认 `E = H = 12`，可选 `E = 15` 加缓冲）；`C = idx(cutoff)`。

```
# 训练集（purge）：标签窗口不得越过 C
train = { a : idx(a) + H ≤ C }              # 等价 idx(a) ≤ C − H

# 隔离带（dropped，既不训练也不测试）：
#   (C − H, C + E]  —— purge 掉的尾部 + embargo 缓冲

# 测试集（embargo 后才开测）：
test  = { a : idx(a) > C + E }  ∩  OOS 区间
```

- **Purge 推导**：训练 as_of 的标签 `[a, a+H]` 末端 `a+H` 必须 `≤ C` 才不碰测试区 → `idx(a) ≤ C − H`。
- **Embargo 推导**：最后一个训练标签止于 `≤ C`；要求首个测试 as_of `idx > C + E`，使测试标签起点与训练标签末端隔开 `≥ E (≥12)` 交易日，规避边界自相关把 IC 抬高。
- **val 同受 purge**：val 取训练尾部 2 个 as_of，本身已满足 `idx ≤ C − H`，无需额外处理。

### 2.3 与既有切分器的差异（必须新建）

`factor_model.WalkForwardSplit.split` 是 `train = dates[:test_idx]`（紧贴），**无 `−H` 的 purge、无 `+E` 的 embargo、不感知交易日索引**。本闸门**新建** `PurgedWalkForwardSplit`（输入：周频 as_of 网格 + SSE 日历索引 + `H` + `E` + cutoffs；输出：含隔离带的 `(train_dates, val_dates, test_dates)`）。**不改** `WalkForwardSplit`（其他季度路径仍依赖）。

### 2.4 工作示例（说明性，精确值实现时计算）

设 `C_0 = 2024-06-30`，`idx(C_0)=1335`，`H=E=12`：
- 末个训练 as_of：`idx ≤ 1323` → 约 **2024-06-12**。
- 隔离带：`idx ∈ (1323, 1347]` → 约 **2024-06-13 → 2024-07-15** 的 as_of 全部丢弃。
- 首个测试 as_of：`idx > 1347` → 约 **2024-07-16**。
- 即 OOS 头部约 2 周 as_of 因 embargo 被砍——**正确代价**，不可省。

---

## 3. 成本模型（A 股，T+1）

### 3.1 复用 + 必要扩展

`ExecutionSimulator`（`backtest/execution.py`）**已实现**：佣金双边（万3）、印花税卖出单边（千1）、过户费沪市双边（万0.2）、滑点（默认 10bp）、**T+1**（`portfolio.can_sell`）、涨跌停拒单、停牌不成交、单笔 ≤ 当日成交额 5%、整手。`BacktestConfig` 已有对应字段（`engine.py:48`）。**直接复用**，但有**两处闸门级缺口必须扩展**（新增配置/逻辑，不改既有默认行为）：

| 缺口 | 现状 | 本闸门要求 | 实现方式（新增，不破坏既有） |
|------|------|-----------|---------------------------|
| **🔴 滑点流动性分层（硬性，本版必做，不得延后）** | `slippage_bp` **单一常量** 10bp（`engine.py:52`） | 中小盘更高 | 新增 `slippage_fn(ticker, bar) → bp`：按流动性分层（用面板内 `amihud_illiquidity` 分位，或 `amount` 近 20 日 ADV 代理）。建议三档：大盘 5bp / 中盘 15bp / 小盘 30bp（可配，事前定，事后不调）。**为何硬性**：幸存者池里中小盘用平 10bp 会**低估成本→高估 alpha**，是**假阳性方向**，必须在本版根除（对比下一行印花税：平 0.1% 是**高估**成本的保守方向，不造假，可后做但仍应做准）。 |
| **印花税时变** | `stamp_duty` 是**常量** `1e-3`（`engine.py:50`），方向上**偏保守（高估成本）** | 区间内有过下调，需**时变** | 新增 `stamp_duty_schedule`（按 `fill_date` 取费率）；实现时**确认确切生效日与费率**——已知线索：**2023-08-28 起印花税由 0.1% 减半至 0.05%**（实现 S0 核官方公告，写 `findings.md`，不得照抄本数字）。OOS 2024-07→2025-12 **全程落在减半后** → OOS 段印花税恒 **0.05%**；训练池/全样本 NAV 仍需时变以保正确。 |
| **🔴 涨跌停板块×时变（硬性，并入 S0 核实）** | `ExecutionSimulator` 硬编码 `±9.95%`（`execution.py:30-31`），**只对主板 10% 正确** | 涨跌停阈值须按**板块 + 时间**正确 | 新增 `limit_pct(ticker, fill_date) → pct`：科创板 `688*` 20%、创业板 `30*` **2020-08-24 起** 20%（之前 10%）、主板 `60*/00*` 10%；ST 5%（本池无退市/ST，按需）。**为何硬性**：错误的涨跌停阈值会让"涨停买不进/跌停卖不出"的摩擦记错 → NAV 失真；科创/创业板用 10% 阈值会**误判可成交**，方向上同样偏高估 alpha。生效日/阈值在 S0 与印花税一起核实。 |

### 3.2 入场 / 出场时点（精确定义）

| 事件 | 时点 | 价格口径 |
|------|------|---------|
| **出信号** | `as_of` **收盘** | 用截至 as_of 收盘已知的 35 因子 → 模型打分 → 选股。**绝不**用 as_of 之后任何信息。 |
| **入场** | `as_of + 1` 交易日（**T+1 约束**） | `execution_mode="open_price"`（`BacktestConfig` 默认）= **次日开盘**成交 + 买方滑点。as_of 收盘→次日开盘的隔夜跳空是**真实实现成本**，由实际成交价体现（不用 12d 标签近似）。 |
| **持有** | **12 交易日** | —— |
| **出场** | 入场后第 12 交易日（即 `as_of + 1 + 12` 的开盘）或该持有末日收盘——**二选一，事前定**，建议 **`as_of+13` 开盘**与入场口径一致 | 卖方滑点 + 卖出印花税（时变）。 |

- **NAV 用实际成交价链算，不用 `forward_return_12d` 标签**。标签只服务训练与 IC；NAV 走 `BacktestEngine` 实际撮合（含 T+1、涨跌停、滑点、费）。两者口径分离已在面板 §10 硬条件 2 与本计划 §12 强制。
- **涨跌停/停牌**：入场日涨停买不进、出场日跌停卖不出 → 由 `ExecutionSimulator` 自然拒单/顺延，记入 NAV（真实摩擦，不修正）。

---

## 4. 组合构造（保持简单透明，为判定 alpha）

| 维度 | 方案 | 理由 |
|------|------|------|
| 方向 | **Long-only** | 判 alpha 有无，不引入做空摩擦/融券约束。 |
| 选股 | **主口径 = Top 五分位（前 20%）等权**；**额外报产品真实口径 = Top-20 / Top-30 固定只数等权** | 五分位更稳、更抗少数幸运股（幸存者池尤需）；客户实际只买几十只，**Top-20/30 才是产品真实可实现口径**，必须并报。 |
| 权重 | **等权** | 透明、无优化器自由度。 |
| 换仓 | **每 12 交易日**（独立换仓日历，锚定 OOS 起点按 12 td 步进）；信号取**换仓日 ≤ 当日的最近 as_of**（PIT，最多滞后 4 个交易日） | 与持有期一致、非重叠，避免重叠收益在 NAV 里重复计数。 |
| 换手 | 计算**实际名单增减**只交易 delta（持仓内留存股不做整轮往返） | 不高估成本；更贴实盘。 |
| 基准 | **沪深300（000300.SH）**；超额 = 组合 − 基准 | `BacktestConfig.benchmark` 默认即此。 |
| 优化器 | **不上 Barra 约束优化器（P2）** | 本版只判信号本身。 |

> **换仓 12 日 vs 网格 5 日不整除**的处理（拍板 F = 非重叠 12 日）：换仓日历独立按 12 td 锚定，信号回看最近 as_of。这字面满足"12 日持有 / 12 日换仓"，且 NAV 非重叠、PnL 独立、正好对齐标签。重叠/交错组合留作 P2 平滑曲线，本版不做。

---

## 5. 度量

| 指标 | 定义 | 复用/新建 |
|------|------|----------|
| **IC** | 每 OOS 截面 `Spearman(pred_score, realized_12d_return)`；报均值 + 序列 | 复用 `lgbm_ranker._evaluate_fold` 的 rank_ic 逻辑（`stats.spearmanr`） |
| **ICIR** | `mean(IC) / std(IC)`（与 `WalkForwardResult.ic_ir` 同口径）；另报年化 `× √(截面/年)` | 复用 `WalkForwardResult.ic_ir`；年化为新增 |
| **分位单调性** | 按预测分 5 组，组均实现收益 Q1<…<Q5 的单调度（建议：`Spearman(组序, 组均收益)` + 单调步数计数） | 部分复用 `_evaluate_fold` 的 quintile spread；单调度为新增 |
| **净超额年化（含成本）** | 组合 NAV vs 沪深300，扣全部成本后的年化超额 | 复用 `PerformanceMetrics`（`annualized_return`/`alpha`）+ 基准对齐；净超额组合为新增编排 |
| **对基准胜率** | **每换仓期**组合收益 > 同期基准收益的比例（**非** `metrics.py` 里逐日正收益口径） | **新建**（`PerformanceMetrics._compute_trading_metrics` 的 win_rate 是逐日正收益，口径不符，不能直接用） |
| **最大回撤** | 组合 NAV 的 MDD（另报相对基准的超额 NAV MDD） | 复用 `PerformanceMetrics.max_drawdown` |
| **死扛基准** | **沪深300 买入持有**，整段 OOS，从 `data/lake/index_daily.parquet` 的 000300.SH **重算**，**不硬编码** | 复用 `BacktestEngine` 基准重算（`engine.py:379` 从 `benchmark_df` 算）+ 显式从 lake 读 000300.SH |

> **采样口径（承 §1.3）**：IC / ICIR / 分位单调性 **用全部周频截面**（~73，统计功效最大）；净超额 / 胜率 / MDD **用非重叠 12 日换仓 PnL**（产品真实口径）。两者勿混采样。
> 辅助：Sharpe/Sortino/Calmar/IR/beta/alpha 由 `PerformanceMetrics.compute_all` 顺带产出，作上下文，不入闸门主线。

---

## 6. 通过线（**事前定死，事后不挪门**）

| 指标 | 基础通过线 | 强信号线（另列） |
|------|-----------|-----------------|
| IC（均值） | **> 0.03** | > 0.05 |
| ICIR | **> 0.4** | > 0.6 |
| 净超额年化（含成本） | **> 5%** | > 10% |
| 对基准胜率（每换仓期） | **> 52%** | > 55% |
| 最大回撤 | **< 25%** | < 20% |
| 分位单调性（辅助门槛） | 5 组组序 vs 收益 Spearman > 0（方向一致即可） | 严格单调 Q1<Q2<Q3<Q4<Q5 |

**判定规则**：基础线**全部满足**=过闸（在幸存者上界意义下，§8）；任一不满足=不过。强信号线仅作分级标注，不改通过/不过。

> ⚠ **A 股短周期 IC 普遍低于美股**。IC **0.03–0.05** 是文献中真实可用水平（A 股周频/月频截面因子常见区间），**不要用美股式（>0.1）的错误预期否定真信号**。本线据此设定，已在评审前冻结。

---

## 7. 模型

- **本版只跑 LGBM，全市场池化截面排序**（LambdaRank，复用 `LGBMRankerModel` 默认超参，`lgbm_ranker.py:183`）——每个 as_of 在全 universe 内排序。**早读用：数据多、单信号、最快拿结论。**
- **科创板 `688*` 纳入池化**（与沪市主板 `60*`、深市主板 `00*`、创业板 `30*` 同池）；**另报每板块 IC 分解**作诊断（看信号是否只来自某板块）。⚠ 科创/创业板的 20% 涨跌停由成本模型 S0 正确处理（§3）。
- **分板块各训独立模型 = 生产/细化步骤，本版不做**（之后再说）。
- **FactorCNN 留待之后**（`models/factor_cnn.py` 不在本版）。
- **`auto_flip` 在本闸门强制关闭或改造**：现版用 OOS IC 定方向（§0.5、§12）。本闸门**方向必须事前固定**（按因子经济先验或仅用训练期 in-sample IC 决定），**严禁用 OOS 数据定方向**。实现时 `walk_forward_evaluate(..., auto_flip=False)` 并独立处理方向。

---

## 8. 幸存者警示（结论性质标注，不可省）

- 面板 v5 源头即**幸存者池**（`weekly_panel_build_plan.md` §0.5：1374 票全交易到 2026、`stock_basic` 100% `L`、`delist_date` 0/1374；A 股 2019–2026 实际退市数以百计却完全缺席）。
- **本闸门一切结论 = 乐观上界（optimistic upper bound）**。短周期反转/动量在"只剩赢家"池里系统性虚高（已实现暴跌的退市票被剔除）。
- **判定的非对称解读（必须写入报告）**：
  - **过闸 ≠ 真有 alpha**：是"必要非充分"；真实（含退市）表现更低，需待退市票行情回补后复核。
  - **不过闸 = 强负面信号**：连灌了水的幸存者池都过不了线，真实环境几乎不可能有 alpha。
- 报告与 `walkforward_gate` 输出 meta **强制含**幸存者上界标注 + "退市票回补为后续项"指针（承接面板 §0.5 第 3 点）。

---

## 9. 复用 vs 新建（依据已读源码）

### 9.1 直接复用（**不修改**）

| 设施 | 位置 | 复用点 |
|------|------|--------|
| `LGBMRankerModel` | `models/lgbm_ranker.py:169` | 训练/预测（`auto_flip=False`） |
| `build_lgbm_arrays` / `CrossSectionalLabel` | `models/factor_model.py:308` / `:192` | X/y/groups 构造、分位标签 |
| `_evaluate_fold` 的 IC/分位逻辑 | `lgbm_ranker.py:348` | rank_ic、quintile（编排复用，必要时抽函数） |
| `ExecutionSimulator` | `backtest/execution.py:79` | A 股撮合：佣金/印花/过户费/滑点/T+1/涨跌停/停牌/量约束 |
| `BacktestEngine` + `BacktestConfig` | `backtest/engine.py` | NAV、基准重算（000300.SH，`:379`） |
| `PerformanceMetrics.compute_all` | `backtest/metrics.py:37` | MDD/Sharpe/IR/alpha/beta/年化 |
| SSE 日历 | `data/sse_calendar.py` | 交易日索引（purge/embargo 的 `idx()`） |
| `data/lake/index_daily.parquet` | （面板 S1 产物） | 000300.SH 取基准 |

### 9.2 新建（集中在新模块，互不污染）

> **路径决定（拍板 G）**：全部放 **`quantmind/backtest/`** 下（复用 `ExecutionSimulator`/`PerformanceMetrics` 最自然），文件名 `wf_*` 前缀，**绝不覆盖既有 `backtest/walk_forward.py` 与 `models/lgbm_ranker.py` 两套 walk-forward**。

| 新增 | 文件 | 内容 |
|------|------|------|
| `PurgedWalkForwardSplit` | `quantmind/backtest/wf_split.py` | §2 交易日索引感知的 purge+embargo 切分（扩展/滚动两模式 + cutoffs） |
| 成本扩展（3 处） | `quantmind/backtest/wf_costs.py`（`BacktestConfig` 钩子） | §3.1：流动性分层滑点【硬】+ 时变印花税 + 板块/时变涨跌停，**不改 `execution.py` 默认行为** |
| 闸门编排器 | `quantmind/backtest/wf_gate.py` | 串：切分→训练(LGBM)→截面 IC→组合构造(§4)→NAV(`BacktestEngine`)→§5 度量→§6 判线→§8 标注 |
| 闸门专属指标 | `quantmind/backtest/wf_metrics.py` | 分位单调性、**每换仓期对基准胜率**、净超额年化编排 |
| 验收 + 无泄漏自证脚本 | `scripts/verify_walkforward.py` | §10 + §12（含 purge 消融/篡改） |
| 测试 | `tests/test_wf_split.py`、`tests/test_wf_gate.py`、`tests/test_wf_costs.py` | purge/embargo 边界、cost 三处扩展（含板块涨跌停/时变印花）、胜率口径 |
| 报告 | `docs/reports/p1_walkforward_<date>.md` | 结果 + 幸存者上界标注 |

> **不新建第三套 NAV 引擎**：组合 NAV 一律走既有 `BacktestEngine`，新模块只产"每换仓日目标持仓"喂给引擎（参考既有 `backtest/lgbm_strategy.py` 的 model→signal→order 桥接模式，实现前先读它确认接口）。

---

## 10. 分步实现顺序 + 每步验收 + 工作量

> 全程前置：面板 v5 verify 通过 + 本计划评审通过。

| 步 | 内容 | 触及/新增 | 验收 | 工作量 |
|----|------|----------|------|-------|
| **S0** | 核验成本事实 + 读桥接代码：① 印花税减半确切生效日/费率；② **涨跌停板块×时变**（科创/创业板 20%、创业板 2020-08-24 起改、主板 10%）确切阈值与生效日；③ 读 `lgbm_strategy.py` 确认 model→order 接口。全部写 `findings.md` | 只读 | findings 记下三组官方来源；接口契约清晰 | 0.5d |
| **S1** | `PurgedWalkForwardSplit`（扩展/滚动 + purge+embargo） | `wf_split.py` | 单测：构造小日历，断言 `train_idx ≤ C−H`、`test_idx > C+E`、零交集；扩展/滚动两模式 | 1d |
| **S2** | Fold 排期落定（§1）：cutoffs、refit、OOS as_of 计数 | `wf_gate.py` | OOS as_of ∈ 区间、refit 次数对、边界 gap ≥ H+E | 0.5d |
| **S3** | 成本扩展（**3 处**）：流动性分层滑点【硬】+ 时变印花税 + 板块/时变涨跌停 | `wf_costs.py` | 单测：amihud 分位→滑点档位（**必过**）；`fill_date` 跨减半日费率切换；688/30/60 阈值随板块+日期正确；既有默认行为不变 | 1.5d |
| **S4** | 训练+IC 编排（复用 LGBM/arrays，`auto_flip=False`，方向只用训练/验证段） | `wf_gate.py` | 每 OOS 截面产 IC（全 ~73 截面）；方向不依赖 OOS（§12 H-A 断言） | 1d |
| **S5** | 组合构造 + NAV（五分位 + Top-20/30 等权、**非重叠 12 日**换仓、delta 换手、过 `BacktestEngine`） | `wf_gate.py` | NAV 用实际成交价（非标签）；换仓非重叠；基准从 lake 重算 | 1.5d |
| **S6** | 闸门指标：分位单调性、**每换仓期对基准胜率**、净超额年化、MDD | `wf_metrics.py` | 与手算抽查一致；胜率口径=每换仓期（非逐日） | 1d |
| **S7** | 判线 + 报告（§6/§8） | `wf_gate.py`、报告 | 全指标对照通过线输出 过/不过 + 强信号分级 + 幸存者上界标注 | 0.5d |
| **S8** | 无泄漏自证（§12，含 purge 消融/篡改） | `scripts/verify_walkforward.py` | §12 全项 PASS；purge-off 时 IC 显著虚高（反证 purge 生效） | 1.5d |

**总估** ≈ 9 人日。**关键路径**：S1→S3→S4→S5→S8（S3 成本三处扩展含两条假阳性根除项，已上调 0.5d）。

---

## 11. 风险点

| 级别 | 风险 | 缓解 |
|------|------|------|
| **高** | **purge/embargo 写错 → 标签跨界泄漏 → IC 虚高** | §2 精确公式 + S1 单测 + §12 purge 消融反证（关掉就虚高才证明它在起作用） |
| **高** | **`auto_flip` 用 OOS 定方向（既有泄漏）** | 强制 `auto_flip=False`；方向仅由先验/训练期 IC 定；§12 专项断言 |
| **高** | **用 `forward_return_12d` 标签当 NAV 收益**（绕过 T+1/滑点） | NAV 一律走 `BacktestEngine` 实际成交价；§12 断言"标签≠NAV来源" |
| **高** | **幸存者偏差（源头级）** | 无法在本版消除；如实标注乐观上界 + 非对称解读（§8） |
| **高** | **滑点用平 10bp（中小盘低估成本）→ 假阳性 alpha** | 流动性分层滑点列为**硬条件本版必做**（§3、§12 H-B）；amihud 分位三档 |
| **高** | **涨跌停阈值错（科创/创业板用 10%）→ 误判可成交、NAV 失真** | 板块×时变 `limit_pct`（§3）；S0 核 2020-08-24 创业板改 20% 等生效日；§12 H-E 抽查 |
| 中 | **印花税时变日期/费率搞错** | S0 先核官方公告写 findings，不照抄；OOS 全程减半后→对 OOS 影响小（保守方向），但全样本 NAV 需正确 |
| 中 | **换仓 12 日 vs 网格 5 日不整除致重叠收益重复计数** | 独立 12 日换仓日历 + 非重叠断言；备选方案 §13 |
| 中 | **refit 引入隐性调参自由度** | 并行跑静态单切基线对照；通过线事前冻结 |
| 中 | **胜率口径误用 `metrics.py` 逐日正收益** | 新建"每换仓期 vs 基准"胜率，单测口径 |
| 低 | **基准被硬编码** | 强制从 `data/lake` 重算 000300.SH；§12 断言 |
| 低 | **早期 2019 训练样本大量 NaN 因子** | LGBM `fillna(0.0)` 容忍；扩展窗口有效起点 ~2020；报告标注 |

---

## 12. 自证无泄漏检查清单（"如何确认闸门没作弊"）

由 `scripts/verify_walkforward.py` 自动执行，全部须 PASS。

**🔒 进实现前的硬条件（评审拍板，全防假 alpha；任一不满足不得进实现/不得宣布过闸）**

- [ ] **H-C（核心闸门）— purge 消融反证**：同一切分跑两次——
  - (a) **purge+embargo 开**（`H=12, E=12`）→ `IC_purged`；
  - (b) **purge 关 + embargo=0**（train 紧贴 test，复刻既有 `WalkForwardSplit` 行为）→ `IC_nopurge`。
  - **期望 `IC_nopurge` 明显 > `IC_purged`**（重叠标签把边界 IC 抬高）。**若两者几乎相等 → purge 没生效（bug）= FAIL**。差值写入报告，等同面板侧 PIT 篡改测试，作"purge 确实在生效"的正式反证。**保留为主拦截。**
- [ ] **H-A — 根除 OOS 方向泄漏**：`auto_flip=False`；任何符号翻转**只能用训练/验证段**决定，**绝不碰 OOS**。断言：把 OOS 标签整体取反重跑，**所选方向不变**。
- [ ] **H-B — 滑点流动性分层（必做，非延后）**：成本含 amihud 分位三档滑点；断言中小盘滑点 > 大盘；关掉分层退回平 10bp 时净超额应**上升**（证明分层在压成本、防假阳性）。
- [ ] **H-D — 胜率按"每换仓期对基准"重建**：**不用** `metrics.py` 逐日正收益口径；断言=每 12 日换仓期 组合收益 vs 同期基准收益，手算抽查 2 期一致。
- [ ] **H-E — 涨跌停板块×时变正确**：抽查科创/创业板某日（创业板须跨 2020-08-24）阈值=20%、主板=10%；错阈值会误判成交→NAV 失真。
- [ ] **L3 — 标签 ≠ NAV 来源**：NAV 收益由 `BacktestEngine` 实际成交价链算；断言 NAV 路径**不引用** `forward_return_12d` 列。

**A. 切分正确性**
- [ ] 训练/测试 as_of **零交集**；`min(test_idx) − max(train_idx) ≥ H + E`。
- [ ] 隔离带 `(C−H, C+E]` 内 as_of **既不在 train 也不在 test**。
- [ ] OOS 测试 as_of 全部 ∈ `[2024-07-01+embargo, 2025-12-31]`；无一来自训练池或 2026 留存区。

**B. PIT 特征**
- [ ] 测试截面 `t` 的特征仅来自面板 `as_of=t` 行（面板已保证 PIT，此处复核索引未串期）。
- [ ] **未来价篡改测试**：把任一 OOS as_of **之后**的价格改成极端值，重算该截面**预测分不变**（特征不含未来）；NAV 仅通过合法的未来成交价合理变化。

**C. 成本/NAV**
- [ ] 关掉全部成本 → 净超额回升到毛超额（成本单调：净 < 毛恒成立）。
- [ ] 印花税在减半日前后费率正确切换（跨样本抽查 fill）。
- [ ] T+1：入场在 `as_of+1`，当日买入当日不可卖（既有 `can_sell` 行为复核）。

**D. 基准**
- [ ] 沪深300 死扛基准由 `data/lake/index_daily.parquet` 000300.SH **重算**，日期与 OOS 对齐；**无硬编码常数**。

**E. 指标口径**
- [ ] 对基准胜率 = **每换仓期** 组合 vs 基准（非逐日正收益）；手算抽查 2 期一致。
- [ ] IC = 每截面 `Spearman(pred, realized_12d)`；抽 1 截面手算一致。

**F. 健全性（sanity）**
- [ ] **标签置换测试**：训练集内打乱标签 → OOS IC 塌到 ~0（证明无索引层泄漏）。
- [ ] 幸存者上界标注存在于报告与 gate meta；退市回补登记为后续项。

**G. 回归**
- [ ] `pytest` 既有用例全绿；新增 `tests/test_wf_split.py`、`tests/test_wf_gate.py`、`tests/test_wf_costs.py` 通过。
- [ ] `git status`：仅新增 `quantmind/backtest/wf_{split,costs,gate,metrics}.py`、`scripts/verify_walkforward.py`、`tests/test_wf_*.py`、`docs/`；**无既有业务/回测/模型文件被修改**（含既有 `backtest/walk_forward.py`、`models/lgbm_ranker.py` 两套 walk-forward）。

---

## 13. 决策记录（全部已拍板 · 事前定线）

**架构级（本计划冻结）**

| # | 决策 | 结论 |
|---|------|------|
| 1 | 判定标签 | `forward_return_12d` |
| 2 | 训练窗口 | 扩展（主）+ 滚动 36mo（鲁棒） |
| 3 | purge/embargo | `H=12`，`E=12`（备选 15 无害）；**真正关键 = purge 确实剔除"标签窗口跨训练截止线"的样本**（§2、§12 H-C） |
| 4 | 通过线 | §6（基础 + 强信号），事后不挪门 |
| 5 | 成本 | 复用 `ExecutionSimulator` + **三处扩展**：流动性分层滑点【硬】+ 时变印花税 + 板块/时变涨跌停 |
| 6 | 入场/出场 | 信号 as_of 收盘；T+1 次日开盘入场；持有 12 日；`as_of+13` 开盘出场 |
| 7 | 模型 | **池化 LGBM**（早读用，最快）+ 每板块 IC 分解；**分板块各训=生产细化，本版不做**；FactorCNN 留后 |
| 8 | 方向 | `auto_flip=False`，**符号翻转只用训练/验证段，绝不碰 OOS**（硬条件 H-A） |
| 9 | 幸存者 | 结论=乐观上界，非对称解读入报告 |

**细节级（拍板 A–G，已落地到正文）**

| # | 决策 | 结论 |
|---|------|------|
| A | embargo 长度 | **`E=12`**（= 标签 H 下限；15 留 buffer 亦可，关键是 purge 生效） |
| B | 组合口径 | **五分位为主**（稳、抗少数幸运股）+ **额外报 Top-20/30 等权**（产品真实口径）；**IC/单调性用全周频截面（统计功效最大），PnL 用非重叠 12 日换仓**（§1.3 度量分两用） |
| C | 模型粒度 | **池化**（数据多、单信号、最快）；分板块之后再说 |
| D | 科创板 688* | **纳入池化**；涨跌停按板块×时变正确处理（成本 S0/§3） |
| E | refit 步进 | **季度** + 静态单切基线对照；月度留作之后细化 |
| F | 换仓对齐 | **非重叠 12 日**（干净、独立 PnL、对齐标签）；重叠/交错=P2 平滑曲线的事 |
| G | 新模块路径 | **`quantmind/backtest/`**（复用 `ExecutionSimulator`/`PerformanceMetrics` 最自然），`wf_*` 前缀，**不覆盖既有两套 walk-forward** |

**进实现前的 5 条硬条件（全防假 alpha，验收见 §12 顶部）**

| 代号 | 硬条件 | 假 alpha 方向 |
|------|--------|--------------|
| **H-C** | purge 消融反证 = 核心闸门：关掉 purge 必见 IC 虚高；相等=未生效=FAIL | 标签跨界泄漏 → IC 虚高 |
| **H-A** | 根除 OOS 方向泄漏：翻转只用训练/验证段 | 用 OOS 定方向 → 数据窥探 |
| **H-B** | 滑点流动性分层（本版必做，非延后） | 中小盘平 10bp 低估成本 → 高估 alpha |
| **H-E** | 涨跌停板块×时变正确（688/30 20%、创业板 2020-08-24 起、主板 10%） | 错阈值误判可成交 → NAV 失真高估 |
| **H-D** | 胜率按"每换仓期对基准"重建 | 既有逐日正收益口径不对 |

> 注：时变印花税（平 0.1%）方向上是**高估**成本的保守做法，不造假，故非"硬条件"但仍做准（S0 核 2023-08-28 减半、OOS 全程 0.05%）。

---

> **本文件为纯设计。实现需等：① 周频面板 v5 verify 通过；② 本计划 code-review 通过。** 实现时遵守：不改任何既有业务/回测/模型代码（含既有两套 walk-forward），新增集中于 `quantmind/backtest/wf_*` + 脚本 + 测试；输出 P1 闸门报告并强制标注幸存者乐观上界。
