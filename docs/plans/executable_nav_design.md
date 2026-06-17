# Executable NAV 回测设计（12d / 63d 产品上线硬前置）

> 状态：**设计文档（不实现）**。B(63d full+fnd) 与 12d 种子均 `research_candidate_pending_nav`；
> 本文定义"研究层净超额 → 可上线 NAV"的升级路径与判定阈值。写完即停，不写代码。
> 关联：`survivorship_p4_verdict.md`(12d) / `survivorship_p63_3_verdict.md`(63d) /
> `ic_vs_net_excess_divergence.md`(读数原则)。

---

## 1. 现状解剖：研究层 net excess 算法（`evaluate_bakeoff` + `wf_metrics`）

当前 leaderboard 的"含成本净超额"是 **label top-quintile proxy**，逐行拆解：

```
holding   = H（12 或 63 交易日）
step      = ceil(H / 5)              # 周频 as_of=5td/格；12d→3, 63d→13
rebal     = all_as_of[::step]        # 非重叠持有期再平衡点
每个 rebal a:
  g        = 截面(score, realized=forward_return_H)          # realized=标签，非真实成交
  top      = g[score >= g.score.quantile(0.80)]              # top-quintile，等权
  port_gross = top.realized.mean()                            # 组合毛收益 = 标签均值
  bench[a]   = g.realized.mean()                              # 基准 = 全池等权
  # 单边换手（membership churn，label 排名集合变化）：
  oneway   = 0.5 · Σ_names |w_t − w_{t-1}|,  w = 1/n if in top else 0  （首期=1 建仓）
  slip     = amihud 分位 → SlippageTiers.bp_for_quantile 均值（分流动性桶 bps）
  cost_bp  = oneway · ( 2·(slip + commission3.0 + transfer0.2) + stamp_duty )  # 买卖两腿×oneway，印花仅卖
  port[a]  = port_gross − cost_bp/1e4
ppy        = 250 / H                  # 每年非重叠再平衡次数（12d≈20.8, 63d≈3.97）
net_excess = annualize(port) − annualize(bench)   # 每期几何累乘后年化（wf_metrics.net_excess_annualized）
```

**"年化换手 ~3.4" 公式**（供核对）：
```
年化单边换手 = avg_oneway_turnover × ppy = avg_oneway × (250/H)
  B(63d):  0.846 × (250/63=3.97) = 3.36 ≈ 3.4    # 单边；双边≈6.7
  A(12d):  0.763 × (250/12=20.8) = 15.9          # 单边
```
- 口径：**单边(one-way)** = `0.5·Σ|Δw|`；年化方式 = **每期单边 × 每年再平衡次数**（非复利，线性年化，因换手是流量）。
- 成本里 `2·(...)` = 买+卖两腿，各按 oneway 比例计；印花税仅卖出侧（`stamp_duty_rate(a)`，时变）。

**三个根本局限（→ 必须升级 executable NAV）**：
1. **port_gross = 标签均值，非真实成交**：用 `forward_return_H`（收盘到收盘、复权、假设全额成交），
   未扣涨跌停不可成交、停牌、T+1、冲击成本。
2. **换手 = label 排名集合变化，非真实 holdings 变化**：真实组合受不可成交约束，持仓≠排名。
3. **样本小**：B 净超额仅 ~6 次非重叠再平衡（81 as_of / step13），置信区间宽（见 §7）。

---

## 2. 升级到 Executable NAV 的要素清单（实现路径，不写代码）

> 原则：**真实 holdings 驱动**——每个再平衡日，基于"上一期真实持仓 + 当日可成交集 + Top-N 目标"
> 解出真实成交，NAV 由真实成交价与持仓推进，换手/成本由真实持仓变化计。

| # | 要素 | 实现路径 |
|---|---|---|
| 2.1 | **涨跌停不可成交剔除** | 每日每票算涨跌停价 = pre_close×(1±limit)。limit **板块时变**：主板 ±10%；科创板(688)/创业板(300) ±20%（创业板 **2020-08-24 起** 20%，之前 10%）；ST ±5%。当日 high==涨停价(封板买不进) → 不可买入；low==跌停价 → 不可卖出。用 lake daily(high/low/pre_close) + stock_basic(板块) + 日期判定 limit。 |
| 2.2 | **停牌 / 复牌处理** | 停牌日(daily 无 bar 或 suspend_d 标记) → 当日不可买卖；持仓票停牌则**强制持有**至复牌；复牌日按**当日开盘价**补建/补平（对齐 E3 的 next-open 语义）。需 suspend_d 或 daily 缺 bar 推断。 |
| 2.3 | **T+1 卖出约束** | 当日买入 T+0 不可卖；卖出在买入次日起可执行。组合状态机记每仓买入日，卖出时校验 ≥T+1。E3 `_exec_price_next_open` 已是 T+1 next-open 语义，直接复用。 |
| 2.4 | **滑点逐笔计（分流动性桶）** | 每笔成交按 `wf_costs.SlippageTiers` + `amihud_to_quantile`(该票该日 amihud 分位) → bp，叠加 commission/transfer/印花(卖)。**逐笔**而非组合级平均，反映大单冲击。 |
| 2.5 | **Top-N 入场/出场对齐 horizon 再平衡日历** | 12d 与 63d **各自再平衡日历**：再平衡日 = OOS as_of[::step]（12d step3 / 63d step13）。每再平衡日：目标 = 当日 score Top-N（N 由组合规模定，如 top-quintile 或固定 50/100）；与上期真实持仓 diff → 买卖单 → 过 2.1/2.2/2.3 过滤 → 真实成交。 |
| 2.6 | **真实换手** | = Σ\|真实持仓权重变化\| / 2（单边），**基于真实 holdings 期间变化**，非 label 排名。不可成交的目标调整会**降低**真实换手（封板买不进则维持原仓）。 |
| 2.7 | **NAV 推进** | 组合 NAV(t) = Σ 持仓市值(真实成交价建仓 × 当日复权价) + 现金；逐日 mark-to-market。超额 = 组合 NAV 曲线 − 基准(等权可投票池 or 指数) NAV 曲线。 |

---

## 3. 12d / 63d 共用同一 NAV 框架
- **同一引擎**，只两处不同：`horizon`(H=12/63 → 再平衡 step=3/13、ppy)、`cost_model`(同 wf_costs，
  但 63d 单边换手高/次数少、12d 反之)。
- 同一份执行约束(2.1-2.4)、同一份真实 holdings 状态机(2.5-2.7)。
- 输出两套产品规格(§6)。

## 4. 与 E3 replay 引擎的复用关系
E3(`quantmind/execution/replay_engine.py`)已是 **parquet 真源 + T+1 next-open + 滑点**的单笔执行底座：
| 组件 | 复用/新写 |
|---|---|
| `_exec_price_next_open`(T+1 次日开盘 + 滑点) | **直接复用**（2.3/2.4 底层） |
| `preload_price_history`(parquet 真源 daily_prices_panel) | **直接复用**（价格源；NAV 改读 v6 全市场 lake daily） |
| `replay_single_order` / `HistoricalReplayEngine` | **复用单笔语义**，但 NAV 需在其上**新写组合编排层** |
| 涨跌停/停牌可成交集过滤(2.1/2.2) | **新写**（E3 单笔回放未含板块时变 limit + 停牌集） |
| Top-N 组合状态机 + 真实换手(2.5/2.6) | **新写**（E3 是逐单回放，无组合层） |
| 分桶滑点逐笔(2.4) | wf_costs 已有 SlippageTiers，**接入即可** |
→ **E3 作执行底座(单笔 T+1+滑点+真源)，NAV = E3 之上的"可成交集过滤 + Top-N 组合编排 + 真实 NAV 推进"。**

## 5. NAV 后判定标准（gate_status → gate_pass=True）
当前 `research_candidate_pending_nav`。**executable NAV 跑出后**，升 `gate_pass=True` 需**同时**满足
（阈值待评审定稿，下为建议）：
| 指标 | 建议阈值 | 说明 |
|---|---|---|
| 年化净超额(真实成交后) | **≥ +5%** | 项目 formal gate line；研究层 proxy 不算 |
| 最大回撤 | ≤ 15%(12d) / ≤ 12%(63d) | 可控；超则降级 |
| 信息比率 / 夏普(超额) | **≥ 1.0** | 超额收益/超额波动 |
| 换手可承受 | 年化双边 ≤ 阈(成本已内含,另查容量) | 真实换手不致冲击侵蚀 |
| 分年/分 regime 不崩 | 每年净超额>0 或可解释 | 防单年顶起 |
- 任一不过 → 维持 `pending_nav` 或降 `retired`，不上线。

## 6. 产品差异化（NAV 给出可上线规格）
| | 12d 短线产品 | 63d 长线产品(基本面) |
|---|---|---|
| 再平衡 | 高频(~每周/双周, step3) | 低频(季度级, step13) |
| 年化单边换手 | ~15.9 | ~3.4（**低 4.5×**） |
| 成本拖累 | 高 | 低 → 净超额相对优势 |
| 信号源 | 量价(35+16+Alpha158) | 量价+**基本面 survivors**（价值+现金流质量主导） |
| 适配客户 | 高频容忍、追短期 alpha | 低换手、税敏感、长持有 |
| 研究层净超额 | +2.75% | +5.33% |
→ NAV 框架对两者分别产出"真实成交后年化净超额 / 回撤 / 容量"，决定各自能否上线及规模上限。

## 7. ⚠ 置信区间诚实标注（NAV 对比段必写）
- **A(12d) +2.75%**：OOS 16 fold / 142 as_of / **~47 次非重叠再平衡** → 相对窄 CI。
- **B(63d) +5.33%**：OOS 7 fold / 81 as_of / **~6 次非重叠再平衡** → **CI 宽**，+5.33% 点估计不确定性大。
- → **63d 的"略过 5%"不能当定论**；executable NAV 用真实逐日 NAV(而非 ~6 个再平衡点)给出更稳的超额
  与 t 统计量，这是升级 NAV 的另一动机（样本从 ~6 期 → 逐日数百点）。

---
**结论**：B/12d 种子的研究层信号成立，但研究层净超额是 **label proxy（无不可成交/真实换手/逐笔成本）**。
上线判定以本文 §5 阈值在 §2 的 executable NAV 上复核；E3 作执行底座（§4）。**实现属 P1，本文只定设计。**
