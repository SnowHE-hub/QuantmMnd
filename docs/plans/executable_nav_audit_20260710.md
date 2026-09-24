# Executable NAV 结果审计与 Formal Gate 定稿（阶段 1.5，2026-07-10）

> 目标：验证 Executable NAV 结果正确性、解释研究 proxy → 可执行净收益的损失来源、明确 formal gate 指标语义。
> **不是模型/组合优化阶段。** 未改 Ridge/LGBM/daily_update；未改 registry（sha256 `50e5cdaf…` 全程不变）；
> 未处理 stash@{0}；未为通过而调阈值；未搜索 Top-N/持有期参数。
> 审计构件（本地，gitignored）：`reports/executable_nav/audit/{reconciliation.json, return_waterfall.json/.csv, rebalance_schedule.csv, cash_insufficient_by_rebalance.csv}`。

---

## 0. 一句话结论

NAV 引擎**数值正确、无影响 NAV 的实现 bug**；发现一个**报告口径缺陷**（拒单事件被每日重试重复计数）——已用**纯增量、不改 NAV/trades/gate** 的方式修复并保留 before/after。
63d 的「7 次再平衡」是**上游 fold 结构 + `as_of[::step]` 位置步进**共同导致的**实际持有期 ≈2× 设计意图**；两条种子的失败**主因不同**（12d 死于成本，63d 死于样本不足 + 度量基准/危机执行）。

---

## 1. NAV 状态机独立核对（逐字段 reconciliation）

对 12d（2022-02-09 / 2024-01-31 / 2026-03-06）与 63d（2022-08-04 / 2024-03-14 / 2025-04-03）各 3 个调仓日独立重建，并逐笔抽查成交。

| 核对项 | 方法 | 结果 |
|---|---|---|
| PIT universe（adv20 top-1500） | 独立用 v6 价格重算 rolling(20) 成交额降序 | 6/6 日 `n_universe=1500` 完全一致 |
| Top-N 目标 | 独立在池内按 score 降序取 300 | 6/6 日与引擎 `overlap=300/300` |
| 目标权重 | `1/n_target` | 等权 1/300=0.00333，一致 |
| T+1 成交价 | 独立 `open×adj_factor` | 逐笔一致（唯一 1 笔差 1e-4，为 float32 存储精度，非逻辑错误） |
| 交易费率 | `(slip+3.0+0.2)/1e4 [+印花(卖)]` | 逐笔 `rate_ok=True`、`cost_ok=True` |
| 停牌/涨跌停判断 | 板块×时变 limit + 一字板兜底 | 与 reason code 分布一致（见 §3） |
| 现金/持仓/NAV 恒等 | `nav_net − (cash+Σ持仓市值)` | max\|err\| = **1.4e-15**（12d）/ **1.8e-15**（63d） |
| gross 恒等 | `nav_gross − (nav_net+cum_cost)` | max\|err\| = **0** |
| benchmark NAV | 同池等权日频 | 见 §5，口径与研究层一致 |

**结论：引擎状态机正确，无不一致。** 唯一技术注记：价格宽表以 `float32` 存储，引入 ~1e-4 相对误差（第 7 位有效数字），对 NAV 可忽略；如需可升 `float64`（属精度增强，非 bug）。

---

## 2. 63d 仅 7 次再平衡的日期级原因

**证据（`rebalance_schedule.csv` + 网格间距测量）：**

| | 12d | 63d |
|---|---|---|
| preds 原始 as_of 数 | 142 | 91 |
| as_of 网格间距分布（交易日） | 5td×126, 25td×15 | 5td×83, 65td×4, 70td×3 |
| `rebalance_step` | 3 | 13 |
| 实际再平衡次数 | 48 | **7** |
| 实际再平衡间距（中位/值） | 15td（15/35 交替） | **125–130td** |
| 设计意图持有期 | 12td | 63td |
| 偏差 | ~1.25×（15 vs 12，可接受） | **~2×（127 vs 63）** |

63d 的 91 个 as_of 由 **7 个 walk-forward OOS fold** 构成，每 fold ~13 个周频点，fold 之间有 **65–70td 大缺口**。设计 §1/§2.5 规定「再平衡日 = OOS `as_of[::step]`」，`step=ceil(63/5)=13` **假设网格是连续 5td 周频**。但按**位置**每 13 个点取一次，恰好落在每个 fold 附近一次 → 7 次，实际持有期翻倍到 ~127td。

**归因：E（多因）——主要是 B + C/D：**
- **B（上游样本结构）**：63d preds 只有 7 个 OOS fold、fold 间大缺口 → 天然稀疏。
- **C/D（实现与设计假设不符）**：`as_of[::step]` 位置步进在**非均匀网格**上产出的持有期是**时间意义上的 2×**；代码严格照设计字面执行，但设计对「均匀周频」的假设被数据违背。**非 NAV bug，属参数/设计假设错配。**

后果：63d 独立再平衡样本仅 7（且实际是半频），**统计证据严重不足**（设计 §7 已预警 ~6 次样本 CI 宽）。

---

## 3. 现金不足拒单审计（63d 3,386 次）

**证据（`cash_insufficient_by_rebalance.csv` + 2025-04-03 重建）：**

| 指标 | 63d | 12d |
|---|---|---|
| insufficient_cash **事件数** | 3,386 | 758 |
| **去重底层订单数**（side,ticker,decision_asof） | **124** | **71** |
| 单订单重试次数 min/中位/max | 1 / 1 / **129** | 1 / 10 / 34 |
| 集中度 | **121/124 来自单次再平衡 2025-04-03**（3,196 事件） | 分散，最大 2024-10-08（28 单/250 事件） |

**2025-04-03 再平衡逐日重建：**
- 换手极高：300 目标中仅 **35 名延续、265 名换手**。
- 成交日 = **2025-04-07**（关税崩盘首日）：当日 NAV 从 0.979 跳空跌到 0.843（−14%）。
- 卖单大面积 **limit_down 锁死**（63d 全期 limit_down 去重 106 单，多数在此窗口），现金无法释放。
- 现金全程 `cash/nav≈0`（满仓），买单排队；随跌停逐日解锁，卖单成交、买单跟进（04-07 买 161/卖 175，04-08 买 81/卖 94…）。
- 121 个 insufficient 买单中 **97 个后续成交、24 个到期未成交**；组合最终稳定在 **279 名**（非 300）。

**关键机制：**
1. **重试统计膨胀（B，主导）**：一个挂单在成交前**每个交易日**都写一条 rejection → 124 单产生 3,386 事件（最长 129 天）。`n_rejection_events` 严重高估真实拒单面。
2. **真实执行约束（A）**：崩盘日卖单跌停锁死、现金无法及时释放，是**合理**的执行约束。
3. **贪婪未按可用现金缩放（C，策略层）**：买单按 rank 顺序各取**全额** `target_value`，现金耗尽后尾部买单归零而非全体按比例缩小 → 组合停在 279 名、集中度上升。**这是组合构建策略选择，不是执行器 bug。**
4. 「买单是否按实际可用现金重新缩放」→ **否**（确认为 C）；「同一订单是否多日重复计入」→ **是**（确认为 B）。

**归因：E（多因）= B（口径膨胀，主导）+ A（真实崩盘约束）+ C（贪婪不缩放，策略层）。**

**修复（仅 B，确认为报告实现缺陷）：** 在 `nav_engine._finalize` **增量新增** `n_distinct_rejected_orders` 与 `rejection_distinct_order_counts`（按 side+ticker+decision_asof+reason 去重）。
- **before/after 已验证：** `nav_daily`、`trades` **逐字节一致**；`nav` 汇总块一致；gate 仍 False；registry sha256 不变。仅执行统计新增两个字段。
- 63d：3,620 事件 → **286 去重单**（insufficient 3,386→124）；12d：1,635 → **424 去重单**（insufficient 758→71）。
- A、C **未修改**（分别属真实约束与策略选择，留待受控实验）。

---

## 4. 收益损失瀑布（`return_waterfall.json/.csv`）

年化**净超额**口径。研究 proxy 为 rebalance-point label top-quintile 净超额（`p4/p63 verdict`）。

| 台阶 | 12d 年化超额 | 边际 | 63d 年化超额 | 边际 |
|---|---|---|---|---|
| 1. 研究 proxy（label，rebalance-point） | **+2.75%** | — | **+5.33%** | — |
| 2–6. 执行约束 + T+1 时点 + 日频 NAV 度量基准（合并） | +0.97% | **−1.78%** | +2.73% | **−2.60%** |
| 7. − 滑点（amihud 分层） | −1.35% | −2.32% | +2.38% | −0.35% |
| 8. − 佣金+过户+印花 | **−2.23%** | −0.89% | **+2.26%** | −0.12% |
| 10. 可执行 net 超额（最终） | **−2.23%** | | **+2.26%** | |

**说明（诚实标注）：** 台阶 2–6 因引擎无「逐约束开关」而**合并**为一格（细分需给引擎加 toggle 或另建平行执行器——后者违反设计「不重复造第二套执行系统」，故本轮不做，列为受控实验建议）。该格同时含 **T+1 时点、可交易性过滤、涨跌停、持仓延续/拒单、现金与整数股，以及从「rebalance-point label」到「逐日 NAV」的度量基准变化**（设计 §7 本身把后者列为升级动机）。成本台阶 7/8 由逐笔 `cost_rate×gross_value` 按构成精确分摊。

**归因结论：**
- **12d 死于成本**：换手年化单边 8.95×，滑点占成本 **72%**，成本拖累 **−3.20%/年**，把 proxy 的 +2.75% 直接打成负。
- **63d 死于「样本/基准/危机执行」**：成本极低（−0.47%，低换手），主要损失在合并格 **−2.60%**（度量基准 + 2025-04 崩盘执行损耗 + 半频再平衡），最终 +2.26% < 5% gate 线。

---

## 5. 基准与回撤定义核对

**当前代码实际口径（`nav_engine._bench_step` / `_finalize`）：**

| 项 | 实现 |
|---|---|
| benchmark universe | PIT adv20 top-1500（每再平衡日更新成员） |
| benchmark weighting | 等权（每日成员日收益**均值**） |
| benchmark rebalance | **日频**（每日按当前成员等权，收益 = 成员 pct_change 均值） |
| benchmark return timing | 收盘到收盘（`adj_close.pct_change`） |
| absolute NAV MaxDD | `min((nav_net − cummax)/cummax)` |
| **gate 采用** | **absolute net MaxDD** |

**三种回撤实测：**

| | absolute net | benchmark | excess（net/bench 比值） |
|---|---|---|---|
| 12d | **−48.88%** | −47.15% | **−15.93%** |
| 63d | **−35.43%** | −38.32% | **−14.44%** |

**15%/12% 阈值语义解读：**
- **若为绝对组合回撤**：代表「多头绝对净值最大亏损 ≤15%」的**产品/客户体验目标**。对满仓多头，2022–2024 熊市中基准自身回撤 47%/38%，该线**结构性不可达**，等于要求择时/对冲/降仓——超出纯 alpha 种子范畴。
- **若为超额/主动回撤**：代表「相对基准的 alpha 曲线最大回撤 ≤12–15%」的**模型目标**，剔除市场 β，衡量信号本身稳定性。
- **当前实现采用绝对口径**；而设计 §5 把 15%/12% 与「年化净超额 ≥5%」「IR≥1」并列为 **alpha 判定**——**文本意图更像超额/主动口径，代码却用绝对口径**。→ **Gate v1 文本与代码语义不一致**（核心问题，见 §7）。按超额口径，12d −15.9%（略破 15%）、63d −14.4%（破 12%、守 15%），与绝对口径的「灾难性」结论截然不同。

（本轮不改阈值，仅指出语义分歧。）

---

## 6. Gate v2 草案（先语义、后阈值；仅提案，不改 registry/gate 代码）

将单层 gate 拆为三层，**指标语义**优先，阈值待评审：

### A. Data Sufficiency Gate（数据充分性——先决，hard）
| 指标 | 语义 | 性质 |
|---|---|---|
| OOS 时长 | 覆盖年数 | hard |
| 独立再平衡次数 | 非重叠持有期数（真实频率，非位置步进） | **hard** |
| 有效交易样本 | 实际成交笔数/覆盖股票 | warning |
| 市场环境覆盖 | bull/neutral/bear 各含足量再平衡 | hard |
> 未过 A → 直接判 `insufficient_oos_evidence`，**不评 B/C**。63d 当前**卡在此层**（独立再平衡=7 且实为半频）。

### B. Model Alpha Gate（信号 alpha——核心，成本后）
| 指标 | 语义 | 性质 |
|---|---|---|
| 成本后年化超额 | 真实成交后净超额 | hard |
| IR（超额） | 超额收益/超额波动 | hard |
| **active/excess drawdown** | **净值/基准 比值的回撤（剔除 β）** | **hard（替代 v1 的绝对回撤误用）** |
| 稳定性 | 分年/分 regime 净超额>0 或可解释 | warning→hard |
| 换手与成本侵蚀 | 成本/毛超额 占比 | warning |

### C. Portfolio Deployment Gate（可部署性——上线前，产品层）
| 指标 | 语义 | 性质 |
|---|---|---|
| absolute MaxDD | 绝对净值回撤（产品体验） | warning（除非明确绝对收益产品→hard） |
| 波动率 / Beta | 市场暴露 | warning |
| 流动性 / 容量 | adv 占用、规模上限 | hard（上规模时） |
| 执行拒单率 | **去重口径**的真实拒单占比 | warning |

**Gate v1 → v2 映射：**
| v1（单层） | v2 归属 | 变化 |
|---|---|---|
| 年化净超额 ≥5% | B.成本后超额 | 不变（hard） |
| MaxDD ≤15%/12% | **拆分**：B.active-DD（hard）+ C.absolute-DD（warning） | **修正语义误用** |
| IR ≥1.0 | B.IR | 不变 |
| 换手阈（未定） | B.成本侵蚀 + C.容量 | 明确为 warning |
| 分年>0 | B.稳定性 | 不变 |
| （无） | **A.数据充分性** | **新增 hard 前置层** |

原则：**不因本轮 63d 结果反向设计阈值**；阈值数值留评审，本提案只固定**层级、语义与 hard/warning 归属**。

---

## 7. 两条种子建议分类（仅报告，不写 registry）

| 种子 | 建议分类 | 依据 |
|---|---|---|
| `ridge_full_12d_v6_seed` | **confirmed_fail（当前形态）** | 成本后净超额 −2.23%，**毛超额也仅 +0.97%**；换手 8.95× 使滑点成本 −3.20%/年结构性吞噬 alpha。数据充分（48 再平衡）→ 结论可靠。**不建议在当前 12d 高频形态继续投入。** |
| `ridge_full_fnd_63d_v6_seed` | **insufficient_oos_evidence（+ implementation_issue_requires_rerun）** | 净超额 +2.26%（<5%，未过），但：①独立再平衡仅 7 且**实际持有期是设计的 2×**（§2 参数/假设错配）；②−2.60% 损失含度量基准变化与单次崩盘执行；③per-fold CI 宽（proxy 7 fold min −7.8%/max +11.9%）。**属样本不足 + 需修正再平衡频率后重跑，不是干净的正式失败。** |

**必须先修复的执行/口径问题（不改策略）：**
- 已修：拒单**去重计数**（报告缺陷）。
- 待评审后受控重跑：63d **再平衡频率**应按**时间（~63td）**而非 `as_of[::13]` 位置步进定义，使实际持有期对齐设计意图并增加独立样本。这是**执行/参数修正**，需与用户确认是否属「可做的执行器修正」还是「策略变化」——因它改变再平衡日集合，判定为**边界项**，本轮**不擅自改**。

---

## 8. 测试与确定性

| 项 | 结果 |
|---|---|
| 新增审计测试 `tests/test_nav_audit.py` | 4 passed（位置步进语义 / NAV 恒等 / 拒单双口径 / 绝对vs超额回撤） |
| Executable NAV 测试 `tests/test_executable_nav.py` | 13 passed |
| 全量默认单元套件 | **1096 passed, 110 deselected**（deselected 含环境 flaky 的 Ollama integration 测试） |
| 跨进程确定性 ×2 | 两次独立进程：7 个数据帧 + summary **完全一致** |
| registry sha256 | `50e5cdaf…` **全程未变** |

> 注：`test_ollama_live`（`@integration @slow`）会实连本机 Ollama 并可能挂起，默认 addopts 的 marker 过滤未含 `integration/slow`；本轮以扩展过滤跳过（与本次改动无关的既有环境问题，建议后续把 `integration/slow` 加入默认排除）。

---

## 9. 代码修复与提交建议（不自动 commit）

**唯一代码改动**：`quantmind/execution/nav_engine.py` 增量新增去重拒单统计（+ `tests/test_nav_audit.py` 新增）。属**报告口径修复**，NAV/trades/gate/registry 不受影响（before/after 已验证）。

建议独立 commit（人工确认后）：
```
fix(nav): report distinct-order rejection counts (dedupe daily retries)

n_rejection_events counted the same queued order once per retry day,
inflating cash-insufficient from 124 distinct orders to 3386 events (63d).
Add n_distinct_rejected_orders + rejection_distinct_order_counts.
NAV / trades / gate / registry byte-identical (verified before/after).
```
建议纳入该 commit 的文件：`quantmind/execution/nav_engine.py`、`tests/test_nav_audit.py`、
`docs/plans/executable_nav_audit_20260710.md`、以及刷新后的 `docs/plans/executable_nav_run_20260710/*`（summary/manifest 含新字段）。
审计 CSV/JSON 在 `reports/`（gitignored），通过 manifest sha256 复现。

---

## 10. 最终结论（对应汇报十问）

1. **NAV 引擎是否正确**：正确。universe/topN/成交价/费用逐笔吻合，NAV 恒等式误差 1e-15，确定性通过。
2. **是否发现实现 bug**：无影响 NAV 的 bug；1 个**报告口径缺陷**（拒单重试重复计数）已增量修复。另有 float32 价格精度（~1e-4，可忽略）。
3. **63d 只有 7 次调仓的准确原因**：E=B+C/D——7 个 OOS fold + `as_of[::13]` 位置步进在非均匀网格上产出 ~127td 实际持有期（设计意图 63td 的 2×）。
4. **现金不足拒单的准确原因**：E=B（每日重试重复计数，124 单→3386 事件，主导）+A（2025-04 崩盘卖单跌停锁死）+C（买单贪婪不按可用现金缩放，策略层）。
5. **12d 收益损失归因**：成本主导——高换手 8.95× × 滑点，成本 −3.20%/年，proxy +2.75% → net −2.23%。
6. **63d 收益损失归因**：非成本（−0.47%）；主要为执行+时点+日频度量基准合并 −2.60% + 半频再平衡，proxy +5.33% → net +2.26%。
7. **Gate v1 语义问题**：15%/12% 回撤线文本意图似为超额/主动口径，代码却用**绝对**口径；对满仓多头绝对口径结构性不可达 → 误判。
8. **Gate v2 提案**：三层（A 数据充分性 hard 前置 / B 成本后 alpha，含 active-DD 替代绝对-DD / C 可部署性），先语义后阈值，不反向设计。
9. **建议研究状态**：12d = confirmed_fail（当前高频形态）；63d = insufficient_oos_evidence（+需按时间重定再平衡频率后重跑）。
10. **下一步**：**先做「执行器/参数修正 + 受控实验」**——(a) 已修拒单口径；(b) 评审确认后，把 63d 再平衡改为**时间对齐 63td** 重跑以补足独立样本；(c) 采用 Gate v2 的 active-DD 口径复核。**暂不做模型/组合参数优化，不接生产。**

**停在此处，等待确认。**
