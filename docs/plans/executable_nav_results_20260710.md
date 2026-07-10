# Executable NAV 首轮结果（2026-07-10）

> 阶段 1「研究 → 可执行 NAV 闭环」的实现记录与首轮 gate 判定。
> 设计依据：`docs/plans/executable_nav_design.md`；两条种子均**未通过 formal gate**，
> registry 保持 `research_candidate_pending_nav` **一字未改**（sha256 前后一致）。
>
> 机器可读摘要与复现 manifest：`docs/plans/executable_nav_run_20260710/`
> （summary / gate_report / `manifest.json` 含本地大文件 sha256）。
> 逐日 NAV / holdings / trades parquet 保留在本地 `reports/executable_nav/`（gitignored），不入库。

---

## 1. 实现架构

```
preds (as_of, ticker, score)          v6 全市场价格 (alpha_prices_panel_v6)
        │                                      │
        ▼                                      ▼
ExecutableNavEngine (quantmind/execution/nav_engine.py)
  ├─ PriceStore：宽表价格仓库 + PIT adv20 top-1500 选股池 + 涨跌停/停牌可成交集
  ├─ 决策日（as_of 收盘）：Top-N 目标 → 与真实持仓 diff → 买卖单（等权调平）
  ├─ 成交日（T+1 次日开盘）：先卖后买；停牌/涨跌停/现金不足 → 记 reason code 重试
  ├─ 逐笔成本：amihud 分位滑点档(5/15/30bp, wf_costs.SlippageTiers)
  │            + 佣金万3 + 过户万0.2 + 时变印花(卖, wf_costs.stamp_duty_rate)
  ├─ 退市处理：>20 交易日无 bar 且无未来数据 → 按末价强制清仓(delisted_writeoff)
  └─ 逐日 mark-to-market → net/gross/bench/excess NAV
        │
        ▼
evaluate_gate (quantmind/execution/nav_gate.py, 设计 §5 五项判定)
        │ 仅 pass 时
        ▼
apply_gate_to_registry（fail → registry 零改动；pass → 只写 metrics，不自动升 production）
```

复用（设计 §4）：`wf_costs` 的板块×时变涨跌停 / 分层滑点 / 时变印花、T+1 next-open 语义、
p4d 的 PIT adv20 top-1500 口径。E3 replay_engine 是单笔止损/止盈回放器，组合编排层按设计"新写"。

## 2. 新增 / 修改文件

| 文件 | 说明 |
|---|---|
| `quantmind/execution/nav_engine.py` | 新增：组合状态机 + 执行约束 + NAV 推进 |
| `quantmind/execution/nav_gate.py` | 新增：gate 判定 + registry 条件更新 + 报告落盘 |
| `scripts/run_executable_nav.py` | 新增：CLI 入口 |
| `tests/test_executable_nav.py` | 新增：13 个测试（七类要求全覆盖） |

## 3. 执行命令

```bash
python scripts/run_executable_nav.py --model 12d           # 只跑 Ridge 12d
python scripts/run_executable_nav.py --model 63d           # 只跑 Ridge 63d
python scripts/run_executable_nav.py --model all           # 两者全跑
python scripts/run_executable_nav.py --model all --dry-run        # 校验输入，不写任何文件
python scripts/run_executable_nav.py --model 12d --report-only    # 重生成报告，不动 registry
python scripts/run_executable_nav.py --model all --no-registry    # 跑全量但绝不更新 registry
```

输出目录：`reports/executable_nav/<model_id>/`（独立目录，不触碰 nav_v4 / sim30d）。
每个模型产出：nav_daily / holdings_daily / trades / rejected_trades / targets / filtered /
turnover（parquet）+ nav_daily.csv + summary.json + gate_report.{json,md}。

## 4. 首轮 gate 结果（2026-07-10，全部**已验证事实**）

| | Ridge 12d (`ridge_full_12d_v6_seed`) | Ridge 63d (`ridge_full_fnd_63d_v6_seed`) |
|---|---|---|
| 区间 | 2022-02-09 → 2026-05-11（1018td, 48 次再平衡） | 2022-08-04 → 2026-05-11（898td, 7 次再平衡） |
| 净年化 / 基准年化 | −0.90% / +1.33% | +12.36% / +10.10% |
| **年化净超额** | **−2.23%**（研究层 proxy +2.75%） | **+2.26%**（研究层 proxy +5.33%） |
| IR（超额） | −0.27 | 0.33 |
| 净 MaxDD | −48.9%（基准 −47.2%，同期市场 β） | −35.4%（基准 −38.3%） |
| 年化单边换手（真实） | 8.95（label proxy 15.9） | 1.50（label proxy 3.4） |
| 累计成本（NAV 单位） | 0.1322（成本拖累 ≈3.2%/年） | 0.0229（≈0.47%/年） |
| 拒单事件 | 1635（cash 758 / 停牌 469 / 涨停 220 / 一字 97 / 跌停 48 / 过期 43） | 3620（cash 3386 / 跌停 121 / 涨停 55 / 停牌 24 / 过期 24 / 一字 10） |
| 退市 write-off | 8 笔 | 3 笔 |
| **gate_pass** | **False**（净超额/回撤/IR/分年 4 项全不过） | **False**（净超额/回撤/IR/分年 4 项不过） |
| registry | 未改动 | 未改动 |

关键读数：
- **研究层 → 可执行层的降幅**：12d +2.75% → −2.23%（成本拖累 3.2%/年 + 不可成交约束）；
  63d +5.33% → +2.26%。方向与 `ic_vs_net_excess_divergence.md` 的预期一致——
  label-proxy 净超额高估真实可实现超额。
- **MaxDD 主因是 β**（满仓多头无对冲）：策略与基准回撤同量级（12d：−48.9% vs −47.2%）。
  设计 §5 的 ≤15%/≤12% 回撤线对"满仓多头绝对 NAV"事实上不可达，更像对冲后/超额口径的线
  ——阈值语义需评审澄清（见 §6）。
- 63d 分年净超额：2022 +4.6% / 2023 −2.6% / 2024 +3.6% / 2025 −2.0% / 2026YTD +4.0%，
  不满足"每年>0"。

## 5. 保守实现决定（设计缺失处，全部可配置）

1. Top-N 未定 → 默认 300 = PIT top-1500 的 top-quintile（与研究层 proxy 同口径）；`--top-n` 可调。
2. 基准未定（"等权可投票池 or 指数"）→ 取等权 PIT top-1500（与研究层 bench 同口径）。
3. 换手 gate 阈值未定 → 仅报告不阻断（gate 报告中注明）。
4. ST 无 PIT 标记 → 板块阈值 + 一字板（high==low）兜底检测。
5. 买单到期 = 下一再平衡日仍未成交 → 撤单（expired_unfilled）；卖单永不过期（停牌强制持有）。
6. gross NAV = net NAV + 累计成本（NAV 单位，近似），成本不改变可成交性判定。
7. gate 通过也不自动升 `production`（人工签收）；本轮未通过，未触及。

## 6. 未解决问题（需人工评审）

- **回撤阈值语义**：绝对 NAV 口径下 ≤15%/≤12% 对 2022-2024 熊市的满仓多头不可达。
  需评审确认是绝对口径（则当前产品形态不成立）还是超额/对冲口径（则需重定基准）。
- **12d 种子在真实执行下净超额为负**：高换手（年化单边 8.95）成本拖累 3.2%/年吞掉全部
  alpha。若无组合层改造（降频/降换手/集中度），12d 不具备上线基础。
- 63d 净超额 +2.26% < 5% gate 线，且仅 7 次再平衡、CI 宽（设计 §7 已预警 6 次样本的
  +5.33% 点估计不可靠——本轮真实 NAV 证实其显著缩水）。

## 7. 建议

**不进入 RecommendationContract / UI 接入阶段。** 两条种子均未通过 formal gate，
按设计 §5 维持 `research_candidate_pending_nav`。后续方向（另立任务）：
63d 组合层调优（集中度/入场分散）、回撤阈值口径评审、或等待更多 OOS 样本。
