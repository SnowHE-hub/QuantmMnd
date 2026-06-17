# Task Plan — 幸存者偏差修复（Survivorship Repair）

## 上一阶段（已关闭）
Bake-off batch-A 定案：**Ridge(full) 赢**（neut IC 0.0340 / 净 +1.9% / 跨流动性均匀）；序列模型 raw IC 追平但优势全在 illiquid、含成本为负；LGBM 落后。详见 `docs/plans/model_bakeoff_plan.md §14`。
收尾修正已落：① 序列模型"全1374"是 3 桶拼接（†标，不与 Ridge 单模型混比）；② illiquid 0.05-0.075 是幸存者+成本双重角落，**最不可信、非 follow-up 线索**。

## 本阶段目标
把幸存者池 universe 修成真 PIT universe（每 as_of 含当时在市、含后来退市的票，按退市日剔除），新建 **v6**，不动 v5、评估侧零改动。修好后第一步只做一件：v6 上重跑 Ridge(full) 看 +0.034/+1.9% 是否还在。

## Phases
- [x] P0 调研：stock_basic 全 L/delist 全空（纯幸存者）；1388 是 SH+SZ 子样本（仅 1137 票 2019前）；backfill 基础设施可复用；Tushare 退市票历史可拉（findings.md）
- [x] P1 **计划** → `docs/plans/survivorship_repair_plan.md`（scope 决策 / 拉数 / PIT 重建 / v6 / 量化 / 验收 / 风险）
- [ ] ⏸ **评审门：等用户拍板 scope（A 全市场 / B 样本+退市；是否含 BSE）+ 数据量 + v6 命名**
- [ ] P2 拉数据（获批后）：全量 stock_basic L/D/P + 退市/缺失票在市期价量（复用 backfill，断点续传，token 防泄漏）
- [ ] P3 PIT universe 重建（list_date≤as_of<delist_date）
- [ ] P4 v6 面板（复用 weekly_panel builder，换 universe/prices，守卫不覆盖 v5）
- [ ] P5 量化影响（票数差/退市占比/加回行数/逐 as_of universe 曲线）
- [ ] P6 验收（PIT 正确性 / 退市票价真实 / v5 字节不变 / 评估零改动）

## Status: P1✅ P2✅ P3✅(独立验收 PASS)。**当前：P4 判定跑（active）**。

## Decisions（已落）
- scope=A（全市场 SH+SZ PIT，L+D+P），不纳 BSE。v6 命名 alpha_prices_panel_v6 / alpha_panel_weekly_v6。
- 退市标签规则A（算到末交易日真实亏损）；seasoning list+120td；ST 保留。
- P4 中性化 = UNKNOWN 行业桶（缺行业不剔除）。

---

# P4 判定跑（active）— v6 上 Ridge(full) 12d 复核

**性质**：单次判定（非优化）。完全复刻 batch-A，唯一变量 = universe/面板（v5→v6）。
出数即停、不调参、不换方案，带证据回评审。

## batch-A 配置（逐字复刻）
- RidgePredictor(alpha=10.0)，数值特征 z-score(train 的 mu/sd)，丢 exposure_industry/area。
- PurgedWalkForwardSplit(cal, horizon=12, embargo=20, mode="rolling", rolling_lookback_td=756, n_val=2)。
- quarterly cutoffs(cal, 2022-01-01, 2025-10-01)；oos_start=2022-01-01；oos_end=末as_of。
- decide_direction(val-only) → score*=d。featset full = WHITELIST_35 + 16 surv + Alpha158(a158_*)。

## P4 Phases
- [x] **P4a** v6 qlib bin 重 dump ✅ 5741 instruments；close_recon 5.7e-08；raw Alpha158 158列0NaN（默认processor smoke的G1 FAIL是假警，已用raw路径证伪）
- [x] **P4b** Alpha158 在 v6 重提取 ✅ alpha158_asof_v6 (1.68M×158, nan1.3%, 5731票)；G1-mini a158_ROC5/MA5 corr=1.00000 PASS
- [x] **P4c** Ridge(full) 训练 ✅ 700,122 preds / 16 folds / 0 flips / 208s
- [x] **P4d** 两口径评估 ✅ 全市场 neut 0.052/净+1.17%；PIT top-1500 neut **0.057**/净**+2.75%**/ICIR0.99/三regime全正；
      分解诊断证伪退市崩盘簇 artifact（Δ=0.0001，将退市行仅0.049%）
- [x] **P4e** 报告 docs/plans/survivorship_p4_verdict.md ✅ **判定=TIER 1**
- [x] **补1** 流动性分桶 IC ✅ 证伪"中小盘"机制（IC 随流动性升而升：top500 0.056>small 0.039；top500>v5）
- [x] **补2** 逐 fold 稳健 ✅ 两口径 16/16 全正；PIT top-1500 丢最好2后=0.054≥0.02（robust）
- [x] **定因** Diag A v5 重评=0.034（基线实）；Diag B v6∩v5的1373票=0.032≈v5
      → **机制实锤：IC 抬升全来自评估票池组成（幸存者压低IC），模型 v5≈v6，无 artifact/泄漏**
- [x] **方法论副产物** 已记入 verdict：幸存者限制对截面排序 IC 是【压低】非抬高
- [x] **解封** productization_backlog.md（顶部写解封记录 + gate 限定：净+2.75%<formal 5%，12d=pending_nav）

---

# T1 三动作（解封后，2026-06-14）

## 动作1 — 契约层 P0（✅ 完成，停下报告）
- [x] quantmind/contracts/：HorizonRegistry（short12/robust21/long63 单一来源）
- [x] ModelRegistry：注册 ridge_full_12d_v6_seed（gate=research_candidate_pending_nav，gate_pass=False）
- [x] RecommendationContract + OutcomeContract（schema 定义，未接生产）
- [x] run_manifest↔ModelRegistry 联动（train manifest 自动注册候选，懒导入）
- [x] tests/test_contracts.py 11 passed。**不含 UI/API/Agent 迁移（P1，等 NAV）**

## 动作2 — 63d 基本面版（P63-1 拉数中）
- [x] **P63-1** 财报 PIT 拉数 ✅ fina_indicator 148,443行 / 5475票（54失败~1%，退市/无财报）；6.9h；0 token泄漏
      **ann_date 验收 PASS**：全市场 0 违规（ann_date>end_date 全成立），滞后 p50=47d（Q1→+30d/年报→+88-107d 教科书级 PIT）
      范围：仅 fina_indicator（估值走 daily_basic 已在 lake，PIT）；现金流因子若 P63-2 要再补拉。
- [x] **P63-2** 基本面因子 + 筛选 ✅（扩池后 **27 因子 → 16 survivors**）
      扩池补拉：fina全字段+income+balancesheet+cashflow（各5475票/0泄漏）→ 解锁全 8 缺失因子。
      因子面板 1.56M×27（26 fundamental.py + 自定义 fcf_yield_cf），PIT by ann_date，逐截面 winsor。
      筛选(in-sample neut ICIR≥0.2 + flip<35% + corr<0.7，63d，UNKNOWN桶) → **16 survivors**：
      新增 accruals(OOS-0.026 站住)/ocf_to_revenue(OOS+0.023 站住)/revenue_accel_q/equity_multiplier/op_yoy。
      **OOS 核心仍是 VALUE+现金流质量**(BM+0.086/DV+0.060/EP+0.049/ocf+0.023/accruals-0.026)；
      growth/accel 入选但 OOS 近零。proper fcf_yield(fcff) 无信号(ICIR-0.06)被弃，自定义 fcf_yield_cf 入选。
      stop 等用户 go P63-3。
- [x] **P63-3** 63d WF 判定 ✅（半年 refit 重跑，fold 自检门 PASS：8cutoff 间距~120td>E63，7有效fold/81as_of）
      首跑季度退化已诊断 → 改半年 refit + 建 wf_horizon_compatibility_checklist.md。
      **ABCD(PIT top1500)**：A(12d)0.057/+2.75% · **B(full+fnd)0.059/+5.33%** · C(no-fnd)0.079/−4.15% · D(fnd-only)0.073/+2.39%。
      硬化全做：逐fold(B6/7·C/D7/7 丢2后均≥0.02) · 分桶(三者top500最高,无小盘红旗) · regime全正 · Diag B(C腰斩=universe-effect,D稳=真alpha)。
      **判定 B=第二产品线种子**(IC0.059≥0.02/净+5.33%≥0/震荡+0.088≥0)，gate=research_candidate_pending_nav，需 NAV 过5%。
      核心交付：基本面 vs C 边际贡献 = rank-IC −0.020 但净超额 +9.5pt、maxDD15.6%→6.4%（多头产品决定性正向）。
      OOS 可信核心 5 因子(BM/DV/EP/ocf/accruals)。注册 ridge_full_fnd_63d_v6_seed。**停，带证据回评审。**

## 动作3 — 不做（显式登记）
- [x] 登记 backlog：12d深化 / 63d-seq-illiquid / Agent·前端·API迁移 = 全部排在 NAV 之后

## ⚠ 缺失文档（已补建/待确认）
- long_horizon_fundamental_plan.md：仓库原缺，本 session 据口头规格补建。
- executable_nav_design.md：仓库**仍缺**，backlog 已标注需补建（NAV 回测前置）。
  - ≥0.02 & 净超额≥0 & 震荡正 → 站住：Phase1过，解封 backlog + 启 63d 基本面版
  - 0.01-0.02 / 净超额近零 / 震荡贴线 → 弱化未死：转 63d 基本面主线，12d 留研究
  - <0.01 或 震荡负 → 日线见顶：转分钟数据评估
- [ ] **顺手** v6 manifest 加 sha256（Item7 漏洞）；以后 v* 面板首次落盘即预录哈希

## 约束
v5 字节不动（每步 md5 守卫）；双环境（dump/Alpha158=qlib_bakeoff，训练/评估=quantmind）；
长跑 nohup setsid + 监控接力；token 防泄漏；判定对 gate 线、不与 v5 攀比。
