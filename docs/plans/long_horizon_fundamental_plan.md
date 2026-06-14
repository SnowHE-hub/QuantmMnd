# 63d 基本面版验证计划（第二产品线）

> 状态：**P63-1 拉数进行中**（2026-06-14 启动）。
> ⚠ 本文档由当前 session 据用户 T1-动作2 的口头规格补建——用户提到的"并行批已写"版本
> **未在仓库找到**（`docs/plans/long_horizon_fundamental_plan.md` / `executable_nav_design.md` 当时缺失）。
> 若存在并行未提交版本，以那份为准并合并。
> 触发：P4 Tier 1 解封（Phase 1 闸门通过）。性质：**独立判定跑**，P63-3 出数前停下评审。

## 背景
12d 短线信号 = 研究层赢家（Tier 1），但净超额 +2.75% < formal gate +5%，状态"待 NAV 回测"。
63d 长持有期 = 第二产品线，基本面因子在长 horizon 上更有信息量、换手更低、成本占比更小。

## Phases

### P63-1 — 财报 PIT 拉数（进行中）
- 拉 fina_indicator / income / balancesheet / cashflow（v6 全市场 5529 票，含退市）。
- **严格保留 ann_date（实际公告日）**；PIT 对齐在因子构建时用 ann_date，**绝不用 end_date**
  （A 股基本面因子最易错处：end_date=报告期，数据要到 ann_date 才可得）。
- 脚本 `scripts/survivorship/p63_1_pull_fundamentals.py`（token 防泄漏、断点续传、落 data/lake/）。
- **验收**：抽 ≥3 家公司的 ≥5 个报告期，核对 ann_date 与公开披露一致（如季报 4 月底、年报 4 月底前）。

### P63-2 — 基本面因子实现
- 价值（PB/PE_ttm/EP/BM）、质量（ROE/ROA/毛利率/负债率）、成长（营收/净利 yoy）、盈利（净利率/资产周转）。
- 复用 `quantmind/features/fundamental.py` 的快照因子签名；as_of 快照只取 ann_date ≤ as_of 的最新一期。
- **复用 short_horizon 同套筛选**：in-sample neut IC 选 → OOS 无偏确认 → 翻转<35% / corr<0.7 去冗余。

### P63-3 — 63d WF 判定（独立判定跑）
- 模型：Ridge(full + 基本面 survivors)，H=63 / E=63 / 季度 cutoff，UNKNOWN 桶中性化。
- 两口径：PIT top-1500 + 全市场。复刻 batch-A 其余配置。
- 输出与 12d 同格式（neut IC / ICIR / 含成本净超额 / 逐 fold / regime）。
- **出数即停评审**，不与 12d 攀比，对 gate 线判定。

## 纪律
- 不同时做 12d 任何深化（12d 等 executable NAV 回测）。
- P63-1 拉数后报覆盖摘要；P63-3 出数前停下。
