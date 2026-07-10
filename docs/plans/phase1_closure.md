# Phase 1 闭幕记录（2026-06-19）

## 一段话总结
Phase 1 从**数据底座**（v6 全市场 PIT 拉数：daily/adj_factor/daily_basic + 财报 fina_indicator/income/
balancesheet/cashflow，含退市票，0 token 泄漏）出发，建起**评估框架**（PurgedWalkForward + UNKNOWN 桶中性化 +
holding-period-aware 含成本净超额 + regime 条件 IC），完成**因子研究**（35 量价 + 16 short_horizon survivors +
16 基本面 survivors + Alpha158）与 **bake-off**（Ridge(full) 胜出），经**四重硬化**（逐 fold 稳健 / 流动性分桶 /
regime 符号 / Diag B 票池反证）层层反证，核心是**幸存者修复**：把今日指数成分的幸存者池修成真 PIT 全市场池（v6），
实证"幸存者限制压低截面 IC、非抬高"，最终产出**两条产品种子**——12d 短线（量价）与 63d 长线（量价+基本面）。
全程纪律：超预期先反证再判断、研究层净超额≠产品判断（IC vs 净超额背离原则）、secrets 绝不入 commit/log/报告。

## 两条种子（ModelRegistry，均 `research_candidate_pending_nav`）
| model_id | horizon | feature_set | PIT top-1500 neut IC | 含成本净超额 | neut ICIR | regime | gate_pass |
|---|---|---|---|---|---|---|---|
| `ridge_full_12d_v6_seed` | short(12d) | full_35_16_158 | **0.057** | **+2.75%** | 0.99 | 三档全正 | False（待 NAV） |
| `ridge_full_fnd_63d_v6_seed` | long(63d) | full_v6_63d_35_16sh_16fnd_158 | **0.059** | **+5.33%** | 0.75 | 三档全正（震荡 +0.088） | False（待 NAV） |
- 配置：Ridge(α=10) / PurgedWFS(12d:E20 季度 refit；63d:E63 半年 refit) / UNKNOWN 桶 + log(circ_mv) 中性化 / decide_direction val-only。
- 基本面边际贡献（63d，B vs C ablation）：rank-IC −0.020 但**含成本净超额 +9.5pp**、maxDD 15.6%→6.4%。
- OOS 真正可信的基本面核心 5 因子：book_to_market / dividend_yield_ttm / earnings_yield / ocf_to_revenue_ttm / accruals。

## 合并信息
- **main SHA**：`67bb891`（67bb891871bfed107add67c53727e731403d8381）
- **PR**：https://github.com/SnowHE-hub/QuantmMnd/pull/1
- **测试**：main 默认 marker 套件 1189 collected / **1121 passed** / 1 skipped / 67 deselected / **0 fail / 0 error**。

## 待续（硬前置 / 暂留，触发条件已记录）
- **executable NAV 实现（C 阶段）**：两条种子 → 可上线产品的**硬前置**。研究层净超额是 label proxy；
  上线须经真实成交 NAV 过项目 formal gate +5%（设计见 `executable_nav_design.md`）。**待用户放行。**
- **暂留**：
  - `stash@{0}`（panel/model refactor）+ 10 个 `stale_panel_fixture` 测试 —— refactor 解封时一起修。
  - 2 个残留 bakeoff smoke + 未分配 untracked docs —— 单独 "chore: misc cleanup" 小 PR。
  - **1b git 历史强推暂缓**：14 个旧 token 全死（1a 完成），不值为改写 106 commit SHA 牺牲 main 审计连续性。

## 关键文档索引
survivorship_repair_plan / survivorship_p4_verdict / survivorship_p63_3_verdict / executable_nav_design /
methodology/ic_vs_net_excess_divergence / wf_horizon_compatibility_checklist / productization_backlog /
data_manifest / security/{token_rotation_2026Q2, secrets_handling_policy} / maintenance/{wip_inventory, test_failure_triage, test_markers}。
