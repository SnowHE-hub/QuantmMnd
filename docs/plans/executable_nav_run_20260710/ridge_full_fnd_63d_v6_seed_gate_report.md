# Executable NAV Gate 报告 — `ridge_full_fnd_63d_v6_seed`

- 判定时间：2026-07-10T18:27:45
- **gate_pass = False**

## 逐项判定

| 指标 | 实际值 | 条件 | 阈值 | 通过 | 备注 |
|---|---|---|---|---|---|
| ann_net_excess | +0.0226 | >= | 0.05 | ❌ |  |
| max_drawdown_net | -0.3543 | >= | -0.12 | ❌ |  |
| information_ratio | +0.3311 | >= | 1.0 | ❌ |  |
| ann_twoway_turnover | +3.0008 | <= | None | ✅ | 设计未定阈值 → 仅报告不阻断（设计缺口） |
| yearly_net_excess_all_positive | 2022:+4.61%; 2023:-2.64%; 2024:+3.55%; 2025:-1.96%; 2026:+3.96% | all>0 | 0.0 | ❌ | 负年份: {'2023': -0.026373151858542054, '2025': -0.01960732870029258} |

## NAV 摘要

- 区间：2022-08-04 → 2026-05-11（898 交易日，7 次再平衡）
- 净年化 +12.36% ｜ 毛年化 +12.83% ｜ 基准年化 +10.10%
- **年化净超额 +2.26%** ｜ IR 0.33 ｜ 净 MaxDD -35.43%
- 年化单边换手 1.50 ｜ 累计成本 0.0229（NAV 单位）
- 拒单事件 3620：{'insufficient_cash': 3386, 'limit_down_locked': 121, 'limit_up_locked': 55, 'suspended_no_bar': 24, 'expired_unfilled': 24, 'one_line_board': 10}

## 设计缺口（保守实现，见 nav_engine 模块 docstring）

- Top-N/基准取研究层同口径（top-quintile of PIT top-1500 / 同池等权）；
- 换手阈值设计未定 → 仅报告不阻断；
- ST 无 PIT 标记 → 板块阈值 + 一字板兜底；
- gate 通过也**不自动升 production**（人工签收）。