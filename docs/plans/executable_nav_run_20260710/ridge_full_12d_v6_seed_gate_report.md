# Executable NAV Gate 报告 — `ridge_full_12d_v6_seed`

- 判定时间：2026-07-10T18:31:18
- **gate_pass = False**

## 逐项判定

| 指标 | 实际值 | 条件 | 阈值 | 通过 | 备注 |
|---|---|---|---|---|---|
| ann_net_excess | -0.0223 | >= | 0.05 | ❌ |  |
| max_drawdown_net | -0.4888 | >= | -0.15 | ❌ |  |
| information_ratio | -0.2668 | >= | 1.0 | ❌ |  |
| ann_twoway_turnover | +17.9000 | <= | None | ✅ | 设计未定阈值 → 仅报告不阻断（设计缺口） |
| yearly_net_excess_all_positive | 2022:-2.49%; 2023:-5.13%; 2024:-1.78%; 2025:-2.85%; 2026:+4.11% | all>0 | 0.0 | ❌ | 负年份: {'2022': -0.0248568101133666, '2023': -0.05130530069299255, '2024': -0.017764431663146896, '2025': -0.028543183127419436} |

## NAV 摘要

- 区间：2022-02-09 → 2026-05-11（1018 交易日，48 次再平衡）
- 净年化 -0.90% ｜ 毛年化 +2.30% ｜ 基准年化 +1.33%
- **年化净超额 -2.23%** ｜ IR -0.27 ｜ 净 MaxDD -48.88%
- 年化单边换手 8.95 ｜ 累计成本 0.1322（NAV 单位）
- 拒单事件 1635：{'insufficient_cash': 758, 'suspended_no_bar': 469, 'limit_up_locked': 220, 'one_line_board': 97, 'limit_down_locked': 48, 'expired_unfilled': 43}

## 设计缺口（保守实现，见 nav_engine 模块 docstring）

- Top-N/基准取研究层同口径（top-quintile of PIT top-1500 / 同池等权）；
- 换手阈值设计未定 → 仅报告不阻断；
- ST 无 PIT 标记 → 板块阈值 + 一字板兜底；
- gate 通过也**不自动升 production**（人工签收）。