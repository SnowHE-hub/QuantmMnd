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

## Status: ⏸ P1 完成，停在评审门（scope 待定），获批前不实现。

## Decisions（计划期）
- 推荐 scope=A（全市场 PIT），不纳入 BSE（单独标注）。
- v6 命名：alpha_prices_panel_v6 / alpha_panel_weekly_v6，新文件不动 v5。
- 评估侧零改动；WF/中性化/成本/p3f 不变。

## 不做（明确推迟）
63d / batch B / illiquid 深度 follow-up —— 全部等幸存者修好、Ridge 在 v6 上复核后再议。
