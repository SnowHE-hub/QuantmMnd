# 产品化 Backlog（Codex 全链路报告的 contract 蓝图，登记 ≠ 实施）

> 状态：**全部推迟，未开工**。来源：Codex 全链路交叉排查报告
> （`quantmind_full_chain_detection_report.md`，Windows 侧）。
> **触发条件（硬）= 幸存者判定通过**：必须先做 `docs/plans/survivorship_repair_plan.md`，
> 在**真实 universe(v6)** 上重跑 **Ridge(full)**，确认 **neut IC +0.034 / 净 +1.9% 仍在**，
> 才围绕"真信号"建以下 contract。**判定为假则全部作废**，改围绕 63d/分钟数据另起。
> 登记目的：防 scope 漂移 + 不丢 Codex 的设计建议。生成：2026-06-09。

---

## 为什么现在不做（scope 纪律）
当前所有 alpha 数字都在幸存者乐观上界上。围绕一个**可能是假象**的信号建生产 contract（注册表/
推荐契约/前端迁移）= 在沙上盖楼。**先判真伪，再谈产品化**。这是用户定案的优先级。

## 推迟的 Codex P0/P1 产品化项（触发条件满足后再逐项开计划）

| 项 | Codex 建议（蓝图，待判定后细化） | 触发条件 |
|---|---|---|
| **HorizonRegistry** | 统一 horizon 口径注册表（12d/21d/63d 的 label/holding/成本/再平衡），消除散落各处的硬编码 horizon | 幸存者判定通过 |
| **ModelRegistry** | 模型注册（model_id → 特征制式/标签/训练口径/版本/权重路径），支持版本化与可复现加载 | 幸存者判定通过 |
| **RecommendationContract** | 推荐输出统一 schema（标的/分数/horizon/置信/成本后净预期/可交易性/生成元数据），前后端共用 | 幸存者判定通过 + 确定围绕哪个真信号 |
| **Ridge 接生产** | 把 batch-A 赢家 Ridge(full) 的训练/打分接入生产推荐流（而非现 demo 口径） | **v6 上 Ridge 复核通过**（最关键前置） |
| **前端口径迁移** | app/ 各页从旧 demo 口径（sim30d/v4 等）迁移到统一 contract + v5/v6 真实口径 | 上述 contract 落地后 |

> ⚠ **schema 细节**：上表为蓝图级；Codex 报告里的**具体字段 schema 建议**应在触发后从
> `quantmind_full_chain_detection_report.md` 原样取入对应子计划（本机当前未取到报告内文，
> 待用户提供路径或内容后补录到此文件的附录）。

## 明确拒绝（记录在案，防再提）
- **「63d sequence illiquid targeted test」（Codex 建议）→ 拒绝**。
  理由（详见 `model_bakeoff_plan.md §14`）：illiquid 桶 = 小盘/ST/最易退市（**幸存者偏差最重**）
  × 滑点最贵（**成本最高**）的双重角落；其 0.05-0.075 是**最不可信、最挖不出钱**的数字，
  **不是 follow-up 线索**。幸存者修复前该测试结果无意义；修复后若 illiquid 信号缩水（大概率），
  此测试自动作废。

## 下一步顺序（与产品化无关，先做的是判真伪）
1. 幸存者修复（v6）→ 2. v6 上重跑 Ridge(full) 看 +0.034/+1.9% → 3. 若仍在：解锁本 backlog；
   若不在：本 backlog 全作废，转 63d/分钟数据方向。
