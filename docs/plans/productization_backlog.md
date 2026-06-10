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

---

## 附录：Codex 报告原文照录（schema 蓝图，触发后据此细化）
来源 `C:\Users\lenovo\Documents\QuantMind\quantmind_full_chain_detection_report.md`。原样收录，**触发条件不变**（=幸存者判定通过）。

### A.1 七大核心 contracts（报告 §12.1）
1. **HorizonRegistry**：`short`=12d(客户短线)、`robust`=21d(只做稳健性验证)、`long`=63d(客户长线)。
2. **DataVersion**：raw source version；lake coverage snapshot；panel version（如 `weekly_v5`）；**survivorship flag**。
3. **FeatureSet**：`tab35` / `tab35_plus16` / `full_35_16_158` / `alpha360_seq60`。
4. **ModelRegistry**：model artifact / feature set / label / horizon / WF config / gate metrics / production status。
5. **RecommendationContract**：每条推荐必须知道自己来自哪个 horizon、模型、特征、数据版本和 gate。
6. **AgentExplanationContract**：Agent 只解释/约束/提示风险；必须引用 base recommendation。
7. **OutcomeContract**：统一收集 12d/21d/63d 真实收益、成本、成交状态。

### A.2 recommendation schema（报告原文）
```json
{
  "as_of": "2026-06-01",
  "ticker": "000001.SZ",
  "horizon_name": "short",
  "horizon_days": 12,
  "label_col": "forward_return_12d",
  "model_id": "ridge_full_12d_v1",
  "feature_set_id": "weekly_v5_35_16_alpha158",
  "score_raw": 0.0,
  "score_neutralized": 0.0,
  "rank": 1,
  "weight": 0.05,
  "liquidity_bucket": "b0",
  "cost_model_id": "wf_costs_v1",
  "gate_status": "research_candidate",
  "agent_summary_id": "agent_explainer_v1"
}
```

### A.3 feedback schema（报告原文）
```json
{
  "as_of": "2026-06-01",
  "ticker": "000001.SZ",
  "horizon_days": 12,
  "model_id": "ridge_full_12d_v1",
  "entry_price": 10.0,
  "exit_price": 10.5,
  "actual_return_12d": 0.05,
  "actual_return_21d": 0.06,
  "actual_return_63d": 0.12,
  "slippage_bps": 15,
  "fees_bps": 10,
  "net_return": 0.0475,
  "fill_status": "filled",
  "liquidity_bucket": "b0"
}
```

### A.4 目标接口（报告 §12.2）
```text
GET /api/recommendations?as_of=YYYY-MM-DD&horizon=short|long
GET /api/models/leaderboard?horizon=12d|21d|63d
GET /api/models/status
GET /api/data/freshness?horizon=short|long
GET /api/agent/explanation?ticker=...&as_of=...&horizon=...
GET /api/feedback/outcomes?model_id=...&horizon=...
# 保留但限制为 admin/local：POST /api/execute、GET /api/stream/{cmd_key}、POST /api/chat
```
> 注：F-08（admin token + localhost）已在 safety-monitoring-fixes 分支落地；其余接口均待触发后实施。
