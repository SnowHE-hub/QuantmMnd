# ROADMAP（C1 维护）

| 阶段 | 目标 | 闸门（全部满足才进入下一阶段） |
|---|---|---|
| **0 止血** | 修坏 import 与 CI；统一 v4 标签价格基准；固化本地推理配置；匿名化消融 | T-001～T-007 全部 accepted；G1 阶段评审无阻断项；你放行 |
| 1 契约层 | EvalContract + 5 个回测入口改签名 + 历史产物回填（Fable 票 F2） | 任意两份报告能自动判定是否可比 |
| 2 A/B 合流 | LGBM v6 进 registry；主线 A 跑 executable NAV；票池统一到 v6 | 产出主线 A 真实成交口径绩效并入 registry |
| 3 推理与决策层 | Bonsai 27B 接入 rerank（前提：T-007 证明 LLM 有正贡献）；PIT-MCP（Fable 票 F3）；数值决策用校准表格分类器 | 各子项 A/B 结果入账 |
| 4 研究循环 | 方法域 RAG；因子挖掘循环；Skills | 持续 |
