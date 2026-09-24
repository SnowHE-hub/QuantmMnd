# 交接 · 阶段 0 开始（写于 2026-09-24）

## 你是谁
QuantMind 升级项目的 **C1 总控（Codex CLI · GPT-6 Sol）**。先读本文件、`../ledger.md`、仓库根目录的 `AGENTS.md`。
你不写生产代码；你写任务卡、派发、维护账本。

## 参考文档（仓库外，**不要复制进仓库**——仓库是公开的，这些文档含私人评估内容）
- `/mnt/d/MCS_26S2课程资料/quantmind/docs/architecture.html` — 现状说明（第 8 节是问题清单）
- `/mnt/d/MCS_26S2课程资料/quantmind/docs/upgrade-plan.html` — 改造方案
- `/mnt/d/MCS_26S2课程资料/quantmind/docs/collaboration-plan.html` — 分工与十项实测结果

## 已确认的事实（全部 [实测] 或 [代码]；出处见 collaboration-plan §1）
1. v4 标签有实质缺陷：2024-09-30 / 2024-12-31 截面约 87% 标签误差 > 10%；v6 不受影响
2. z-score 被加 % 喂给 LLM（8 处）；DPO 训练分布与推理分布不一致；漏斗 Layer 4 运行时猜单位
3. 本地推理：Bonsai 27B + KV q4 + FA 约 29.7 tok/s；JSON schema 约束零代价且 100% 合法；
   推测解码全无效；-np 不增加总吞吐；任意两个大模型不能同时驻留；换模型 8–16 秒
4. Laya 零样本在数值卡片上坍缩（40/40 同一答案）→ 数值决策改用校准表格分类器
5. Bonsai 知识截止估计 2025-01～03（8 条探针，不是证明）
6. 测试：本地 1130 passed / 2 failed（wf_gate.py:64 坏 import）；覆盖率 55%。**GitHub CI 64/64 次失败、从未成功，当前已停用**（T-002）
6b. executable NAV 审计（`docs/plans/executable_nav_audit_20260710.md`）：12d 死于成本（−3.20%/年）；63d 实际持有期是设计的 2 倍、只有 7 次独立再平衡，属样本不足而非干净失败；拒单 1635 次是重试膨胀后的事件数，去重后 424 张订单
7. 未提交项已处理：在分支 `chore/pending-wip-and-agent-setup` 上提交，等用户合并进 main；剩 `.mcp.json`（本机配置，不提交）与 `scripts/db_migration/_update_cron.sh`（改 crontab 的一次性脚本，待用户决定）

## 未决（需要用户）
- 合并分支 `chore/pending-wip-and-agent-setup` 进 main
- integration job 的 `TUSHARE_TOKEN` secret 是否更新（T-002 第 4 步）
- `_update_cron.sh` 去留

## 今天建议
1. 派发 T-001（K2 Composer）、T-005（K1 Grok 与 G2 Codex 各做一份，模型校准题）、T-006a（C3）
2. T-001 验收后派 T-002
3. 起草 T-007 的实验设计，交 C2 · Opus 5.5 评审（跨家族），再交 G1 做统计复核
4. Fable 票 F1（T-003）在 T-001 合并后开
