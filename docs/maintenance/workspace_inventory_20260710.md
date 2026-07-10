# 工作区清单 2026-07-10（阶段 0 收口）

> 基准：`main @ 67bb891`；本次盘点**未删除任何文件**、未动 `stash@{0}`。
> 前序盘点：`wip_inventory_2026Q2.md`（2026-06-18）、`wip_cleanup_2026Q2.md`（Q5 候选删表，仍待用户确认）。
> 本次盘点时未跟踪共 **69 项**（`git status --porcelain -u`），modified 仅 `task_plan.md`（+8 行）。

## Git 状态摘要

| 项 | 值 |
|---|---|
| HEAD | `67bb891871bfed107add67c53727e731403d8381`（= origin/main） |
| Modified | `task_plan.md`（+8 行 Phase 1 合并说明，见 §E） |
| Stash | `stash@{0}` WIP panel/model refactor（factor_model / lgbm_ranker / build_full_panel）——**本轮不动** |
| 分支 | `main`、`safety-monitoring-fixes`（已合入，可后续删除，本轮不动） |

---

## A. 建议提交（源码 / 测试 / 正式文档）

### A1. 被已跟踪代码引用的脚本（不提交则 clean clone 语义不完整）

| 文件 | 被谁引用（已跟踪） | 理由 |
|---|---|---|
| `scripts/verify_weekly_panel.py` | `quantmind/features/weekly_panel.py` 文档、`tests/test_weekly_panel.py` 注释 | v5 面板端到端验收脚本，panel 契约的一半 |
| `scripts/build_data_lake.py` | `tests/test_lake.py`（"依赖已建好的真实湖表（scripts/build_data_lake.py 的产物）"） | lake 表的唯一正式构建入口 |
| `scripts/run_wf_full_v1.py` | `scripts/_diag_wf_v2.py` | v5 LGBM walk-forward 正式跑批入口 |
| `scripts/bakeoff/dump_bin.py`、`p1_dump_bin.py` | `scripts/survivorship/p4a_dump_v6.py`、`model_bakeoff_plan.md` | v6 qlib bin 重建依赖 |
| `scripts/bakeoff/p3a_alpha158_asof.py` | `scripts/survivorship/p4b_alpha158_v6.py` | Alpha158 as_of 抽取，P4b 依赖 |
| `scripts/bakeoff/p3d_leaderboard.py`、`p3e_harden.py` | `progress.md`（bake-off 正式 harness） | leaderboard / 硬化评估的正式脚本 |
| `scripts/screen_short_horizon.py` | `progress.md`（R3 正式筛选） | 16 survivors 的产生脚本，复现凭据 |

### A2. 正式文档（决策记录 / 验收报告，task_plan 与 phase1_closure 相互引用）

| 文件 | 理由 |
|---|---|
| `docs/plans/phase1_closure.md` | **Phase 1 闭幕权威记录**；`task_plan.md` 未提交的 8 行直接引用它，必须一起提交 |
| `docs/plans/data_lake_build_plan.md`、`data_lake_completion_report.md` | 数据湖建设计划+完成报告（progress.md 引用） |
| `docs/plans/walkforward_design_plan.md`、`weekly_panel_build_plan.md`、`panel_code_review.md` | WF/面板设计与评审记录 |
| `docs/plans/wf_v2_diagnostics.md`、`docs/wf_review_notes.md` | Step 1 诊断结论（短周期因子任务的承接依据） |
| `docs/plans/data_sufficiency_audit.md`、`data_sufficiency_audit_plan.md`、`tushare_probe_report.md` | 数据充分性审计与探针报告 |
| `docs/plans/tooling_setup_plan.md`、`docs/TOOLING_SETUP.md`、`docs/LABEL_AUDIT.md` | 工具链与标签审计 |
| `docs/maintenance/baseline_20260710.md`、`workspace_inventory_20260710.md`（本文件） | 阶段 0 基线产物 |

### A3. 训练脚本（弱引用，建议随 A1 一起提交）

| 文件 | 理由 |
|---|---|
| `scripts/train_lgbm_12d.py` | 无已跟踪引用，但为 12d LGBM 正式训练脚本（非 `_` 前缀实验件）；若确认废弃可转 D 类 |

## B. 建议加入 .gitignore（数据 / 日志 / 缓存 / 跑批输出）

> 本轮已生成最小 .gitignore diff（见文末），**未删除任何文件**。

| 路径 | 类型 | 理由 |
|---|---|---|
| `data/loss_signals/`、`data/loss_signals_v4/`（json/csv/jsonl） | 运行时信号状态 | `dispatch_loss_signals.py` 每跑必写；与已 ignore 的 paper_trading 运行态同类 |
| `data/watchlist/scores/` | 运行时评分输出 | watchlist 每日产物 |
| `data/qlib_cn_daily/`、`data/qlib_cn_daily_v6/` | qlib bin 数据 | `dump_bin.py` 可重建；calendars/instruments 是数据不是源码 |
| `.codegraph/` | 本地索引缓存 | 工具生成，机器相关 |

## C. 暂留、需人工确认

| 路径 | 说明 | 待决问题 |
|---|---|---|
| `.mcp.json` | 本机 MCP server 配置（260B） | 含本机路径；入库（团队共享）还是 gitignore？ |
| `.cursor/rules/karpathy-guidelines.mdc` | Cursor 行为规则 | 建议 commit（对所有 agent 生效），待确认 |
| `scripts/bakeoff/p0_env_setup.sh` | qlib_bakeoff 双环境搭建脚本 | 可复用基建，建议归 A；确认后移 |
| `scripts/bakeoff/p0_probe_result.json`、`p1_g1_result.json`、`scripts/survivorship/_p4a_g1_v6.json` | 探针/G1 验证结果快照 | 证据文件：入库存档 or 依赖 verdict 文档后删除？ |
| `scripts/db_migration/_update_cron.sh` | dual-write cron 注入工具 | PG 迁移是否继续决定去留 |

## D. 明确可删（本轮不执行删除；多数已在 wip_cleanup_2026Q2.md Q5 删表中登记）

| 路径 | 类型 |
|---|---|
| `scripts/bakeoff/_memmon.sh`、`_next_bucket.sh`、`_orch_check.sh`、`_orchestrator.sh`、`_run_chain9.sh`、`_run_gru_test.sh`、`_run_one.sh` | bake-off 一次性编排脚本 |
| `scripts/bakeoff/_qlib_model_smoke.py`、`_seq_smoke.py`、`_qlib_smoke.out`、`_seq_smoke.out`、`_orchestrator_status.txt` | smoke 脚本与输出快照 |
| `scripts/db_migration/_commit_msg.txt`、`_commit_msg_e1_ops.txt` | 一次性 commit message 草稿 |

## E. task_plan.md 未提交改动审查

- 内容：文件头新增 8 行「Phase 1 闸门已合 main」段（SHA `67bb891`、PR #1、指向 `phase1_closure.md`）。
- 核验：SHA 与 `git rev-parse HEAD` 一致 ✅；merge 事实与 `git log --graph` 一致 ✅；与 `phase1_closure.md` 内容一致 ✅。
- **结论：建议保留并提交**，且须与 `docs/plans/phase1_closure.md`（A2）**同一 commit**（该 8 行引用了它）。

## 附：本轮 .gitignore 最小 diff（已应用）

```
+ data/loss_signals/
+ data/loss_signals_v4/
+ data/watchlist/scores/
+ data/qlib_cn_daily/
+ data/qlib_cn_daily_v6/
+ .codegraph/
```
