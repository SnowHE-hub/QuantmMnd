# 工作区盘点 2026 Q2（阶段 A1）—— 只读，待用户决策

> 状态：19 modified + 102 untracked。**不动任何文件**，仅盘点 + 建议。每项标 commit/stash/discard + 理由，待拍板。

## 🚨 头号发现：核心生产模块未提交，main 是坏 clone
以下模块**未跟踪（从未 commit）且 main 缺失**，但**已提交代码（含 Phase 1）import 它们** → 任何人 clone main 会 ImportError：

| 未跟踪核心模块 | 被谁依赖 | 建议 |
|---|---|---|
| `quantmind/backtest/wf_split.py` / `wf_metrics.py` / `wf_costs.py` / `wf_gate.py` | evaluate_bakeoff / p4c / p63_3（已提交） | **必须 commit** |
| `quantmind/data/lake.py` | 全 survivorship 链 | **必须 commit** |
| `quantmind/features/weekly_panel.py` / `short_horizon_factors.py` | p4/p63 训练评估 | **必须 commit** |
| `quantmind/utils/silence_provider_logging.py` | 拉数脚本 token 防泄漏 | **必须 commit** |
| `scripts/backfill_tushare.py` | 全部拉数 | **必须 commit** |
→ **这是合并 main 前的真正 must-fix**（B 阶段优先）。它们非 gitignore，只是漏 `git add`。
→ 也解释用户看到的"40 errors"：clean checkout 缺这些模块 → 40 个 import/collection 错误级联。

## 19 MODIFIED（含 EOL 噪声判定）
> 多数 data_pipeline 脚本是 **CRLF/LF 行尾churn**（+N/-N 近相等），忽略空白后仅 17~36 行真实改动。

| 文件 | 真实改动(-w) | mtime | 建议 | 理由 |
|---|---|---|---|---|
| `.env.example` | 4 | 06-06 | **commit** | 配置模板（占位字段） |
| `.gitignore` | 2 | 06-06 | **commit** | 应含新增 data/ 忽略规则（核对后提交） |
| `HANDOVER.md` / `README.md` | 4 / 2 | 06-06 | **commit** | 小幅文档更新 |
| `docs/plans/survivorship_repair_plan.md` | 17 | 06-10 | **commit** | survivorship 计划残留编辑（属 Phase 1） |
| `data/paper_trading/forward_positions.json` | 282 | **06-18 17:15** | **discard + gitignore** | paper-trade 运行时状态，每跑必churn |
| `data/paper_trading/performance.json` / `strategy_config.json` | 2 / 2 | 06-18 | **discard + gitignore** | 同上运行时产物 |
| `data/sim30d/summary.json` | 2 | 06-06 | **discard + gitignore** | 运行时产物 |
| `quantmind/models/factor_model.py` | 49 | 06-06 | **review→stash或commit** | 在途模型改动；test_lgbm_training subject |
| `quantmind/models/lgbm_ranker.py` | 10 | 06-06 | **review→stash或commit** | 同上在途 |
| `scripts/build_full_panel.py` | 18(+EOL) | 06-06 | **review→stash或commit(规范EOL)** | 在途 panel refactor；test_full_panel subject |
| `scripts/data_pipeline/patch_adj_close.py` | 17(+EOL) | 06-06 | **review** | 在途 + EOL；核实是否 token 相关 |
| `scripts/data_pipeline/setup_api_config.sh` | 36(+EOL) | 06-06 | **review（重点：含 token 配置）** | 改动较大，**审有无 secret 字段** |
| `scripts/data_pipeline/step1/2/4_*.py` | 17~19(+EOL) | 06-06 | **review→stash或commit** | 在途数据管线 + EOL |
| `scripts/run_2025q1_full_demo.py` | 17(+EOL) | 06-06 | **review→stash或commit** | 在途 demo + EOL |
| `scripts/verify_tokens.py` | 19(+EOL) | 06-06 | **review（token 脚本，审 secret）** | 与 token 轮换相关，先审再定 |

## 102 UNTRACKED（按类别）
| 类别 | 数量/大小 | 示例 | 建议 |
|---|---|---|---|
| **核心生产模块**（见上表） | ~8 | wf_*.py / lake.py / weekly_panel.py / backfill_tushare.py | **commit（必须）** |
| **在途测试**（测上面核心模块） | tests/ 5 | test_lake / test_weekly_panel / test_wf_split / test_wf_costs / test_wf_gate | **commit（随核心模块一起）** |
| **关键文档** | docs/ 14 | **p3_independent_review.md**(P3验收!) / data_lake_*plan / panel_code_review / tooling_setup | **commit 有用的**（p3_review 必提） |
| **research 数据产物** | data/ 28 / **480 MB** | alpha_panel_weekly_v6 / fundamental_factors_v6 / bakeoff preds | **gitignore**（不入库，太大；已有 sha256 manifest 溯源） |
| **codegraph 索引** | .codegraph/ 13 MB | 索引缓存 | **gitignore** |
| **survivorship/bakeoff 脚本** | scripts/ 部分 | scripts/survivorship 残留 / bakeoff helpers | **commit 正式脚本**，删 `_*.py`/`_*.out` 实验件 |
| **临时实验/诊断** | scripts/_*.py, *.out | `_compare_augment.py` `_diag_*.py` `_g1.out` `_save_cnn_v2.py` | **删除**（一次性诊断） |
| **杂项/异常** | 根级 | `8}`（疑似误生成文件）/ `run_valuation.py` / `tmp/` / `.mcp.json` | `8}`+`tmp/`**删**；`run_valuation.py` review；`.mcp.json` review（可能含配置） |

## 待用户拍板的关键决策
1. **核心模块 + 在途测试 + p3_review 文档**：确认全部 commit（B 阶段必做，否则 main 仍坏）。
2. **在途 panel/model refactor**（factor_model/lgbm_ranker/build_full_panel/weekly_panel + 相关测试）：
   是**随 Phase 1 一起合**，还是**单独 stash 成另一 PR**？（影响 test_full_panel/test_lgbm 是否必修）
3. **data/ 480MB 产物**：确认 gitignore（不入库）。
4. **token 相关脚本**（setup_api_config.sh / verify_tokens.py / patch_adj_close.py）：先审有无 secret 字段再 commit。
5. **临时实验件**（`_*.py` / `_*.out` / `8}` / `tmp/`）：确认删除。
