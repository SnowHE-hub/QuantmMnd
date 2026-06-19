# 待清理清单 2026 Q2（阶段 B / Q5）—— 不自动删，待用户 5 秒扫确认

> 列出候选 + 用途猜测 + 建议。**未删任何文件。** 确认后我按"删除"类执行。
> ⚠ 已排除依赖：`scripts/_diag_wf_v2.py`（build_regime 被 p4/p63 eval import）已 commit，**不在删表**。

## A. 运行日志 / 输出（建议：全删）
| 路径 | 用途 | 建议 |
|---|---|---|
| `scripts/survivorship/*.log`（_p1_prices / _p2_* / _p4* / _p63_* 共 ~13 个） | 各阶段拉数/训练/评估 run 日志 | **删** |
| `scripts/bakeoff/*.log`（_orchestrator/_orch_*/_p3b_*/_p3c_*/_memmon/_one_* 共 ~20+） | bake-off 编排 run 日志 | **删** |
| `scripts/bakeoff/_g1.out` / `_g1_v2.out` | G1 自检输出快照 | **删** |
| `__pycache__/`（多处） | 字节码缓存 | **删 + gitignore** |
| `tmp/qlib_mlruns/`、`tmp/p3_review/` | 临时 mlflow/review 产物 | **删** |

## B. 一次性诊断/验证脚本（建议：多数删，少数保留为工具）
### B1. scripts/ 顶层 `_*.py`（研究期 diag，多已过时）
| 文件 | 用途猜测 | 建议 |
|---|---|---|
| `_check_env.py` | 环境自检 | 删（一次性） |
| `_compare_augment.py` / `_compare_augment_v2.py` | 增广特征对比实验 | 删（研究废弃） |
| `_diag_board.py` / `_diag_board_ic.py` / `_diag_meta.py` / `_diag_mom2.py` / `_diag_momentum.py` | 板块/元/动量诊断 | 删（研究期） |
| `_download_bge_m3.py` | 下 BGE 嵌入模型 | 删或移 setup（一次性） |
| `_save_cnn_v2.py` | 存 CNN 模型实验 | 删（废弃实验） |
| `_sim_attribution.py` | 归因模拟 | 删（研究期） |
| `_test_kb_status.py` | KB 状态测试 | 删（临时） |
| `_verify_board_router.py` / `_verify_momentum_fix.py` / `_verify_valuation_fix.py` | 修复验证 | 删（一次性验证，已落定） |
| ~~`_diag_wf_v2.py`~~ | **build_regime 依赖** | **已 commit，保留** |

### B2. scripts/bakeoff `_*.py`（debug）
| 文件 | 用途 | 建议 |
|---|---|---|
| `_dbg_pred.py` / `_dedup_survivors.py` / `_g1_recheck.py` / `_g1_v2.py` | 预测调试/去重/G1 复查 | 删（一次性 debug） |

### B3. scripts/survivorship `_*.py`（本轮验证证据，部分可留作工具）
| 文件 | 用途 | 建议 |
|---|---|---|
| `_token_rotation_verify.py` | token 轮换验证（可复跑） | **保留**（安全工具，可 commit 到 scripts/maintenance） |
| `_p63_3_foldcheck.py` | WF fold 间距自检（模板工具） | **保留**（可 commit，配 wf_horizon_compatibility_checklist） |
| `_p4d_decomp.py` / `_p4f_cause.py` / `_p2_delist_spotcheck.py` / `_p4b_g1mini.py` / `_p63_1_anndate_check.py` / `_p1_verify.py` | 各 verdict 的验证证据脚本 | 留或删（研究证据，verdict 已记结论；倾向删，留 verdict 即可） |
| `_p63_2_smoke.py` / `_contracts_report.py` / `_registry_verify.py` / `_p63_3_detail.py` | 一次性 helper | 删 |

## C. 怪文件 / 杂项（建议：删）
| 路径 | 用途 | 建议 |
|---|---|---|
| `8}` | 0 字节误生成文件（May 27） | **删** |
| `run_valuation.py`（根级，无引用，May 26） | 旧估值 runner，无代码引用 | **删**（确认后） |
| `tmp/` 整目录 | 临时 | **删** |

## 决策
- **A + C + B1 + B2 全删** = OK?
- **B3**：`_token_rotation_verify.py` + `_p63_3_foldcheck.py` 我建议**移到 `scripts/maintenance/` 并 commit**（复用工具）；其余 B3 删。同意?
- 确认后我执行删除 + （若同意）把 2 个工具脚本归档 commit。
