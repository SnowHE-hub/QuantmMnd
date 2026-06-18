# 测试回归分流 2026 Q2（阶段 A2）—— 待评审

> 本地实跑（untracked 模块在盘上）：**18 failed / 1170 passed / 1 skipped**（294s，带 25s/测试 signal 超时）。
> 用户报告的"80 fail / 40 errors"= **clean checkout 状态**（缺未跟踪核心模块 → ~40 import/collection 错误级联），
> 提交未跟踪核心模块（见 wip_inventory）即可消除那 40 errors。本地 18 failed 分流如下。

## 分流总览
| 类别 | 数量 | 处理 | 阻塞合并? |
|---|---|---|---|
| (a) 网络依赖（akshare/tushare 实连） | 7 | 加 `@requires_network` marker | 否 |
| (b) DB 未配置（PG/Mongo） | 3 | 加 `@requires_db` marker | 否 |
| (c) 在途 panel/model refactor 债 | 8 | 取决于在途工作去留（决策2） | 否（非 Phase 1 引入） |
| **真·v6 survivorship 回归** | **0** | — | — |

**关键结论：18 个失败无一由 Phase 1（survivorship/contracts/safety）已提交代码引入。**

## (a) 网络依赖 → marker（环境，非回归）
- `test_lazy_data_engine::test_akshare_sleep_between_batches` / `test_get_spot_data_network_failure_graceful`
- `test_pit_correctness::test_tushare_price_pit_strict` / `test_akshare_income_notice_date_pit` /
  `test_universe_changes_over_time` / `test_universe_cross_validate_current` /
  `test_tushare_vs_akshare_f_ann_date_alignment_post_ipo`
→ 全是实连 akshare/tushare 的测试，离线/限频即失败。B 阶段加 `requires_network`，默认套件排除。

## (b) DB 未配置 → marker（环境，非回归）
- `test_db_backend_parity::test_pnl_parity` / `test_positions_structure` / `test_forward_positions_df`
→ 需 PG/Mongo 实例。B 阶段加 `requires_db`，默认排除。

## (c) 在途 panel/model refactor 债（非 Phase 1，非 v6 回归）
| 测试 | 实测错误 | 根因 | 归类 |
|---|---|---|---|
| `test_full_panel::test_expansion_nonempty_after_2021` | `KeyError: ['relative_strength_vs_csi500_60d','volume_price_corr_20d'] not in index` | panel 缺这两个 expansion 因子列 = 在途 weekly_panel refactor 未产出 | **在途债**（c） |
| `test_full_panel::test_expansion_all_nan_2019_2020` / `test_expansion_many_nonempty_columns_modern` | 同上 KeyError 族 | 同上 expansion 因子缺列 | **在途债** |
| `test_full_panel::test_full_panel_rowcount_vs_snapshots` | `row count 6905 vs expected 28680, ratio=0.76` | panel 数据 fixture 偏少（旧/部分）vs 快照预期 | **数据 fixture 债** |
| `test_full_panel::test_split_no_date_overlap` | AssertionError | 同 full_panel 数据/切分 | **在途/数据债** |
| `test_lgbm_training::test_train_lgbm_base_produces_model_and_metrics` / `test_predict_rankings_csv_columns` | AttributeError | 在途 factor_model.py(+49) / lgbm_ranker.py(+10) 改动 | **在途债** |
| `test_backfill_realism::test_nav_ratio_reasonable` | NAV ratio 断言 | backfill/数据 fixture | **数据/在途债** |

**判定**：这 8 个 subject（build_full_panel / weekly_panel / factor_model / lgbm_ranker / 全 panel 数据）
全在**未提交的在途 refactor 集**里，**不在 Phase 1 已提交 commit**。它们引用的 expansion 因子列
（relative_strength_vs_csi500_60d / volume_price_corr_20d）是在途 panel 重构尚未产出的列。
→ **是 pre-v6 就在演进的 panel 工作债，不是 v6 survivorship 引入的回归，不阻塞 Phase 1 合并。**

## 必修清单（合并 main 前）
1. **【必修·最高】提交未跟踪核心模块**（wf_*/lake/weekly_panel/short_horizon_factors/silence_provider_logging/backfill_tushare）
   —— 否则 main 仍是坏 clone、那 40 errors 不消。**这是唯一真正阻塞合并的 must-fix。**
2. 不在"必修"内：(a)(b) 加 marker 默认排除；(c) 视决策2（在途 refactor 去留）处理——
   若在途 refactor 随 Phase 1 合，则 8 个测试需修或 mark；若 stash 单独 PR，则它们随该工作走。

## 待评审确认
- 确认必修清单 = 仅"提交未跟踪核心模块"。
- 确认 (a)(b) 走 marker；(c) 的去留绑定 wip_inventory 决策 2（在途 refactor 是否随 Phase 1 合）。
