# Walk-Forward 模块独立 Code-Review 笔记

> 评审文件：`quantmind/backtest/wf_split.py`、`wf_costs.py`、`wf_metrics.py`、`wf_gate.py`
> 测试文件：`tests/test_wf_split.py`、`tests/test_wf_costs.py`、`tests/test_wf_gate.py`
> 评审时间：2026-06-06
> 结论：**无 Blocker，放行**。Minor 4 项，不阻塞。

---

## 检查清单结论

### wf_split.py (PurgedWalkForwardSplit)

| # | 检查项 | 结论 |
|---|--------|------|
| 1 | purge 边界：train idx ≤ C−H，test idx > C+E，零交集 | ✅ `train_hi = C - H`; `test: idx > C + E`; H>0 时 train_hi < C+E 保证零交集；H=E=0 时 train≤C、test>C 紧贴无交集 |
| 2 | val ⊂ train 且满足 purge | ✅ `val = train[-n_val:]`，所有 val 元素已满足 `idx ≤ C-H` |
| 3 | E < H 时抛 ValueError | ✅ `if horizon > 0 and embargo < horizon: raise ValueError` (line 80-83) |
| 4 | rolling 下界正确 | ✅ `train_lo = C - rolling_lookback_td`（仅 rolling 模式），与 `train_hi = C-H` 组成双端窗口 |
| 5 | H=0, E=0 退化无 purge | ✅ train: idx≤C, test: idx>C, 紧贴；`test_purge_ablation_counterproof` 验证 IC 虚高 |

### wf_costs.py

| # | 检查项 | 结论 |
|---|--------|------|
| 1 | Amihud 三档单调，NaN→最贵 | ✅ 5<15<30 bp；NaN 检测用 `q!=q`（IEEE NaN），返回 `small_bp=30` |
| 2 | 印花税 2023-08-28 切换 | ✅ `d >= STAMP_DUTY_HALVING_DATE`；2023-08-27→0.1%，2023-08-28→0.05% |
| 3 | 涨跌停：科创20/主板10/创业板跨2020-08-24/ST5 | ✅ 全部正确；ST 检查在板块判断之前（优先级正确）|
| 4 | T+1：entry=as_of+1, exit=as_of+1+holding_td | ✅ `entry_fill_index = as_of_idx+1`; `exit_fill_index = as_of_idx+1+holding_td` |

### wf_metrics.py

| # | 检查项 | 结论 |
|---|--------|------|
| 1 | 胜率=相对基准超额（非绝对正收益） | ✅ `(port > bench).mean()`；`test_win_rate_is_not_daily_positive` 专项反证 |
| 2 | 分位单调性 Q1→Q5 | ✅ `pd.qcut(..., labels=False)` 给 0-4 标签；`Spearman(group_idx, group_means)`；`strictly_monotone = steps == n_groups-1` |
| 3 | IC=截面 Spearman，date 对齐 | ✅ `pd.concat([pred, realized], axis=1)` 按 ticker index 对齐；调用方保证 pred/realized 来自同一截面 |

### wf_gate.py

| # | 检查项 | 结论 |
|---|--------|------|
| 1 | H-A：decide_direction 只读 val | ✅ 仅迭代 `val_dates`；`test_direction_uses_only_val_not_oos` 把 OOS 标签取反后断言方向不变 |
| 2 | CarryForwardLabelProbe：无purge→IC虚高 | ✅ memo=最后训练日标签窗口；purge关时窗口与测试重叠→高IC；purge开时窗口与测试不重叠→IC≈0 |
| 3 | gate_meta：幸存者上界强制标注 | ✅ `survivorship_caveat` 硬编码在 meta 里，含"乐观上界 optimistic upper bound"字样 |
| 4 | evaluate_gate：PASS/FAIL 判线 | ✅ `all(bool(v))`；NaN 比较返回 False→FAIL（保守正确）；6 项全满足才 PASS |

---

## Minor 项（不阻塞放行）

### M1. `assert_no_leakage` 不做 idx-gap 完整校验（仅日期序）

- **位置：** `wf_split.py:175-184`
- `assert_no_leakage` 的注释承认"用调用方提供的索引无法在此重算"，实际只做 `max(train) < min(test)` 的日期序检查，不验证 `min(test_idx) - max(train_idx) >= H + E`。
- 完整的 gap 校验只在 `test_purge_embargo_boundaries_expanding` 从调用方侧做（`assert min(test_idx) - max(train_idx) >= 12 + 12`）。
- **建议：** 把 idx-gap 检查加进 `assert_no_leakage`，或改名为 `assert_no_date_overlap` 避免语义误导。当前行为不影响生产正确性（split() 时已保证），但 assertion 名称暗示了更强的保证。

### M2. `make_folds` 的 `oos_start` 对所有 fold 共用（晚期 fold 通常非约束，注释缺失）

- **位置：** `wf_split.py:150-170`
- 所有 fold 共享同一个 `oos_start`，但 embargo 条件（`idx > C+E`）通常比 `oos_start` 下界更强，`oos_start` 对晚期 fold 实际无约束。这不是 bug，但文档未说明。
- **建议：** docstring 补一句"oos_start 对各 fold 起全局下界；实际约束通常来自各 fold 的 embargo"。

### M3. `evaluate_gate` 中 max_drawdown NaN 时 FAIL（`float("nan") < 0.25` → False），但错误信息不显示是哪项 NaN

- **位置：** `wf_gate.py:255-265`
- `evaluate_gate` 的 `checks` 字典里，NaN 指标会静默给出 False，只能在外层通过 `checks` 逐项排查。
- **建议：** 在 verdict 字符串里附注哪些指标是 NaN（可选）。

### M4. `LGBMPredictor.fit` 中 `train_core` 极端情况：若 val_dates == train_dates，回退到全 train

- **位置：** `wf_gate.py:62`，`train_core = [d for d in train_dates if d not in set(val_dates)] or train_dates`
- 当 `n_val >= len(train_dates)`（训练数据极少时），`train_core` 会回退为 `train_dates`，模型用 val 数据训练再用 val 验证，过拟合。
- 实际 rolling_lookback_td=756 下训练折叠总有大量样本，不会触发；但应加注释或最小 train_core 大小校验。

### M5. `build_lgbm_arrays` 不支持 string 列（运行时发现）

- **位置：** `quantmind/models/factor_model.py:344`（既有代码，评审范围外）
- WHITELIST_35 包含 `exposure_industry`（string, 110 类）和 `exposure_area`（string, 32 类），
  `build_lgbm_arrays` 直接 `astype(np.float32)` 崩溃。
- **全量运行处理方式：** 脚本内排除这两列，使用 33 数值特征，并在报告中标注。
- **建议：** `LGBMPredictor.fit` 加前处理：检测 object/string 列，自动 label-encode 或 drop + 记录 warning；
  或在 `build_lgbm_arrays` 里加对 `object` 列的 label-encoding 支持（并传 `categorical_feature` 给 LGBM）。

---

## 运行参数说明（用户规格 H=4週/E=2週 修正）

用户规格 `H=4週, E=2週` 在当前代码中**无法通过校验**（E=10 < H=20 → `ValueError`）。
面板标签为 `forward_return_12d`（12 交易日），正确参数应为：

- `horizon H = 12`（与标签一致）
- `embargo E >= 12`；取 E=20（4 周 = 20 交易日，比最小值更保守）
- rolling mode，`rolling_lookback_td=756`

全量运行采用 H=12, E=20。
