# T-017 · v6 周频面板在完整交易日历上重建：执行与核验报告

- 执行：Claude Code 桌面版 · Opus 5.5，2026-10-11
- 代码：分支 `t/017`（`p2_build_v6.py` 新增路径 / 日历参数，默认行为不变；`p63_2_build_factors.py` 新增 `--data-root/--panel/--out`）
- 证据：`docs/work/evidence/T-017/`

## 1. 结论

| 项 | 结果 |
|---|---|
| 新面板 | `data/panel/alpha_panel_weekly_v6_1.parquet`：1,683,972 行 × 54 列（35 因子 + 3 个标签 + 16 个短周期因子），5,406 只票，as_of 2019-01-08 → 2026-09-15，共 374 个截面 |
| 网格 | 与 `build_asof_grid(trade_cal_sse)` 完全一致（374 / 374），面板里没有网格外日期 |
| 标签 | 独立写法重算 40 个截面的 12d / 21d / 63d 标签，各约 17 万个值，**差异 > 1e-9 的为 0** |
| 因子 | 与旧 v6 在共同的 110 个截面上逐值比较：回看窗口不含任何旧数据缺日的 892,582 个值**全部相同**；差异只出现在窗口含缺日的位置（§3.3） |
| 原文件 | 构建前后 sha256 不变：`alpha_panel_weekly_v6`、`short_horizon_factors_v6`、`alpha_prices_panel_v6`、`alpha_prices_panel`、`alpha_panel_weekly_v5`、`stock_basic_full`（`sha256_originals_recheck_20261011.txt`） |
| 测试 | `tests/test_p2_build_v6_calendar.py` 6 passed |
| 耗时 | 主体 05:15 → 13:22（约 8 小时，单进程 CPU 满载）；短周期因子 13:22 → 13:33 |

## 2. 产物 `[实测]`

| 文件 | sha256 |
|---|---|
| `panel/alpha_panel_weekly_v6_1_base.parquet`（35 因子 + 标签） | 495d643e… |
| `panel/short_horizon_factors_v6_1.parquet`（1,683,972 × 16） | 3115e7ab… |
| `panel/alpha_panel_weekly_v6_1.parquet`（合并） | c7bd2c7c… |

完整哈希见 `sha256_v6_1_products.txt`。命令见 `~/c3_scratch/run_t017_build.sh`（`--prices-file data/raw/alpha_prices_panel_v6_20261009.parquet --calendar-file data/lake/trade_cal_sse.parquet --stock-basic data/lake/stock_basic_full_merged_20261009.parquet`）。

## 3. 核验 `[实测]`（`validate_v6_1.py` → `v6_1_validation.json`）

### 3.1 网格
`build_asof_grid` 以"最后一个带 63d 标签的交易日"为锚点、每 5 个交易日回退取点。因此网格的相位取决于数据末日和日历本身：
旧 v6（日历缺 18 天、数据止于 2026-05-11）与 v6_1 只有 **110 / 350 个截面日期相同**（主要在 2021-03 第一个缺日之前）。
**含义**：v6 与 v6_1 的结果应按时间窗口比较，不能按"同一 as_of"逐日对齐。T-018 的评估按窗口比较。

### 3.2 标签
独立实现：目标日 = `trade_cal_sse` 上基准日之后第 h 个交易日，label = adj_close(目标) / adj_close(基准) − 1；
只比较未退市、两端都有价格的票（退市分支由单元测试覆盖）。随机抽 40 个截面：12d 170,415 个、21d 170,451 个、63d 170,430 个值，差异全为 0。

### 3.3 因子与旧版的差异全部来自数据缺日
首轮核验按"回看 260 个交易日内是否含旧**价格表**缺日"分组：未受影响组仍有 0.82% 的值不同，集中在 beta_252d、beta_60d、
market_volatility_60d、relative_strength_vs_csi300/500、market_momentum_60d——都依赖指数序列。

追查（`explain_untouched_diffs.py` → `.json`）：旧 `index_daily` 四个指数**都缺 2019-01-02 → 2019-02-22 的 33 个交易日**（T-015 已补回）。
把旧指数缺日与旧价格缺日合并、回看取 300 个交易日（beta_252d 对齐后实际回看超过 252 日）后：

| 组 | 比较的值 | 不同 |
|---|---:|---:|
| 回看窗口不含任何旧缺日 | 892,582 | **0** |
| 回看窗口含旧缺日 | 1,187,006 | 238,118（20.1%） |

即新旧差异**全部**来自旧数据缺日，代码路径本身没有引入差异。

### 3.4 非空率
新面板各列非空率与旧版接近或略高（如 momentum_12m_skip_1m 0.852 → 0.862、volatility_1y 0.854 → 0.864、forward_return_63d 0.958 → 0.961），
原因是补回的交易日让更多票满足最小观测数。融资融券类因子非空率约 0.22～0.24（只有两融标的才有值），与旧版（约 0.21）一致。

## 4. 对主线 B 已有结论的影响（旧预测、新标签）`[实测]`

在**不重训**的前提下，只把评估标签 / 价格换成完整版本（`b_ic_label_fix.py`、`line_b_gate_complete_prices.py`）：

| 种子 | IC 均值（旧标签 → 新标签） | 可执行 NAV 年化净超额（原 → 同窗口完整价格） | IR | 闸门 |
|---|---|---|---|---|
| 63d（`ridge_full_fnd_63d_v6_seed`，7 次再平衡） | 0.1051 → 0.1029 | +2.26% → +2.22% | 0.331 → 0.329 | 未过 |
| 12d（`ridge_full_12d_v6_seed`，48 次再平衡） | 0.0697 → 0.0692 | −2.23% → −2.34% | −0.267 → −0.283 | 未过 |

标签修正对旧结论的影响很小，结论方向不变。把同一批旧预测持有到 2026-10-09（预测止于 2025-10，后段为持仓不更新）时，
63d 为 −0.96%、12d 为 −3.43%——这只说明旧预测不能外推，不是新的模型评估；在新数据上的重训与评估见 T-018。

## 5. 未做与后续

1. **基本面因子 v6_1**（`fundamental_factors_v6_1.parquet`）：由 T-018 用本分支的 `p63_2_build_factors.py --panel alpha_panel_weekly_v6_1_base.parquet` 生成。
2. **主线 B 重训**：T-018（预登记 R / X / C / P 四种方案）。
3. **默认路径未切换**：`load_trading_calendar()` 仍从旧价格表反推日历；建议验收后由 C1 开卡统一改读 `trade_cal_sse.parquet`。
4. 构建耗时 8 小时的主因是每个截面都整表读取融资融券等 lake 表（`read_lake_window`），可以改成一次读入后按窗口切片；本卡不改，以免影响逐值复现。
