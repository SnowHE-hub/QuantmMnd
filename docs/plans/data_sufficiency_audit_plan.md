# 数据充分性审计计划（周频重建训练面板）

> 类型：**只读审计**。严禁修改任何业务代码 / 数据 / 模型。
> 唯一写入目标：`docs/plans/` 下的两份 Markdown（本计划 + 结论报告）。
> 验收：`git status` 仅应新增 `docs/` 下文件。

## 0. 背景与目标

- 现状：训练面板 `data/panel/alpha_panel_v4.parquet` 为**季度采样**（30 个 `as_of` 截面，2019-03-31 → 2026-06-30）。
- 目标：改为**周采样**（每 5 个交易日一个 `as_of`），以支持 `forward_return_12d` 前瞻标签。
- 关键问题：哪些因子能从已有行情**按任意频率直接重算**，哪些依赖的外部日频序列**本地缺少日粒度**、必须回补。

## 1. 审计范围（73 个因子列）

`alpha_panel_v4.parquet` 共 75 列 = 73 因子 + 2 标签（`forward_return_21d` / `forward_return_63d`）。
按数据依赖分三类：
- **(a) 纯价格派生** —— 可从 `data/raw/alpha_prices_panel.parquet` 在任意 `as_of` 重算。
- **(b) 依赖外部日频序列** —— north_* / margin_* / beta_* / 指数相对强弱等。
- **(c) 依赖低频基本面** —— 估值（daily_basic）+ 财报指标（fina_indicator / 三大表）。

## 2. 执行步骤（全部只读）

| # | 步骤 | 方法 | 产出 |
|---|------|------|------|
| S1 | 读取面板列名 / 索引 / `as_of` 频率 | `pandas.read_parquet` | 确认 30 季度截面、73 因子 |
| S2 | 行情主源边界 | 读 `alpha_prices_panel` / `daily_prices_panel` / `index_daily_panel` 的 `trade_date` 范围与交易日历 | 确认行情到 2026-05-11 |
| S3 | 因子 → 计算函数映射 | grep `quantmind/features/*.py` 定义；读 `expansion.py`/`fundamental.py`/`technical.py`/`sentiment.py` 函数签名 | 每因子标注文件:行号 + 底层依赖 |
| S4 | 外部序列本地存量 | 枚举 `data/snapshots/<as_of>/{north_bound,margin,index_daily,hk_hold,daily_basic}.parquet`，逐文件量 `trade_date` 窗口；跨快照拼接求并集，统计日历缺口 | 区分"日频已在本地（可拼接重采样）" vs "仅季度截面物化（须回补）" |
| S5 | 基本面 PIT | 检查 `financial_indicators` / `financials_*` 是否带 `ann_date` / `f_ann_date` | 判定能否做周频 PIT 对齐 |
| S6 | 回补清单 | 汇总：序列 / Tushare 接口名 / 2019-2026 估算量 / PIT 注意；同时列出"已有日频、无需回补"项 | 一张表 |
| S7 | 标签可计算边界 | 用 `alpha_prices` 交易日历回数 12/21/63 个交易日 | 三档最后可训练 `as_of` |

## 3. 工具

- 只读 `pandas`、`grep`、`Read`。可用 codegraph（`codegraph_*`）做结构查询。
- 不运行任何写盘 / 重建脚本。

## 4. 验收清单

- [ ] `docs/plans/data_sufficiency_audit_plan.md`（本文件）已生成。
- [ ] `docs/plans/data_sufficiency_audit.md` 结论报告已生成，每条结论附文件路径+行号或实测数字。
- [ ] 回答了任务 5 个问题：因子分类 / 外部序列存量 / 基本面 PIT / 回补清单 / 标签边界。
- [ ] `git status` 仅显示 `docs/` 新增，无业务代码/数据改动。
