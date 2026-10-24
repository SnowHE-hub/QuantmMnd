# QuantMind 数据清单（T-055）

生成时间：2026-10-24T13:25:53　·　脚本：`scripts/line_a/t55_inventory.py`

> 只读盘点。parquet 的列名/类型/行数来自 footer 元数据，日期范围只读日期列；json/md 只读最新一份做结构抽样。`_llm_cache`、`.env*`、`*secret*`、`*credential*`、`account_profiles.yaml` 未读取（缓存目录只计入体积）。

## 总览

| 根 | 体积 | 文件数 |
|---|---:|---:|
| `/home/lenovo/projects/quantmind/data` | 62.5 GB | 424,552 |
| `/home/lenovo/qm_fwd_root` | 10.3 GB | 417 |
| **合计** | **72.8 GB** | **424,969** |

- 目录数：43,414；跳过/未解析的缓存目录：`_llm_cache`×30, `_gold_llm_cache`×1, `_ds_cache`×2（约 47.6 MB，只计入总量、不解析内容）。
- 数据条目（catalog entries）：44 个，覆盖 6 个分组。
- 体积按 1024 进制显示（与 `du -h` 的 G 一致；脚本 JSON 里是精确的 `size_bytes`）。

## 怎么读这份清单

- **coverage** 的 `source` 说明范围来自哪里：文件名/目录名里的日期（含月分区），或抽样 parquet 的日期列；两者都会写进 JSON（`file_date_min/max`、`sample_date_min/max`）。
- **purpose/producer/consumers** 中标注 `推测` 的条目没有找到直接写入代码，依据是目录名、文件名或文档；`confirmed` 表示在 `qm-t035/quantmind` 或 `scripts` 里找到读写引用。
- `regenerable` 是“能否由上游数据/代码重算”的判断，**不代表可以删除**；删除前先看是否有任务/审计依赖。
- 每个条目的结构抽样最多 8 个文件模式；重点模式（如 `f1..f9`、`day_*`、`a158/base/sh`）强制纳入。

## 市场数据

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| Tushare 日频数据湖（核心表） | `data:lake/*.parquet`<br>`data:lake/bse` | 823.6 MB | 33 | 2019-01-02 → 2026-10-23 | fina_indicator_full.parquet<br>balancesheet.parquet<br>income.parquet | A 股日频行情、估值、财务、两融、北向持股的连续数据湖；以 trade_date 为键做严格 PIT 窗口读取，是回测/面板的底层来源。 |
| 数据湖每日备份 / 隔离副本 | `data:lake/_backup_*`<br>`data:lake/_quarantine_*` | 4.7 GB | 73 | 2026-10-10 → 2026-10-24 | fina_indicator_full.parquet<br>fina_indicator.parquet<br>balancesheet.parquet | lake 下每日 _backup_YYYYMMDD 与 _quarantine_* 快照副本，用于回滚和事故隔离。 |
| 财务数据时点（PIT）审计集 | `data:lake_pit` | 357.7 MB | 22,243 | 2019-04-08 → 2026-10-22 | 000903.SZ.parquet<br>600735.SH.parquet<br>300237.SZ.parquet | 按公告日/报告期保存的历史财务版本，用于 T-039 的 PIT 审计（同一报告期的多版本）。 |
| 原始行情面板与批量快照 | `data:raw` | 2.4 GB | 20 | 2018-10-08 → 2026-10-23 | alpha_prices_panel_v6_20261023.parquet<br>alpha_prices_panel_v6_20261022.parquet<br>alpha_prices_panel_v6_20261021.parquet | 从 Tushare 拉取后拼接的原始价格面板（含复权价/复权因子）、bulk_v1 批量快照和每日数据校验报告。 |
| 训练/验证因子面板（周频 v1–v6.1） | `data:panel` | 2.6 GB | 50 | 2019-01-08 → 2026-09-15 | alpha_panel_weekly_v6_1.parquet<br>alpha_panel_weekly_v6.parquet<br>alpha_panel_weekly_v6_1_base.parquet | 按季度/周频时点切片的宽表因子面板（估值、质量、动量、北向、两融等），以及 train/val/test 切分与 v6.1 扩展因子。 |
| 指数/基准日线 | `data:prices` | 19.8 MB | 4 | 2019-01-02 → 2026-05-11 | csi300_daily_ohlcv.parquet<br>index_daily.parquet<br>csi300_daily_adj_close.parquet | 沪深 300 及指数日线 OHLCV/复权收盘，用作基准与市场状态输入。 |
| Qlib 格式导出（v6 / v6.1） | `data:qlib_cn_daily_v6`<br>`data:qlib_cn_daily_v6_1` | 538.2 MB | 92,436 | 2019-01-02 → 2026-10-09 | amount.day.bin<br>low.day.bin<br>close.day.bin | 给 Qlib/深度学习流程用的 per-ticker 二进制特征与日历，由 v6/v6.1 面板导出。 |
| Qlib 格式导出（基础版） | `data:qlib_cn_daily` | 69.6 MB | 10,994 | 2019-01-02 → 2026-05-11 | amount.day.bin<br>low.day.bin<br>close.day.bin | 较早的 Qlib 日频导出，特征版本早于 v6。 |
| 前瞻/隔离用 Qlib 导出 | `data:qlib_cn_daily_fwd_20261020`<br>`data:qlib_cn_daily_fwd_20261021`<br>`data:qlib_cn_daily_fwd_20261022`<br>`data:qlib_cn_daily_fwd_20261023`<br>`data:_quarantine_20261022_qlib_fwd_g1_selection` | 1.4 GB | 232,748 | 2019-01-02 → 2026-10-23 | amount.day.bin<br>low.day.bin<br>close.day.bin | T-030 前瞻滚动时为 2026-10-20..23 各日单独 dump 的 Qlib 数据；quarantine 是 G1 选择口径出错后被隔离的一份。 |
| GPU 模型日频面板（T-030） | `data:gpu_v1/panel`<br>`data:gpu_v1/seq` | 16.6 GB | 591 | 2019-01-01 → 2026-10-31 | <month>.parquet<br>batch_<k>.parquet<br>seq_raw_f32.npy | 把周频面板拆成日频、按 (as_of, ticker) 对齐的训练用面板：Alpha158、基础量价、行业/板块、财务、事件，以及序列张量 seq_raw_f32.npy。 |
| 数据湖写入检查点 / 财务队列 | `data:lake/_checkpoints` | 1.8 MB | 33 | 2025-09-15 → 2026-10-24 | fundamentals_update_20261019.v2.json<br>fundamentals_update_20261009.v2.json<br>fundamentals_update_q_20261022_20261023_20261022.v2.json | 数据湖各序列与财务更新的检查点（每序列一个 json）和逐日 fin_queue parquet，用于断点续跑与审计。 |
| 季度 PIT 快照（研究用） | `data:snapshots` | 2.0 GB | 655 | 2019-03-29 → 2026-10-19 | universe.parquet<br>prices.parquet<br>financials_income.parquet | 按 as_of 日期冻结的一份 universe + 日线/估值/财务/两融/北向，用于可复现研究与 KB 建库。 |
| Alpha 研究universe（成分/财务合并表） | `data:alpha_universe` | 48.4 MB | 1,381 | 2019-01-02 → 2026-05-11 | alpha_prices_wide.parquet<br>fina_indicator_combined.parquet<br>alpha_daily_basic_combined.parquet | Alpha 研究用股票池（含 CSI300/500/800 标记）及合并后的财务指标/日频估值，供迁移/对照。 |
| 日频 alpha 特征切片 | `data:features` | 19.0 MB | 58 | 2019-03-31 → 2026-03-31 | factor_ic_heatmap.md<br>factor_ic_analysis.csv<br>csi300_full_panel.parquet | 按交易日切片的 alpha 特征小表（2025-12 起）与 regime 特征、因子方向/选择结果。 |
| 30 日模拟盘选股结果 | `data:sim30d` | 93.3 MB | 305 | 2025-07-01 → 2026-05-11 | <date8>.json<br>prices_all.parquet<br>daily_basic_30d.parquet | 30 日持有期模拟的持仓、已实现收益与评分（T-025 一类实验）。 |
| 策略迭代轮次记录 | `data:iteration` | 93.3 MB | 308 | 2025-07-01 → 2026-06-01 | <date8>.json<br>prices_all.parquet<br>daily_basic_30d.parquet | 自迭代优化器的每轮参数、净值、持仓与总结。 |
| 前瞻影子数据根 qm_fwd_root | `fwd:20261021..20261023/{main,a158}` | 10.3 GB | 417 | 2019-01-01 → 2026-10-23 | <month>.parquet<br>batch_<k>.parquet<br><month>.X.npy | T-030/主线 A 前瞻运行时用的“影子数据根”：只含截止某日的面板/预测/模型副本，按日复制，避免污染主数据。 |

### Tushare 日频数据湖（核心表）

- 路径：`data:lake/*.parquet`, `data:lake/bse`
- 体积 / 文件数：823.6 MB / 33
- 覆盖：2019-01-02 → 2026-10-23（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-23
- 用途：A 股日频行情、估值、财务、两融、北向持股的连续数据湖；以 trade_date 为键做严格 PIT 窗口读取，是回测/面板的底层来源。
- 生产者：scripts/build_data_lake.py、scripts/daily_update.py、scripts/backfill_tushare.py → quantmind/data/lake.py::write_lake（幂等 merge）
- 消费者：quantmind/data/lake.py::read_lake_window；quantmind/features/weekly_panel.py；quantmind/line_a/{snapshot,f6_research}.py、harness/tools_data.py
- 置信度：confirmed；可再生：否（可通过 Tushare 重下，但历史逐日回补成本高）
- 备注：lake/_backup_* 是每日副本、_checkpoints 是写入检查点，单独归类。
- 主要文件模式：`fina_indicator_full.parquet`×2; `balancesheet.parquet`×2; `income.parquet`×2; `cashflow.parquet`×2; `daily_basic.parquet`×1; `daily.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/daily.parquet` — 9,052,349 行 × 12 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `ticker`:string
  - `/home/lenovo/projects/quantmind/data/lake/daily_basic.parquet` — 8,994,519 行 × 19 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `close`:double, `turnover_rate`:double, `turnover_rate_f`:double, `volume_ratio`:double, `pe`:double, `pe_ttm`:double, `pb`:double, `ps`:double, `ps_ttm`:double, `dv_ratio`:double, `dv_ttm`:double, `total_share`:double …（共 19 列）
  - `/home/lenovo/projects/quantmind/data/lake/stock_basic.parquet` — 1,388 行 × 15 列
    - 日期列 `list_date`：1990-12-01 → 2026-04-21
    - 列：`ts_code`:string, `symbol`:string, `name`:string, `area`:string, `industry`:string, `fullname`:string, `enname`:string, `market`:string, `exchange`:string, `curr_type`:string, `list_status`:string, `list_date`:timestamp[ns], `delist_date`:timestamp[ns], `is_hs`:string …（共 15 列）
  - `/home/lenovo/projects/quantmind/data/lake/margin.parquet` — 2,079,008 行 × 11 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-22
    - 列：`trade_date`:timestamp[ns], `ts_code`:string, `rzye`:double, `rqye`:double, `rzmre`:double, `rqyl`:double, `rzche`:double, `rqchl`:double, `rqmcl`:double, `rzrqye`:double, `ticker`:string
  - `/home/lenovo/projects/quantmind/data/lake/north_bound.parquet` — 1,834 行 × 7 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`trade_date`:timestamp[ns], `ggt_ss`:string, `ggt_sz`:string, `hgt`:string, `sgt`:string, `north_money`:string, `south_money`:string
  - `/home/lenovo/projects/quantmind/data/lake/income.parquet` — 159,382 行 × 8 列
    - 日期列 `ann_date`：2019-01-02 → 2026-10-23
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `total_revenue`:double, `revenue`:double, `n_income`:double, `n_income_attr_p`:double, `operate_profit`:double
  - `/home/lenovo/projects/quantmind/data/lake/index_daily.parquet` — 7,568 行 × 11 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `close`:double, `open`:double, `high`:double, `low`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double
  - `/home/lenovo/projects/quantmind/data/lake/fina_indicator_full.parquet` — 153,818 行 × 29 列
    - 日期列 `ann_date`：2019-04-08 → 2026-10-23
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `assets_to_eqt`:double …（共 29 列）

### 数据湖每日备份 / 隔离副本

- 路径：`data:lake/_backup_*`, `data:lake/_quarantine_*`
- 体积 / 文件数：4.7 GB / 73
- 覆盖：2026-10-10 → 2026-10-24（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2019-01-02 → 2026-10-22
- 用途：lake 下每日 _backup_YYYYMMDD 与 _quarantine_* 快照副本，用于回滚和事故隔离。
- 生产者：scripts/data_pipeline/update_lake_latest.py、update_fundamentals_latest.py（首次写入前复制到 _backup_<YYYYMMDD>）
- 消费者：人工恢复/审计（正常读写路径不读）
- 置信度：inferred；可再生：是（源表仍在 lake 时可重建）
- 备注：只在文件名层归类，不做 schema 抽样。
- 主要文件模式：`fina_indicator_full.parquet`×8; `fina_indicator.parquet`×8; `balancesheet.parquet`×7; `income.parquet`×7; `cashflow.parquet`×7; `daily_basic.parquet`×6
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/fina_indicator_full.parquet` — 153,798 行 × 29 列
    - 日期列 `ann_date`：2019-04-08 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `assets_to_eqt`:double …（共 29 列）
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/fina_indicator.parquet` — 153,798 行 × 17 列
    - 日期列 `ann_date`：2019-04-08 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double …（共 17 列）
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/balancesheet.parquet` — 156,622 行 × 8 列
    - 日期列 `ann_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `total_assets`:double, `total_liab`:double, `total_hldr_eqy_exc_min_int`:double, `total_cur_assets`:double, `total_cur_liab`:double
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/income.parquet` — 159,109 行 × 8 列
    - 日期列 `ann_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `total_revenue`:double, `revenue`:double, `n_income`:double, `n_income_attr_p`:double, `operate_profit`:double
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/cashflow.parquet` — 158,951 行 × 7 列
    - 日期列 `ann_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `n_cashflow_act`:double, `c_paid_invest`:double, `c_pay_acq_const_fiolta`:double, `free_cashflow`:double
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/daily_basic.parquet` — 8,988,947 行 × 19 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `close`:double, `turnover_rate`:double, `turnover_rate_f`:double, `volume_ratio`:double, `pe`:double, `pe_ttm`:double, `pb`:double, `ps`:double, `ps_ttm`:double, `dv_ratio`:double, `dv_ttm`:double, `total_share`:double …（共 19 列）
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/daily.parquet` — 9,046,777 行 × 12 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `ticker`:string
  - `/home/lenovo/projects/quantmind/data/lake/_backup_20261024/margin.parquet` — 2,074,524 行 × 11 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-21
    - 列：`trade_date`:timestamp[ns], `ts_code`:string, `rzye`:double, `rqye`:double, `rzmre`:double, `rqyl`:double, `rzche`:double, `rqchl`:double, `rqmcl`:double, `rzrqye`:double, `ticker`:string

### 财务数据时点（PIT）审计集

- 路径：`data:lake_pit`
- 体积 / 文件数：357.7 MB / 22,243
- 覆盖：2019-04-08 → 2026-10-22（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-04-08 → 2026-10-22
- 用途：按公告日/报告期保存的历史财务版本，用于 T-039 的 PIT 审计（同一报告期的多版本）。
- 生产者：scripts/data_pipeline/fetch_fin_versions.py（--out data/lake_pit）
- 消费者：scripts/audit_*、research/pit_audit；研究性检查
- 置信度：confirmed；可再生：否（重新拉取 Tushare 可近似重建）
- 主要文件模式：`000903.SZ.parquet`×4; `600735.SH.parquet`×4; `300237.SZ.parquet`×4; `002726.SZ.parquet`×4; `300437.SZ.parquet`×4; `300398.SZ.parquet`×4
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake_pit/fina_indicator.parquet` — 253,018 行 × 32 列
    - 日期列 `ann_date`：2019-04-08 → 2026-10-22
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/lake_pit/raw/fina_indicator/000903.SZ.parquet` — 49 行 × 32 列
    - 日期列 `ann_date`：2019-04-30 → 2026-08-26
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/lake_pit/raw/fina_indicator/300437.SZ.parquet` — 50 行 × 32 列
    - 日期列 `ann_date`：2019-04-26 → 2026-10-21
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/lake_pit/raw/fina_indicator/300528.SZ.parquet` — 50 行 × 32 列
    - 日期列 `ann_date`：2019-04-29 → 2026-08-24
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/lake_pit/raw/fina_indicator/600735.SH.parquet` — 55 行 × 32 列
    - 日期列 `ann_date`：2019-04-30 → 2026-10-20
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/lake_pit/universe.json`
    - 顶层键：`built`, `lake_tables`, `stock_basic_full`, `union`, `tickers`
  - `/home/lenovo/projects/quantmind/data/lake_pit/_provenance.json`
    - 顶层键：`runs`
  - `/home/lenovo/projects/quantmind/data/lake_pit/raw/fina_indicator/300398.SZ.parquet` — 51 行 × 32 列
    - 日期列 `ann_date`：2019-04-26 → 2026-08-27
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）

### 原始行情面板与批量快照

- 路径：`data:raw`
- 体积 / 文件数：2.4 GB / 20
- 覆盖：2018-10-08 → 2026-10-23（抽样 parquet 日期列）
- 抽样文件日期列范围：2018-10-08 → 2026-10-23
- 用途：从 Tushare 拉取后拼接的原始价格面板（含复权价/复权因子）、bulk_v1 批量快照和每日数据校验报告。
- 生产者：scripts/build_daily_price_panel.py、scripts/download_data.py、scripts/validate_daily_price_panel.py
- 消费者：quantmind/data/handler.py、quantmind/features/*、line_a 评分与账本
- 置信度：confirmed；可再生：部分（可从 lake 重建，但 bulk_v1 是供应商原始抓取）
- 备注：文件名带 2026100x 后缀的是按日快照。
- 主要文件模式：`alpha_prices_panel_v6_20261023.parquet`×1; `alpha_prices_panel_v6_20261022.parquet`×1; `alpha_prices_panel_v6_20261021.parquet`×1; `alpha_prices_panel_v6_20261020.parquet`×1; `alpha_prices_panel_v6_20261009.parquet`×1; `alpha_prices_panel_v6.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel.parquet` — 2,273,529 行 × 13 列
    - 日期列 `trade_date`：2019-01-02 → 2026-05-11
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/daily_prices_panel.parquet` — 728,370 行 × 11 列
    - 日期列 `trade_date`：2018-10-08 → 2024-12-31
    - 列：`trade_date`:timestamp[ns], `ts_code`:string, `open`:double, `high`:double, `low`:double, `close`:double, `vol`:double, `amount`:double, `pct_chg`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel_v6_20261023.parquet` — 9,052,349 行 × 13 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel_v6_20261022.parquet` — 9,046,777 行 × 13 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-22
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel_v6_20261021.parquet` — 9,041,207 行 × 13 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-21
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel_v6_20261020.parquet` — 9,035,639 行 × 13 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-20
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double
  - `/home/lenovo/projects/quantmind/data/raw/price_panels_20261020.validation.json`
    - 顶层键：`end`, `v6`, `v5`, `v5_universe_size`, `v5_universe_tickers_with_rows`
  - `/home/lenovo/projects/quantmind/data/raw/price_panels_20261021.validation.json`
    - 顶层键：`end`, `v6`, `v5`, `v5_universe_size`, `v5_universe_tickers_with_rows`

### 训练/验证因子面板（周频 v1–v6.1）

- 路径：`data:panel`
- 体积 / 文件数：2.6 GB / 50
- 覆盖：2019-01-08 → 2026-09-15（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-08 → 2026-09-15
- 用途：按季度/周频时点切片的宽表因子面板（估值、质量、动量、北向、两融等），以及 train/val/test 切分与 v6.1 扩展因子。
- 生产者：scripts/build_full_panel.py、scripts/build_panel.py、scripts/build_monthly_features.py、scripts/build_train_test_split.py
- 消费者：scripts/train_*.py、scripts/champion_challenger.py、quantmind/features/weekly_panel.py、backtest/*
- 置信度：confirmed；可再生：是（由 lake/raw 重算）
- 备注：同一目录有多个版本（v1..v6_1），旧版仅为可复现保留。
- 主要文件模式：`alpha_panel_weekly_v6_1.parquet`×1; `alpha_panel_weekly_v6.parquet`×1; `alpha_panel_weekly_v6_1_base.parquet`×1; `short_horizon_factors_v6_1.parquet`×1; `short_horizon_factors_v6.parquet`×1; `fundamental_factors_v6_1.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/panel/alpha_panel_v4.parquet` — 29,406 行 × 77 列
    - 日期列 `as_of`：2019-03-31 → 2026-06-30
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 77 列）
  - `/home/lenovo/projects/quantmind/data/panel/train.parquet` — 4,770 行 × 69 列
    - 日期列 `as_of`：2019-03-31 → 2022-12-30
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 69 列）
  - `/home/lenovo/projects/quantmind/data/panel/test.parquet` — 898 行 × 69 列
    - 日期列 `as_of`：2024-03-29 → 2024-12-31
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 69 列）
  - `/home/lenovo/projects/quantmind/data/panel/alpha_panel_weekly_v6_1.parquet` — 1,683,972 行 × 56 列
    - 日期列 `as_of`：2019-01-08 → 2026-09-15
    - 列：`momentum_1m`:double, `momentum_3m`:double, `momentum_6m`:double, `momentum_12m_skip_1m`:double, `reversal_1w`:double, `volatility_3m`:double, `volatility_1y`:double, `downside_volatility_3m`:double, `max_drawdown_3m`:double, `amihud_illiquidity`:double, `volume_spike_5_30`:double, `rsi_14`:double, `bollinger_position`:double, `distance_to_52w_high`:double …（共 56 列）
  - `/home/lenovo/projects/quantmind/data/panel/alpha_panel_weekly_v6.parquet` — 1,561,617 行 × 56 列
    - 日期列 `as_of`：2019-01-08 → 2026-04-20
    - 列：`momentum_1m`:double, `momentum_3m`:double, `momentum_6m`:double, `momentum_12m_skip_1m`:double, `reversal_1w`:double, `volatility_3m`:double, `volatility_1y`:double, `downside_volatility_3m`:double, `max_drawdown_3m`:double, `amihud_illiquidity`:double, `volume_spike_5_30`:double, `rsi_14`:double, `bollinger_position`:double, `distance_to_52w_high`:double …（共 56 列）
  - `/home/lenovo/projects/quantmind/data/panel/alpha_panel_weekly_v6_1_base.parquet` — 1,683,972 行 × 40 列
    - 日期列 `as_of`：2019-01-08 → 2026-09-15
    - 列：`momentum_1m`:double, `momentum_3m`:double, `momentum_6m`:double, `momentum_12m_skip_1m`:double, `reversal_1w`:double, `volatility_3m`:double, `volatility_1y`:double, `downside_volatility_3m`:double, `max_drawdown_3m`:double, `amihud_illiquidity`:double, `volume_spike_5_30`:double, `rsi_14`:double, `bollinger_position`:double, `distance_to_52w_high`:double …（共 40 列）
  - `/home/lenovo/projects/quantmind/data/panel/short_horizon_factors_v6_1.parquet` — 1,683,972 行 × 18 列
    - 日期列 `as_of`：2019-01-08 → 2026-09-15
    - 列：`sh_resid_rev_10`:double, `sh_resid_vol_10`:double, `sh_rskew_21`:double, `sh_rkurt_21`:double, `sh_wq101`:double, `sh_wq6_corr_open_vol_10`:double, `sh_wq12_volsign_rev`:double, `sh_wq3_rkopen_rkvol_10`:double, `sh_wq42_vwap_close`:double, `sh_wq44_corr_high_rkvol_5`:double, `sh_gtja40_updown_volratio`:double, `sh_wq2_corr_dvol_intra`:double, `sh_wq5_vwap_open_close`:double, `sh_wq11_vwap_close_vol`:double …（共 18 列）
  - `/home/lenovo/projects/quantmind/data/panel/split_meta.json`
    - 顶层键：`panel_file`, `label_column`, `drop_nan_label`, `train_max_year`, `val_year`, `test_year`, `holdout_min_year`, `val_periods`, `test_periods`, `predict_periods`, `snapshots_root`, `track_a_snapshots_expected_ready`, `semantics`, `feature_columns`, `n_periods_total`, `train`, `val`, `test`, `predict`, `holdout`

### 指数/基准日线

- 路径：`data:prices`
- 体积 / 文件数：19.8 MB / 4
- 覆盖：2019-01-02 → 2026-05-11（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-05-11
- 用途：沪深 300 及指数日线 OHLCV/复权收盘，用作基准与市场状态输入。
- 生产者：scripts/build_daily_price_panel.py / download_data.py
- 消费者：quantmind/line_a/m_market.py、f8_books.py、backtest 基准
- 置信度：confirmed；可再生：是（Tushare 重下）
- 主要文件模式：`csi300_daily_ohlcv.parquet`×1; `index_daily.parquet`×1; `csi300_daily_adj_close.parquet`×1; `failed_tickers.txt`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/prices/csi300_daily_adj_close.parquet` — 400 行 × 2 列
    - 列：`600519.SH`:double, `__index_level_0__`:timestamp[ns]
  - `/home/lenovo/projects/quantmind/data/prices/csi300_daily_ohlcv.parquet` — 509,481 行 × 12 列
    - 日期列 `trade_date`：2019-01-02 → 2026-05-11
    - 列：`open`:double, `high`:double, `low`:double, `close`:double, `vol`:double, `amount`:double, `pre_close`:double, `adj_factor`:double, `adj_close`:double, `volume`:double, `trade_date`:timestamp[ns], `ts_code`:string
  - `/home/lenovo/projects/quantmind/data/prices/index_daily.parquet` — 3,560 行 × 8 列
    - 日期列 `trade_date`：2019-01-02 → 2026-05-11
    - 列：`close`:double, `pct_chg`:double, `open`:double, `high`:double, `low`:double, `vol`:double, `trade_date`:timestamp[ns], `ts_code`:string
  - `/home/lenovo/projects/quantmind/data/prices/failed_tickers.txt`

### Qlib 格式导出（v6 / v6.1）

- 路径：`data:qlib_cn_daily_v6`, `data:qlib_cn_daily_v6_1`
- 体积 / 文件数：538.2 MB / 92,436
- 覆盖：2019-01-02 → 2026-10-09（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-09
- 用途：给 Qlib/深度学习流程用的 per-ticker 二进制特征与日历，由 v6/v6.1 面板导出。
- 生产者：scripts/survivorship/p4a_dump_v6.py（复用 scripts/bakeoff/p1_dump_bin.py）；v6.1 重指向后重新 dump
- 消费者：scripts/bakeoff/*、scripts/survivorship/p4b_alpha158_v6.py（qlib.init provider_uri 指向该目录）
- 置信度：confirmed；可再生：是（由 panel 重新 dump）
- 主要文件模式：`amount.day.bin`×11554; `low.day.bin`×11554; `close.day.bin`×11554; `vwap.day.bin`×11554; `factor.day.bin`×11554; `open.day.bin`×11554
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/calendars/day.txt`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/amount.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/low.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/close.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/vwap.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/factor.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/open.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_v6_1/features/sz302132/volume.day.bin`

### Qlib 格式导出（基础版）

- 路径：`data:qlib_cn_daily`
- 体积 / 文件数：69.6 MB / 10,994
- 覆盖：2019-01-02 → 2026-05-11（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-05-11
- 用途：较早的 Qlib 日频导出，特征版本早于 v6。
- 生产者：scripts/bakeoff/p1_dump_bin.py（alpha_prices_panel → qlib bin + G1 自检）
- 消费者：scripts/bakeoff/p3a_alpha158_asof.py、p3c_sequence.py（qlib.init provider_uri 指向该目录）
- 置信度：confirmed；可再生：是
- 主要文件模式：`amount.day.bin`×1374; `low.day.bin`×1374; `close.day.bin`×1374; `vwap.day.bin`×1374; `factor.day.bin`×1374; `open.day.bin`×1374
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/calendars/day.txt`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/amount.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/low.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/close.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/vwap.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/factor.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/open.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily/features/sz301498/volume.day.bin`

### 前瞻/隔离用 Qlib 导出

- 路径：`data:qlib_cn_daily_fwd_20261020`, `data:qlib_cn_daily_fwd_20261021`, `data:qlib_cn_daily_fwd_20261022`, `data:qlib_cn_daily_fwd_20261023`, `data:_quarantine_20261022_qlib_fwd_g1_selection`
- 体积 / 文件数：1.4 GB / 232,748
- 覆盖：2019-01-02 → 2026-10-23（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-23
- 用途：T-030 前瞻滚动时为 2026-10-20..23 各日单独 dump 的 Qlib 数据；quarantine 是 G1 选择口径出错后被隔离的一份。
- 生产者：T-030/T-036 前瞻滚动时重跑 p1_dump_bin / p4a_dump_v6（推断；目录名带 20261020..23）
- 消费者：前瞻模型训练/诊断（一次性）
- 置信度：inferred；可再生：是（由当日面板重 dump）
- 备注：隔离目录保留是为了审计，不建议删除。
- 主要文件模式：`amount.day.bin`×29092; `low.day.bin`×29092; `close.day.bin`×29092; `vwap.day.bin`×29092; `factor.day.bin`×29092; `open.day.bin`×29092
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/calendars/day.txt`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/features/sz302132/amount.day.bin`
  - `/home/lenovo/projects/quantmind/data/_quarantine_20261022_qlib_fwd_g1_selection/ticker_qsym_map_fwd_20261021.json`
    - 顶层键：`SZ000001`, `SZ000002`, `SZ000004`, `SZ000005`, `SZ000006`, `SZ000007`, `SZ000008`, `SZ000009`, `SZ000010`, `SZ000011`, `SZ000012`, `SZ000014`, `SZ000016`, `SZ000017`, `SZ000018`, `SZ000019`, `SZ000020`, `SZ000021`, `SZ000023`, `SZ000025`
  - `/home/lenovo/projects/quantmind/data/_quarantine_20261022_qlib_fwd_g1_selection/qlib_fwd_20261021_g1.json`
    - 顶层键：`features_rows`, `close_recon_maxrelerr`, `close_recon_ok`, `handler_ok`, `handler_err`, `pit_note`, `g1_pass`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/features/sz302132/low.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/features/sz302132/close.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/features/sz302132/vwap.day.bin`
  - `/home/lenovo/projects/quantmind/data/qlib_cn_daily_fwd_20261023/features/sz302132/factor.day.bin`

### GPU 模型日频面板（T-030）

- 路径：`data:gpu_v1/panel`, `data:gpu_v1/seq`
- 体积 / 文件数：16.6 GB / 591
- 覆盖：2019-01-01 → 2026-10-31（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-09
- 用途：把周频面板拆成日频、按 (as_of, ticker) 对齐的训练用面板：Alpha158、基础量价、行业/板块、财务、事件，以及序列张量 seq_raw_f32.npy。
- 生产者：T-030 的 build_daily_{base,fnd,seq,sector,sh,events,a158}.py（证据脚本在 docs/work/evidence/T-030/）
- 消费者：T-030 训练（quantmind/research/gpu/data.py、dl.py）、line_a/ridge_scoring.py、line_a/pipeline.py
- 置信度：confirmed；可再生：是（由 lake/raw 重算，但耗时）
- 备注：a158 每月一个文件，_a158_parts 是分块中间件。
- 主要文件模式：`<month>.parquet`×564; `batch_<k>.parquet`×24; `seq_raw_f32.npy`×1; `market_daily.parquet`×1; `meta.json`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/a158/2026-10.parquet` — 11,117 行 × 160 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-09
    - 列：`a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float, `a158_ROC5`:float …（共 160 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/base/2026-06.parquet` — 108,419 行 × 42 列
    - 日期列 `as_of`：2026-06-01 → 2026-06-30
    - 列：`momentum_1m`:double, `momentum_3m`:double, `momentum_6m`:double, `momentum_12m_skip_1m`:double, `reversal_1w`:double, `volatility_3m`:double, `volatility_1y`:double, `downside_volatility_3m`:double, `max_drawdown_3m`:double, `amihud_illiquidity`:double, `volume_spike_5_30`:double, `rsi_14`:double, `bollinger_position`:double, `distance_to_52w_high`:double …（共 42 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/sh/2026-10.parquet` — 11,626 行 × 18 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-09
    - 列：`sh_resid_rev_10`:double, `sh_resid_vol_10`:double, `sh_rskew_21`:double, `sh_rkurt_21`:double, `sh_wq101`:double, `sh_wq6_corr_open_vol_10`:double, `sh_wq12_volsign_rev`:double, `sh_wq3_rkopen_rkvol_10`:double, `sh_wq42_vwap_close`:double, `sh_wq44_corr_high_rkvol_5`:double, `sh_gtja40_updown_volratio`:double, `sh_wq2_corr_dvol_intra`:double, `sh_wq5_vwap_open_close`:double, `sh_wq11_vwap_close_vol`:double …（共 18 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/fnd/2026-09.parquet` — 122,778 行 × 29 列
    - 日期列 `as_of`：2026-09-01 → 2026-09-30
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 29 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/sector/2026-10.parquet` — 10,360 行 × 14 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-09
    - 列：`sw_l1`:string, `sw_l2`:string, `sw_l3`:string, `ind_mom_1m`:double, `ind_mom_3m`:double, `ind_mom_6m`:double, `rel_mom_3m`:double, `rel_mom_6m`:double, `ind_dm_roe_ttm`:double, `ind_dm_pb`:double, `ind_dm_pe_ttm`:double, `ind_dm_operating_profit_yoy`:double, `as_of`:timestamp[ns], `ticker`:string
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/market/market_daily.parquet` — 1,882 行 × 23 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-09
    - 列：`csi300_ret_1`:double, `csi300_ret_5`:double, `csi300_ret_20`:double, `csi300_ret_60`:double, `csi500_ret_1`:double, `csi500_ret_5`:double, `csi500_ret_20`:double, `csi500_ret_60`:double, `chinext_ret_1`:double, `chinext_ret_5`:double, `chinext_ret_20`:double, `chinext_ret_60`:double, `csi300_vol_20`:double, `csi300_dd_252`:double …（共 23 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/event/2026-10.parquet` — 10,360 行 × 13 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-09
    - 列：`fc_pchg_mid`:double, `fc_type_score`:double, `express_np_growth`:double, `holder_num_chg`:double, `insider_net_ratio_90d`:double, `repurchase_flag_90d`:double, `unlock_ratio_next60`:double, `inst_net_buy_20d`:double, `block_discount_20d`:double, `mf_lg_net_20d`:double, `limit_up_cnt_20d`:double, `as_of`:timestamp[ns], `ticker`:string
  - `/home/lenovo/projects/quantmind/data/gpu_v1/panel/_a158_parts/batch_023.parquet` — 21,410 行 × 160 列
    - 日期列 `as_of`：2019-01-02 → 2026-10-09
    - 列：`as_of`:timestamp[ns], `a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float …（共 160 列）

### 数据湖写入检查点 / 财务队列

- 路径：`data:lake/_checkpoints`
- 体积 / 文件数：1.8 MB / 33
- 覆盖：2025-09-15 → 2026-10-24（文件名/目录名中的完整日期/年月）
- 用途：数据湖各序列与财务更新的检查点（每序列一个 json）和逐日 fin_queue parquet，用于断点续跑与审计。
- 生产者：quantmind/data/lake.py::CHECKPOINT_DIR、scripts/data_pipeline/update_lake_latest.py、update_fundamentals_latest.py
- 消费者：scripts/data_pipeline/update_*（续跑）、审计脚本
- 置信度：confirmed；可再生：是（重跑即可重建）
- 备注：体积很小，但不属于行情表，单独列出以免遗漏。
- 主要文件模式：`fundamentals_update_20261019.v2.json`×1; `fundamentals_update_20261009.v2.json`×1; `fundamentals_update_q_20261022_20261023_20261022.v2.json`×1; `fundamentals_update_q_20261009_20261020_20261009.v2.json`×1; `fundamentals.json`×1; `fina_indicator_full.json`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fin_queue_20261022_20261023.parquet` — 529 行 × 4 列
    - 列：`ts_code`:string, `fetch_since`:string, `queues`:string, `reasons`:string
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fundamentals_update_20261019.v2.json`
    - 顶层键：`version`, `done`, `partial`, `no_data`, `empty_attempts`, `failed`, `since`, `end`, `target`, `empty_retries`, `updated`, `added_rows_this_run`, `summary`
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fundamentals_update_20261009.v2.json`
    - 顶层键：`version`, `done`, `partial`, `no_data`, `empty_attempts`, `failed`, `legacy_v1_done_ignored`, `since`, `end`, `target`, `empty_retries`, `updated`, `added_rows_this_run`, `summary`
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fin_queue_20261009_20261020.parquet` — 493 行 × 4 列
    - 列：`ts_code`:string, `fetch_since`:string, `queues`:string, `reasons`:string
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fin_queue_20261021_20261022.parquet` — 369 行 × 4 列
    - 列：`ts_code`:string, `fetch_since`:string, `queues`:string, `reasons`:string
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fin_queue_20261020_20261021.parquet` — 315 行 × 4 列
    - 列：`ts_code`:string, `fetch_since`:string, `queues`:string, `reasons`:string
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fundamentals_update_q_20261022_20261023_20261022.v2.json`
    - 顶层键：`version`, `done`, `partial`, `no_data`, `empty_attempts`, `failed`, `since`, `end`, `target`, `empty_retries`, `updated`, `added_rows_this_run`, `summary`
  - `/home/lenovo/projects/quantmind/data/lake/_checkpoints/fundamentals_update_q_20261009_20261020_20261009.v2.json`
    - 顶层键：`version`, `done`, `partial`, `no_data`, `empty_attempts`, `failed`, `since`, `end`, `target`, `empty_retries`, `updated`, `added_rows_this_run`, `summary`

### 季度 PIT 快照（研究用）

- 路径：`data:snapshots`
- 体积 / 文件数：2.0 GB / 655
- 覆盖：2019-03-29 → 2026-10-19（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2023-12-31 → 2026-10-16
- 用途：按 as_of 日期冻结的一份 universe + 日线/估值/财务/两融/北向，用于可复现研究与 KB 建库。
- 生产者：scripts/backfill_snapshot_v1.py、scripts/backfill_kb_snapshots.py
- 消费者：scripts/build_full_panel.py、scripts/build_daily_price_panel.py、KB 构建与审计脚本
- 置信度：confirmed；可再生：是（按日重拉/重算）
- 备注：2019-03 起，2026 年快照最密。
- 主要文件模式：`universe.parquet`×64; `prices.parquet`×54; `financials_income.parquet`×54; `financials_balance_sheet.parquet`×53; `financials_cashflow.parquet`×52; `daily_basic.parquet`×52
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/daily_basic.parquet` — 1,368 行 × 18 列
    - 日期列 `trade_date`：2026-10-16 → 2026-10-16
    - 列：`ticker`:string, `trade_date`:timestamp[ns], `close`:double, `turnover_rate`:double, `turnover_rate_f`:double, `volume_ratio`:double, `pe`:double, `pe_ttm`:double, `pb`:double, `ps`:double, `ps_ttm`:double, `dv_ratio`:double, `dv_ttm`:double, `total_share`:double …（共 18 列）
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-19/universe.parquet` — 1,371 行 × 2 列
    - 列：`ticker`:string, `weight`:double
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/meta.json`
    - 顶层键：`as_of`, `universe_name`, `universe_index_arg`, `tickers_override_policy`, `snapshot_dir`, `rows_per_table`, `files`, `built_at`, `elapsed_sec`, `n_tickers`, `max_tickers`, `failures`, `providers`, `data_expansion_version`, `snapshot_strict`, `include_flags`, `modules`
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/prices.parquet` — 370,002 行 × 18 列
    - 日期列 `trade_date`：2025-08-22 → 2026-10-16
    - 列：`trade_date`:timestamp[ns], `ticker`:string, `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `volume`:double, `amount`:double, `pct_change`:double, `turnover_rate`:double, `turnover_rate_f`:double, `pe`:double, `pe_ttm`:double …（共 18 列）
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/financials_income.parquet` — 10,968 行 × 85 列
    - 日期列 `report_date`：2023-12-31 → 2026-09-30
    - 列：`ticker`:string, `ann_date`:timestamp[ns], `f_ann_date`:timestamp[ns], `report_date`:timestamp[ns], `report_type`:string, `comp_type`:string, `end_type`:string, `basic_eps`:double, `diluted_eps`:double, `total_revenue`:double, `revenue`:double, `int_income`:double, `prem_earned`:double, `comm_income`:double …（共 85 列）
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/financials_balance_sheet.parquet` — 10,968 行 × 152 列
    - 日期列 `report_date`：2023-12-31 → 2026-09-30
    - 列：`ticker`:string, `ann_date`:timestamp[ns], `f_ann_date`:timestamp[ns], `report_date`:timestamp[ns], `report_type`:string, `comp_type`:string, `end_type`:string, `total_share`:double, `cap_rese`:double, `undistr_porfit`:double, `surplus_rese`:double, `special_rese`:double, `money_cap`:double, `trad_asset`:double …（共 152 列）
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/financials_cashflow.parquet` — 10,968 行 × 97 列
    - 日期列 `report_date`：2023-12-31 → 2026-09-30
    - 列：`ticker`:string, `ann_date`:timestamp[ns], `f_ann_date`:timestamp[ns], `report_date`:timestamp[ns], `comp_type`:string, `report_type`:string, `end_type`:string, `net_profit`:double, `finan_exp`:double, `c_fr_sale_sg`:double, `recp_tax_rends`:double, `n_depos_incr_fi`:double, `n_incr_loans_cb`:double, `n_inc_borr_oth_fi`:double …（共 97 列）
  - `/home/lenovo/projects/quantmind/data/snapshots/2026-10-16/financial_indicators.parquet` — 10,960 行 × 108 列
    - 日期列 `report_date`：2024-09-30 → 2026-09-30
    - 列：`ticker`:string, `ann_date`:timestamp[ns], `report_date`:timestamp[ns], `eps`:double, `dt_eps`:double, `total_revenue_ps`:double, `revenue_ps`:double, `capital_rese_ps`:double, `surplus_rese_ps`:double, `undist_profit_ps`:double, `extra_item`:double, `profit_dedt`:double, `gross_margin`:double, `current_ratio`:double …（共 108 列）

### Alpha 研究universe（成分/财务合并表）

- 路径：`data:alpha_universe`
- 体积 / 文件数：48.4 MB / 1,381
- 覆盖：2019-01-02 → 2026-05-11（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-05-11
- 用途：Alpha 研究用股票池（含 CSI300/500/800 标记）及合并后的财务指标/日频估值，供迁移/对照。
- 生产者：研究期脚本（scripts/db_migration/_inspect_parquet.py 引用；原始构建脚本未在仓库）
- 消费者：scripts/db_migration/*（导入 PostgreSQL 校验）
- 置信度：inferred；可再生：是（从 lake 重算）
- 主要文件模式：`alpha_prices_wide.parquet`×1; `fina_indicator_combined.parquet`×1; `alpha_daily_basic_combined.parquet`×1; `alpha_universe.parquet`×1; `000759.SZ.parquet`×1; `000547.SZ.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/alpha_universe/alpha_universe.parquet` — 1,374 行 × 8 列
    - 日期列 `list_date`：19901201 → 2026-04-21
    - 列：`ts_code`:string, `name`:string, `industry`:string, `market`:string, `list_date`:string, `in_csi300`:bool, `in_csi500`:bool, `in_csi800`:bool
  - `/home/lenovo/projects/quantmind/data/alpha_universe/fina_indicator_combined.parquet` — 62,931 行 × 14 列
    - 日期列 `ann_date`：2019-04-09 → 2026-04-30
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double
  - `/home/lenovo/projects/quantmind/data/alpha_universe/alpha_prices_wide.parquet` — 1,762 行 × 1375 列
    - 日期列 `trade_date`：2019-01-02 → 2026-05-11
    - 列：`000001.SZ`:double, `000002.SZ`:double, `000004.SZ`:double, `000006.SZ`:double, `000007.SZ`:double, `000008.SZ`:double, `000009.SZ`:double, `000010.SZ`:double, `000011.SZ`:double, `000012.SZ`:double, `000014.SZ`:double, `000016.SZ`:double, `000017.SZ`:double, `000019.SZ`:double …（共 1375 列）
  - `/home/lenovo/projects/quantmind/data/alpha_universe/alpha_daily_basic_combined.parquet` — 37,494 行 × 10 列
    - 日期列 `trade_date`：2019-03-29 → 2026-03-31
    - 列：`ts_code`:string, `trade_date`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ratio`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate`:double, `close`:double
  - `/home/lenovo/projects/quantmind/data/alpha_universe/fundamentals/000759.SZ.parquet` — 167 行 × 25 列
    - 日期列 `ann_date`：2019-04-27 → 2026-04-30
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double …（共 25 列）
  - `/home/lenovo/projects/quantmind/data/alpha_universe/fundamentals/000547.SZ.parquet` — 181 行 × 25 列
    - 日期列 `ann_date`：2019-04-26 → 2026-04-30
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double …（共 25 列）
  - `/home/lenovo/projects/quantmind/data/alpha_universe/fundamentals/000698.SZ.parquet` — 177 行 × 25 列
    - 日期列 `ann_date`：2019-04-30 → 2026-04-25
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double …（共 25 列）
  - `/home/lenovo/projects/quantmind/data/alpha_universe/fundamentals/601118.SH.parquet` — 165 行 × 25 列
    - 日期列 `ann_date`：2019-04-30 → 2026-04-29
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `roe`:double, `roa`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double, `ebit_ps`:double, `bps`:double, `eps`:double …（共 25 列）

### 日频 alpha 特征切片

- 路径：`data:features`
- 体积 / 文件数：19.0 MB / 58
- 覆盖：2019-03-31 → 2026-03-31（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-03-31 → 2026-03-31
- 用途：按交易日切片的 alpha 特征小表（2025-12 起）与 regime 特征、因子方向/选择结果。
- 生产者：scripts/build_features.py、scripts/build_regime_features.py、scripts/analyze_factors.py
- 消费者：scripts/champion_challenger.py、run_sequential_validation.py、quantmind/regime/*
- 置信度：confirmed；可再生：是
- 主要文件模式：`factor_ic_heatmap.md`×2; `factor_ic_analysis.csv`×2; `csi300_full_panel.parquet`×1; `csi300_<yyyy>Q1_<yyyy>Q2.parquet`×1; `alpha_2026-05-22.parquet`×1; `alpha_2026-05-27.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/features/alpha_2026-10-16.parquet` — 1,371 行 × 74 列
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 74 列）
  - `/home/lenovo/projects/quantmind/data/features/regime_features.parquet` — 25 行 × 9 列
    - 日期列 `as_of`：2020-03-31 → 2026-03-31
    - 列：`csi300_63d_return`:double, `csi500_csi300_63d`:double, `chiext_csi300_63d`:double, `csi300_20d_vol`:double, `small_large_63d_spread`:double, `breadth_20d`:double, `regime_label`:int64, `regime_small_prob`:double, `as_of`:date32[day]
  - `/home/lenovo/projects/quantmind/data/features/top_factors_v4_63d.json`
    - 顶层键：`selected_factors`, `removed_factors`, `weak_watchlist`, `top_improving`, `version`, `based_on`, `removal_reason`, `n_factors`, `n_removed`
  - `/home/lenovo/projects/quantmind/data/features/top_factors_v3_63d.json/factor_ic_heatmap.md`
    - 标题：# Factor IC Heatmap（Rank IC 按截面 Spearman）
  - `/home/lenovo/projects/quantmind/data/features/top_factors_v3_63d.json/factor_ic_analysis.csv` — 72 行
    - 表头：factor, IC_mean, IC_std, ICIR, IC_positive_rate
  - `/home/lenovo/projects/quantmind/data/features/csi300_full_panel.parquet` — 6,905 行 × 69 列
    - 日期列 `as_of`：2019-03-31 → 2024-12-31
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 69 列）
  - `/home/lenovo/projects/quantmind/data/features/csi300_2019Q1_2024Q2.parquet` — 5,760 行 × 45 列
    - 日期列 `as_of`：2019-03-31 → 2024-06-30
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 45 列）
  - `/home/lenovo/projects/quantmind/data/features/alpha_2026-05-22.parquet` — 1,374 行 × 74 列
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 74 列）

### 30 日模拟盘选股结果

- 路径：`data:sim30d`
- 体积 / 文件数：93.3 MB / 305
- 覆盖：2025-07-01 → 2026-05-11（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2025-07-01 → 2026-05-11
- 用途：30 日持有期模拟的持仓、已实现收益与评分（T-025 一类实验）。
- 生产者：scripts/run_30day_sim.py、scripts/optimize_30day_results.py
- 消费者：quantmind/features/{north_flow,text_sentiment,irm_sentiment,analyst_revision}.py、quantmind/iteration/analyzer.py、watchlist/manager.py
- 置信度：confirmed；可再生：是
- 主要文件模式：`<date8>.json`×60; `prices_all.parquet`×1; `daily_basic_30d.parquet`×1; `basic_20251015.parquet`×1; `basic_20251021.parquet`×1; `basic_20251010.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/sim30d/positions.parquet` — 299 行 × 21 列
    - 日期列 `date`：2025-10-09 → 2025-11-19
    - 列：`date`:string, `day_index`:int64, `ticker`:string, `name`:string, `industry`:string, `rank`:int64, `composite_score`:double, `rating`:string, `suggested_horizon`:string, `hist_win_rate`:double, `hist_sharpe`:double, `hist_maxdd`:double, `risk_level`:string, `portfolio_1w_mean`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/sim30d/stock_returns.parquet` — 450 行 × 19 列
    - 日期列 `date`：2025-10-09 → 2025-11-19
    - 列：`date`:timestamp[ns], `ticker`:string, `in_final`:bool, `lgbm_score`:double, `composite_score`:double, `value_score`:double, `momentum_score`:double, `quality_score`:double, `technical_score`:double, `rating`:string, `suggested_horizon`:string, `hist_win_rate`:double, `hist_sharpe`:double, `hist_maxdd`:double …（共 19 列）
  - `/home/lenovo/projects/quantmind/data/sim30d/nav/20251119.json`
    - 顶层键：`entry_date`, `stocks`, `portfolio_nav`, `portfolio_final_return`, `portfolio_max_drawdown`, `with_stop_loss`, `stop_loss`, `trailing`, `n_stocks`
  - `/home/lenovo/projects/quantmind/data/sim30d/raw/prices_all.parquet` — 1,127,103 行 × 9 列
    - 日期列 `trade_date`：2025-07-01 → 2026-05-11
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `vol`:double, `amount`:double
  - `/home/lenovo/projects/quantmind/data/sim30d/raw/daily_basic_30d.parquet` — 162,969 行 × 9 列
    - 日期列 `trade_date`：2025-10-09 → 2025-11-19
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/sim30d/raw/basic_20251015.parquet` — 5,424 行 × 9 列
    - 日期列 `trade_date`：2025-10-15 → 2025-10-15
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/sim30d/raw/basic_20251021.parquet` — 5,429 行 × 9 列
    - 日期列 `trade_date`：2025-10-21 → 2025-10-21
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/sim30d/raw/trade_cal_30d.json`

### 策略迭代轮次记录

- 路径：`data:iteration`
- 体积 / 文件数：93.3 MB / 308
- 覆盖：2025-07-01 → 2026-06-01（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2025-07-01 → 2026-05-11
- 用途：自迭代优化器的每轮参数、净值、持仓与总结。
- 生产者：scripts/run_iteration.py、quantmind/iteration/*
- 消费者：quantmind/iteration/analyzer.py、comparator.py
- 置信度：confirmed；可再生：是
- 主要文件模式：`<date8>.json`×60; `prices_all.parquet`×1; `daily_basic_30d.parquet`×1; `basic_20251015.parquet`×1; `basic_20251021.parquet`×1; `basic_20251010.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/summary.json`
    - 顶层键：`generated_at`, `simulation_period`, `n_trading_days`, `total_positions`, `avg_stocks_per_day`, `horizons`, `system_evaluation`, `factor_insights`, `optimization`
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/stock_returns.parquet` — 450 行 × 19 列
    - 日期列 `date`：2025-10-09 → 2025-11-19
    - 列：`date`:timestamp[ns], `ticker`:string, `in_final`:bool, `lgbm_score`:double, `composite_score`:double, `value_score`:double, `momentum_score`:double, `quality_score`:double, `technical_score`:double, `rating`:string, `suggested_horizon`:string, `hist_win_rate`:double, `hist_sharpe`:double, `hist_maxdd`:double …（共 19 列）
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/nav/20251119.json`
    - 顶层键：`entry_date`, `stocks`, `portfolio_nav`, `portfolio_final_return`, `portfolio_max_drawdown`, `with_stop_loss`, `stop_loss`, `trailing`, `n_stocks`
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/raw/prices_all.parquet` — 1,127,103 行 × 9 列
    - 日期列 `trade_date`：2025-07-01 → 2026-05-11
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `vol`:double, `amount`:double
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/raw/daily_basic_30d.parquet` — 162,969 行 × 9 列
    - 日期列 `trade_date`：2025-10-09 → 2025-11-19
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/raw/basic_20251015.parquet` — 5,424 行 × 9 列
    - 日期列 `trade_date`：2025-10-15 → 2025-10-15
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/iteration/round_001_20260601_1329/raw/basic_20251021.parquet` — 5,429 行 × 9 列
    - 日期列 `trade_date`：2025-10-21 → 2025-10-21
    - 列：`ts_code`:string, `pe_ttm`:double, `pb`:double, `ps_ttm`:double, `dv_ttm`:double, `total_mv`:double, `circ_mv`:double, `turnover_rate_f`:double, `trade_date`:string
  - `/home/lenovo/projects/quantmind/data/iteration/config_round_001_20260601_1329.json`
    - 顶层键：`version`, `updated_at`, `source`, `holding_period`, `system1_updates`, `system2_updates`, `system3_updates`, `industry_regime`, `layer6_overweight`, `meta_learner_update`, `disclosure_contrarian_weight`, `_disclosure_comment`, `disclosure_window_days`, `north_flow_weight`, `_north_flow_comment`, `north_flow_hmm_threshold_m_yuan`, `_north_flow_hmm_comment`, `ensemble_lgbm_weight`, `ensemble_cnn_weight`, `ensemble_version`

### 前瞻影子数据根 qm_fwd_root

- 路径：`fwd:20261021..20261023/{main,a158}`
- 体积 / 文件数：10.3 GB / 417
- 覆盖：2019-01-01 → 2026-10-23（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2025-07-01 → 2026-10-23
- 用途：T-030/主线 A 前瞻运行时用的“影子数据根”：只含截止某日的面板/预测/模型副本，按日复制，避免污染主数据。
- 生产者：docs/work/evidence/T-036/run_builders_fwd.sh 一类前瞻构建脚本
- 消费者：line_a pipeline 的 model_root（quantmind/line_a/pipeline.py::Ctx.model_root）
- 置信度：confirmed；可再生：是（从主数据根按日重建）
- 备注：3 天 × (main 2.1G + a158 1.4G)，几乎全是主数据的副本。
- 主要文件模式：`<month>.parquet`×312; `batch_<k>.parquet`×72; `<month>.X.npy`×6; `<month>.meta.parquet`×6; `seq_raw_f32.npy`×3; `alpha_prices_panel_v6_20261009.parquet`×3
- 结构抽样：
  - `/home/lenovo/qm_fwd_root/20261023/fwd_root_manifest.json`
    - 顶层键：`end`, `cut`, `a158_start`, `frozen_tag`, `data_root`, `redirected_inputs`, `frozen_links`
  - `/home/lenovo/qm_fwd_root/20261023/main/data/gpu_v1/panel/sh/2026-10.parquet` — 69,840 行 × 18 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-23
    - 列：`sh_resid_rev_10`:double, `sh_resid_vol_10`:double, `sh_rskew_21`:double, `sh_rkurt_21`:double, `sh_wq101`:double, `sh_wq6_corr_open_vol_10`:double, `sh_wq12_volsign_rev`:double, `sh_wq3_rkopen_rkvol_10`:double, `sh_wq42_vwap_close`:double, `sh_wq44_corr_high_rkvol_5`:double, `sh_gtja40_updown_volratio`:double, `sh_wq2_corr_dvol_intra`:double, `sh_wq5_vwap_open_close`:double, `sh_wq11_vwap_close_vol`:double …（共 18 列）
  - `/home/lenovo/qm_fwd_root/20261023/main/data/gpu_v1/panel/base/2026-09.parquet` — 108,671 行 × 42 列
    - 日期列 `as_of`：2026-09-01 → 2026-09-30
    - 列：`momentum_1m`:double, `momentum_3m`:double, `momentum_6m`:double, `momentum_12m_skip_1m`:double, `reversal_1w`:double, `volatility_3m`:double, `volatility_1y`:double, `downside_volatility_3m`:double, `max_drawdown_3m`:double, `amihud_illiquidity`:double, `volume_spike_5_30`:double, `rsi_14`:double, `bollinger_position`:double, `distance_to_52w_high`:double …（共 42 列）
  - `/home/lenovo/qm_fwd_root/20261023/a158/data/gpu_v1/panel/_a158_parts/batch_023.parquet` — 16,648 行 × 160 列
    - 日期列 `as_of`：2025-07-01 → 2026-10-23
    - 列：`as_of`:timestamp[ns], `a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float …（共 160 列）
  - `/home/lenovo/qm_fwd_root/20261023/a158/data/gpu_v1/panel/a158/2026-10.parquet` — 66,836 行 × 160 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-23
    - 列：`a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float, `a158_ROC5`:float …（共 160 列）
  - `/home/lenovo/qm_fwd_root/20261023/main/data/gpu_v1/model/v1/2026-10.X.npy`
    - shape=[62189, 250] dtype=float16
  - `/home/lenovo/qm_fwd_root/20261023/main/data/gpu_v1/model/v1/2026-10.meta.parquet` — 62,189 行 × 7 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-23
    - 列：`as_of`:timestamp[ns], `ticker`:string, `y20`:double, `y63`:double, `top1500`:bool, `sw_l1`:int64, `seq_idx`:int64
  - `/home/lenovo/qm_fwd_root/20261023/a158/data/raw/alpha_prices_panel_v6_20261009.parquet` — 1,746,534 行 × 13 列
    - 日期列 `trade_date`：2025-07-01 → 2026-10-23
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `open`:double, `high`:double, `low`:double, `close`:double, `pre_close`:double, `change`:double, `pct_chg`:double, `vol`:double, `amount`:double, `adj_factor`:double, `adj_close`:double

## 文本与事件

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| 巨潮公告文本 / PDF | `data:lake/text` | 970.1 MB | 3,216 | 2019-01-02 → 2026-10-24 | <date8>.parquet<br>1209841615.pdf<br>1207703912.pdf | 巨潮资讯公告元数据与正文（含 PDF 抽取），供事件监控、公告分类、F6/F7 证据检索。 |
| 龙虎榜 / 事件日频表 | `data:lake/events` | 6.3 GB | 37,899 | 2018-01-02 → 2026-10-23 | <date8>.parquet<br>202002.parquet<br>201902.parquet | 龙虎榜（top_list）与其它事件明细的按日 parquet，供事件监控和 bypass 队列。 |
| 采购的事件数据（闲鱼批次） | `data:lake/events_purchased` | 2.3 GB | 10 | 2026-10-22 → 2026-10-22 | tushare_report_rc.csv<br>tushare_research_report.csv<br>research_report.parquet | 外部采购的事件/公告数据批次（2026-10-22 的 xianyu 包），用于扩展事件覆盖。 |
| 券商研报中转数据 report_rc | `data:lake/events_relay` | 5.3 MB | 59 | 2019-01-01 → 2026-10-16 | _checkpoint.json<br>part_00000.parquet<br>part_00001.parquet | T-038 的 report_rc 研报/分析师数据中转，供 X4/X5 研究。 |
| 文本/北向因子表 | `data:text` | 3.4 MB | 6 | 2024-12-31 → 2026-05-22 | announcements.parquet<br>forecast_disclosure.parquet<br>north_flow.parquet | 公告情感、披露意外、北向资金等从文本/事件源算出的因子表。 |
| 因子失效/损失信号（v4） | `data:loss_signals_v4` | 4.3 MB | 15 | 2022-02-09 → 2026-04-20 | wf_v2_oos_predictions.parquet<br>factor_health.json<br>wf_v2_neutralization.csv | Walk-forward 因子健康度、失效信号与动作计划、OOS 预测，用于因子淘汰/换手控制。 |
| DPO / 推荐结果数据 | `data:dpo`<br>`data:recommendations` | 776.4 KB | 80 | 2024-06-28 → 2026-10-16 | <date>.json<br>top10.json<br>synthetic_100.jsonl | DPO 合成偏好样本与每日 top10 推荐 JSON（展示层可直接消费）。 |
| 模拟盘状态与净值 | `data:paper_trading`<br>`data:paper_trading_20261009` | 116.7 KB | 19 | 2026-10-09 → 2026-10-09 | performance.json<br>positions.parquet<br>nav_curve_2w.parquet | 旧版模拟盘（2 周/季度前瞻）的配置、持仓、净值与结算；20261009 是某次快照。 |

### 巨潮公告文本 / PDF

- 路径：`data:lake/text`
- 体积 / 文件数：970.1 MB / 3,216
- 覆盖：2019-01-02 → 2026-10-24（文件名/目录名中的完整日期/年月）
- 用途：巨潮资讯公告元数据与正文（含 PDF 抽取），供事件监控、公告分类、F6/F7 证据检索。
- 生产者：T-031 采集脚本、scripts/ingest_research_reports.py、scripts/data_pipeline/*
- 消费者：quantmind/line_a/event_monitor.py、evidence/ann_classifier.py、f6_research.py、harness/tools_data.py
- 置信度：confirmed；可再生：部分（公告原文可重爬，PDF 重下慢）
- 主要文件模式：`<date8>.parquet`×2380; `1209841615.pdf`×1; `1207703912.pdf`×1; `1207885771.pdf`×1; `1207387715.pdf`×1; `1219329946.pdf`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo/_parts/20261023.parquet` — 1,765 行 × 18 列
    - 列：`announcementId`:string, `secCode`:string, `secName`:string, `orgId`:string, `announcementTitle`:string, `announcementTime`:int64, `adjunctUrl`:string, `adjunctSize`:int64, `adjunctType`:string, `announcementType`:string, `columnId`:string, `pageColumn`:string, `important`:null, `storageTime`:null …（共 18 列）
  - `/home/lenovo/projects/quantmind/data/lake/text/news_fwd/ths_global/20261024.parquet` — 69 行 × 7 列
    - 列：`source`:string, `item_id`:string, `publish_time_bj`:string, `title`:string, `content`:string, `url`:string, `first_seen_ts_bj`:string
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2022-04/1212959432.json`
    - 顶层键：`ann_id`, `url`, `sha256`, `bytes`, `fetched_utc`
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2020-06/1207885771.json`
    - 顶层键：`ann_id`, `url`, `sha256`, `bytes`, `fetched_utc`
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2021-04/1209841615.pdf`
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2020-05/1207703912.pdf`
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2020-06/1207885771.pdf`
  - `/home/lenovo/projects/quantmind/data/lake/text/cninfo_pdf/2020-03/1207387715.pdf`

### 龙虎榜 / 事件日频表

- 路径：`data:lake/events`
- 体积 / 文件数：6.3 GB / 37,899
- 覆盖：2018-01-02 → 2026-10-23（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-23
- 用途：龙虎榜（top_list）与其它事件明细的按日 parquet，供事件监控和 bypass 队列。
- 生产者：T-019 数据下载、scripts/daily_update.py、events/_parts 合并
- 消费者：quantmind/line_a/event_monitor.py、f6_research.py
- 置信度：confirmed；可再生：是（Tushare 重拉）
- 备注：events/_parts 是分块中间件。
- 主要文件模式：`<date8>.parquet`×14517; `202002.parquet`×6; `201902.parquet`×6; `201802.parquet`×6; `202104.parquet`×6; `202504.parquet`×6
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/events/top_list_20261023.parquet` — 143,432 行 × 15 列
    - 日期列 `trade_date`：2019-01-02 → 2026-10-23
    - 列：`trade_date`:string, `ts_code`:string, `name`:string, `close`:double, `pct_change`:double, `turnover_rate`:double, `amount`:double, `l_sell`:double, `l_buy`:double, `l_amount`:double, `net_amount`:double, `net_rate`:double, `amount_rate`:double, `float_values`:double …（共 15 列）
  - `/home/lenovo/projects/quantmind/data/lake/events/_parts/report_rc/20261020_20261023.parquet` — 1,138 行 × 21 列
    - 日期列 `report_date`：2026-10-20 → 2026-10-23
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/lake/events/_parts/pledge_stat/20261016.parquet` — 2,215 行 × 7 列
    - 日期列 `end_date`：2026-10-16 → 2026-10-16
    - 列：`ts_code`:string, `end_date`:string, `pledge_count`:int64, `unrest_pledge`:double, `rest_pledge`:double, `total_share`:double, `pledge_ratio`:double
  - `/home/lenovo/projects/quantmind/data/lake/events/_parts/repurchase/202002.parquet` — 238 行 × 9 列
    - 日期列 `ann_date`：2020-02-03 → 2020-02-29
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `proc`:string, `exp_date`:string, `vol`:double, `amount`:double, `high_limit`:double, `low_limit`:double
  - `/home/lenovo/projects/quantmind/data/lake/events/_parts/repurchase/201902.parquet` — 434 行 × 9 列
    - 日期列 `ann_date`：2019-02-01 → 2019-02-28
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `proc`:string, `exp_date`:string, `vol`:double, `amount`:double, `high_limit`:double, `low_limit`:double
  - `/home/lenovo/projects/quantmind/data/lake/events/_parts/repurchase/201802.parquet` — 152 行 × 9 列
    - 日期列 `ann_date`：2018-02-01 → 2018-02-28
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `proc`:string, `exp_date`:string, `vol`:double, `amount`:double, `high_limit`:double, `low_limit`:double
  - `/home/lenovo/projects/quantmind/data/lake/events/disclosure_date_20261020.manifest.json`
    - 顶层键：`api`, `file`, `rows`, `exact_duplicate_rows_dropped`, `columns`, `date_col`, `date_min`, `date_max`, `chunk_states`, `failed_or_truncated`, `sha256`
  - `/home/lenovo/projects/quantmind/data/lake/events/express_20261020.manifest.json`
    - 顶层键：`api`, `file`, `rows`, `exact_duplicate_rows_dropped`, `columns`, `date_col`, `date_min`, `date_max`, `chunk_states`, `failed_or_truncated`, `sha256`

### 采购的事件数据（闲鱼批次）

- 路径：`data:lake/events_purchased`
- 体积 / 文件数：2.3 GB / 10
- 覆盖：2026-10-22 → 2026-10-22（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2011-05-24 → 2026-10-16
- 用途：外部采购的事件/公告数据批次（2026-10-22 的 xianyu 包），用于扩展事件覆盖。
- 生产者：外部采购后落盘（无仓库脚本）
- 消费者：事件研究（按需读取）
- 置信度：inferred；可再生：否（采购数据）
- 主要文件模式：`tushare_report_rc.csv`×1; `tushare_research_report.csv`×1; `research_report.parquet`×1; `report_rc.parquet`×1; `report_rc_extended.parquet`×1; `report_rc_clean.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/report_rc_extended.parquet` — 2,689,037 行 × 26 列
    - 日期列 `report_date`：2011-05-24 → 2026-10-16
    - 列：`author_name`:string, `classify`:string, `create_time`:string, `eps`:double, `ev_ebitda`:double, `imp_dg`:string, `max_price`:double, `min_price`:double, `name`:string, `np`:double, `op_pr`:double, `op_rt`:double, `org_name`:string, `pe`:double …（共 26 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/raw/tushare_report_rc.csv`
    - 表头：﻿author_name, classify, create_time, eps, ev_ebitda, imp_dg, max_price, min_price, name, np, op_pr, op_rt, org_name, pe, quarter, rating, rd, report_date, report_title, report_type
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/research_report.parquet` — 317,660 行 × 11 列
    - 日期列 `trade_date`：2017-01-01 → 2026-05-24
    - 列：`abstr`:string, `author`:string, `ind_name`:string, `inst_csname`:string, `name`:string, `report_type`:string, `title`:string, `trade_date`:string, `ts_code`:string, `url`:string, `row_hash`:string
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/report_rc.parquet` — 2,639,290 行 × 24 列
    - 日期列 `report_date`：2011-05-24 → 2026-05-22
    - 列：`author_name`:string, `classify`:string, `create_time`:string, `eps`:double, `ev_ebitda`:double, `imp_dg`:string, `max_price`:double, `min_price`:double, `name`:string, `np`:double, `op_pr`:double, `op_rt`:double, `org_name`:string, `pe`:double …（共 24 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/report_rc_clean.parquet` — 2,639,015 行 × 26 列
    - 日期列 `report_date`：2011-05-24 → 2026-05-22
    - 列：`author_name`:string, `classify`:string, `create_time`:string, `eps`:double, `ev_ebitda`:double, `imp_dg`:string, `max_price`:double, `min_price`:double, `name`:string, `np`:double, `op_pr`:double, `op_rt`:double, `org_name`:string, `pe`:double …（共 26 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/_validation_official.json`
    - 顶层键：`profile`, `windows`, `column_diffs`, `identical`, `totals`
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/_provenance.json`
    - 顶层键：`source`, `files`
  - `/home/lenovo/projects/quantmind/data/lake/events_purchased/xianyu_20261022/raw/tushare_research_report.csv`
    - 表头：﻿abstr, author, ind_name, inst_csname, name, report_type, title, trade_date, ts_code, url, row_hash

### 券商研报中转数据 report_rc

- 路径：`data:lake/events_relay`
- 体积 / 文件数：5.3 MB / 59
- 覆盖：2019-01-01 → 2026-10-16（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-01 → 2026-10-16
- 用途：T-038 的 report_rc 研报/分析师数据中转，供 X4/X5 研究。
- 生产者：scripts/data_pipeline/（report_rc 采集）
- 消费者：quantmind/line_a/p4/*、f6_research
- 置信度：inferred；可再生：部分（可重拉）
- 主要文件模式：`_checkpoint.json`×1; `part_00000.parquet`×1; `part_00001.parquet`×1; `part_00005.parquet`×1; `part_00014.parquet`×1; `part_00013.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00054.parquet` — 156 行 × 21 列
    - 日期列 `report_date`：2026-05-05 → 2026-10-06
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/_checkpoint.json`
    - 顶层键：`segments`
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00000.parquet` — 9,263 行 × 21 列
    - 日期列 `report_date`：2019-01-01 → 2019-12-31
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00001.parquet` — 7,481 行 × 23 列
    - 日期列 `report_date`：2019-01-01 → 2019-12-31
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 23 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00005.parquet` — 3,894 行 × 21 列
    - 日期列 `report_date`：2026-05-02 → 2026-10-16
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00014.parquet` — 3,659 行 × 21 列
    - 日期列 `report_date`：2026-05-02 → 2026-10-16
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/_provenance.json`
    - 顶层键：`source`, `url`, `started_utc`, `api`, `layout`, `rule`
  - `/home/lenovo/projects/quantmind/data/lake/events_relay/report_rc/by_ticker/part_00013.parquet` — 3,597 行 × 21 列
    - 日期列 `report_date`：2026-05-02 → 2026-10-16
    - 列：`ts_code`:string, `name`:string, `report_date`:string, `report_title`:string, `report_type`:string, `classify`:string, `org_name`:string, `author_name`:string, `quarter`:string, `op_rt`:double, `op_pr`:double, `tp`:double, `np`:double, `eps`:double …（共 21 列）

### 文本/北向因子表

- 路径：`data:text`
- 体积 / 文件数：3.4 MB / 6
- 覆盖：2024-12-31 → 2026-05-22（抽样 parquet 日期列）
- 抽样文件日期列范围：2024-12-31 → 2026-05-22
- 用途：公告情感、披露意外、北向资金等从文本/事件源算出的因子表。
- 生产者：quantmind/features/text_sentiment.py、north_flow.py、irm_sentiment.py、analyst_revision.py 及 scripts/build_sentiment_factor.py
- 消费者：scripts/train_sentiment_agent_v3.py、面板构建
- 置信度：confirmed；可再生：是（由原始文本/北向重算）
- 主要文件模式：`announcements.parquet`×1; `forecast_disclosure.parquet`×1; `north_flow.parquet`×1; `ann_sentiment_factor.parquet`×1; `market_north_flow.parquet`×1; `disclosure_surprise_factor.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/text/announcements.parquet` — 76,908 行 × 3 列
    - 日期列 `ann_date`：2025-01-25 → 2026-05-22
    - 列：`ts_code`:string, `ann_date`:timestamp[ns], `title`:string
  - `/home/lenovo/projects/quantmind/data/text/ann_sentiment_factor.parquet` — 21,491 行 × 3 列
    - 日期列 `trade_date`：2025-01-25 → 2026-05-22
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `ann_sentiment_5d`:double
  - `/home/lenovo/projects/quantmind/data/text/north_flow.parquet` — 13,879 行 × 5 列
    - 日期列 `trade_date`：2024-12-31 → 2025-09-30
    - 列：`ts_code`:string, `trade_date`:timestamp[ns], `vol`:int64, `ratio`:double, `exchange`:string
  - `/home/lenovo/projects/quantmind/data/text/disclosure_surprise_factor.parquet` — 3 行 × 3 列
    - 日期列 `ann_date`：2025-07-15 → 2025-07-17
    - 列：`ts_code`:string, `ann_date`:timestamp[ns], `disclosure_contrarian_30d`:double
  - `/home/lenovo/projects/quantmind/data/text/forecast_disclosure.parquet` — 4,418 行 × 13 列
    - 日期列 `ann_date`：2024-11-22 → 2025-11-14
    - 列：`ts_code`:string, `ann_date`:timestamp[ns], `end_date`:string, `type`:string, `p_change_min`:double, `p_change_max`:double, `net_profit_min`:double, `net_profit_max`:double, `last_parent_net`:double, `first_ann_date`:string, `summary`:string, `change_reason`:string, `update_flag`:string
  - `/home/lenovo/projects/quantmind/data/text/market_north_flow.parquet` — 209 行 × 3 列
    - 日期列 `trade_date`：2025-01-02 → 2025-11-19
    - 列：`trade_date`:timestamp[ns], `north_money`:double, `daily_net_flow`:double

### 因子失效/损失信号（v4）

- 路径：`data:loss_signals_v4`
- 体积 / 文件数：4.3 MB / 15
- 覆盖：2022-02-09 → 2026-04-20（抽样 parquet 日期列）
- 抽样文件日期列范围：2022-02-09 → 2026-04-20
- 用途：Walk-forward 因子健康度、失效信号与动作计划、OOS 预测，用于因子淘汰/换手控制。
- 生产者：scripts/screen_short_horizon.py（short_horizon_*）、scripts/survivorship/p63_2_screen.py（fundamental_*）；其它 json 由 survivorship 训练链登记
- 消费者：quantmind/line_a/ridge_scoring.py（读 short_horizon_survivors/fundamental_survivors）、scripts/survivorship/*、db_migration
- 置信度：confirmed；可再生：是（重跑评估）
- 主要文件模式：`wf_v2_oos_predictions.parquet`×1; `factor_health.json`×1; `wf_v2_neutralization.csv`×1; `wf_v2_diag_summary.json`×1; `wf_v2_per_date_ic.csv`×1; `short_horizon_final.csv`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/history.jsonl` — 9 行
    - 元素键：`run_ts`, `ranking_loss`, `factor_decay`, `strategy_loss`, `realized_gap`, `n_actions`, `overall_health`
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/factor_health.json`
    - 顶层键：`turnover_3m_avg`, `margin_buy_intensity`, `turnover_rate_quantile`, `volume_price_corr_20d`, `amplitude_quantile`, `volume_spike_5_30`, `book_to_market`, `price_to_52w_low`, `dividend_yield_ttm`, `momentum_1m`, `turnover_acceleration`, `accruals`, `earnings_yield`, `bollinger_position`, `rsi_14`, `relative_strength_vs_csi300_60d`, `relative_strength_vs_csi500_60d`, `momentum_3m`, `pb`, `reversal_1w`
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/wf_v2_oos_predictions.parquet` — 191,744 行 × 8 列
    - 日期列 `as_of`：2022-02-09 → 2026-04-20
    - 列：`ticker`:string, `pred`:double, `realized`:double, `industry`:string, `log_mktcap`:double, `as_of`:timestamp[ms], `fold`:int64, `regime`:string
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/wf_v2_neutralization.csv` — 143 行
    - 表头：as_of, raw_ic, neu_ic, regime, n
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/wf_v2_diag_summary.json`
    - 顶层键：`params`, `overall_ic`, `overall_ir`, `n_oos_dates`, `n_direction_flips`, `fold_directions`, `fold_meta`, `regime_meta`, `regime_ic`, `regime_fold_ic`, `fold_ic`, `neutralization_overall`, `neutralization_by_regime`, `gain_importance_norm`, `univariate_ic`
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/wf_v2_per_date_ic.csv` — 143 行
    - 表头：as_of, ic, regime, fold, n
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/short_horizon_final.csv` — 48 行
    - 表头：factor, is_raw_ic, is_neut_ic, is_neut_icir, is_flip, oos_neut_ic, oos_neut_icir, maxcorr_base_is, selected, redundant_with
  - `/home/lenovo/projects/quantmind/data/loss_signals_v4/short_horizon_screen.csv` — 48 行
    - 表头：factor, is_raw_ic, is_neut_ic, is_neut_icir, is_flip, oos_neut_ic, oos_neut_icir, maxcorr_base, maxcorr_cand, keep

### DPO / 推荐结果数据

- 路径：`data:dpo`, `data:recommendations`
- 体积 / 文件数：776.4 KB / 80
- 覆盖：2024-06-28 → 2026-10-16（文件名/目录名中的完整日期/年月）
- 用途：DPO 合成偏好样本与每日 top10 推荐 JSON（展示层可直接消费）。
- 生产者：scripts/evaluate_dpo.py、scripts/backfill_recommendations.py
- 消费者：DPO 训练/评估、前端展示
- 置信度：confirmed；可再生：是
- 主要文件模式：`<date>.json`×24; `top10.json`×23; `synthetic_100.jsonl`×1; `report_300316.SZ.md`×1; `report_002142.SZ.md`×1; `report_000651.SZ.md`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/dpo/synthetic_100.jsonl` — 100 行
    - 元素键：`prompt`, `chosen`, `rejected`, `index`, `as_of`, `source`
  - `/home/lenovo/projects/quantmind/data/recommendations/2026-10-16/top10.json`
    - 顶层键：`as_of`, `top10`
  - `/home/lenovo/projects/quantmind/data/recommendations/2026-10-16.json`
    - 顶层键：`as_of`, `generated_at`, `top10`, `market_summary`, `runtime_seconds`, `step_status`
  - `/home/lenovo/projects/quantmind/data/recommendations/2026-05-26/rag_reports/report_300316.SZ.md`
    - 标题：# 300316.SZ 投资分析报告 / ## 1. 执行摘要 / ## 2. 公司概况 / ## 3. 行情与估值 / ### 最新行情（2025-03-27 至 2025-03-31） / ### 估值水平（as_of=2025-03-31） / ### 估值多期趋势分析（旧→新） / ## 4. 财务分析
  - `/home/lenovo/projects/quantmind/data/recommendations/2024-12-31/rag_reports/report_002142.SZ.md`
    - 标题：# 002142.SZ 投资分析报告 / ## 1. 执行摘要 / ## 2. 公司概况 / ## 3. 行情与估值 / ## 4. 财务分析 / ## 5. 北向资金 / ## 6. 融资融券 / ## 7. 市场环境
  - `/home/lenovo/projects/quantmind/data/recommendations/2024-12-31/rag_reports/report_000651.SZ.md`
    - 标题：# 000651.SZ 投资分析报告 / ## 1. 执行摘要 / ## 2. 公司概况 / ## 3. 行情与估值 / ## 4. 财务分析 / ## 5. 北向资金 / ## 6. 融资融券 / ## 7. 市场环境
  - `/home/lenovo/projects/quantmind/data/recommendations/2024-12-31/rag_reports/report_688599.SH.md`
    - 标题：# 688599.SH 投资分析报告 / ## 1. 执行摘要 / ## 2. 公司概况 / ## 3. 行情与估值 / ## 4. 财务分析 / ## 5. 北向资金 / ## 6. 融资融券 / ## 7. 市场环境
  - `/home/lenovo/projects/quantmind/data/recommendations/2024-12-31/rag_reports/report_300308.SZ.md`
    - 标题：# 300308.SZ 投资分析报告 / ## 1. 执行摘要 / ## 2. 公司概况 / ## 3. 行情与估值 / ## 4. 财务分析 / ## 5. 北向资金 / ## 6. 融资融券 / ## 7. 市场环境

### 模拟盘状态与净值

- 路径：`data:paper_trading`, `data:paper_trading_20261009`
- 体积 / 文件数：116.7 KB / 19
- 覆盖：2026-10-09 → 2026-10-09（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2024-06-28 → 2026-10-08
- 用途：旧版模拟盘（2 周/季度前瞻）的配置、持仓、净值与结算；20261009 是某次快照。
- 生产者：scripts/paper_trading_sim.py、scripts/settle_forward_positions.py
- 消费者：scripts/track_realized_pnl.py、前端展示
- 置信度：confirmed；可再生：部分（净值可由行情重算）
- 主要文件模式：`performance.json`×2; `positions.parquet`×2; `nav_curve_2w.parquet`×2; `nav_curve_1w.parquet`×2; `nav_curve_21d.parquet`×2; `nav_curve_3m.parquet`×2
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/paper_trading/strategy_config.json`
    - 顶层键：`version`, `updated_at`, `data_source`, `holding_period`, `position_sizing`, `factor_signals`, `risk_controls`, `optimization_triggers`
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/nav_curve_3m.parquet` — 15 行 × 3 列
    - 日期列 `as_of`：2024-06-28 → 2026-06-18
    - 列：`as_of`:timestamp[ns], `nav`:double, `period_return`:double
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/positions.parquet` — 220 行 × 14 列
    - 日期列 `as_of`：2024-06-28 → 2026-10-08
    - 列：`as_of`:timestamp[ns], `entry_date`:timestamp[ns], `ticker`:string, `predicted_rank`:int64, `predicted_score`:double, `return_1w`:double, `excess_1w_vs_300`:null, `return_2w`:double, `excess_2w_vs_300`:null, `return_21d`:double, `excess_21d_vs_300`:null, `return_3m`:double, `excess_3m_vs_300`:null, `portfolio_date`:string
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/performance.json`
    - 顶层键：`generated_at`, `total_positions`, `n_dates`, `date_range`, `horizons`, `per_date`, `signals`
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/nav_curve_2w.parquet` — 18 行 × 3 列
    - 日期列 `as_of`：2024-06-28 → 2026-07-15
    - 列：`as_of`:timestamp[ns], `nav`:double, `period_return`:double
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/nav_curve_1w.parquet` — 18 行 × 3 列
    - 日期列 `as_of`：2024-06-28 → 2026-07-15
    - 列：`as_of`:timestamp[ns], `nav`:double, `period_return`:double
  - `/home/lenovo/projects/quantmind/data/paper_trading_20261009/nav_curve_21d.parquet` — 18 行 × 3 列
    - 日期列 `as_of`：2024-06-28 → 2026-07-15
    - 列：`as_of`:timestamp[ns], `nav`:double, `period_return`:double
  - `/home/lenovo/projects/quantmind/data/paper_trading/forward_positions.json`
    - 顶层键：`generated_at`, `tracking_horizon`, `n_open_positions`, `positions`, `next_review`

## 模型与预测

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| GPU 训练矩阵 model/v1 | `data:gpu_v1/model` | 4.1 GB | 191 | 2019-01-01 → 2026-10-31 | <month>.X.npy<br><month>.meta.parquet<br>market.parquet | T-030 的模型就绪矩阵：每月 X.npy（float16 特征）+ meta.parquet（ticker/y20/y63/top1500/sw_l1/seq_idx）+ 市场特征，是 GPU 训练的直接输入。 |
| GPU 模型折外预测 | `data:gpu_v1/preds` | 3.4 GB | 885 | 2022-08-04 → 2026-10-09 | fold_<k>.parquet | 各模型/种子的 walk-forward 折外预测（fold_*.parquet），用于 ENS 组合、评估与前端展示。 |
| GPU 训练运行产物（runs） | `data:gpu_v1/runs` | 7.8 GB | 6,172 | ? → ? | k06_s0.npy<br>k07_s0.npy<br>k05_s0.npy | 训练过程的权重/日志/中间数组（每个模型一个子目录），用于复现与诊断。 |
| 冻结的前瞻模型包（ENS 20d/63d） | `data:gpu_v1/forward_models` | 212.7 MB | 46 | 2026-07-30 → 2026-10-09 | norm.npz<br>k18_s2_state.pt<br>k18_s3_state.pt | T-034/T-047 冻结的四族模型包（xgb/cat/tabm/seqfusion），主线 A 每日 ens 打分直接加载，不重训。 |
| GPU 模型诊断/归因/P0 审计 | `data:gpu_v1/attrib`<br>`data:gpu_v1/diag`<br>`data:gpu_v1/p0` | 64.8 MB | 13 | 2022-02-14 → 2026-09-15 | ridge_20d_A0.parquet<br>ridge_20d_A2.parquet<br>ridge_20d_A1.parquet | Ridge 复现、fold 一致性、资源探测、目标金丝雀等 P0 审计证据。 |
| 模型 bakeoff / survivorship 实验 | `data:bakeoff` | 2.3 GB | 99 | 2019-01-08 → 2026-09-15 | alpha158_asof_v6_1.parquet<br>alpha158_asof_v6.parquet<br>alpha158_asof.parquet | Ridge/LGBM/LSTM/Transformer 等模型在 v6/v6.1 上的批量对比：Alpha158 宽表、折外预测、排行榜与稳健性报告。 |
| 研究性数据（PIT 审计 / T-004 分解） | `data:research` | 500.9 MB | 1,862 | 2019-04-23 → 2026-08-29 | <month>.parquet<br>round_result.json<br>factor_ic.csv | PIT 审计的原始/结果、T-004 标签缺陷的按板块/序列验证中间件。 |

### GPU 训练矩阵 model/v1

- 路径：`data:gpu_v1/model`
- 体积 / 文件数：4.1 GB / 191
- 覆盖：2019-01-01 → 2026-10-31（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2019-01-02 → 2026-10-09
- 用途：T-030 的模型就绪矩阵：每月 X.npy（float16 特征）+ meta.parquet（ticker/y20/y63/top1500/sw_l1/seq_idx）+ 市场特征，是 GPU 训练的直接输入。
- 生产者：docs/work/evidence/T-030/build_model_ready.py
- 消费者：quantmind/research/gpu/data.py::ModelData、dl.py、line_a/f2f4.py
- 置信度：confirmed；可再生：是（由 gpu_v1/panel 重建，耗时）
- 备注：X.npy 是最大的可再生中间产物之一。
- 主要文件模式：`<month>.X.npy`×94; `<month>.meta.parquet`×94; `market.parquet`×1; `features.json`×1; `sw_l1_codes.json`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/model/v1/2026-10.X.npy`
    - shape=[10360, 250] dtype=float16
  - `/home/lenovo/projects/quantmind/data/gpu_v1/model/v1/2026-10.meta.parquet` — 10,360 行 × 7 列
    - 日期列 `as_of`：2026-10-08 → 2026-10-09
    - 列：`as_of`:timestamp[ns], `ticker`:string, `y20`:double, `y63`:double, `top1500`:bool, `sw_l1`:int64, `seq_idx`:int64
  - `/home/lenovo/projects/quantmind/data/gpu_v1/model/v1/features.json`
    - 顶层键：`groups`, `a158_all_nan_dropped`, `stock_features`, `market_features`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/model/v1/market.parquet` — 1,882 行 × 27 列
    - 日期列 `as_of`：2019-01-02 → 2026-10-09
    - 列：`market_momentum_60d`:double, `market_volatility_60d`:double, `market_drawdown_60d`:double, `north_bound_30d_net_inflow`:double, `csi300_ret_1`:double, `csi300_ret_5`:double, `csi300_ret_20`:double, `csi300_ret_60`:double, `csi500_ret_1`:double, `csi500_ret_5`:double, `csi500_ret_20`:double, `csi500_ret_60`:double, `chinext_ret_1`:double, `chinext_ret_5`:double …（共 27 列）
  - `/home/lenovo/projects/quantmind/data/gpu_v1/model/v1/sw_l1_codes.json`
    - 顶层键：`交通运输`, `传媒`, `公用事业`, `农林牧渔`, `医药生物`, `商贸零售`, `国防军工`, `基础化工`, `家用电器`, `建筑材料`, `建筑装饰`, `房地产`, `有色金属`, `机械设备`, `汽车`, `煤炭`, `环保`, `电力设备`, `电子`, `石油石化`

### GPU 模型折外预测

- 路径：`data:gpu_v1/preds`
- 体积 / 文件数：3.4 GB / 885
- 覆盖：2022-08-04 → 2026-10-09（抽样 parquet 日期列）
- 抽样文件日期列范围：2022-08-04 → 2026-10-09
- 用途：各模型/种子的 walk-forward 折外预测（fold_*.parquet），用于 ENS 组合、评估与前端展示。
- 生产者：T-030 训练脚本（xgb/mlp/tabm/seqfusion/cat/master/…）
- 消费者：line_a/scoring.py、export_reference_contracts.py、bakeoff 评估
- 置信度：confirmed；可再生：是（重训，代价高）
- 备注：每个成员一个目录，10 折文件。
- 主要文件模式：`fold_<k>.parquet`×885
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/preds/xgb_63d_N1s55/fold_0.parquet` — 560,193 行 × 4 列
    - 日期列 `as_of`：2022-08-04 → 2023-02-03
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double, `top1500`:bool
  - `/home/lenovo/projects/quantmind/data/gpu_v1/preds/xgb_63d_N1s55/fold_8.parquet` — 222,450 行 × 4 列
    - 日期列 `as_of`：2026-08-04 → 2026-10-09
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double, `top1500`:bool

### GPU 训练运行产物（runs）

- 路径：`data:gpu_v1/runs`
- 体积 / 文件数：7.8 GB / 6,172
- 覆盖：None → None（未发现日期线索）
- 用途：训练过程的权重/日志/中间数组（每个模型一个子目录），用于复现与诊断。
- 生产者：T-030 训练脚本
- 消费者：诊断/复现（非线上读取）
- 置信度：inferred；可再生：是（重训）
- 备注：大量 .npy 小文件。
- 主要文件模式：`k06_s0.npy`×26; `k07_s0.npy`×26; `k05_s0.npy`×26; `k04_s0.npy`×26; `k03_s0.npy`×26; `k02_s0.npy`×26
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k08_s0.npy`
    - shape=[222450] dtype=float32
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k08_record.json`
    - 顶层键：`fold`, `n_train_rows`, `n_train_days`, `train_keys_sha256`, `perm_paths_identical`, `perm_index_reproduces_labels`, `perm_index_sha256`, `y_shuffled_sha256`, `train_matrix_sha256`, `pred_matrix_sha256`, `model_file_sha256_repro`, `model_file_sha256_orig`, `orig_model_repredict_max_abs_diff_vs_orig_npy`, `repro_vs_orig_pred_max_abs_diff`, `repro_vs_orig_pred_exact_share`, `repro_vs_orig_pred_pearson`, `n_pred_rows`, `model_files_identical`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k06_s0.npy`
    - shape=[625555] dtype=float32
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_gF1F2F3/ledger/k00_final_s0.json`
    - 顶层键：`config`, `rounds`, `seed`, `n_train`, `n_train_days`, `day_stride`, `day_offset`, `seconds`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_gF1F2F3/ledger/k07_final_s0.json`
    - 顶层键：`config`, `rounds`, `seed`, `n_train`, `n_train_days`, `day_stride`, `day_offset`, `seconds`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k07_s0.npy`
    - shape=[618723] dtype=float32
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k05_s0.npy`
    - shape=[616818] dtype=float32
  - `/home/lenovo/projects/quantmind/data/gpu_v1/runs/xgb_63d_N1s00_repro/k04_s0.npy`
    - shape=[613894] dtype=float32

### 冻结的前瞻模型包（ENS 20d/63d）

- 路径：`data:gpu_v1/forward_models`
- 体积 / 文件数：212.7 MB / 46
- 覆盖：2026-07-30 → 2026-10-09（manifest.json 的 pred_days（20d 全部 / 63d 前段））
- 用途：T-034/T-047 冻结的四族模型包（xgb/cat/tabm/seqfusion），主线 A 每日 ens 打分直接加载，不重训。
- 生产者：T-034/T-047 冻结脚本
- 消费者：quantmind/line_a/scoring.py::score_ens20、pipeline.py
- 置信度：confirmed；可再生：是（重训并重新冻结）
- 备注：20d_fold18、63d_fold8。
- 主要文件模式：`norm.npz`×4; `k18_s2_state.pt`×2; `k18_s3_state.pt`×2; `k18_s1_state.pt`×2; `k18_s0_state.pt`×2; `k18_s4_state.pt`×2
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/63d_fold8/manifest.json`
    - 顶层键：`what`, `fold`, `horizon`, `cutoff`, `pred_days`, `n_pred_rows`, `utc_start`, `env`, `families`, `ens_rebuilt_vs_stored`, `features_json_sha256`, `utc_end`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/63d_fold8/seqfusion/norm.npz`
    - 数组：`mkt_mu`(26,), `mkt_sd`(26,), `seq_mu`(16,), `seq_sd`(16,), `seq_lo`(16,), `seq_hi`(16,)
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/20d_fold18/seqfusion/k18_s2_state.pt`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/20d_fold18/seqfusion/k18_s3_state.pt`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/20d_fold18/seqfusion/k18_s1_state.pt`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/20d_fold18/seqfusion/k18_s0_state.pt`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/20d_fold18/seqfusion/k18_s4_state.pt`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/forward_models/63d_fold8/seqfusion/k08_s2_state.pt`

### GPU 模型诊断/归因/P0 审计

- 路径：`data:gpu_v1/attrib`, `data:gpu_v1/diag`, `data:gpu_v1/p0`
- 体积 / 文件数：64.8 MB / 13
- 覆盖：2022-02-14 → 2026-09-15（抽样 parquet 日期列）
- 抽样文件日期列范围：2022-02-14 → 2026-09-15
- 用途：Ridge 复现、fold 一致性、资源探测、目标金丝雀等 P0 审计证据。
- 生产者：T-030 P0/P6 诊断脚本
- 消费者：审计/报告（非线上）
- 置信度：confirmed；可再生：是（重跑诊断）
- 主要文件模式：`ridge_20d_A0.parquet`×1; `ridge_20d_A2.parquet`×1; `ridge_20d_A1.parquet`×1; `ridge_20d_A0r.parquet`×1; `ridge_repro_20d.parquet`×1; `ridge_repro_63d.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/gpu_v1/p0/ridge_repro_20d.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/gpu_v1/p0/consistency_v2.json`
    - 顶层键：`base`, `labels`, `sh`, `fnd`, `a158`, `sector`, `event`, `all_ok`, `partial`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/attrib/ridge_20d_A2.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/gpu_v1/attrib/ridge_20d_A0.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/gpu_v1/attrib/ridge_20d_A1.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/gpu_v1/attrib/ridge_20d_A0r.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/gpu_v1/p0/consistency.json`
    - 顶层键：`base`, `labels`, `sh`, `fnd`, `a158`, `sector`, `event`, `all_ok`, `partial`
  - `/home/lenovo/projects/quantmind/data/gpu_v1/p0/ridge_repro_63d.parquet` — 1,000,377 行 × 3 列
    - 日期列 `as_of`：2022-08-09 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double

### 模型 bakeoff / survivorship 实验

- 路径：`data:bakeoff`
- 体积 / 文件数：2.3 GB / 99
- 覆盖：2019-01-08 → 2026-09-15（抽样 parquet 日期列）
- 抽样文件日期列范围：2019-01-08 → 2026-09-15
- 用途：Ridge/LGBM/LSTM/Transformer 等模型在 v6/v6.1 上的批量对比：Alpha158 宽表、折外预测、排行榜与稳健性报告。
- 生产者：scripts/bakeoff/p3a_alpha158_asof.py、p3b_tabular.py、scripts/survivorship/*
- 消费者：scripts/build_model_comparison_md.py、champion_challenger.py、报告
- 置信度：confirmed；可再生：是（重算/重训）
- 备注：2.4G，主要为 alpha158 宽表。
- 主要文件模式：`alpha158_asof_v6_1.parquet`×1; `alpha158_asof_v6.parquet`×1; `alpha158_asof.parquet`×1; `ridge_20d_v6_1_C.parquet`×1; `ridge_20d_v6_1_EVT.parquet`×1; `ridge_20d_v6_1_BASE.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/bakeoff/alpha158_asof_v6_1.parquet` — 1,814,158 行 × 160 列
    - 日期列 `as_of`：2019-01-08 → 2026-09-15
    - 列：`a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float, `a158_ROC5`:float …（共 160 列）
  - `/home/lenovo/projects/quantmind/data/bakeoff/leaderboard.csv` — 5 行
    - 表头：model, fmt, period, refit, seed, raw_ic, neut_ic, neut_icir, neut_ic_posfrac, net_excess_cost, annualized_turnover, max_drawdown, win_rate, n_rebalances
  - `/home/lenovo/projects/quantmind/data/bakeoff/preds_v6_1_t028/ridge_20d_v6_1_EVT.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/bakeoff/alpha158_asof_v6.parquet` — 1,683,466 行 × 160 列
    - 日期列 `as_of`：2019-01-08 → 2026-04-20
    - 列：`a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float, `a158_ROC5`:float …（共 160 列）
  - `/home/lenovo/projects/quantmind/data/bakeoff/alpha158_asof.parquet` — 452,347 行 × 160 列
    - 日期列 `as_of`：2019-01-08 → 2026-04-20
    - 列：`a158_KMID`:float, `a158_KLEN`:float, `a158_KMID2`:float, `a158_KUP`:float, `a158_KUP2`:float, `a158_KLOW`:float, `a158_KLOW2`:float, `a158_KSFT`:float, `a158_KSFT2`:float, `a158_OPEN0`:float, `a158_HIGH0`:float, `a158_LOW0`:float, `a158_VWAP0`:float, `a158_ROC5`:float …（共 160 列）
  - `/home/lenovo/projects/quantmind/data/bakeoff/preds_v6_1/ridge_20d_v6_1_C.parquet` — 1,108,869 行 × 3 列
    - 日期列 `as_of`：2022-02-14 → 2026-09-15
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/bakeoff/ticker_qsym_map_fwd_20261023.json`
    - 顶层键：`SZ000001`, `SZ000002`, `SZ000004`, `SZ000005`, `SZ000006`, `SZ000007`, `SZ000008`, `SZ000009`, `SZ000010`, `SZ000011`, `SZ000012`, `SZ000014`, `SZ000016`, `SZ000017`, `SZ000018`, `SZ000019`, `SZ000020`, `SZ000021`, `SZ000023`, `SZ000025`
  - `/home/lenovo/projects/quantmind/data/bakeoff/ticker_qsym_map_fwd_20261022.json`
    - 顶层键：`SZ000001`, `SZ000002`, `SZ000004`, `SZ000005`, `SZ000006`, `SZ000007`, `SZ000008`, `SZ000009`, `SZ000010`, `SZ000011`, `SZ000012`, `SZ000014`, `SZ000016`, `SZ000017`, `SZ000018`, `SZ000019`, `SZ000020`, `SZ000021`, `SZ000023`, `SZ000025`

### 研究性数据（PIT 审计 / T-004 分解）

- 路径：`data:research`
- 体积 / 文件数：500.9 MB / 1,862
- 覆盖：2019-04-23 → 2026-08-29（抽样 fnd_evaldates/财务原始文件 的 ann_date/as_of）
- 用途：PIT 审计的原始/结果、T-004 标签缺陷的按板块/序列验证中间件。
- 生产者：scripts/audit_*、docs/work/evidence/T-004
- 消费者：审计报告/研究
- 置信度：inferred；可再生：部分
- 主要文件模式：`<month>.parquet`×168; `round_result.json`×24; `factor_ic.csv`×24; `ic_per_period.csv`×24; `model.pkl`×12; `603003.SH.parquet`×8
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/research/pit_audit/fnd_evaldates/pit_v1/2026-08.parquet` — 23,360 行 × 29 列
    - 日期列 `as_of`：2026-08-04 → 2026-08-25
    - 列：`pe_ttm`:double, `pb`:double, `ps_ttm`:double, `book_to_market`:double, `earnings_yield`:double, `dividend_yield_ttm`:double, `log_market_cap`:double, `log_circ_market_cap`:double, `roe_ttm`:double, `roa_ttm`:double, `gross_margin`:double, `net_margin`:double, `debt_to_assets`:double, `current_ratio`:double …（共 29 列）
  - `/home/lenovo/projects/quantmind/data/research/t004/seqval_v4_2/round_7/round_result.json`
    - 顶层键：`round_id`, `name`, `note`, `n_train_periods`, `train_end`, `test_dates`, `direction`, `ic_mean`, `icir`, `ic_per_period`, `champion_ic_before`, `champion_decision`, `loss_signals`, `top5_factors_by_icir`, `bottom5_factors_by_icir`
  - `/home/lenovo/projects/quantmind/data/research/t004/seqval_v4_2/round_7/factor_ic.csv` — 39 行
    - 表头：factor, ic_mean, ic_std, icir, positive_rate, n_periods
  - `/home/lenovo/projects/quantmind/data/research/pit_audit/x13_cases/raw/fina_indicator/603003.SH.parquet` — 47 行 × 32 列
    - 日期列 `ann_date`：2019-04-26 → 2026-08-29
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/research/pit_audit/x13_cases/raw/fina_indicator/002639.SZ.parquet` — 51 行 × 32 列
    - 日期列 `ann_date`：2019-04-25 → 2026-08-24
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/research/pit_audit/x13_cases/raw/fina_indicator/600529.SH.parquet` — 49 行 × 32 列
    - 日期列 `ann_date`：2019-04-23 → 2026-08-25
    - 列：`ts_code`:string, `ann_date`:string, `end_date`:string, `update_flag`:string, `roe`:double, `roa`:double, `roe_yearly`:double, `roa_yearly`:double, `grossprofit_margin`:double, `netprofit_margin`:double, `assets_turn`:double, `current_ratio`:double, `quick_ratio`:double, `debt_to_assets`:double …（共 32 列）
  - `/home/lenovo/projects/quantmind/data/research/pit_audit/t038_cases/results/summary.json`
    - 顶层键：`sample`, `analysed`, `income`, `balancesheet`, `cashflow`, `fina_indicator`, `fina_avail_how`, `fina_avail_how_multi_version`, `fina_avail_ambiguous_multi_version`
  - `/home/lenovo/projects/quantmind/data/research/t004/seqval_v4_2/round_7/ic_per_period.csv` — 2 行
    - 表头：as_of, rank_ic

## 主线 A 每日产物

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| 每日运行产物 runs/<as_of>/ | `data:line_a_v2/runs` | 12.7 MB | 162 | 2026-10-12 → 2026-10-23 | f3_ens_20d_<hash>.parquet<br>f1_instruments_<hash>.parquet<br>f3_ridge_63d_<hash>.parquet | 主线 A v2 每个交易日的 F0–F9 产物：市场状态、证券状态、支持域、Ridge/ENS 打分、候选、风险卡、bypass 事件队列、模板报告与运行清单。文件名带任务键哈希，输入/代码变化不覆盖旧产物。 |
| 账本 books/<as_of>/ | `data:line_a_v2/books` | 1.5 MB | 222 | 2026-10-14 → 2026-10-23 | <strategy>__all_boards__targets.parquet<br><strategy>__no_bse__targets.parquet<br><strategy>__main_only__targets.parquet | 前瞻网格账本：各参考/挑战策略的 NAV、容量、目标持仓、拒单与汇总，F8 账本输出。 |
| 研究报告 research/daily/<as_of>/ | `data:line_a_v2/research` | 40.0 MB | 573 | 2024-04-10 → 2026-10-23 | A2_<hash>.json<br>A6_<hash>.json<br>A1_<hash>.json | F6 研究案例选择/角色运行、F7 证据 claim 与风险旗标、F9 研究报告与摘要；cases/ 下保留每只股票的 A1–A6 角色 JSON。 |
| F0 输入快照 snapshots/<as_of>/ | `data:line_a_v2/snapshots` | 115.1 KB | 13 | 2026-10-12 → 2026-10-23 | snapshot_b21c544299cd69ff.json<br>snapshot_e8d5e18c0c530494.json<br>snapshot_22e749c88293b7cc.json | F0 快照清单：记录当日输入文件的内容哈希、行数、水位与缺失/过期项；是任务键与可复现性的锚点。 |
| P5 扩展域与择时实验 | `data:line_a_v2/p5` | 15.0 MB | 36 | 2022-01-10 → 2026-10-09 | nav_20d_R.parquet<br>nav_63d_R.parquet<br>forecasts.parquet | P5 X2（1500 名外扩展域）与 X12（择时/小盘配额）的净值、容量、状态与汇总。 |
| Ridge 参考臂存档 | `data:line_a_v2/refs` | 70.6 KB | 2 | ? → ? | ridge_20d_lastfold.npz<br>ridge_63d_lastfold.npz | 冻结的 Ridge 20d/63d 最后一折模型（系数/特征名/标准化参数），每日 f3_ridge 直接加载。 |
| 任务表 jobs.sqlite | `data:line_a_v2/jobs.sqlite` | 148.0 KB | 1 | ? → ? | jobs.sqlite | JobStore 持久化：每个 (stage, as_of, snapshot, code, model) 任务键的状态、重试、产物路径与 sha256；支持断点续跑与并发互斥。 |

### 每日运行产物 runs/<as_of>/

- 路径：`data:line_a_v2/runs`
- 体积 / 文件数：12.7 MB / 162
- 覆盖：2026-10-12 → 2026-10-23（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2026-10-23 → 2026-10-23
- 用途：主线 A v2 每个交易日的 F0–F9 产物：市场状态、证券状态、支持域、Ridge/ENS 打分、候选、风险卡、bypass 事件队列、模板报告与运行清单。文件名带任务键哈希，输入/代码变化不覆盖旧产物。
- 生产者：scripts/line_a/daily_shadow.sh → quantmind/line_a/pipeline.py::run_day（m_market/f1/f2f4/ridge_scoring/scoring/event_monitor/risk_card/report）
- 消费者：line_a/pipeline.py::run_books/run_research、line_a/mcp_server.py、前端
- 置信度：confirmed；可再生：是（同输入可重跑；但任务键会变）
- 备注：同一日期有多个哈希文件 = 多次运行/输入变化；不要手工删除。
- 主要文件模式：`f3_ens_20d_<hash>.parquet`×16; `f1_instruments_<hash>.parquet`×16; `f3_ridge_63d_<hash>.parquet`×16; `f3_ridge_20d_<hash>.parquet`×16; `f2_domains_<hash>.parquet`×16; `f4_candidates_<hash>.parquet`×16
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f1_instruments_6cc8682ad544.parquet` — 5,224 行 × 22 列
    - 列：`board`:string, `listed_td`:int64, `name`:string, `is_st`:bool, `delisting_consolidation`:bool, `suspended`:bool, `close`:double, `has_bar`:bool, `limit_up_close`:bool, `limit_down_close`:bool, `one_line_up`:bool, `one_line_down`:bool, `c_st`:bool, `c_listed_lt_min`:bool …（共 22 列）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f2_domains_66c67c081d77.parquet` — 5,185 行 × 4 列
    - 列：`top1500`:bool, `nan_share`:double, `domain`:string, `ticker`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f3_ens_20d_a24c01909404.parquet` — 5,185 行 × 7 列
    - 日期列 `as_of`：2026-10-23 → 2026-10-23
    - 列：`as_of`:timestamp[ns], `ticker`:string, `xgb`:double, `cat`:double, `tabm`:double, `seqfusion`:double, `ENS`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f3_ridge_20d_bd11a9c58b6f.parquet` — 5,185 行 × 3 列
    - 日期列 `as_of`：2026-10-23 → 2026-10-23
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f4_candidates_29a6d14a90b4.parquet` — 1,897 行 × 9 列
    - 列：`ticker`:string, `horizon`:string, `reasons`:string, `domain`:string, `priority`:int64, `ens20`:double, `ridge20`:double, `ens63`:double, `ridge63`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/f5c_risk_cards_95aee199ba59.parquet` — 963 行 × 29 列
    - 日期列 `as_of`：2026-10-23 → 2026-10-23
    - 列：`ticker`:string, `as_of`:string, `evidence_id`:string, `last_bar`:string, `industry_tushare`:string, `log_total_mv`:double, `beta_252`:double, `vol_60_ann`:double, `mom_20`:double, `mom_60`:double, `mom_250`:double, `adv20_yuan`:double, `gap_p50_250`:double, `gap_p95_250`:double …（共 29 列）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/bypass_events_57826049bb6a.parquet` — 318 行 × 8 列
    - 列：`ticker`:string, `importance`:string, `n_events`:int64, `rules`:string, `evidence_ids`:list<element: string>, `summaries`:list<element: string>, `rule_version`:string, `__index_level_0__`:int64
  - `/home/lenovo/projects/quantmind/data/line_a_v2/runs/2026-10-23/day_f12629ac212e.json`
    - 顶层键：`as_of`, `code`, `snapshot`, `outputs`, `unavailable`

### 账本 books/<as_of>/

- 路径：`data:line_a_v2/books`
- 体积 / 文件数：1.5 MB / 222
- 覆盖：2026-10-14 → 2026-10-23（抽样 parquet 日期列）
- 抽样文件日期列范围：2026-10-14 → 2026-10-23
- 用途：前瞻网格账本：各参考/挑战策略的 NAV、容量、目标持仓、拒单与汇总，F8 账本输出。
- 生产者：quantmind/line_a/pipeline.py::run_books（f8_books 计算）
- 消费者：pipeline.py::books_lines（F9 报告）、前端展示
- 置信度：confirmed；可再生：是（按日重算）
- 主要文件模式：`<strategy>__all_boards__targets.parquet`×20; `<strategy>__no_bse__targets.parquet`×20; `<strategy>__main_only__targets.parquet`×20; `<strategy>__main_only__nav.parquet`×20; `<strategy>__no_bse__nav.parquet`×20; `<strategy>__all_boards__nav.parquet`×20
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/books_summary.json`
    - 顶层键：`as_of`, `grid`, `inputs`, `prices`, `prices_sha256`, `arms_missing_scores`, `challengers`, `books`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/holdings_latest.parquet` — 864 行 × 1 列
    - 列：`ticker`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__main_only__nav.parquet` — 8 行 × 10 列
    - 日期列 `date`：2026-10-14 → 2026-10-23
    - 列：`date`:timestamp[ns], `nav_net`:double, `nav_gross`:double, `nav_bench`:double, `cash`:double, `n_positions`:int64, `cum_cost`:double, `daily_ret_net`:double, `excess_net`:double, `cum_return_net`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__main_only__capacity.parquet` — 3 行 × 8 列
    - 日期列 `as_of`：2026-10-14 → 2026-10-14
    - 列：`as_of`:date32[day], `aum`:double, `n_targets`:int64, `n_unbuyable_one_lot`:int64, `cash_residual_share`:double, `max_participation`:double, `n_over_participation`:int64, `min_commission_drag_bp`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__all_boards__targets.parquet` — 1 行 × 8 列
    - 日期列 `as_of`：2026-10-14 → 2026-10-14
    - 列：`as_of`:string, `n_scored`:int64, `n_universe`:int64, `n_candidates`:int64, `n_target`:int64, `nav_at_decision`:double, `target_weight_each`:double, `targets`:list<element: string>
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__no_bse__targets.parquet` — 1 行 × 8 列
    - 日期列 `as_of`：2026-10-14 → 2026-10-14
    - 列：`as_of`:string, `n_scored`:int64, `n_universe`:int64, `n_candidates`:int64, `n_target`:int64, `nav_at_decision`:double, `target_weight_each`:double, `targets`:list<element: string>
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__main_only__targets.parquet` — 1 行 × 8 列
    - 日期列 `as_of`：2026-10-14 → 2026-10-14
    - 列：`as_of`:string, `n_scored`:int64, `n_universe`:int64, `n_candidates`:int64, `n_target`:int64, `nav_at_decision`:double, `target_weight_each`:double, `targets`:list<element: string>
  - `/home/lenovo/projects/quantmind/data/line_a_v2/books/2026-10-23/ens_q_ext_63d__no_bse__nav.parquet` — 8 行 × 10 列
    - 日期列 `date`：2026-10-14 → 2026-10-23
    - 列：`date`:timestamp[ns], `nav_net`:double, `nav_gross`:double, `nav_bench`:double, `cash`:double, `n_positions`:int64, `cum_cost`:double, `daily_ret_net`:double, `excess_net`:double, `cum_return_net`:double

### 研究报告 research/daily/<as_of>/

- 路径：`data:line_a_v2/research`
- 体积 / 文件数：40.0 MB / 573
- 覆盖：2024-04-10 → 2026-10-23（文件名日期 ∪ 抽样日期列）
- 抽样文件日期列范围：2026-10-23 → 2026-10-23
- 用途：F6 研究案例选择/角色运行、F7 证据 claim 与风险旗标、F9 研究报告与摘要；cases/ 下保留每只股票的 A1–A6 角色 JSON。
- 生产者：quantmind/line_a/f6_research.py、pipeline.py::run_research
- 消费者：line_a/mcp_server.py、前端研究页
- 置信度：confirmed；可再生：是（重跑研究，LLM 成本另计）
- 备注：_llm_cache 为 LLM 缓存，未纳入 schema 抽样。
- 主要文件模式：`A2_<hash>.json`×81; `A6_<hash>.json`×80; `A1_<hash>.json`×80; `case_<hash>.json`×80; `A3_<hash>.json`×79; `A4_<hash>.json`×77
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f6_cases.parquet` — 40 行 × 13 列
    - 日期列 `as_of`：2026-10-23 → 2026-10-23
    - 列：`ticker`:string, `category`:string, `roles_run`:string, `roles_reuse`:string, `as_of`:string, `cost_rmb`:double, `role_paths`:string, `role_as_of_map`:string, `case_path`:string, `error`:null, `a6_verdicts`:string, `check_status`:string, `status`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f6_plan.parquet` — 40 行 × 6 列
    - 列：`ticker`:string, `category`:string, `fresh`:bool, `last`:string, `roles_run`:string, `roles_reuse`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f7_claims.parquet` — 3,315 行 × 11 列
    - 列：`ticker`:string, `ref`:string, `role_as_of`:string, `type`:string, `text`:string, `value`:double, `unit`:string, `period`:string, `check`:string, `a6`:string, `column`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f7_risk_flags.parquet` — 827 行 × 4 列
    - 列：`ticker`:string, `role`:string, `role_as_of`:string, `flag`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f9_research.md`
    - 标题：# 影子研究报告 · 2026-10-23 / ## 案例一览 / ## 688383.SH 新益昌（持仓 × 重大事件） / ## 688507.SH 索辰科技（持仓 × 重大事件） / ## 002614.SZ 奥佳华（持仓 × 重大事件）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/f9_summaries.json`
    - 元素键：`ticker`, `n_points`, `n_dropped`, `dropped`, `mode`, `errors`, `first_pass`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/status.json`
    - 顶层键：`as_of`, `rule_version`, `cfg`, `n_selected`, `n_ok`, `n_reused_only`, `by_status`, `cost_rmb`, `stopped`, `wall_s`, `state`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/research/daily/2026-10-23/cases/20261023/301269.SZ/A2_7829b067f0f5.json`
    - 顶层键：`as_of`, `checks`, `cost_rmb`, `errors`, `evidence`, `model`, `output`, `role`, `status`, `steps`, `ticker`

### F0 输入快照 snapshots/<as_of>/

- 路径：`data:line_a_v2/snapshots`
- 体积 / 文件数：115.1 KB / 13
- 覆盖：2026-10-12 → 2026-10-23（文件名/目录名中的完整日期/年月）
- 用途：F0 快照清单：记录当日输入文件的内容哈希、行数、水位与缺失/过期项；是任务键与可复现性的锚点。
- 生产者：quantmind/line_a/snapshot.py::build
- 消费者：line_a/pipeline.py（任务键）、审计
- 置信度：confirmed；可再生：是（按日重建）
- 主要文件模式：`snapshot_b21c544299cd69ff.json`×7; `snapshot_e8d5e18c0c530494.json`×3; `snapshot_22e749c88293b7cc.json`×1; `snapshot_f12629ac212e84e4.json`×1; `snapshot_03f012b5d9dbcc9a.json`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/snapshots/2026-10-23/snapshot_f12629ac212e84e4.json`
    - 顶层键：`as_of`, `data_cutoff_ts`, `created_utc`, `inputs`, `notes`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/snapshots/2026-10-19/snapshot_b21c544299cd69ff.json`
    - 顶层键：`as_of`, `data_cutoff_ts`, `created_utc`, `inputs`, `notes`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/snapshots/2026-10-21/snapshot_e8d5e18c0c530494.json`
    - 顶层键：`as_of`, `data_cutoff_ts`, `created_utc`, `inputs`, `notes`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/snapshots/2026-10-21/snapshot_22e749c88293b7cc.json`
    - 顶层键：`as_of`, `data_cutoff_ts`, `created_utc`, `inputs`, `notes`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/snapshots/2026-10-19/snapshot_03f012b5d9dbcc9a.json`
    - 顶层键：`as_of`, `data_cutoff_ts`, `created_utc`, `inputs`, `notes`

### P5 扩展域与择时实验

- 路径：`data:line_a_v2/p5`
- 体积 / 文件数：15.0 MB / 36
- 覆盖：2022-01-10 → 2026-10-09（抽样 parquet 日期列）
- 抽样文件日期列范围：2022-01-10 → 2026-10-09
- 用途：P5 X2（1500 名外扩展域）与 X12（择时/小盘配额）的净值、容量、状态与汇总。
- 生产者：scripts/line_a/run_x*.py（X2/X12）
- 消费者：P5 报告/前端
- 置信度：confirmed；可再生：是（重跑实验）
- 主要文件模式：`nav_20d_R.parquet`×2; `nav_63d_R.parquet`×2; `forecasts.parquet`×1; `nav_20d_C5.parquet`×1; `nav_20d_C4.parquet`×1; `nav_20d_C1.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x2_v1/x2_63d.json`
    - 顶层键：`horizon`, `n_test_asof`, `adv_floor_rmb`, `ext_eligible_per_date_median`, `pool_per_date_median`, `arms`, `reproduction`, `paired`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x12_v1/x12_63d.json`
    - 顶层键：`horizon`, `n_rebalances`, `state_share`, `arms`, `paired`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x12_v1/states_63d.csv` — 17 行
    - 表头：, trend, vol, crowd, crowd_known
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x1_v1/nav_20d_R.parquet` — 1,128 行 × 3 列
    - 日期列 `date`：2022-02-14 → 2026-10-09
    - 列：`nav_net`:double, `nav_bench`:double, `date`:timestamp[ns]
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x1_v1/nav_63d_R.parquet` — 1,008 行 × 3 列
    - 日期列 `date`：2022-08-09 → 2026-10-09
    - 列：`nav_net`:double, `nav_bench`:double, `date`:timestamp[ns]
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x3/forecasts.parquet` — 328,159 行 × 8 列
    - 日期列 `as_of`：2022-01-10 → 2026-09-01
    - 列：`y`:double, `B1`:double, `B2`:double, `B3`:double, `M`:double, `year`:int64, `as_of`:timestamp[ns], `ticker`:string
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x1_v1/nav_20d_C5.parquet` — 1,128 行 × 3 列
    - 日期列 `date`：2022-02-14 → 2026-10-09
    - 列：`nav_net`:double, `nav_bench`:double, `date`:timestamp[ns]
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p5/x2_v1/x2_20d.json`
    - 顶层键：`horizon`, `n_test_asof`, `adv_floor_rmb`, `ext_eligible_per_date_median`, `pool_per_date_median`, `arms`, `reproduction`, `paired`

### Ridge 参考臂存档

- 路径：`data:line_a_v2/refs`
- 体积 / 文件数：70.6 KB / 2
- 覆盖：None → None（未发现日期线索）
- 用途：冻结的 Ridge 20d/63d 最后一折模型（系数/特征名/标准化参数），每日 f3_ridge 直接加载。
- 生产者：scripts/line_a/export_reference_contracts.py
- 消费者：quantmind/line_a/ridge_scoring.py::load_ref、pipeline.py
- 置信度：confirmed；可再生：是（重训并导出）
- 主要文件模式：`ridge_20d_lastfold.npz`×1; `ridge_63d_lastfold.npz`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/refs/ridge_63d_lastfold.npz`
    - 数组：`mu`(223,), `sd`(223,), `w`(223,), `b`(), `direction`(), `feats`(223,), `horizon`(), `cutoff`(), `train_dates`()
  - `/home/lenovo/projects/quantmind/data/line_a_v2/refs/ridge_20d_lastfold.npz`
    - 数组：`mu`(223,), `sd`(223,), `w`(223,), `b`(), `direction`(), `feats`(223,), `horizon`(), `cutoff`(), `train_dates`()

### 任务表 jobs.sqlite

- 路径：`data:line_a_v2/jobs.sqlite`
- 体积 / 文件数：148.0 KB / 1
- 覆盖：None → None（运行时间戳在表内，未用于覆盖）
- 用途：JobStore 持久化：每个 (stage, as_of, snapshot, code, model) 任务键的状态、重试、产物路径与 sha256；支持断点续跑与并发互斥。
- 生产者：quantmind/line_a/jobs.py::JobStore
- 消费者：quantmind/line_a/pipeline.py、catchup.py
- 置信度：confirmed；可再生：否（运行历史；删了会重跑）
- 备注：只读打开（mode=ro）抽样表结构。
- 主要文件模式：`jobs.sqlite`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/jobs.sqlite`

## 评测实验

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| Agent/Harness 评测（eval/） | `data:line_a_v2/eval` | 21.8 MB | 773 | 2023-03-10 → 2026-10-25 | A1_<hash>.json<br>A1_single.json<br>llm_ledger.jsonl | 按任务号组织的 Agent 评测：A1/X7/X8/X10/X11/T-051/T-054 等的 cases、逐条结果、LLM ledger 与汇总；_aborted_* 为中断留证。 |
| P4 窄模型实验（p4/） | `data:line_a_v2/p4` | 2.5 GB | 364 | 2019-01-01 → 2026-10-16 | x4_<yyyy>.parquet<br>config.json<br>manifest.json | T-038/T-049 的 X4 公告分类（规则/DeepSeek/MacBERT/蒸馏）、X5 实体、X6 检索、X13 PDF 抽取、T-054 蒸馏数据的训练集、金标准与分数。 |
| T-052 可执行模拟盘回测 | `data:line_a_v2/t052_sim` | 2.2 MB | 121 | 2026-07-28 → 2026-10-23 | trades.parquet<br>nav.parquet<br>targets.parquet | T-052 在最近 3/6 个月按 10 万/100 万/1000 万规模的可执行模拟：网格汇总、主结果、阶段检查与报告；v1_before_t053_review 是 T-053 复审前快照。 |
| 外部工程线程输出 dsh_out/ | `data:line_a_v2/dsh_out` | 20.2 MB | 33 | 2026-10-24 → 2026-10-24 | limit_rounding_mismatch.parquet<br>amihud.parquet<br>ens20_scores.parquet | 其他工程线程（t52 复核、t53 复审）的独立产物，与主产物隔离；本次 t55 输出目录自身不计入。 |

### Agent/Harness 评测（eval/）

- 路径：`data:line_a_v2/eval`
- 体积 / 文件数：21.8 MB / 773
- 覆盖：2023-03-10 → 2026-10-25（文件名/目录名中的完整日期/年月）
- 用途：按任务号组织的 Agent 评测：A1/X7/X8/X10/X11/T-051/T-054 等的 cases、逐条结果、LLM ledger 与汇总；_aborted_* 为中断留证。
- 生产者：scripts/line_a/run_eval_a1.py、run_t051.py、run_t054.py、run_x10.py、run_x11.py、run_x7_x8.py、run_x14_injection.py
- 消费者：评测报告/论文式对比、前端实验页
- 置信度：confirmed；可再生：是（重跑，LLM 成本另计）
- 备注：_llm_cache 不计入 schema。
- 主要文件模式：`A1_<hash>.json`×597; `A1_single.json`×80; `llm_ledger.jsonl`×27; `results.jsonl`×24; `summary.json`×17; `cases.json`×12
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/t051_L31_holdout3/summary.json`
    - 顶层键：`arm`, `set`, `model`, `v3`, `structured_a1`, `n_cases`, `n_ok`, `abstain_rate`, `mean_recall`, `mean_verified_share`, `mean_problem_share`, `completeness_asked`, `cost_rmb_total`, `secs_mean`, `status_counts`, `seed`, `as_ofs`, `missed_fact_counts`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/t054_R_holdout4/results.jsonl` — 28 行
    - 元素键：`abstained`, `as_of`, `case_id`, `cost_rmb`, `errors`, `missed`, `n_claims`, `n_gold`, `n_hit`, `path`, `problem_share`, `recall`, `repair`, `secs`, `status`, `ticker`, `verified_share`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/t054_R_holdout4/llm_ledger.jsonl` — 28 行
    - 元素键：`cached`, `content_sha256`, `cost_rmb`, `finish_reason`, `latency_s`, `meta`, `model`, `model_version`, `request_hash`, `status`, `usage`, `utc`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/t054_S1_holdout4/cases.json`
    - 元素键：`as_of`, `ticker`, `board`, `case_id`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/t054_R_holdout4/runs/20250310/000958.SZ/A1_187e58a7255a.json`
    - 顶层键：`aids`, `as_of`, `checks`, `cost_rmb`, `errors`, `evidence`, `model`, `output`, `repair`, `role`, `status`, `steps`, `ticker`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/x7b_local_single_heldout_20261023/runs/20250512/688310.SH/A1_single.json`
    - 顶层键：`as_of`, `checks`, `cost_rmb`, `errors`, `evidence`, `model`, `output`, `role`, `status`, `steps`, `ticker`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/x11_v22_20261025/perturbation_rows.csv` — 359 行
    - 表头：﻿kind, ticker, ref, orig, correct, perturbed, v1_errors, v2_errors
  - `/home/lenovo/projects/quantmind/data/line_a_v2/eval/x11_v22_20261025/real_rewrites.csv` — 42 行
    - 表头：﻿ticker, status, cached, v1, error, ref, claim_text, clause

### P4 窄模型实验（p4/）

- 路径：`data:line_a_v2/p4`
- 体积 / 文件数：2.5 GB / 364
- 覆盖：2019-01-01 → 2026-10-16（x4_v1 年度分片 x4_2019..x4_2026 及其日期列）
- 用途：T-038/T-049 的 X4 公告分类（规则/DeepSeek/MacBERT/蒸馏）、X5 实体、X6 检索、X13 PDF 抽取、T-054 蒸馏数据的训练集、金标准与分数。
- 生产者：scripts/line_a/build_x4_final.sh、x4v21_split.py、x4_hard_examples.py、run_x6.py、run_t054.py、t054_distill_data.py
- 消费者：line_a/evidence/ann_classifier.py、event_monitor.py、T-049 采用决策
- 置信度：confirmed；可再生：部分（LLM 标注与训练可复现但成本高）
- 备注：2.6G，大头是 4 份 MacBERT 检查点（各约 391M）。
- 主要文件模式：`x4_<yyyy>.parquet`×15; `config.json`×8; `manifest.json`×6; `model.safetensors`×4; `scores.jsonl`×4; `llm_ledger.jsonl`×4
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v1/x4_2026.parquet` — 492,842 行 × 34 列
    - 日期列 `date`：2026-01-01 → 2026-10-16
    - 列：`announcementId`:string, `ts_code`:string, `date`:string, `title`:string, `rule_type`:string, `rule_stage`:string, `is_meta`:bool, `split`:string, `struct_performance_forecast`:bool, `struct_performance_express`:bool, `struct_repurchase`:bool, `struct_holder_reduction`:bool, `struct_holder_increase`:bool, `struct_unlock`:bool …（共 34 列）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v22/predictions_gold_hard.parquet` — 4,200 行 × 9 列
    - 列：`announcementId`:string, `title`:string, `type`:string, `y`:bool, `part`:string, `R_rule`:bool, `M2`:bool, `M21`:bool, `D_deepseek`:bool
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v21/gold_hard_labeled.parquet` — 300 行 × 3 列
    - 列：`announcementId`:string, `title`:string, `types`:list<element: string>
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v22/macbert_multilabel_v22/x4v2_config.json`
    - 顶层键：`cfg`, `types`, `pos_weight`, `n_train`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v1/results_local/ft__macbert-base/scores.jsonl` — 11,531 行
    - 元素键：`cell`, `sha`, `p_yes`, `status`, `mode`, `model_key`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x4_v22/macbert_multilabel_v22/config.json`
    - 顶层键：`add_cross_attention`, `architectures`, `attention_probs_dropout_prob`, `bos_token_id`, `classifier_dropout`, `directionality`, `dtype`, `eos_token_id`, `gradient_checkpointing`, `hidden_act`, `hidden_dropout_prob`, `hidden_size`, `id2label`, `initializer_range`, `intermediate_size`, `is_decoder`, `label2id`, `layer_norm_eps`, `max_position_embeddings`, `model_type`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x13_test_v2/manifest.json`
    - 顶层键：`version`, `seed`, `n_excluded_tickers`, `years`, `per_board_year`, `n_cases`, `est_download_mb`, `n_gold`, `n_gold_usable`, `cases_sha256`, `gold_sha256`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/p4/x13_test_v2/gold.csv` — 441 行
    - 表头：﻿ann_id, ts_code, year, field, value, prior, excluded

### T-052 可执行模拟盘回测

- 路径：`data:line_a_v2/t052_sim`
- 体积 / 文件数：2.2 MB / 121
- 覆盖：2026-07-28 → 2026-10-23（抽样 parquet 日期列）
- 抽样文件日期列范围：2026-07-28 → 2026-10-23
- 用途：T-052 在最近 3/6 个月按 10 万/100 万/1000 万规模的可执行模拟：网格汇总、主结果、阶段检查与报告；v1_before_t053_review 是 T-053 复审前快照。
- 生产者：scripts/line_a/run_x*.py / T-052 模拟器（dsh_out/t52 为独立复核）
- 消费者：T-053 复审、前端回测页
- 置信度：confirmed；可再生：是（重跑模拟）
- 主要文件模式：`trades.parquet`×24; `nav.parquet`×24; `targets.parquet`×24; `rejects.parquet`×24; `holdings.parquet`×12; `grid_summary.parquet`×2
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/primary.json`
    - 顶层键：`W6_ENS20_n10_s4_b2.0_open_100000_main+chinext`, `W3_ENS20_n10_s4_b2.0_open_100000_main+chinext`, `W6_ENS20_n30_s4_b2.0_open_1000000_all`, `W3_ENS20_n30_s4_b2.0_open_1000000_all`, `W6_ENS20_n100_s4_b2.0_open_10000000_all`, `W3_ENS20_n100_s4_b2.0_open_10000000_all`, `W6_ENS20_n10_s4_b2.0_open_100000_main_only`, `W3_ENS20_n10_s4_b2.0_open_100000_main_only`, `W6_ENS20_n30_s4_b2.0_open_1000000_main_only`, `W3_ENS20_n30_s4_b2.0_open_1000000_main_only`, `W6_ENS20_n100_s4_b2.0_open_10000000_main_only`, `W3_ENS20_n100_s4_b2.0_open_10000000_main_only`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/grid_summary.parquet` — 4,038 行 × 42 列
    - 列：`signal`:string, `base`:string, `quota`:list<element: double>, `n`:int64, `step`:int64, `buffer`:double, `exec`:string, `aum`:double, `window`:string, `drift`:double, `boards`:string, `primary`:bool, `key`:string, `days`:int64 …（共 42 列）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/phase_check.parquet` — 21 行 × 11 列
    - 列：`aum`:double, `n`:int64, `step`:int64, `offset`:int64, `start`:string, `total_return`:double, `bench`:double, `excess_exposure_adj`:double, `turnover_x`:double, `cost_share`:double, `avg_cash`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/report.md`
    - 标题：## 主配置（事先指定，每档资金一个） / ## 网格分布（全部组合；按信号 × 资金 × 窗口，账户收益的中位数与四分位） / ## 单个维度的影响（W6，按仓位调整的超额*，其余维度取全部组合的中位数）
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/runs/W3_ENS20_n30_s4_b2.0_open_1000000_main_only/trades.parquet` — 114 行 × 9 列
    - 日期列 `date`：2026-07-29 → 2026-09-24
    - 列：`date`:timestamp[ns], `ticker`:string, `side`:string, `shares`:double, `price`:double, `gross`:double, `cost`:double, `decided`:timestamp[ns], `participation`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/runs/W3_ENS20_n30_s4_b2.0_open_1000000_main_only/nav.parquet` — 58 行 × 7 列
    - 日期列 `date`：2026-07-28 → 2026-10-23
    - 列：`equity`:double, `cash`:double, `invested`:double, `n_pos`:int64, `bench_pool_ew`:double, `max_weight`:double, `date`:timestamp[ns]
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/runs/W3_ENS20_n30_s4_b2.0_open_1000000_main_only/targets.parquet` — 3 行 × 4 列
    - 日期列 `as_of`：2026-07-28 → 2026-09-22
    - 列：`as_of`:timestamp[ns], `targets`:list<element: string>, `equity`:double, `per_name`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/t052_sim/runs/W3_ENS20_n30_s4_b2.0_open_1000000_main_only/rejects.parquet` — 18 行 × 4 列
    - 日期列 `date`：2026-08-26 → 2026-09-28
    - 列：`date`:timestamp[ns], `ticker`:string, `side`:string, `reason`:string

### 外部工程线程输出 dsh_out/

- 路径：`data:line_a_v2/dsh_out`
- 体积 / 文件数：20.2 MB / 33
- 覆盖：2026-10-24 → 2026-10-24（文件名/目录名中的完整日期/年月）
- 抽样文件日期列范围：2026-03-02 → 2026-10-23
- 用途：其他工程线程（t52 复核、t53 复审）的独立产物，与主产物隔离；本次 t55 输出目录自身不计入。
- 生产者：各任务脚本（如 scripts/line_a/t52_crosscheck.py）
- 消费者：评审/对照
- 置信度：confirmed；可再生：是
- 备注：t55 的 data_catalog.* 写在本目录的 t55/ 子目录。
- 主要文件模式：`limit_rounding_mismatch.parquet`×1; `amihud.parquet`×1; `ens20_scores.parquet`×1; `dividends.parquet`×1; `st_intervals.parquet`×1; `trades_1000w_n100_band0.0.parquet`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t53/limit_rounding_summary.json`
    - 顶层键：`rows`, `up_mismatch`, `dn_mismatch`, `buy_false_block`, `buy_false_allow`, `sell_false_block`, `sell_false_allow`, `primary_sell_fills_on_false_allow`, `primary_buy_fills_on_false_block`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t53/limit_rounding_mismatch.parquet` — 674,291 行 × 14 列
    - 日期列 `date`：2026-04-23 → 2026-10-23
    - 列：`ticker`:string, `date`:timestamp[ns], `pct`:double, `pre`:double, `hi`:double, `lo`:double, `up_py`:double, `up_true`:double, `dn_py`:double, `dn_true`:double, `buy_false_block`:bool, `buy_false_allow`:bool, `sell_false_block`:bool, `sell_false_allow`:bool
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t52/amihud.parquet` — 820,964 行 × 3 列
    - 日期列 `as_of`：2026-03-02 → 2026-10-23
    - 列：`as_of`:timestamp[ns], `ticker`:string, `amihud_illiquidity`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t52/ens20_scores.parquet` — 124,027 行 × 3 列
    - 日期列 `as_of`：2026-04-29 → 2026-10-21
    - 列：`as_of`:timestamp[ns], `ticker`:string, `score`:double
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t52/dividends.parquet` — 58,768 行 × 6 列
    - 列：`ts_code`:string, `div_proc`:string, `ex_date`:string, `cash_div_tax`:double, `stk_div`:double, `__index_level_0__`:int64
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t52/dsh_summary.json`
    - 顶层键：`task`, `period`, `initial_cash`, `final_equity`, `total_pnl`, `total_return`, `total_return_pct`, `n_trades`, `n_buy_trades`, `n_sell_trades`, `commission`, `transfer_fee`, `stamp_tax`, `slippage`, `dividend_tax`, `total_cost`, `cash_end`, `n_positions_end`, `rejected`, `notes`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t53/drift_primary_all.json`
    - 顶层键：`10w_n10_band0.5`, `10w_n10_band0.0`, `100w_n30_band0.5`, `100w_n30_band0.0`, `1000w_n100_band0.5`, `1000w_n100_band0.0`
  - `/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t52/st_intervals.parquet` — 2,835 行 × 3 列
    - 列：`ts_code`:string, `s`:timestamp[ns], `e`:timestamp[ns]

## 其他

| 条目 | 路径 | 体积 | 文件数 | 覆盖 | 主要模式 | 用途 |
|---|---|---:|---:|---|---|---|
| KB/上下文共享缓存 | `data:cache` | 3.8 MB | 120 | 2024-12-31 → 2024-12-31 | prices_2024-12-31_20_4ceffff6.parquet<br>spot_full_a_2024-12-31.parquet<br>fundamentals_2024-12-31_45f39cbc.parquet | KB 检索上下文与元数据的文件缓存，避免重复构建。 |

### KB/上下文共享缓存

- 路径：`data:cache`
- 体积 / 文件数：3.8 MB / 120
- 覆盖：2024-12-31 → 2024-12-31（缓存键的 as_of 日期（抽样文件））
- 用途：KB 检索上下文与元数据的文件缓存，避免重复构建。
- 生产者：quantmind/kb/*、scripts/build_kb.py
- 消费者：研究/报告生成
- 置信度：inferred；可再生：是（缓存）
- 主要文件模式：`prices_2024-12-31_20_4ceffff6.parquet`×1; `spot_full_a_2024-12-31.parquet`×1; `fundamentals_2024-12-31_45f39cbc.parquet`×1; `spot_2024-12-31.parquet`×1; `fundamentals_2024-12-31_dd649023.parquet`×1; `kb_context_601818.SH_2024-12-31.meta.json`×1
- 结构抽样：
  - `/home/lenovo/projects/quantmind/data/cache/shared/kb_context_601166.SH_2024-12-31.meta.json`
    - 顶层键：`written_at`, `ttl_hours`, `key`
  - `/home/lenovo/projects/quantmind/data/cache/shared/prices_2024-12-31_20_4ceffff6.parquet` — 85 行 × 5023 列
    - 列：`000001.SZ`:double, `000002.SZ`:double, `000006.SZ`:double, `000007.SZ`:double, `000008.SZ`:double, `000009.SZ`:double, `000011.SZ`:double, `000012.SZ`:double, `000014.SZ`:double, `000017.SZ`:double, `000019.SZ`:double, `000020.SZ`:double, `000021.SZ`:double, `000025.SZ`:double …（共 5023 列）
  - `/home/lenovo/projects/quantmind/data/cache/shared/spot_full_a_2024-12-31.parquet` — 5,515 行 × 11 列
    - 日期列 `list_date`：1990-12-01 → 2026-05-11
    - 列：`ticker`:string, `name`:string, `industry`:string, `total_mv`:double, `circ_mv`:double, `pe_ttm`:double, `pb`:double, `list_date`:timestamp[ns], `is_st`:bool, `close`:double, `__index_level_0__`:int64
  - `/home/lenovo/projects/quantmind/data/cache/shared/fundamentals_2024-12-31_45f39cbc.parquet` — 63 行 × 4 列
    - 列：`ticker`:string, `pe_ttm`:double, `pb`:double, `__index_level_0__`:int64
  - `/home/lenovo/projects/quantmind/data/cache/shared/spot_2024-12-31.parquet` — 0 行 × 11 列
    - 日期列 `list_date`：None → None
    - 列：`ticker`:null, `name`:null, `industry`:null, `total_mv`:null, `circ_mv`:null, `pe_ttm`:null, `pb`:null, `list_date`:null, `is_st`:null, `close`:null, `__index_level_0__`:int64
  - `/home/lenovo/projects/quantmind/data/cache/shared/kb_context_601818.SH_2024-12-31.meta.json`
    - 顶层键：`written_at`, `ttl_hours`, `key`
  - `/home/lenovo/projects/quantmind/data/cache/shared/kb_context_000938.SZ_2024-12-31.meta.json`
    - 顶层键：`written_at`, `ttl_hours`, `key`
  - `/home/lenovo/projects/quantmind/data/cache/shared/fundamentals_2024-12-31_dd649023.parquet` — 18 行 × 4 列
    - 列：`ticker`:string, `pe_ttm`:double, `pb`:double, `__index_level_0__`:int64

## 附：代码目录结构（只读）

- `quantmind`：22 个子目录，2 个文件
  - 子目录：`agents`, `analysis`, `backtest`, `contracts`, `core`, `data`, `execution`, `features`, `iteration`, `kb`, `line_a`, `models`, `notify`, `portfolio`, `regime`, `research`, `risk`, `selection`, `ui`, `utils`, `watchlist`, `workflow`
- `scripts`：6 个子目录，116 个文件
  - 子目录：`bakeoff`, `data_pipeline`, `db_migration`, `line_a`, `maintenance`, `survivorship`
- `configs`：1 个子目录，3 个文件
  - 子目录：`line_a_v2`
- `docs/work/tasks`：0 个子目录，50 个文件

- 任务卡：50 个（T-001…），标题见 `data_catalog.json → code_structure.task_index`。

## 体积大头（Top 15）

### 顶层目录（depth 1，完整子树）

| # | 目录 | 体积 | 文件数 | 可再生性（推测） |
|---:|---|---:|---:|---|
| 1 | `data:gpu_v1` | 32.2 GB | 7,898 | — |
| 2 | `data:lake` | 15.1 GB | 41,323 | — |
| 3 | `fwd:20261023` | 3.5 GB | 139 | 影子数据根副本（推测可由主数据根重建） |
| 4 | `fwd:20261022` | 3.4 GB | 139 | 影子数据根副本（推测可由主数据根重建） |
| 5 | `fwd:20261021` | 3.4 GB | 139 | 影子数据根副本（推测可由主数据根重建） |
| 6 | `data:line_a_v2` | 2.7 GB | 2,306 | — |
| 7 | `data:panel` | 2.6 GB | 50 | 因子面板（推测可由 lake 重算） |
| 8 | `data:raw` | 2.4 GB | 20 | — |
| 9 | `data:bakeoff` | 2.3 GB | 99 | 模型对比实验产物（推测可重算） |
| 10 | `data:snapshots` | 2.0 GB | 655 | — |
| 11 | `data:research` | 500.9 MB | 1,862 | — |
| 12 | `data:lake_pit` | 357.7 MB | 22,243 | — |
| 13 | `data:qlib_cn_daily_fwd_20261023` | 280.8 MB | 46,562 | Qlib dump（推测可由 panel 重建） |
| 14 | `data:qlib_cn_daily_fwd_20261022` | 280.6 MB | 46,554 | Qlib dump（推测可由 panel 重建） |
| 15 | `data:_quarantine_20261022_qlib_fwd_g1_selection` | 280.6 MB | 46,548 | — |

### 二级目录（depth 2，按完整子树聚合）

| # | 目录 | 体积 | 文件数 | 可再生性（推测） |
|---:|---|---:|---:|---|
| 1 | `data:gpu_v1/panel` | 16.0 GB | 589 | GPU 训练/预测中间产物（推测可由面板重训/重算） |
| 2 | `data:gpu_v1/runs` | 7.8 GB | 6,172 | GPU 训练/预测中间产物（推测可由面板重训/重算） |
| 3 | `data:lake/events` | 6.3 GB | 37,899 | — |
| 4 | `data:gpu_v1/model` | 4.1 GB | 191 | GPU 训练/预测中间产物（推测可由面板重训/重算） |
| 5 | `data:gpu_v1/preds` | 3.4 GB | 885 | GPU 训练/预测中间产物（推测可由面板重训/重算） |
| 6 | `data:line_a_v2/p4` | 2.5 GB | 364 | P4 实验/模型检查点（推测可重跑/重训） |
| 7 | `data:lake/events_purchased` | 2.3 GB | 10 | — |
| 8 | `fwd:20261023/main` | 2.1 GB | 111 | 影子数据根副本（推测可由主数据根重建） |
| 9 | `fwd:20261022/main` | 2.1 GB | 111 | 影子数据根副本（推测可由主数据根重建） |
| 10 | `fwd:20261021/main` | 2.1 GB | 111 | 影子数据根副本（推测可由主数据根重建） |
| 11 | `fwd:20261023/a158` | 1.4 GB | 27 | 影子数据根副本（推测可由主数据根重建） |
| 12 | `fwd:20261022/a158` | 1.4 GB | 27 | 影子数据根副本（推测可由主数据根重建） |
| 13 | `fwd:20261021/a158` | 1.4 GB | 27 | 影子数据根副本（推测可由主数据根重建） |
| 14 | `data:lake/text` | 970.1 MB | 3,216 | — |
| 15 | `data:lake/_backup_20261024` | 810.3 MB | 11 | 数据湖备份/隔离副本（推测可重建） |

### 单文件

| # | 文件 | 体积 | 可再生性（推测） |
|---:|---|---:|---|
| 1 | `data:bakeoff/alpha158_asof_v6_1.parquet` | 990.1 MB | 模型对比实验产物（推测可重算） |
| 2 | `data:bakeoff/alpha158_asof_v6.parquet` | 925.8 MB | 模型对比实验产物（推测可重算） |
| 3 | `data:lake/events_purchased/xianyu_20261022/raw/tushare_report_rc.csv` | 759.9 MB | — |
| 4 | `data:lake/events_purchased/xianyu_20261022/raw/tushare_research_report.csv` | 738.8 MB | — |
| 5 | `data:lake/events/moneyflow_20261023.parquet` | 707.2 MB | — |
| 6 | `data:lake/events/moneyflow_20261022.parquet` | 706.7 MB | — |
| 7 | `data:lake/events/moneyflow_20261021.parquet` | 706.3 MB | — |
| 8 | `data:lake/events/moneyflow_20261020.parquet` | 705.9 MB | — |
| 9 | `data:lake/events/moneyflow_20261009.parquet` | 703.0 MB | — |
| 10 | `fwd:20261023/main/data/gpu_v1/seq/seq_raw_f32.npy` | 672.1 MB | 影子数据根副本（推测可由主数据根重建） |
| 11 | `fwd:20261022/main/data/gpu_v1/seq/seq_raw_f32.npy` | 671.6 MB | 影子数据根副本（推测可由主数据根重建） |
| 12 | `fwd:20261021/main/data/gpu_v1/seq/seq_raw_f32.npy` | 671.1 MB | 影子数据根副本（推测可由主数据根重建） |
| 13 | `data:gpu_v1/seq/seq_raw_f32.npy` | 667.7 MB | GPU 训练/预测中间产物（推测可由面板重训/重算） |
| 14 | `data:panel/alpha_panel_weekly_v6_1.parquet` | 528.6 MB | 因子面板（推测可由 lake 重算） |
| 15 | `data:panel/alpha_panel_weekly_v6.parquet` | 488.1 MB | 因子面板（推测可由 lake 重算） |

说明：`_backup_*`、`gpu_v1/panel/_a158_parts`、`gpu_v1/model/v1/*.X.npy`、`gpu_v1/runs`、`p4/*/macbert_*`、`bakeoff`、`qm_fwd_root`、`dsh_out` 基本都是**可再生的中间产物/副本**（推测，依据目录名与代码引用）；`lake/*.parquet`、`lake_pit`、`events_purchased`、`snapshots` 则是较难重得的源数据/冻结快照。
