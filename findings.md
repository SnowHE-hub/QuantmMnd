# Findings — 短周期特异性因子（Step 2 计划期调研）

> 任务：从现有日线数据构造短周期特异性因子，用中性化 IC 筛选。本文件记录构造计划所需的事实依据。

## 数据 schema（只用这三个源）

### 1. `data/raw/alpha_prices_panel.parquet`（2,273,529 × 13，长表）
列：`ts_code, trade_date(datetime64), open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, adj_close`
- `open/high/low/close/pre_close` = **原始价**（未复权，元）。
- `adj_close` = 后复权收盘；`adj_factor` = 复权因子；故 **adj_open = open×adj_factor**、adj_high/adj_low 同理（数据里只有 adj_close，需重建 adj OHLC）。
- `vol` 单位 = **手**（1 手 = 100 股）；`amount` 单位 = **千元**（1 千元 = 1000 元）。
- 日历：2019-01-02 → 2026-05-11。

### 2. `data/lake/daily_basic.parquet`（8,288,000 × 19）
列含 `ts_code, trade_date, circ_mv, total_mv, turnover_rate, pe, pb, ...`
- **规模用 `circ_mv`（流通市值，万元）** → `log_market_cap = log(circ_mv)`。
- PIT：as_of 是交易日，circ_mv 在 as_of 收盘已知 → 按 `trade_date == as_of` join 安全。

### 3. `data/features/alpha_panel_weekly_v5.parquet`（452,439 × 38）
- MultiIndex `(as_of, ticker)`，as_of = 周频（每 5 交易日，相位锚定 fwd_12d，末截面 2026-04-20，共 350 as_of）。
- 已含 `exposure_industry`（行业，110 类，string）、`exposure_area`、35 因子、`forward_return_{12,21,63}d`。
- **新因子目标网格 = 此面板的 (as_of,ticker)**，经 `merge_increment` 增量 join。

## VWAP 单位自查（关键，覆盖用户的粗略提示）

成交额(元)=amount×1000，成交股数=vol×100 →
**vwap(元/股) = amount×1000 / (vol×100) = 10 × amount / vol。**
（用户提示 "amount/(vol×100)" 漏了 amount 的千元单位；因 WQ 用 `vwap - close`、close 是元，vwap 必须是元才有意义，故取 `10×amount/vol`。rank/相对算子里绝对尺度会约掉，但差值算子必须对齐量纲。）

## 复权口径规则（PIT + 公司行动一致性）

| 场景 | 用价 | 理由 |
|---|---|---|
| 同日价内关系（vwap/open/high/low vs close，同 t） | **原始价** | 同日同 adj_factor，比值/差值不受影响 |
| 跨日收益 / delta / 动量 / 价格时序相关 | **复权价**（adj_close、adj_open=open×adj_factor…） | 防止除权跳变污染 |
| 日内 = close_t/open_t − 1（同日） | 原始价 | 同 t 约掉 |
| 隔夜 = open_t/close_{t-1} − 1（跨日） | 复权：adj_open_t / adj_close_{t-1} | 跨日须复权 |
| 成交量 / adv = MEAN(vol,20) | 原始 vol | tushare vol 未复权；rank/比率稳健（标注） |

`returns` = adj_close.pct_change()。`cap` = circ_mv。`adv{n}` = MEAN(vol, n)。

## merge_increment 契约（`quantmind/features/weekly_panel.py:514`）
- base 与 increment 均须 MultiIndex(as_of, ticker)；index names 一致。
- increment 键必须**唯一**；**列名不得与 base 冲突**（否则 raise）；join 后行数不变。
- → 新因子表须严格落在 v5 的 (as_of,ticker) 上，列名加前缀（如 `sh_`）避免碰撞。

## 中性化筛选逻辑（复用 `scripts/_diag_wf_v2.py`）
- 逐 OOS 截面：因子（标准化）对 `C(exposure_industry) + z(log circ_mv)` OLS，取残差，残差 vs forward_return_12d 算 Spearman IC。
- 已验证：v2 模型预测中性化后 retention 仅 12%。同一逻辑直接套到单因子。
- WF 切分参数：H=12, E=20, rolling=756, n_val=2, OOS 2022-01→2026-04，16 fold，142 截面（与 _diag_wf_v2.py 一致）。

## 可复用算子（`quantmind/features/expr_factors.py`）
已有：`Ref(delay), Mean, Std, Abs, Greater, Log, Rank(截面), RollingMax, RollingMin`。
需新增（WQ/GTJA 用）：`delta, ts_rank, ts_argmax/argmin, correlation, covariance, sign, sum, product, scale, decay_linear, ts_min/ts_max, signedpower`。全部时序算子**只回看**。

## 幸存者警示
v5 = 幸存者池（delist_date 全空），任何正结果均为**乐观上界**；客户面前须先补退市票复核。

---

# Findings — 幸存者修复（2026-06-09 调研）

## 现状（survivorship 根因实测）
- `data/lake/stock_basic.parquet`：**1388 票，全 list_status='L'，delist_date 0/1388（全空）** → 零退市票 = 纯幸存者池。
- 交易所：SZSE 826 + SSE 562；**无 BSE（北交所）**。板块：主板 1143 / 创业板 147 / 科创板 98。
- list_date：1990→2026；**仅 1137 票在 2019 前上市**（远低于真实 A 股 ~3500+ 2019前存续票）→ **当前 1388 是一个来源不明的 ~1374 子样本，不是全 A 股**。
- `alpha_prices_panel.parquet`：1374 票，2019-01-02→2026-05-11（universe 实际来源 = 此原始价格文件，为何是 1374 未知）。
- weekly_panel universe = stock_basic ∩ priced，PIT by `list_date<=as_of`；delist 全空 → 退市过滤不生效（meta 已标）。

## ⚠ 关键 scope 歧义（评审决策点）
当前 1388 既缺退市票，又只是全市场的子集（缺大量存续票）。"修幸存者"有两种口径：
- A. **全市场 PIT**（推荐）：拉全 A 股（SH+SZ，L+D+P），按 list/delist 日建真 PIT universe。唯一无偏、定义清晰，但数据量大（~4800+ 票）。
- B. **样本 + 其退市票**：保持现 1374 样本口径只补它对应的退市票——但样本口径本身来源不明、含未知选择偏差，"对应退市票"难定义。
→ 必须先定 scope 才能动手。

## Tushare 可用性（供实现期）
- `backfill_tushare.py` 基础设施可复用：provider `_call`（指数退避 2/4/8s + 超时）、断点续传 `_checkpoints/`、loguru token 泄漏防护（`logger.remove()`）。
- Tushare `stock_basic(list_status='L'/'D'/'P')` 分别返回 在市/退市/暂停；含 list_date + delist_date。
- Tushare `daily` + `adj_factor` **对退市票的历史在市期数据仍可拉**（退市后历史保留）。
- ⚠ 退市票 adj_factor / 停牌段缺口需处理；速率限制（积分档）。

---

# Findings — 模型 Bake-off（Step 3 设计期环境调研，2026-06-06）

## ⚠ 环境实况 vs 任务书（重要差异）

| 项 | 任务书所述 | **实测** | 影响 |
|---|---|---|---|
| 计算环境 | WSL | **WSL Ubuntu**（conda `quantmind`，py 3.11.15）；本会话 Bash 工具默认走 **Git Bash/Windows**（`python`=WindowsApps，torch **cpu**，qlib 无） | 所有 GPU/训练命令必须 `wsl -e bash -lic '...'` 走 WSL，**不能用默认 Bash 的 python** |
| GPU | RTX 5060 **Ti 16GB** | **RTX 5060 Laptop GPU, 8151 MiB(~8GB)**，driver 595.79 | **显存只有一半**；batch/序列模型/Transformer 配置须按 **8GB** 设计，不是 16GB |
| PyTorch | CUDA | torch **2.11.0+cu128**，`cuda_available=True`，cap **sm_120 (Blackwell)**，cuda 12.8 | GPU 训练可行；Blackwell sm_120 需 cu128（已满足） |
| qlib | 用 Qlib | **未安装**（`ModuleNotFoundError`） | 须先装 pyqlib；且本 env 是 **numpy 2.0.2 / pandas 2.3.3**，经典 pyqlib 常 pin numpy<2 → **依赖冲突风险** |
| 其它 | — | lightgbm 4.6.0、statsmodels 0.14.6、scipy 1.17.1、sklearn 1.8.0、pyarrow 23 | LGBM/中性化 OLS 现成可用 |

WSL python 绝对路径：`/home/lenovo/miniforge3/envs/quantmind/bin/python`。

## 由实况推导的关键设计约束
1. **双 env 隔离**：`quantmind`（numpy2/pandas2.3，数据+特征+WF评估，**不可被 qlib 降级**）｜ 新建 `qlib_bakeoff`（pyqlib + 兼容 numpy/pandas + cu128 torch，仅训练+导出预测）。桥 = **预测 parquet (date,ticker,pred)**。
2. **8GB 显存预算**：Alpha360 = 6×60=360 维/股/日；GRU/LSTM hidden≤64、batch≤800、AMP fp16；Transformer d_model≤64、nhead 4、layers 2、batch≤512、AMP。OOM 时降 batch / 梯度检查点。
3. **评估不在 qlib 内**：用我们的 `PurgedWalkForwardSplit` 驱动 fold（cutoff = refit 频率），qlib 模型逐 fold `.fit/.predict`，导出 OOS 预测；再过 `_diag_wf_v2` 中性化 + `wf_costs` 打分，与 v2 apples-to-apples。
4. **训练标签**用我们面板的 `forward_return_{12,63}d`（PIT 已算），不用 qlib 自算标签，保证目标完全一致。
5. **wf_costs.WFCostModel.holding_td** 可参数化（默认 12）→ 63d 跑设 63。`exit_fill_index` 用 holding_td。

## 计算量量级（驱动分阶段）
模型×制式 = Linear(tab)/LGBM(tab)/GRU(seq)/LSTM(seq)/Transformer(seq) = 5；×周期{12d,63d}=10；×refit{季度~16fold, 月度~48fold}；深度模型×≥3 seed。
→ 全网格深度训练上千次 = **多日 GPU**。须分阶段（pilot→季度全网格→月度仅 baseline+优胜者）。

## qlib↔我们数据的桥接要点（dump_bin）
- `alpha_prices_panel` → qlib bin：字段 `$open/$high/$low/$close/$volume` + `$factor`（复权因子）；calendar = 交易日历；instruments 用 universe（带 list_date 起点；delist 全空=幸存者，标注）。
- Alpha158（tabular 补充给 Linear/LGBM）；Alpha360（序列：过去60日 OHLCV，逐日截面 CSZScoreNorm）。
- 自检门：抽 1 只票，qlib handler 输出 vs 手算对齐；PIT 反证（qlib Ref/Mean 只回看）。
