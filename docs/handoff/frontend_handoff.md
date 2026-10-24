# QuantMind 前端转接文件（给 Antigravity）

> 版本 2026-10-24。用途：设计一个**个人使用**的全新展示前端。本文说明系统做什么、每天产出什么、数据在哪里、怎么取（只读 REST 接口），以及哪些内容值得展示。
> 完整的数据清单（约 64 GB 主数据 + 11 GB 影子数据根，共 44 个数据集，含每个文件模式的列名、类型、行数、日期范围）见同目录的 `data_catalog.md` / `data_catalog.json`。

---

## 1. 系统一句话

A 股选股系统，主线 A v2。每天收盘后自动完成：
1. **数据**：Tushare 行情、财报、事件，巨潮公告元数据，卖方盈利预测。
2. **量化打分**：
   - 四族模型等权集成 ENS（XGBoost、CatBoost、TabM、序列融合），20 日和 63 日两个周期；
   - 冻结的 Ridge 作对照。
3. **漏斗**：
   - F0 时点快照；
   - F1 证券身份与可交易状态；
   - F2 支持域，即研究池 = 按时点成交额前 1500；
   - F3 打分；
   - F4 研究候选；
   - 旁路：全市场事件监测，公告由 MacBERT 分类；
   - F5 风险与执行卡；
   - F6 Agent 研究，6 个角色，DeepSeek；
   - F7 代码校验每条声明；
   - F8 前瞻账本；
   - F9 研究报告。
4. **正式荐股日报**（2026-10-24 起）：三个资金档（10 万 / 100 万 / 1000 万元）的纸面账户持仓、明日开盘委托。
5. **Agent 大脑**：
   - 读当天的一切，给正式持仓和委托打风险标记（只提示，不改持仓）；
   - 安排第二天要研究的股票。

所有结论都有证据编号，数字由代码核对。系统不连接券商、不下单。

## 2. 数据在哪里
| 位置 | 内容 | 体积 |
|---|---|---|
| WSL `/home/lenovo/projects/quantmind/data/`<br>（Windows：`\\wsl.localhost\Ubuntu\home\lenovo\projects\quantmind\data\`） | 主数据根 | 约 64 GB |
| `…/data/lake/` | Tushare 数据湖：日线、复权、每日指标、财报三表、事件表（`events/`）、巨潮公告（`text/cninfo/`）、指数 | 15 GB |
| `…/data/gpu_v1/` | GPU 研究栈：日频特征面板、各模型各折的样本外预测（`preds/<模型>_<期限>/fold_*.parquet`）、模型包 | 32 GB（最大头，多数是可重算的中间产物） |
| `…/data/raw/alpha_prices_panel_v6_<YYYYMMDD>.parquet` | 价格面板（每天一份，2019 至今，约 900 万行） | 每份约 300 MB |
| **`…/data/line_a_v2/`** | **主线 A v2 的全部每日产物，前端主要读这里** | 2.8 GB |
| `/home/lenovo/qm_fwd_root/<YYYYMMDD>/` | 每天的影子数据根（冻结构建器的输出） | 每份约 3.5 GB |
| Windows `D:\MCS_26S2课程资料\quantmind\formal_reports\` | 正式荐股日报的 Markdown 副本 | — |

**不要展示、也不要读**：
- `.env`、`api_key.txt`、任何 `*secret*` / `*credential*`；
- `configs/line_a_v2/account_profiles.yaml`；
- 任何账户规模、真实持仓。

前端只读 `line_a_v2/` 和接口返回的内容即可。

### `line_a_v2/` 的结构
| 路径 | 每天一份？ | 内容 |
|---|---|---|
| `runs/<日期>/day_<哈希>.json` | 是 | 当天日任务的清单：`outputs` 指向下面各文件 |
| `runs/<日期>/m_market_*.json` | 是 | 市场状态：指数趋势、风格、成交额、波动 |
| `runs/<日期>/f1_instruments_*.parquet` | 是 | 全市场证券身份与交易状态（约 5,200 行；`name`、`board`、`is_st`、`suspended`、涨跌停……） |
| `runs/<日期>/f2_domains_*.parquet` | 是 | 支持域（`domain` = research_pool / extended / …） |
| `runs/<日期>/f3_ens_20d_*.parquet`、`f3_ens_63d_*`、`f3_ridge_*` | 是 | 打分（`ENS` 是集成分数；`xgb`、`cat`、`tabm`、`seqfusion` 是成员） |
| `runs/<日期>/f4_candidates_*.parquet` | 是 | 研究候选（约 1,900 行；`reasons`、`priority`、`ens20/ridge20/ens63/ridge63`） |
| `runs/<日期>/bypass_events_*.parquet` | 是 | 旁路事件（按股票聚合；`importance` major/watch，`summaries` 公告标题，`evidence_ids`，`rules`） |
| `runs/<日期>/f5c_risk_cards_*.parquet` | 是 | 风险与执行卡（约 960 只；`beta_252`、`vol_60_ann`、动量、跳空、涨跌停次数、`har_next_day_vol_ann`、异常波动距离、除权……） |
| `runs/<日期>/f9_report_*.md` | 是 | 量化部分的模板报告 |
| `books/<日期>/` | 是 | 前瞻影子账本：`<臂>__<权限线>__{nav,targets,capacity,rejected}.parquet`，`books_summary.json`，`holdings_latest.parquet` |
| `research/daily/<日期>/` | 是 | 影子研究：`f9_research.md`（研究报告）、`f9_summaries.json`（每只股票的要点）、`f7_claims.parquet`（每条声明及校验状态）、`f7_risk_flags.parquet`、`f6_cases.parquet`（研究了谁、为什么）、`cases/`（每个角色的完整输出与证据）、`brain_memo.json`（Agent 大脑） |
| `formal/<日期>/report.json`、`report.md` | 是 | **正式荐股日报**（结构见第 4 节） |
| `brain/flags_ledger.jsonl` | 追加 | 大脑标记的台账（前瞻评估用） |
| `t052_sim/` | 一次性 | 模拟盘回测（4,038 个组合；`grid_summary.parquet`、`primary.json`、`runs/<组合>/nav.parquet` 等） |
| `eval/`、`p4/`、`p5/` | 一次性 | 各项实验（任务卡 T-0xx） |

## 3. 只读接口（推荐前端只用它）
- 启动（WSL 里）：
  ```bash
  cd ~/projects/qm-t035 && PYTHONPATH=. /home/lenovo/miniforge3/envs/qm-gpu/bin/python -m quantmind.line_a.api_server --port 8780
  ```
- 地址：`http://127.0.0.1:8780`（WSL 的 localhost 在 Windows 上可直接访问）。
  - 交互式文档：`/docs`；OpenAPI：`/openapi.json`。
- 全部是 GET、只读，CORS 已放开 GET。日期参数写 `YYYY-MM-DD` 或 `latest`。

| 接口 | 参数 | 返回 |
|---|---|---|
| `/api/health` | — | 是否可用、最新日期 |
| `/api/dates` | — | 有数据的日期：`runs`、`books`、`research`、`formal` |
| `/api/formal/{d}` | — | 正式荐股日报 JSON（第 4 节） |
| `/api/formal/{d}/markdown` | — | 正式荐股日报 Markdown |
| `/api/brain/{d}` | — | Agent 大脑：`memo`（summary、market_view、flags、research_requests、questions_for_user）和 `run`（状态、费用、工具调用数、被剔除的标记） |
| `/api/market/{d}` | — | 市场状态 |
| `/api/candidates/{d}` | `horizon` = 20d/63d，`top` | 研究候选 |
| `/api/scores/{d}` | `model` = ens_20d/ens_63d/ridge_20d/ridge_63d，`top` | 打分排行（含四个成员分数） |
| `/api/bypass/{d}` | `importance` = major/watch | 旁路事件 |
| `/api/risk_cards/{d}` | `ticker` 可选 | 风险与执行卡 |
| `/api/research/{d}` | — | 研究状态、每只股票要点、研究名单、研究报告 Markdown（较大，约 400 KB） |
| `/api/research/{d}/{ticker}` | — | 这只股票的每条声明（类型、文字、数值、期间、**校验状态** check、A6 审查意见）和风险提示 |
| `/api/books/{d}` | — | 前瞻账本摘要（每个臂 × 权限线的净值、超额、换手……） |
| `/api/books/{d}/{book}/{kind}` | book 如 `ens_native_20d__all_boards`；kind = nav/targets/capacity/rejected | 账本明细 |
| `/api/stock/{ticker}/prices` | `start`、`end` | 日线（不复权 OHLC、成交量额、复权因子） |
| `/api/stock/{ticker}/announcements` | `as_of`、`lookback_days` | 按时点的公告列表（含模型分类的类型、阶段、重要性） |
| `/api/backtest/t052` | — | 模拟盘主配置结果与报告 |
| `/api/backtest/t052/nav/{key}` | key 见 primary.json | 模拟盘净值序列（账户、前 1500 等权基准、现金、持仓数） |
| `/api/experiments` | — | 全部任务卡（标题、判定行） |
| `/api/experiments/{id}` | 如 `T-052` | 任务卡全文（Markdown，记录了每个实验的登记、结果和判定） |
| `/api/catalog` | — | 数据清单 JSON |

## 4. 关键数据结构
### 正式荐股日报 `formal/<日期>/report.json`
```text
as_of, mode ("preview" 首次调仓前 / "live"), today_is_rebalance, next_rebalance, go_live, kill_switch,
approval {research_report_approved, capital_gate, approved_on, evidence_at_approval},
tiers {
  aum_1e5 | aum_1e6 | aum_1e7: {
    aum, top_n, boards, equity, cash, return_since_start, pool_ew_since_start,
    holdings [ {ticker, name, board, shares, price, value, weight} ],
    orders_next_open [ {side buy/sell, ticker, name, board, est_shares, ref_close, est_value, rank_in_target, decided} ],
    costs {commission, transfer, stamp, slippage, dividend_tax}, targets_latest [ticker…]
  }
}
```
- `est_shares = 0` 的买单表示买不起一手（按规则跳过，现金留着），前端应单独标出。

### Agent 大脑 `research/daily/<日期>/brain_memo.json`
```text
as_of, summary, market_view,
flags [ {ticker, action caution/avoid_new_buy/review_holding, reason, evidence_ids[]} ],   ← 已经过代码核对
research_requests [ {ticker, roles[A1..A4], reason} ], questions_for_user [ … ]
```

### 研究声明 `f7_claims.parquet`
- `ticker`；`ref`（如 `A1:c7`，角色 : 声明编号）；`type`（reported_fact / computed / inference / counter_evidence / open_question）；`text`；`value`；`unit`；`period`；
- `check`：代码校验状态，取值 verified / minor_mismatch / mismatch / missing_evidence / unverifiable / weak_basis。前端用徽标区分，**这是系统可信度的核心**；
- `a6`：审查角色的意见。

### 账本净值 `books/<日期>/<臂>__<线>__nav.parquet`
- 列：`date`、`nav_net`、`nav_gross`、`nav_bench`、`cash`、`n_positions`、`cum_cost`、`daily_ret_net`、`excess_net`、`cum_return_net`。
- 臂：
  - 参考臂：`ens_native_20d`、`ens_native_63d`、`ridge_native_20d`、`ridge_native_63d`；
  - 挑战臂：`ens_c5_20d`、`ens_c5_63d`、`ens_q_ext_63d`。
- 权限线：`all_boards`、`no_bse`、`main_only`。

## 5. 建议展示的页面（具体由你和用户商定）
1. **今日**：
   - 三个资金档的纸面账户；
   - 明日委托：买 / 卖、股数、参考价，买不起一手的单列；
   - Agent 大脑的结论和标记；
   - 市场状态卡片；
   - 下一个调仓日。
2. **个股**：
   - K 线（prices）；
   - 模型分数及排名变化（scores）；
   - 风险卡；
   - 研究声明列表，带校验徽标，点开看证据编号；
   - 公告与事件链；
   - 旁路事件。
3. **研究**：当天研究报告，研究名单（为什么研究它：重大事件 / 大脑请求 / 两个期限都靠前 / 抽样……），费用与校验统计。
4. **账本与回测**：
   - 前瞻账本净值曲线（参考臂 vs 挑战臂 vs 基准），容量情景；
   - T-052 模拟盘的三档曲线和网格分布。
5. **实验日志**：任务卡列表（T-0xx），每张卡的登记、结果、判定。这是系统"怎么一步步变成现在这样"的记录。
6. **系统状态**：
   - 每日运行各步骤的状态：`/home/lenovo/t030_scratch/logs/daily/<YYYYMMDD>/status`，每行是 `<步骤> start|exit <rc> <时间>`；
   - 数据日期；
   - 研究与大脑的费用。

## 6. 更新节奏
- 每个交易日北京时间约 21:30 开始：
  - 数据和量化部分约 2 小时；
  - 研究约 1 小时（需要能访问 DeepSeek 的网络）；
  - 之后生成正式日报和大脑意见。
- 某一步失败不会影响已经生成的部分。可以读 status 文件，看到哪一步失败。

## 7. 术语
| 术语 | 含义 |
|---|---|
| ENS | 四族模型等权集成分数 |
| 20d / 63d | 预测期限（交易日） |
| 研究池 | 按时点 20 日均成交额前 1500 |
| 旁路 | 全市场事件监测 |
| 证据编号 | 例如 `tushare:income:600000.SH:20260630:20260828`、`cninfo:ann:<公告号>`、`doc:<公告号>:p<页>:<段>`、`chain:<…>` |
| 声明 | Agent 输出的最小单位 |
| 校验 | 代码逐条核对数值、单位、引文 |
| 影子 | 只记录、不形成交易指令 |
| 纸面账户 | 按规则模拟成交的账户 |
| 调仓 | 每 4 个周频网格点一次 |
