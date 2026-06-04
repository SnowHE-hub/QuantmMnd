# QuantMind 存储分工（真实现状）

> ## ⚠️ 2026-06-04 更正
>
> 本报告中关于"PostgreSQL 12/13 张表为空 / 波将金村"的判断是**错的**。
>
> **根因**：报告依据 `pg_stat_user_tables.n_live_tup` 判断行数。这是一个**估算值**，在 `ANALYZE` 跑之前一直显示 0。真实的 `COUNT(*)` 显示 PG 表自 2026-06-02 迁移以来一直有数据（`daily_prices_panel`=728370 行，与 parquet 一致）。
>
> **修正后的事实**：
> - PG 不是"空的"，迁移确实执行了；
> - 但 9 张表的"零生产代码引用"判断是对的（经精确 SQL grep 重新确认）；
> - 这 9 张表已被 DROP，理由从"空表" → "零引用 + parquet 是 SoT + 维护成本"；
> - E3 的真实问题是 `daily_prices_panel` 在 PG 和 parquet 中都只到 2024-12-31，缺 70/80 笔最新数据。已通过 E3 改读 `alpha_prices_panel.parquet` 修复。
>
> 下方原报告内容保留，作为发现过程的记录。

> **类型**：基于代码实证的现状描述（只读调研，未改代码）
> **日期**：2026-06-03
> **原则**：只写**当前真实分工**，不写"将来要灌 PG"。结论均附 `文件:行号` 证据。

---

## TL;DR — 三套存储真实分工

| 存储 | 真实角色 | 证据 |
|---|---|---|
| **Parquet** | **唯一真源（source of truth）+ 默认服务层**。所有行情/因子/财务/训练/反馈数据都在这里；`DATA_BACKEND` 默认 `parquet`。 | **251 处** `read_parquet`（app 27 / quantmind 73 / scripts 151，非测试） |
| **MongoDB** | **文档型数据的"可用副本"**。7 个 collection 都已灌入，但**仅当 `DATA_BACKEND=postgres`（非默认）时**才被 `DataService` 读取。 | 写：`app/db/writers.py`；读：`data_service.py:959-1086`（全部在 `_mongo_*` 分支） |
| **PostgreSQL** | **实际上只拥有 1 张表：`simulated_orders`（E3 执行层，100 行）**。其余 12 张表全空——3 张被 E3 代码查询但为空，9 张只有迁移脚本引用（死表）。 | `pg_stat_user_tables`：仅 `simulated_orders`=100，其余=0 |

一句话：**Parquet 是真源也是默认服务面；Mongo 是文档数据的可切换备用后端；PostgreSQL 现实中是 E3 执行层的"单表数据库"，另外 12 张表是一次没做完的迁移残留。**

---

## Step 1-3 实测结果

### Step 1 — 实际查询 PG 表的**生产**代码（剔除 `scripts/db_migration/*` 与 tests）

| PG 表 | 生产查询位置 | 该表行数 |
|---|---|---|
| `simulated_orders` | `data_service.py:675,776`、`execution/manager.py:158,271,279,336`、`backfill_executions.py:368` | **100 ✅** |
| `realized_pnl` | `data_service.py:781,944`、`execution/replay_engine.py:71`、`backfill_executions.py:39`、`writers.py:201`(DELETE) | **0 ❌** |
| `daily_prices_panel` | `app/pages/14_执行管理.py:315`、`replay_engine.py:102`、`manager.py:119`、`backfill_executions.py:111` | **0 ❌** |
| `alpha_universe` | `backfill_executions.py:56`（name/industry 映射） | **0 ❌** |

> 其余 PG 表（`price_daily / alpha_panel / daily_basic / index_daily / fina_indicator / regime_features / nav_curve / snapshot_daily_basic / snapshot_financials`）**只**出现在 `scripts/db_migration/02_import_pg.py`、`04_verify_consistency.py` 等迁移/校验脚本里，**生产代码零引用**。

### Step 2 — 实际查询 Mongo collection 的代码

- 全部集中在 `app/services/data_service.py`（读，`_mongo_*` 方法，第 959-1086 行）和 `app/db/writers.py`（写，第 146-308 行）。
- 读取的 collection：`recommendations`、`positions`、`agent_analysis`、`loss_signals`。
- **关键**：`DataService` 里这些 `_mongo_*` 读取**只在 `self._backend == "postgres"` 分支被调用**（见 `data_service.py:142-153, 295, 326, 423, 439, 626`）。默认 `parquet` 后端下，Mongo **完全不被读**。
- Mongo 7 个 collection 文档数：`recommendations`=13、`agent_analysis`=51、`sim_daily`=60、`positions`=32、`watchlist_scores`=2、`strategy_config`=1、`loss_signals`=2（均已填充）。

### Step 3 — `read_parquet` 计数（非测试）

```
app/:       27
quantmind/: 73
scripts/:  151
总计:      251
```
分布极广：features/、agents/、models/、watchlist/、selection/、regime/、data/、app/pages/、app/utils/ 全都直接读 parquet。**这是事实上的主数据通路。**

---

## Step 4 — 直球回答

### A. PG 的 12 张空表里，哪些表代码里**真的有查询**？

**只有 3 张**被生产代码查询（且都返回空）：

1. **`daily_prices_panel`** — E3 执行层的价格来源（`backfill_executions`/`manager`/`replay_engine`）+ 执行管理页直接查。
2. **`realized_pnl`** — E3 回放/回填的推荐池来源 + `DataService` 在 postgres 后端下的 PnL 来源 + 执行vs死扛对比。
3. **`alpha_universe`** — E3 回填取 name/industry 映射。

**另外 9 张是死表**（生产零引用，仅迁移脚本提到）：
`price_daily`、`alpha_panel`、`daily_basic`、`index_daily`、`fina_indicator`、`regime_features`、`nav_curve`、`snapshot_daily_basic`、`snapshot_financials`。

> 注意：这 3 张"被查询"的表**当前也是空的**，所以相关功能现在就已经在"静默返回空/退化"。例如 `backfill_executions._simulate_exit_with_intraday`（`backfill_executions.py:118-127`）在 `daily_prices_panel` 查空时直接退化成 `time_expired`@入场价——**无 parquet 兜底**。

### B. 如果 DROP 掉那 12 张空表，会有什么实际功能受影响？

**不会破坏任何当前正常工作的功能。** 细分：

- **9 张死表**：DROP → 生产**零影响**。只有 `02_import_pg.py`/`04_verify_consistency.py`（开发期迁移工具）再跑时会报错，而 `02_import_pg.py` 本就用 `if_exists="replace"` 会重建它们。
- **3 张被查询但空的表**（`daily_prices_panel`/`realized_pnl`/`alpha_universe`）：它们现在已经是空的→相关功能现在就返回空/退化。DROP 后，这些查询会从"**静默返回空**"变成"**明确报错（表不存在）**"。受影响的全是 **E3 执行层**：
  - `backfill_executions.py`（手动重跑回填 simulated_orders）——现在产出退化结果，DROP 后变硬报错；
  - `replay_engine.py`（参数回放）——读 `realized_pnl`+`daily_prices_panel`，现在空，DROP 后报错；
  - `data_service.get_execution_vs_hold_comparison`（执行 vs 死扛对比 UI）——"死扛"一侧读 `realized_pnl` 已是空曲线，DROP 后报错；
  - `_pg_get_realized_pnl`（仅 `DATA_BACKEND=postgres` 时）——DROP 后报错而非空；
  - `app/pages/14_执行管理.py:315` 直接查 `daily_prices_panel`——现在空，DROP 后报错。
- **必须保留 `simulated_orders`**：它是 PG 唯一有数据、且被执行管理页（`get_simulated_orders`/`get_execution_stats`，`data_service.py:650-747`，**不分后端、永远走 PG**）真实读取的表。

**结论**：12 张空表可以全部 DROP 而不影响任何**当前能用**的功能；DROP 后只是把 E3 那 3 张表的"静默空"变成"显式报错"——对排错反而更友好。唯一要留的是 `simulated_orders`。

### C. 现在的真实存储分工是什么？

- **Parquet**：行情、因子面板、财务、快照、训练标签、realized_pnl、NAV、推荐/分析的文件版——**全部真源**，且是默认服务面（`DATA_BACKEND=parquet`）。251 处 `read_parquet`。
- **Mongo**：7 个文档型 collection 的**已填充镜像**，是一个**真正能用的备用文档后端**——但只有显式 `DATA_BACKEND=postgres` 才会被读；默认通路不碰它。
- **PG**：**事实上的单表库**——只有 `simulated_orders`（E3 执行层）有数据并被持续读取。其余 12 张表是中断的迁移残留（3 张被 E3 代码引用但空，9 张完全没用）。E3 的"写/刷新"路径（回填/回放）依赖的 3 张源表为空且无兜底，所以 **E3 的写侧当前是坏的**，现存 100 行 `simulated_orders` 是 2026-06-02 迁移窗口的一次性产物。

---

## 各数据的 Single Source of Truth 与消费路径

| 数据 | **唯一真源（SoT）** | 默认消费路径 | Mongo（仅 postgres 后端） | PG 现状 |
|---|---|---|---|---|
| 全A股日行情（2.27M 行） | `data/raw/alpha_prices_panel.parquet` | `read_parquet` 直读（features/snapshot/agents…） | — | `price_daily` 空·死表 |
| Alpha 因子面板（29k×75） | `data/panel/alpha_panel_v4.parquet` | `read_parquet`（模型/选股/打分） | — | `alpha_panel` 空·死表 |
| 估值/财务/指数/快照 | `data/snapshots/{date}/*.parquet`、`data/alpha_universe/*.parquet` | `read_parquet` | — | `daily_basic`/`fina_indicator`/`index_daily`/`snapshot_*` 空·死表 |
| 股票池 name/industry | `data/alpha_universe/alpha_universe.parquet` | `read_parquet` | — | `alpha_universe` 空·被 E3 查询 |
| Regime 历史 | `data/regime/regime_history.parquet` | `read_parquet`（`data_service._regime_snapshot`） | — | `regime_features` 空·死表 |
| 每日推荐 | `data/recommendations/{date}/`（JSON/parquet） | `rec_data` + `DataService`（parquet 分支） | `recommendations`(13) | — |
| 六维 Agent 分析 | `reports/investment_pipeline/{date}/strategies.json` | `DataService._load_strategies`（parquet 分支） | `agent_analysis`(51) | — |
| 前向持仓 | `data/paper_trading/forward_positions.json` | `DataService`（parquet 分支） | `positions`(32) | — |
| 已实现 PnL | `data/feedback/realized_pnl.parquet` | `read_parquet`（`get_realized_pnl` parquet 分支） | — | `realized_pnl` 空·被 E3 查询 |
| 损失信号 | `data/loss_signals_v4/*.json` | `DataService`（parquet 分支） | `loss_signals`(2) | — |
| 策略配置 | `data/paper_trading/strategy_config_v2.json` | `read`/`DataService` | `strategy_config`(1) | — |
| 30日模拟/NAV | `data/sim30d/`、`data/iteration/*/` | `read_parquet`/`sim_data` | `sim_daily`(60) | `nav_curve` 空·死表 |
| 自选股打分 | `data/watchlist/scores/{date}.json` | `read`/`DataService` | `watchlist_scores`(2) | — |
| **模拟执行订单（E3）** | **PostgreSQL `simulated_orders`（唯一）** | `DataService.get_simulated_orders/get_execution_stats`（**永远走 PG，不分后端**） | — | **`simulated_orders`=100 ✅（PG 真正拥有的唯一数据）** |

**两条铁律（现状）**：
1. 除了 `simulated_orders` 在 PG，**其它一切数据的 SoT 都是 `data/` 下的文件（parquet/json）**。
2. **Mongo 不是任何数据的 SoT**——它是文档数据的镜像副本，只服务于 `DATA_BACKEND=postgres` 这个非默认开关。

---

## 据此的诚实建议（不含"将来灌 PG"的空话）

| 优先级 | 建议 | 理由 |
|---|---|---|
| P0 | **DROP 那 9 张死表**（`price_daily`/`alpha_panel`/`daily_basic`/`index_daily`/`fina_indicator`/`regime_features`/`nav_curve`/`snapshot_daily_basic`/`snapshot_financials`） | 生产零引用，留着只会让 `pg_ping`/监控页显示一堆"0 行"假象，误导"有 DB 层" |
| P0 | **给 E3 的 3 张源表查询（`daily_prices_panel`/`realized_pnl`/`alpha_universe`）改成显式报错或 parquet 兜底** | 现在 E3 写侧静默退化（空价格→`time_expired`@入场价），是"假装在工作"；要么从 parquet 读，要么明确报"E3 源表未就绪" |
| P1 | **把 `simulated_orders` 也落一份 parquet/json 备份** | 它是 PG 唯一真源却无文件副本，PG 一挂就只能靠手动回填（而回填源又是空的）→ 实际不可恢复 |
| P1 | **文档/监控里停止把 PG 表述为"主存储"** | 真实只有 1 张表在用；`db_schema_postgres.sql` 的 13 表与现实不符 |
| P2 | 决定 Mongo 去留：要么把 `DATA_BACKEND=postgres` 正式作为"文档后端"维护（它确实能用），要么连同 PG 一起降级为实验特性 | 避免"三套并存"长期空耗 |

> 本文件只描述 2026-06-03 的真实状态，未改任何代码、未 commit。
