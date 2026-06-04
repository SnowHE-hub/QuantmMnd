# QuantMind 数据工程底座审计报告

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

> **类型**：只读审计（本轮未修改任何代码、未 commit）
> **日期**：2026-06-03
> **审计范围**：数据工程六层架构 + 六个横向维度 + 诚实诊断 + 路线图
> **方法**：源码精读 + 配置核对 + 运行时实证（PG/Mongo 行数、cron、日志、连接探测均为读取）

本报告的立场是 **诚实评估优先于表扬**。下面凡是结论，尽量都附了文件路径或实测证据，便于复核。

---

## 摘要（先看这段）

QuantMind 的 **软件工程纪律是真的好**（ruff + mypy + 1097 个测试 + CI + PIT 测试标记），
**PIT 快照机制和 DataService 统一访问层是真正经过思考的好设计**。

但 **"数据底座"层面存在一个核心的"账实不符"**：

- 项目对外呈现"parquet / PostgreSQL / MongoDB 三套存储 + 可切换后端 + 双写"的完整数据平台形象；
- **实测**：PostgreSQL 13 张表里 **12 张为空，只有 `simulated_orders`（100 行）有数据**；
  2.27M 行的 `price_daily`、29406 行的 `alpha_panel`、80 行的 `realized_pnl` 在 PG 里 **全是 0 行**；
- MongoDB 反而是真的灌进去了（7 个 collection 都有数据），但其设计文档 `docs/db_schema_mongo.md` 开头仍写着"**MongoDB 尚未安装**"——文档与现实相反；
- 双写在 cron 里以 `WRITE_MODE=dual` 常态开启，但 `logs/db_write_failures.log` 显示自 2026-05-28 后**只有失败、没有成功**，且失败信息可疑（见下）。

也就是说：**PG 这一层目前是"波将金村"——脚手架齐全、测试通过、但里面基本是空的**。
一旦 `DATA_BACKEND=postgres`，绝大多数页面会因为 `try/except` 静默返回空 DataFrame 而"看起来没数据"。

定级：**单人研究级，处于"小团队可用"的上沿，尚未到准生产。**

---

# 第一部分：数据工程六层架构对照

## Layer 1 — Data Generation（数据生成）

| 项 | 现状 |
|---|---|
| 主要外部源 | **Tushare Pro**（2000 分套餐）——财报、披露日、复权因子、北向、融资融券、指数。见 `quantmind/data/tushare_provider.py` |
| 备用源 | **AKShare**（`quantmind/data/akshare_provider.py`，16KB，已实现）；`configs/default.yaml` 写 `primary_provider: akshare, fallback: [tushare]` |
| 用户数据 | 自选股 `data/watchlist/`、推荐反馈 `data/feedback/`、人工标注 `data/dpo/` |
| Agent 生成 | 六维辩论结果 `reports/investment_pipeline/{date}/strategies.json`、推荐 `data/recommendations/{date}/` |
| 模型生成 | IC/SHAP `data/features/top_factors_v3*.json`、Regime `data/regime/`、meta-learner `data/meta_learner/` |

**真实外部 vs 内部派生**：真正"外生"的只有 Tushare/AKShare 拉来的量价+财报+北向；
其余（因子面板、推荐、Agent 信号、NAV、执行订单）**全是系统内部派生**。

**⚠️ 单点风险（确认存在）**：虽然配置层写了 akshare 主 / tushare 备，但 **PIT 数据采集主路径 `quantmind/data/snapshot.py:41` 直接 `from quantmind.data.tushare_provider import TushareProvider`，硬编码 Tushare，没有走可配置 provider，也没有 akshare 兜底**。所以快照构建（整个训练/回测数据的源头）**完全依赖 Tushare 单一数据源**。Tushare 挂了或积分耗尽 → 整条历史数据链断。

---

## Layer 2 — Data Acquisition / Ingestion（数据采集）

| 项 | 现状（含文件路径） |
|---|---|
| Provider 接口 | `quantmind/data/base.py`（抽象基类 `DataProvider`、列名映射、ticker 规范化）；`tushare_provider.py` / `akshare_provider.py` 实现 |
| 限频 | `tushare_provider.py:99 _rate_limit()` —— 进程内最小间隔 0.25s（≈200 次/分），线程锁保护 |
| 超时 | 双重保险：`socket.setdefaulttimeout(30)`（TCP 层）+ `DataApi(timeout=15)`（HTTP 层）。注释解释得很清楚（`tushare_provider.py:24-28,72-78`） |
| 重试 | `_TRANSIENT_KEYWORDS`（频率/timeout/502/503/504…）+ `tenacity`（依赖里有）；客户端单例懒加载 + 双 token（`TUSHARE_HI_URL` 高频代理） |
| 调度 | **cron**（WSL，见 Layer/编排）——非触发式、无 DAG |
| 增量 vs 全量 | 快照是 **按 `as_of` 全量冻结**（`snapshot.py`）；价格面板 `scripts/data_pipeline/step2_download_prices.py` 支持分批 |

**幂等性**：快照写入按 `data/snapshots/{as_of}/` 目录覆盖，meta.json 记录 row_count/列名/日期范围/PIT 列（`snapshot.py:50 module_manifest_for_dataframe`）。同一日期重拉 = 覆盖，**不会重复堆积**。这点做得好。

**ingestion 审计**：snapshot 的 meta.json 是一个不错的"局部血缘"记录（provider、PIT 列、行数、schema 版本）。但**没有统一的 ingestion 日志/审计表**，无法回答"上周二这批价格是几点拉的、拉了多少行、有没有缺"。

---

## Layer 3 — Storage（存储层）

### 三套存储的"账实对照"（核心发现）

实测（2026-06-03，只读查询）：

**PostgreSQL（`localhost:5432`，PG16，进程在跑）**——`pg_stat_user_tables`：

| 表 | PG 行数 | parquet/源头 真实行数 | 状态 |
|---|---:|---:|---|
| `simulated_orders` | **100** | （E3 直写 PG）| ✅ 唯一有数据 |
| `price_daily` | 0 | 2,273,529 | ❌ 空 |
| `alpha_panel` | 0 | 29,406 | ❌ 空 |
| `realized_pnl` | 0 | 80 | ❌ 空 |
| `daily_basic` / `index_daily` / `fina_indicator` / `regime_features` / `alpha_universe` / `snapshot_*` / `nav_curve` / `daily_prices_panel` | 0 | 各有 | ❌ 全空 |

> 结论：**PG 的"迁移"只建了 schema（`docs/db_schema_postgres.sql`，13 张表），关系型大表的数据从未真正灌入**。唯一例外 `simulated_orders` 是 E3 执行层直接写 PG 的（`app/services/data_service.py:650` 读、`scripts/backfill_executions.py` 写）。

**MongoDB（`localhost:27017`，mongod 在跑）**——实测 7 个 collection 都有数据：

| collection | 文档数 | 文件源头 |
|---|---:|---|
| `recommendations` | 13 | `data/recommendations/{date}/` |
| `agent_analysis` | 51 | `reports/.../strategies.json` |
| `sim_daily` | 60 | `data/sim30d/` |
| `positions` | 32 | `data/paper_trading/forward_positions.json` |
| `watchlist_scores` | 2 | `data/watchlist/scores/` |
| `strategy_config` | 1 | `data/paper_trading/strategy_config*.json` |
| `loss_signals` | 2 | `data/loss_signals_v4/` |

> 矛盾点：`docs/db_schema_mongo.md` 开头写"**状态：纯规划文档…MongoDB 尚未安装**"，但实测 Mongo 已安装、已填充、连接正常。**文档严重滞后于现实**。

**Parquet（真正的 source of truth）**：`data/` 下 27 个子目录；最大文件 `data/raw/alpha_prices_panel.parquet`（77.2MB / 2.27M 行）。

### 三者关系（诚实版）

- **parquet = 真源**；**Mongo = 文档类数据的真实副本（已用）**；**PG = 半成品/僵尸层（除执行订单外为空）**。
- 不是"冗余"也不是"互补"，是"**一次没做完的迁移**"，且没有明确的完成线（finish line）。

### PIT 快照机制 ✅

`data/snapshots/{date}/` 把整个市场冻结到磁盘（prices/financials×3/indicators/universe/north_bound/stock_basic/hk_hold/margin/index_daily + meta.json）。
共 ~80 个快照日期（2019Q1 → 2026-06-02）。**这是本项目数据工程里最扎实的一块**，保证回测/Agent 可 100% 复现。

### 命名规范不一致（确认）

- 季末快照用 `{date}/`（目录），推荐用 `{date}/top10.json`，因子用 `{date}.json`（文件）——**目录 vs 文件混用**。
- 因子列名跨层不统一：表达式因子 `momentum_21d` ↔ alpha_panel `momentum_1m`（`CLAUDE.md` 因子表里就并列了两套名），靠 `panel_equiv` 字段对齐。
- `agent_signals` 存在 **rich（含 confidence/summary）/ flat（纯 float）两种格式**，`data_service.py:259 _shape_strategy` 用 isinstance 兼容——典型的 schema 漂移。

### 周期不一致

`alpha_panel` 季频（29406 行，~30 个季末截面）vs daily 数据日频。对齐靠 `as_of` 主键 + PIT 快照"取 ≤ as_of 最近一期"，逻辑上自洽，但**没有显式文档说明对齐规则**。

---

## Layer 4 — Transformation / Cleaning（转换/清洗）

| 项 | 现状 |
|---|---|
| 因子计算 | `quantmind/features/`（17 个模块）：`expr_factors.py`（Qlib 风格表达式）、`fundamental.py`/`em_fundamental.py`、`technical.py`、`north_flow.py`、`analyst_revision.py`、`text_sentiment.py` 等 |
| 标准化 | `standardize.py`（截面 zscore）；`neutralize.py`（行业去均值 + 可选市值 OLS 残差，`neutralize_cross_section`） |
| 表达式一致性 | `expr_factors.py` 自带 `validate_expr_consistency`（expr vs 参考 Python，要求 Spearman ≥ 0.99）——**这是好实践** |
| PIT 严格性 | `configs/default.yaml: pit_strict: true`；财报用 `f_ann_date`（实际公告日，`tushare_provider.py:5-7`）；pytest 有专门的 `pit` marker（`pyproject.toml:248`） |

**因子依赖/血缘风险**：因子链是 `量价/财报 → 单因子 → standardize → neutralize → alpha_panel → 模型`。
**没有显式的因子 DAG**，所以"某个上游因子算错会污染哪些下游"只能靠读代码推断。`neutralize_cross_section` 对所有数值列做行业去均值——如果 `industry` 列脏（缺失/错分类），会**静默地**污染当截面所有因子（`neutralize.py:34` 行业列缺失时直接 return 原样，不报错）。

**缺失/异常值**：`neutralize.py` 用 `pd.to_numeric(errors="coerce")` 把异常转 NaN 后保留；标准化层处理 zscore。但**没有看到统一的异常值识别规则（如 winsorize/MAD 阈值）做成可配置的清洗层** —— 散落在各因子模块里。

---

## Layer 5 — Serving（服务层）

**`app/services/data_service.py`（`DataService`，单例 `get_data_service()`）—— 本项目最好的设计之一。**

- 统一读路径：推荐/六维分析/PnL/持仓/模型状态/Regime/损失信号/执行订单/数据新鲜度，全部一个类。
- 实例级缓存 `self._cache` + `clear_cache()`。
- **后端切换**：`DATA_BACKEND ∈ {parquet, postgres}`（`data_service.py:69`）。每个方法按 backend 分派到 `_parquet_*` / `_pg_*` / `_mongo_*`。
- 防御式：每个方法 `try/except` → 失败返回空 DataFrame/None，不崩页面。

**双写**：`app/db/writers.py`（`DataWriter`，`WRITE_MODE ∈ {parquet_only, dual, db_only}`）。
设计原则正确：**parquet 先写（失败中断业务）→ DB 后写（失败只 warn + 记 `logs/db_write_failures.log`，业务不中断）**。失败隔离这点是对的。

### ⚠️ "所有页面只走 DataService" 的说法不成立

`data_service.py` 文件头宣称"所有前端页面只通过 DataService 读数据，不再直接 read_parquet / json.load"。
**实测：14 个页面里有 5 个仍直接读文件**（`grep` 命中）：

- `app/pages/7_系统控制台.py`、`8_Regime状态.py`、`9_持仓跟踪.py`、`10_持仓详情.py`、`14_执行管理.py`

即统一访问层的"统一"是 ~64% 完成度，宣称值得修正。

### 后端切换的隐患

因为 `DATA_BACKEND=postgres` 时，空表会经由 `try/except` 静默返回空 DF，**切到 postgres 后端不会报错，只会"没数据"**——这种"静默失败"是最难排查的一类问题。

---

## Layer 6 — Analytics / ML（分析与机器学习）

| 消费方 | 读什么 / 怎么读 |
|---|---|
| LGBM（4 个：main/gem/star/alpha）| `models/lgbm_v6_*.pkl`；训练读 `data/panel/alpha_panel_v4.parquet`；`scripts/train_lgbm_model.py` |
| FactorCNN | `models/factor_cnn_v2_augmented.pkl`（含 val_ic/val_icir）|
| HMM Regime | `data/regime/regime_history.parquet`（DataService 取末行，避免重 fit，`data_service.py:560`）|
| Meta-Learner | `data/meta_learner/meta_learner_v*.meta.json`（取最新版本）|
| 6-Agent / DebateOrchestrator | `daily_update.py:1143 step7a`，上下文来自快照 raw_factors + LGBM 分 + regime |
| 实时 vs 批量 | 批量训练读 parquet 面板；实时打分 `compute_agent_analysis_live`（`data_service.py:356`）现算（fast 模式）|

**模型输出回流**：预测分 → `data/recommendations/`；Agent 信号 → `strategies.json`；执行订单 → PG `simulated_orders`；NAV → `data/sim30d/`、`data/iteration/`。
回流路径清晰，但**回流目标分散在 parquet/json/PG/Mongo 四处**，没有统一的"实验结果存储"。

---

# 第二部分：六个横向维度诊断

## ⭐ 评分总览（雷达图数据）

| 维度 | 评分 | 一句话 |
|---|:---:|---|
| Security 安全 | **3 / 5** | 单人级卫生不错，离最小权限/密钥托管还远 |
| Data Management 治理 | **2 / 5** | 无数据字典、文档过期、无血缘、无质量监控 |
| DataOps 运维 | **2 / 5** | 有 cron/CI/失败日志，但告警没接、监控没人看、日志可信度存疑 |
| Data Architecture 架构 | **2 / 5** | 三套存储半迁移，PG 空表成僵尸层（DataService 设计本身是加分项）|
| Orchestration 编排 | **2 / 5** | 纯时间 cron 无依赖门控，且有月末任务 cron 语义 bug |
| Software Engineering 软件工程 | **4 / 5** | 真正的强项：ruff/mypy/1097 测试/CI/PIT 标记 |

```
雷达图（0-5）：
Security              ███████░░░  3
Data Management       █████░░░░░  2
DataOps               █████░░░░░  2
Data Architecture     █████░░░░░  2
Orchestration         █████░░░░░  2
Software Engineering  █████████░  4
```

---

## Security（安全性）— 3/5

**做得对**：
- `.env` / `api_key.txt` / `*.key` / `secrets/` 均在 `.gitignore`；`git ls-files` 确认 **未被跟踪**（只跟踪了 `.env.example` 模板）。
- `git log -S password` 未发现历史里 commit 过明文密码。
- PG 角色 `quantmind`：`rolsuper=f, rolcreatedb=f, rolcreaterole=f` —— **不是集群超级用户**。
- PG/Mongo 都只监听 `127.0.0.1`，未对外暴露。
- cron 跑 daily_update 带 `--no-llm --agent-provider none` → **当前无 LLM 数据外发**。

**缺口**：
- PG 角色虽非超管，但 **OWNER 整个 quantmind 库** → 对所有表有 DDL/DML（`writers.py` 里真的在 `TRUNCATE realized_pnl`）。**非 per-table 最小权限**。
- `api_key.txt`（242B）裸放在仓库根目录（虽 gitignore），是松散密钥文件；密钥用明文 `.env`，**无密钥托管/轮转**。
- 无显式的"Agent 调 LLM 不得带 PII"策略——目前靠"关掉 LLM"规避，不是治理。

---

## Data Management（数据治理）— 2/5

- **数据字典**：只有 `docs/db_schema_postgres.sql`（带 COMMENT）+ `docs/db_schema_mongo.md`。**parquet 层没有字段字典**。且 `alpha_panel` 实测 **75 列**，schema 文档只列了 ~35 列 → **账实不符**。
- **文档可信度**：Mongo 文档说"尚未安装"（实际已用）；这类过期文档会直接误导新人。
- **血缘**：无 lineage 工具；snapshot 的 meta.json 是唯一的局部血缘亮点。从 Tushare 到一个推荐结果**无法机器化追溯**。
- **Schema 演进**：靠文件名后缀（`alpha_panel_v4`、`lgbm_v6`、`meta_learner_v*`）做版本，**无迁移脚本/版本注册表**；`agent_signals` 两种格式靠 isinstance 兼容（已是技术债）。
- **质量监控**：只有 `get_data_freshness`（新鲜度）+ 一次性 `scripts/audit_*.py`。**无持续的行数突变/空值率/分布漂移检查**。（`loss_signals/factor_health` 是模型表现监控，不是数据质量监控）。

---

## DataOps（数据运维）— 2/5

- **自动化**：5 个 cron（见编排）；CI `.github/workflows/ci.yml`（push/PR/每周日）。
- **监控/告警**：`app/db/writers.py` 有失败/审计双日志基础设施，且有双写监控页。**但**：
  - `logs/db_write_audit.log` **最后一次成功是 2026-05-28**；此后（05-30、06-02、06-03）`db_write_failures.log` **只有失败**。
  - 失败信息可疑：2026-06-03 的 08:59/10:58/11:31 写着 `PG down` / `Mongo down` / `DB down`，**但我实测此刻 PG/Mongo 都连得上**（`select 1` OK、`db.command('ping')` OK）。真实的 psycopg2/pymongo 报错应该是长串描述（早期条目 "ConnectionRefusedError could not connect to MongoDB" 才像真的）。这些短消息**像是合成/测试注入**，用来喂监控页。
  - 审计日志里出现 **`date=2026-00-01`（不可能的月份 00）** → 日志本身有数据完整性 bug。
  - cron 全程 `--no-alert`；`quantmind/notify/` 模块存在但**失败没有真正 page 给任何人**。
- 回答记忆里的问题"双写监控页是不是真在被看"：**证据指向"没人盯"**——失败堆了一周无人处置，且日志被合成数据污染。
- **测试**：CI 只跑 `pytest -m "not integration"`，**无数据契约测试、无覆盖率门槛**。

---

## Data Architecture（数据架构）— 2/5

- **范式**：纯批（pure batch），无流式。对"日频再平衡的量化研究"**这是合理选择**，不需要 lambda/kappa。
- **三套存储的合理性**：现状不合理——parquet（真源）+ Mongo（已用）+ PG（除执行订单外空表）。**PG 当前是僵尸层**，flip 后端即"无数据"。三套并存缺少明确分工文档（谁主谁备、PG 是要替代 parquet 还是补充）。
- **DataService backend 切换**：抽象本身是**好设计**（单读路径、可测、防御式）。但它当前**掩盖了 PG 空表的事实**（静默返回空）——好抽象被用来藏问题，是"过度设计先行于数据落地"的典型。
- **历史包袱 vs 深思熟虑**：PIT 快照、DataService、expr_factor 一致性校验 = 深思熟虑；三套存储半迁移 + 切换开关先于数据 = 包袱。

---

## Orchestration（任务编排）— 2/5

实测 WSL crontab（5 个任务）：

| 时间 | 任务 | 备注 |
|---|---|---|
| `30 16 * * 1-5` | `daily_update.py`（dual）| 工作日 16:30，主流程 |
| `45 16 * * 1` | `track_realized_pnl.py`（dual）| **周一 16:45，仅比 daily_update 晚 15 分钟** |
| `05 17 * * 1` | `dispatch_loss_signals.py`（dual）| 周一 17:05 |
| `10 17 28-31 * 1-5` | `paper_trading_sim.py` | **见下方 cron bug** |
| `15 17 28-31 * 1-5` | `update_sim_strategy.py`（dual）| 同上 |

- **无依赖门控（确认）**：track_realized_pnl 在 daily_update 后 15 分钟无条件启动。**若 daily_update 失败或还没跑完，下游照样在旧数据上跑**——没有"上游成功才跑下游"的机制。
- **🐛 cron 语义 bug（确认）**：`28-31 * 1-5` —— 当 **day-of-month 和 day-of-week 同时被限定（都不是 `*`）时，cron 取"或"**。即"日∈28-31 **或** 周∈一-五"→ 实际上**每个工作日都会跑**，根本不是预期的"月末"。这两个本该月末跑的任务每天都在跑。
- **无重试策略**（cron 层）；API 层有 tenacity，但任务级失败不重试。
- **长任务进度不可见**：daily_update 1922 行 / 9 步，跑很久时只能 `tail` 日志，无进度上报/心跳。
- **未引入编排器**：无 Airflow/Prefect/Dagster；以现规模 cron 尚可，但依赖关系一多就危险。

---

## Software Engineering（软件工程）— 4/5

**真正的强项**：
- 包结构清晰：`quantmind/` 22 个子包，职责分明（data/features/models/agents/portfolio/regime/risk/execution/...）。
- 工具链完整：`pyproject.toml` 配了 **ruff（E/W/F/I/B/C4/UP/N/SIM）+ mypy（core.* 开 strict）+ pytest + pytest-cov + pre-commit**；依赖**单一真源**（pyproject 的 optional-dependencies 分组，无 requirements.txt/environment.yml 漂移问题）。
- 测试：**68 文件 / 1097 个 test 函数**；有专门的 `pit`、`integration`、`slow`、`gpu` marker。
- 类型注解：全项目 `from __future__ import annotations`，类型覆盖较好。

**缺口**：
- **覆盖率 56%**（`htmlcov/index.html`）——中等；`ui/` 已 omit；关键路径有 pit 标记是好的，但 56% 离"准生产"差一截。
- **吞异常严重**：`quantmind/` + `app/` 里 **~409 处宽 except，其中 ~60 处 `except: pass`**。配合 DataService"失败返回空"模式，**真实数据 bug 会被伪装成"没数据"**。
- **仓库卫生**：根目录有垃圾文件 `8}`（0B）、两个 `*:Zone.Identifier`（Windows 下载标记）、`scripts/_*.py`（十几个 `_diag_*`/`_compare_*`/`_verify_*` 临时脚本）、`scripts/db_migration/_commit_msg*.txt` 等未清理。
- 文档量其实不少（README/HANDOVER/METHODOLOGY/Spec/blog×3/QUICKSTART），但**部分过期**（mongo 文档、schema 列数）。

---

# 第三部分：诚实诊断（6 个直球问题）

### 1. 这个"数据工程底座"现在是什么水平？

**单人研究级，站在"小团队可用"的上沿，未到准生产。**
往上拉的是软件工程纪律 + PIT 快照 + DataService；往下拽的是空 PG、半迁移、纯时间编排、宽 except 静默失败、零备份。
准生产至少还差：可用的存储层、备份/DR、依赖编排、被人看的告警。

### 2. 最危险的三个问题（不粉饰）

1. **PG 是"波将金村"**：schema + 后端切换 + 双写 + 测试全有，但表基本是空的（除 simulated_orders）。`DATA_BACKEND=postgres` 一开，绝大多数页面经 `try/except` 静默返回空——**"看起来有 DB 层，功能上没有"**。这是最危险的，因为它**伪装成已完成**。
2. **编排无正确性保证 + 一个真 cron bug**：下游任务不管上游成败按点就跑；`28-31 * 1-5` 因 cron 的 DOM/DOW 或语义，月末任务**实际每个工作日都在跑**。
3. **零备份 + 没人看的监控**：`data/` 与 Mongo **没有任何自动备份**；双写失败堆了一周无人处置，且失败日志含**合成痕迹（"PG down" 实际 PG 在线）和不可能日期 `2026-00-01`**——**监控数据本身不可信**。

### 3. 如果今天数据库挂了，能多快恢复？parquet 备份够吗？

- **DB 挂了 = 低影响**：PG 本来就基本空（simulated_orders 100 行可由 `backfill_executions.py` 从 parquet+推荐重建）；Mongo 7 个 collection **全部有文件级真源**（recommendations/{date}、strategies.json、forward_positions.json、loss_signals_v4/），可重跑 writer 重建。
- **真正的单点是 parquet/json 文件树本身**：它才是 source of truth，却**没有任何自动备份**。`data/raw/alpha_prices_panel.parquet`（77MB/2.27M 行）+ ~80 个 PIT 快照一旦丢失，**只能重新从 Tushare 拉**（数小时到数天，且部分历史 PIT 未必能完美复现）。
- **结论**：parquet "够当 DB 的备份"，但 **parquet 自己没备份**——这才是要补的洞。

### 4. 新人多久能搞清 Tushare→推荐 的完整链路？

**约 2–4 天**（有经验的量化工程师）。
有利：文档不少 + CodeGraph 索引 + snapshot meta.json。
不利：链路长（tushare_provider → snapshot → build_full_panel/features → daily_update **1922 行/9 步** → recommendations.json → DataService → 页面），artifact 带 v4/v6 版本后缀，**无单张血缘图**，且会被"配置写 akshare、实际跑 tushare""文档说 Mongo 没装、其实在用"这类**账实不符**绊倒。

### 5. 要"云上 7×24 服务 100 用户"，还差什么？

差很多，本质上它是研究系统不是服务：
(a) **没有真正的存储层**（PG 空，靠单机文件）；(b) 无鉴权/多租户/API 网关（Streamlit 单进程）；(c) 无备份/DR；(d) 无编排器/重试/告警；(e) 密钥在 .env 非密钥托管；(f) 跑在工作站 WSL+cron，未容器化/上云；(g) 缓存是实例级，无水平扩展；(h) 无可观测性/SLO。
**保守估计数月工程量。**

### 6. 哪些是"小聪明"，哪些是"真正的好设计"？

**真正的好设计**：
- PIT 快照（冻结到盘 + meta.json 清单 + schema 版本）
- DataService 统一访问层（单读路径、防御式、可缓存）
- expr_factor 的 expr-vs-Python 一致性校验（≥0.99）
- 双写的**失败隔离**（parquet 永不被 DB 拖垮）
- pyproject 单源依赖 + ruff/mypy/pit-marker 纪律

**小聪明（聪明但脆）**：
- `DATA_BACKEND`/`WRITE_MODE` 切换开关**先于 PG 数据落地**就上线 → 制造"有 DB 层"的错觉
- 满地 `try/except 返回空` → 把失败伪装成"无数据"
- BRIN 替 TimescaleDB（本身没错，但表是空的，讨论 moot）
- 用 sed 往 cron 注入 `WRITE_MODE=dual`（`_update_cron.sh`）
- 看起来"活着"的合成失败/审计日志，让监控页显得在工作

---

# 第四部分：路线图建议

## 🔴 必须做（P0，最多 5 条）

| # | 现状问题 | 建议方案 | 工作量 | 价值/风险 |
|---|---|---|---|---|
| 1 | **PG 空表 + 静默返回空**，后端切换是假象 | 二选一并收口：**要么**真正灌 `price_daily/alpha_panel/realized_pnl/...` 进 PG 并加行数一致性校验（用现成 `db-migration` skill）；**要么**把 `DATA_BACKEND=postgres` 标为"实验性"，在 DataService 对空表**显式告警**而非静默返回空 | 2–4 天 | 价值高 / 风险中 |
| 2 | cron `28-31 * 1-5` 实际每天跑 + 无上游门控 | 修月末语义（用 `[ $(date +\%d) -ge 28 ]` 守卫，DOW 设 `*`）；daily_update 成功才触发 track_pnl/loss_signals（落地一个 `.ok` 标记文件门控）| 0.5–1 天 | 价值高 / 风险低 |
| 3 | `data/` 与 Mongo **零备份** | 每日 `rsync` 快照 `data/` + `mongodump`，至少落到另一块盘/对象存储；保留 N 份滚动 | 1 天 | 价值高 / 风险低 |
| 4 | 双写失败没人看 + 日志被合成数据污染 | 接通告警（`quantmind/notify/` → 企业微信/邮件）：双写失败/ cron 失败即推送；清理审计日志里的合成条目与 `2026-00-01` 脏数据 | 1 天 | 价值高 / 风险低 |
| 5 | 无数据字典 + 文档过期 | 写一份 parquet+PG+Mongo 统一字段字典；修正 `db_schema_mongo.md`（已安装）、`alpha_panel` 列数（75≠35）、配置里 akshare/tushare 主备与现实对齐 | 1–2 天 | 价值中高 / 风险低 |

## 🟡 应该做（P1）

- 引入 **Prefect/Dagster**（或先用 Makefile-DAG + `.ok` 门控）做依赖编排 + 任务级重试 + 进度可见。
- **数据质量监控**：行数突变 / 空值率 / 分布漂移，纳入 daily 或 CI（pandera / great-expectations）。
- **收紧宽 except**：区分"数据缺失（返回空合理）"与"真实异常（应抛/告警）"，至少给 60 处 `except: pass` 加日志。
- CI 加**数据契约测试** + **覆盖率门槛**（56% → 70%）。
- PG 改 **per-table 最小权限角色**，应用账户去掉 owner/DDL 权限。

## 🟢 可以做（P2）

- 仓库清理：`8}`、`*:Zone.Identifier`、`scripts/_*.py` 临时脚本、`_commit_msg*.txt`。
- 把 **akshare fallback 真正接入 `snapshot.py`**，消除 Tushare 单点。
- 用**版本注册表**取代 `v4/v6/meta_v*` 文件名后缀。
- 在 parquet 之上加 **DuckDB** 查询层，替代部分 pandas 全量读盘。

---

## 附：本次审计的实证命令（均为只读）

- PG 行数：`SELECT relname, n_live_tup FROM pg_stat_user_tables`
- PG 角色：`SELECT rolname, rolsuper, ... FROM pg_roles WHERE rolname='quantmind'`
- Mongo：`db.getCollectionNames().forEach(c => print(c, db[c].countDocuments({})))`
- 连接探测：`select 1` / `db.command('ping')`（确认 2026-06-03 两库均在线）
- 端口/进程：`ss -tlnp | grep 5432/27017`、`pgrep postgres/mongod`
- parquet 真值：`pd.read_parquet(...).shape`
- 秘密扫描：`git ls-files | grep -iE '\.env|api_key'`、`git log -S password`
- 编排：`crontab -l`（WSL）
- 测试/覆盖：`grep -rc def test_`、`htmlcov/index.html`

> 报告完。本轮只读，未改任何代码、未 commit。
