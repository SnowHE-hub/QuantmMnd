# QuantMind 数据库迁移规划

> **状态**：纯规划文档，2026-06-02 起草。所有 DDL/Schema 已写出但**尚未执行**。
> 执行前须经过人工确认。

---

## Step 0：环境现状确认

### PostgreSQL
| 项目 | 状态 |
|------|------|
| 版本 | PostgreSQL 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1) ✅ |
| 服务 | 运行中（/var/run/postgresql/.s.PGSQL.5432） ✅ |
| TimescaleDB | **未安装**（pg_available_extensions 中不存在）❌ |
| 角色 `lenovo` | **不存在**，目前只有 `postgres` 超级用户 ❌ |

### MongoDB
| 项目 | 状态 |
|------|------|
| mongod | **未安装** ❌ |
| mongosh | **未安装** ❌ |

> ⚠️ **关键差距**：MongoDB 并未安装。用户在任务描述中说"本机已装 PostgreSQL 和 MongoDB"，
> 经实际检测只有 PostgreSQL。MongoDB 安装是迁移前置条件。

### Python 驱动（conda env: quantmind）
| 包 | 状态 |
|----|------|
| sqlalchemy 2.0.49 | ✅ 已安装 |
| psycopg2 | ❌ 未安装（需 `pip install psycopg2-binary`） |
| pymongo | ❌ 未安装（需 `pip install pymongo`） |
| motor（异步 pymongo） | ❌ 未安装（可选） |

### 前置条件清单（执行迁移前必须完成）
```bash
# 1. 安装 MongoDB Community
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org
sudo systemctl enable --now mongod

# 2. 创建 PostgreSQL 角色和数据库
sudo -u postgres psql -c "CREATE USER quantmind WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "CREATE DATABASE quantmind OWNER quantmind;"

# 3. 安装 Python 驱动
conda run -n quantmind pip install psycopg2-binary pymongo python-dotenv

# 4. 创建 .env 文件（不提交到 git）
cat >> .env << 'EOF'
POSTGRES_DSN=postgresql://quantmind:your_password@localhost:5432/quantmind
MONGO_URI=mongodb://localhost:27017/quantmind
EOF
```

---

## Step 1：数据源盘点与分类

### A. 进入 PostgreSQL（结构化时序数据）

| 数据源 | 当前路径 | 规模 | 主要字段 | 索引策略 |
|--------|----------|------|---------|---------|
| **alpha_prices_panel** | `data/raw/alpha_prices_panel.parquet` | 2,273,529 行 × 13 列 | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, adj_close | (ts_code, trade_date) 复合主键；BRIN on trade_date |
| **daily_prices_panel** | `data/raw/daily_prices_panel.parquet` | 728,370 行 × 11 列 | trade_date, ts_code, open, high, low, close, vol, amount（+ adj 列） | 同上，Alpha universe 子集 |
| **index_daily_panel** | `data/raw/index_daily_panel.parquet` | 6,064 行 × 8 列 | trade_date, ts_code, close, pct_chg, open, high, low, vol | (ts_code, trade_date) |
| **alpha_daily_basic** | `data/alpha_universe/alpha_daily_basic_combined.parquet` | 37,494 行 × 10 列 | ts_code, trade_date, pe_ttm, pb, ps_ttm, dv_ratio, total_mv, circ_mv | (ts_code, trade_date) |
| **fina_indicator** | `data/alpha_universe/fina_indicator_combined.parquet` | 62,931 行 × 14 列 | ts_code, ann_date, end_date, roe, roa, grossprofit_margin, netprofit_margin, assets_turn | (ts_code, end_date) |
| **alpha_universe** | `data/alpha_universe/alpha_universe.parquet` | 1,374 行 × 8 列 | ts_code, name, industry, market, list_date, in_csi300, in_csi500, in_csi800 | ts_code PK（参考表） |
| **alpha_panel**（历史截面）| `data/features/alpha_*.parquet` + `csi300_full_panel.parquet` | 6,905 行 × 67 列（合并后更多）| as_of, ticker, 65 个因子列 | (as_of, ticker) 复合主键；idx on ticker |
| **regime_features** | `data/features/regime_features.parquet` | 25 行 × 8 列 | as_of, csi300_63d_return, csi500_csi300_63d, chiext_csi300_63d, csi300_20d_vol... | as_of PK |
| **realized_pnl** | `data/feedback/realized_pnl.parquet` | 80 行 × 16 列 | as_of_date, ticker, entry_date, exit_date, holding_days, predicted_rank, predicted_score, entry_price, exit_price, actual_return | (as_of_date, ticker) |
| **nav_curve** | `data/sim30d/nav/*.json` + `data/iteration/*/nav/*.json` | ~60 files（每文件单天）| run_id, trade_date, nav, daily_return, positions | (run_id, trade_date) |
| **snapshot_daily_basic** | `data/snapshots/{date}/daily_basic.parquet` | ~35 日期 × ~1374 行 | ts_code, as_of, close, pe_ttm, pb, total_mv, circ_mv | (as_of, ts_code) |

**归入 PG 的理由**：行×列结构固定、有时间索引、DataService 主要做 WHERE/ORDER BY/聚合查询（IC 计算、因子截面、NAV 曲线），SQL 优于文档查询。

---

### B. 进入 MongoDB（半结构化文档）

| 数据源 | 当前路径 | 规模 | 结构特征 |
|--------|----------|------|---------|
| **recommendations** | `data/recommendations/{date}/top10.json` | ~15 日期 × 10 股 | 嵌套：`top10[].key_factors`, `top10[].agent_signals`，字段随版本变化 |
| **forward_positions** | `data/paper_trading/forward_positions.json` | 20 条持仓 | 嵌套 position 对象，含 null 字段（open/closed） |
| **strategy_config** | `data/paper_trading/strategy_config_v2.json` | 1 doc，深嵌套 | 多层嵌套：holding_period.rationale, system2_updates.weights_calibrated, reversed_dim_factors[] |
| **loss_signals** | `data/loss_signals_v4/*.json` | 4 files | 可变字段：signal_1_ranking_loss, factor_decay 含动态数组（improving, decaying） |
| **agent_analysis** | `reports/investment_pipeline/{date}/strategies.json` | 多 stock per date | 六维 agent_signals 嵌套字典，字段随模型迭代变化 |
| **sim_daily** | `data/sim30d/daily/*.json` + `data/iteration/*/daily/*.json` | ~60 files | system1_candidates 数组，每天不同长度 |
| **iteration_config** | `data/iteration/*.json` | 2 files | 版本化参数，schema 频繁演化 |
| **watchlist_scores** | `data/watchlist/scores/*.json` | 每日一文件 | 自由格式评分，字段不固定 |

**归入 MongoDB 的理由**：字段集随模型版本变化（无固定 schema）、深层嵌套结构 JSON 天然映射 BSON document、前端读取方式是 `json.loads(p.read_text())` 而非 SQL 聚合。

---

## Step 4：迁移策略——六大问题回答

### Q1. 一次性全量迁移 vs 双写过渡？

**推荐：双写过渡，过渡期 2 周**

理由：QuantMind 是实时交易辅助系统，parquet 文件是当前真实数据源，DataService 层已经 稳定运行。一次性切换风险高——万一迁移出现字段精度问题、时区问题、数据截断，会直接影响每日推荐和前端展示。

**两阶段方案**：
```
Phase 1（1 周）：全量历史数据导入 DB → 验证一致性 → 仍从 parquet 读取
Phase 2（1 周）：写入脚本改为双写（parquet + DB 同时写）→ DataService 切换为从 DB 读 → 观察
Phase 3：确认无误后，停止写 parquet，parquet 作为只读备份
```

---

### Q2. daily_update.py / track_realized_pnl.py 迁移后写哪里？

**推荐：改为双写，之后逐步过渡到纯 DB 写入**

```python
# 模式：同时写 parquet（保留备份）+ DB
def save_realized_pnl(df: pd.DataFrame):
    # 原来的 parquet 写入保留
    df.to_parquet(PNL_PATH)
    # 新增 DB 写入
    engine = get_pg_engine()
    df.to_sql("realized_pnl", engine, if_exists="append", index=False,
              method="multi", chunksize=500)
```

对于 JSON 类写入脚本（forward_positions, recommendations）：
```python
# 原来的 json.dump 保留
json.dump(data, f)
# 新增 MongoDB 写入
db["forward_positions"].replace_one({"_id": date}, doc, upsert=True)
```

---

### Q3. DataService 怎么改？抽象层 vs 直接换 DB？

**推荐：加 storage backend 抽象层（可切换）**

在 DataService 内部引入轻量 backend 概念，不改变任何外部接口：

```python
# app/services/data_service.py 内部新增
class _StorageBackend:
    PARQUET = "parquet"
    POSTGRES = "postgres"

class DataService:
    def __init__(self, ..., backend: str = "parquet"):
        self._backend = backend
        ...

    def get_realized_pnl(self) -> pd.DataFrame:
        if self._backend == "postgres":
            return self._pg_load_realized_pnl()
        return self._parquet_load_realized_pnl()  # 原逻辑
```

环境变量控制切换：
```
DATA_BACKEND=postgres  # .env 中配置
```

这样：
- 前端零改动（只调用 DataService 方法）
- 可随时回滚（改环境变量即可）
- CI/test 可继续用 parquet backend

---

### Q4. 迁移后如何验证数据一致性？

**三层验证**：

```python
# Layer 1：行数一致
assert len(df_parquet) == db.execute("SELECT COUNT(*) FROM table").scalar()

# Layer 2：关键列均值/最大最小抽样比对（浮点精度 1e-6）
for col in ["close", "pe_ttm", "total_mv"]:
    pq_mean = df_parquet[col].mean()
    db_mean = db.execute(f"SELECT AVG({col}) FROM table").scalar()
    assert abs(pq_mean - db_mean) / pq_mean < 1e-5, f"{col} 均值不一致"

# Layer 3：随机抽取 100 行精确比对
sample = df_parquet.sample(100)
for _, row in sample.iterrows():
    db_row = db.execute("SELECT * FROM table WHERE ts_code=:c AND trade_date=:d",
                        {"c": row.ts_code, "d": row.trade_date}).fetchone()
    assert abs(db_row.close - row.close) < 0.001
```

完整验证脚本规划放在 `scripts/db_migration/verify_consistency.py`。

---

### Q5. 回滚方案

**DataService backend 切换**（零停机回滚）：
```bash
# .env 中改一行，重启服务
DATA_BACKEND=parquet  # 从 postgres 回退
```

parquet 文件在双写过渡期**不删除**，作为热备份保留至少 30 天。

数据库侧回滚：
- PostgreSQL：DROP TABLE 或 TRUNCATE（数据库不是唯一真实数据源，所以可以直接丢弃）
- MongoDB：db.collection.drop()

若数据损坏：从 parquet 重新导入（全量迁移脚本可重复执行，使用 `if_exists="replace"`）。

---

### Q6. 连接管理与密码存储

**连接池配置（PostgreSQL）**：
```python
# app/db/postgres.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import os

_engine = None

def get_pg_engine():
    global _engine
    if _engine is None:
        dsn = os.environ["POSTGRES_DSN"]  # 从 .env 读取
        _engine = create_engine(
            dsn,
            poolclass=QueuePool,
            pool_size=5,          # Streamlit 并发不高
            max_overflow=10,
            pool_pre_ping=True,   # 自动检测断连
            pool_recycle=3600,    # 1h 回收连接
        )
    return _engine
```

**连接池配置（MongoDB）**：
```python
# app/db/mongo.py
from pymongo import MongoClient
import os

_client = None

def get_mongo_db():
    global _client
    if _client is None:
        uri = os.environ["MONGO_URI"]
        _client = MongoClient(
            uri,
            maxPoolSize=10,
            serverSelectionTimeoutMS=5000,
        )
    return _client["quantmind"]
```

**密码存储规则**：
- 密码放 `.env`（已在 `.gitignore` 中）
- 使用 `python-dotenv` 加载：`from dotenv import load_dotenv; load_dotenv()`
- 生产/CI 环境用系统环境变量，不依赖 `.env` 文件
- `.env.example` 提交到 git 作为模板（密码用占位符）

```
# .env.example（提交到 git）
POSTGRES_DSN=postgresql://quantmind:CHANGE_ME@localhost:5432/quantmind
MONGO_URI=mongodb://localhost:27017/quantmind
DATA_BACKEND=parquet
```

---

## 数据库目录规划

```
app/
  db/
    postgres.py      # get_pg_engine() 连接池
    mongo.py         # get_mongo_db() 连接管理
    __init__.py

scripts/
  db_migration/
    00_prereqs.sh            # 安装 MongoDB、创建 PG 角色/数据库
    01_pg_create_tables.sql  # 执行 DDL
    02_pg_import_parquet.py  # parquet → PostgreSQL 批量导入
    03_mongo_import_json.py  # json → MongoDB 批量导入
    04_verify_consistency.py # 一致性校验
    05_switch_backend.sh     # .env 切换 DATA_BACKEND

docs/
  DB_MIGRATION_PLAN.md      # 本文档
  db_schema_postgres.sql    # PostgreSQL DDL
  db_schema_mongo.md        # MongoDB schema 设计
```
