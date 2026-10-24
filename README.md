# QuantMind — A 股截面选股研究系统

> 一个把"如何不骗自己"当作核心问题的 A 股量化研究项目：时点正确（PIT）的数据湖、Purged Walk-Forward、可执行净值回测、预登记评估、负对照与跨家族验收。
>
> **研究用途，不构成任何投资建议。截至 2026-10-25，研究闸门 v2 仍未通过，也没有模型用于真实资金；主线 A v2 的正式荐股只在三档纸面账户里运行（用户 2026-10-24 知情批准，go_live 2026-10-26）。**

[![CI](https://github.com/SnowHE-hub/QuantmMnd/actions/workflows/ci.yml/badge.svg)](https://github.com/SnowHE-hub/QuantmMnd/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## 现状一览（2026-10-25）

| 问题 | 答案 | 证据 |
|---|---|---|
| 有模型上线吗 | **没有**。模型注册表里两条研究种子都是 `research_candidate_pending_nav` | `data/registry/model_registry.json`（本地数据，不在 git） |
| 主线 A（每日运行的链路）在真实成交口径下 | 2021-12 → 2026-10 年化净超额 **−5.0%**，IR −0.48，不过闸门 | [`docs/work/reports/T-010-line-a-oos.md`](docs/work/reports/T-010-line-a-oos.md) |
| 主线 B 最好的 Ridge 种子 | 63d 连续样本外 **+5.03%/年**，IR 0.64，16 次再平衡，不过闸门 | [`T-018`](docs/work/reports/T-018-line-b-v6-1.md)、[`T-026`](docs/work/tasks/T-026.md) |
| GPU 模型族（研究分支） | 20d：四族等权集成对 Ridge 的日均 RankIC **+0.053**，预登记主比较**检测到增益**。63d：RankIC +0.073，但只有一个块长的下界 > 0，**未检测到增益（块长敏感）**。两个期限的净值配对差都未检测到，相对回撤明显更高，都不过闸门 | 分支 `t/030-gpu-models`：`docs/work/reports/T-030-gpu-models.md` |
| LLM 重排有用吗 | 19 期、437 次请求的消融：**无可检测增益**，不如一条固定的简单因子规则；生产中关闭 | [`T-007`](docs/work/reports/T-007-anonymization-v3.md) |
| 下一份干净的证据 | 2026-10-09 冻结之后的前瞻预测。前瞻模型包已与历史预测逐位一致，前瞻数据在刷新 | 分支 `t/033-attribution-forward`：`docs/work/tasks/T-034.md` |
| 主线 A v2（每日自动运行的完整系统） | 数据 → 时点快照 → 量化打分（冻结的 ENS 模型包）→ 漏斗 → 6 个 Agent 研究 → **代码逐条核验声明** → 前瞻影子账本 → 正式荐股日报；上面一层 Agent 大脑（只有研究权，没有投资权） | 分支 `t/035-line-a-v2-p0`：`docs/work/tasks/T-036`–`T-059` |
| 正式荐股（纸面账户） | 10 万 / 100 万 / 1000 万三档，规则全部来自事先验证的 T-052 主配置；**用户 10-24 知情批准**（批准时的证据：模拟盘区间包含 0、研究闸门未通过）；首次调仓 2026-10-28 | [`T-057`](docs/work/tasks/T-057.md) |
| 按股数记账的模拟盘（T-052） | 近 6 个月普跌行情：10 万 **+5.0%**、100 万 **+3.6%**、1000 万 **−10.5%**，研究池等权 −21.5%；**95% 区间全部包含 0**，超额集中在 7–8 月、对调仓日敏感 | [`T-052`](docs/work/tasks/T-052.md) |
| 前瞻跟踪（冻结之后） | ENS 20 日：已到期的 5 个运行日，事后 5 日 Rank IC 平均 **−0.02**（研究期 0.145）；样本太少、持有期重叠，只跟踪不检验 | `/api/forward_ic` |
| Agent 研究的代码核验 | 每条声明的数值、单位、引文、计算、证据编号由校验器核对（不由 Agent 兼任）；校验器 v3.1 修掉 5 类误报，抽查 16 条翻转全对 | [`T-050`](docs/work/tasks/T-050.md)、[`T-051`](docs/work/tasks/T-051.md) |
| 本机小模型能替代 DeepSeek 吗 | 9B + 代码辅助（数字全部交给代码算）在新留出集上通过登记的 5 条：召回 100%、有问题声明 0.7%；但按事后补充的严格口径，它自己补充的事实只有 DeepSeek 的约 37%，**只能作降级后端** | [`T-054`](docs/work/tasks/T-054.md) |

> **关于旧版首页的"净 Sharpe 0.996、净年化 +24.46%"**：那组数字用的是扁平 13bp 成本，没有可成交性检查，而且大部分落在训练期内。
> 同一模型样本外 265 个交易日的 Sharpe 是 0.247；放进真实成交约束（T+1、涨跌停、停牌、分层滑点）后，年化净超额是 −5.0%。
> 所以从 2026-10-20 起，首页不再展示那组数字（[`T-010`](docs/work/reports/T-010-line-a-oos.md) §2、§3.5）。

---

## 两条主线

**主线 A — 既有运行链路**（每个交易日定时运行，`scripts/daily_update.py`）

全 A 股 → 6 层漏斗（基础质量 / 流动性 / 趋势 / 基本面 / LGBM 打分 / 行业分散）→ 四维评分 → 历史验证 → 约 10 只候选。
仓位用 HRP；6-Agent 生成研究报告，但报告不进入推荐；LLM 重排关闭。

**主线 A v2 — 每日自动运行的完整系统**（分支 `t/035-line-a-v2-p0`，`scripts/line_a/daily_shadow.sh`）

| 层 | 做什么 | 实现 |
|---|---|---|
| F0–F2 | 时点快照、证券身份与可交易性、研究池（PIT 成交额前 1500） | 规则；PIT 审计 |
| F3 | 量化打分：XGBoost / CatBoost / TabM / 序列模型等权集成（20 日、63 日），Ridge 对照 | 冻结模型包 |
| 旁路 | 全市场公告事件监测 | MacBERT v2.2 多标签分类（金标准 F1 0.981） |
| F4–F5 | 研究候选、风险与执行卡（HAR 波动、beta、跳空、涨跌停、除权） | 代码 |
| F6 | Agent 研究：财务 / 公告 / 股东 / 卖方预期 / 行业 / 审查 6 个角色 | DeepSeek + 受约束的工具循环 |
| F7 | 声明逐条核验（数值、单位、引文、公式重算、证据编号） | 代码校验器 v3.1 |
| F8–F9 | 前瞻影子账本（参考臂 / 挑战臂 / 观察臂）、研究报告 | 冻结引擎 + 按股数记账的模拟器 |
| L10 | 正式荐股日报：三档纸面账户、明日开盘委托 | T-052 主配置；买不起一个申报单位的跳过 |
| A0 | Agent 大脑：看全局、给持仓打风险标记、安排第二天的研究 | 标记经代码核对；不改持仓，积累 60 条、40 个交易日后前瞻判定 |

**主线 B — 研究主线**

v6 / v6_1 全市场 PIT 面板（含退市票）→ Purged Walk-Forward（方案 C：连续样本外，季度重训）→ Ridge 与 GPU 模型族 → 配对 RankIC 评估 → 可执行净值 → 研究闸门 v2。

两条主线共享数据底座，但评估口径完全不同，**绩效数字不能互相引用**。

---

## 怎么防止自己骗自己

| 陷阱 | 做法 | 位置 |
|---|---|---|
| 前视偏差 | 财报按公告日过滤；事件与文本必须**严格早于**决策时点；行业分类按发布日可用；信号与成交分在不同交易日 | `quantmind/data/`、`quantmind/features/` |
| 幸存者偏差 | v6 全市场股票池保留退市票，退市时强制清仓 | `quantmind/data/` |
| 标签泄漏 | Purged + Embargo 切分：训练标签窗口与测试期之间留隔离带 | `quantmind/backtest/wf_split.py` |
| 风格伪装成 alpha | 行业 + 规模中性化后的 IC 作为硬性判据 | `quantmind/backtest/` |
| 纸面收益 | 可执行净值：T+1 开盘成交、板块与时变涨跌停、停牌、流动性分层滑点、时变印花税、现金约束 | `quantmind/execution/nav_engine.py`、`quantmind/backtest/wf_costs.py` |
| 研究者自由度 | 预登记（脚本与判定规则在看结果之前提交）、每个期限只设一个确认性比较、负对照与金丝雀、等价检验、错误率披露、跨家族验收 | `docs/work/tasks/`、`docs/work/reviews/` |
| 闸门放水 | 研究闸门 v2 用相对财富回撤，与资金闸门分开；资金闸门一律未批准 | `quantmind/execution/nav_gate.py` |

---

## 系统结构

```
数据源：Tushare（主）/ Akshare（备）/ 巨潮公告元数据 / 财经快讯
   │
   ▼
PIT 数据湖 data/lake/（行情、财报、事件、文本元数据；parquet；只追加，写前备份）
   │
   ├──► 主线 A：每日快照 → 6 层漏斗 → LGBM → 四维评分 → 推荐 + 6-Agent 报告 → Streamlit 面板
   │
   └──► 主线 B：v6_1 周频面板 / 日频面板 → Purged Walk-Forward（方案 C）
              → Ridge ｜ XGBoost · CatBoost · TabM · 序列模型（GPU）→ 等权集成
              → 配对 RankIC → 可执行净值 → 研究闸门 v2 → 模型注册表
```

## 仓库结构（主要部分）

```
quantmind/             核心库
  data/                数据接入、PIT 快照、数据湖读写
  features/            因子与面板
  backtest/            wf_split（Purged / Embargo）、wf_costs（A 股成本）、wf_gate / wf_metrics
  execution/           nav_engine（可执行净值）、nav_gate（闸门判定）
  contracts/           model_registry（模型注册表）
  models/ regime/ portfolio/ risk/ selection/ agents/ kb/
scripts/               日更 daily_update.py、回测、数据管道 data_pipeline/
app/                   Streamlit 面板（app/main.py + app/pages/）与 FastAPI 后端
frontend/              个人展示前端（React + Vite），读 quantmind/line_a/api_server.py 的只读接口
tests/                 单元测试（CI 只跑离线部分）
docs/work/             任务卡 tasks/、报告 reports/、证据脚本与结果 evidence/、验收与评审 acceptance/ reviews/
```

## 展示前端与只读接口

- `quantmind/line_a/api_server.py`：FastAPI，只监听 127.0.0.1:8780，全部只读。除了每日产物，还有后端算好的聚合接口：
  - `/api/forward`、`/api/redteam`：T+1 开盘复权入场的事后收益、相对研究池等权与指数的超额；红队规则与基准跑输率比较；
  - `/api/consensus`：四个模型的截面百分位与分歧度；`/api/claim`：F7 声明的真实核验记录；`/api/funnel`：每层真实数量；
  - `/api/ledger/*`：按股数逐笔记账的模拟账户（持仓、T+1 可用、成本、交割单与逐项税费、已实现 / 未实现盈亏、盈亏日历），会计恒等式逐日核对，T-052 账户与存档净值逐日一致（[`T-059`](docs/work/tasks/T-059.md)）。
- 接口与数据说明：[`docs/handoff/frontend_handoff.md`](docs/handoff/frontend_handoff.md)；前端审查报告：[`docs/handoff/forensic_audit_20261024.md`](docs/handoff/forensic_audit_20261024.md)。
- 个人展示前端在 [`frontend/`](frontend/)（React + Vite，Windows 上运行：`cd frontend && npm install && npm run dev`，需要先在 WSL 里启动只读接口）。测试：`npm run test:contract`（接口契约与数值）、`npm run test:e2e`（Edge 无头浏览器，逐项对照接口数值）。

## 分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | 截至 2026-10-12 已验收合入的工作 | — |
| `t/030-gpu-models` | 日频面板上的 GPU 模型族重建（树模型、TabM、序列模型、集成）与负对照 | `implemented`，统计复核中，未合入 |
| `t/031-text-data` | 公告元数据、分析师预期、新闻前瞻采集 | `implemented`，下载进行中，未合入 |
| `t/033-attribution-forward` | 20d 增益归因（探索性）与前瞻验证模型包 | `implemented` / 进行中，未合入 |
| `t/035-line-a-v2-p0` | 主线 A v2：每日自动运行的完整系统、Agent 研究与代码核验、Agent 大脑、正式荐股纸面账户、T-052 模拟盘、本机小模型、只读接口与模拟账户账本 | 每日运行中，未合入 |

---

## 快速开始

**环境**（Python 3.11）：

```bash
conda create -n quantmind python=3.11
conda activate quantmind
pip install -e ".[data,ml,viz,ui,dev]"
```

全量依赖可以用 `pip install -e ".[all]"`。

**配置**：

```bash
cp .env.example .env
```

在 `.env` 里填 `TUSHARE_TOKEN`。
- `.env` 和 `api_key.txt` 已在 `.gitignore` 中，**不要提交任何密钥**，见 [`docs/security/secrets_handling_policy.md`](docs/security/secrets_handling_policy.md)；
- 默认存储后端是 `DATA_BACKEND=parquet`、`WRITE_MODE=parquet_only`。

**离线测试**（与 CI 相同）：

```bash
python -m pytest tests/ -m "not integration and not requires_network and not requires_db and not requires_optional_deps and not stale_panel_fixture"
```

**不需要数据和密钥的体验**：见 [`docs/QUICKSTART.md`](docs/QUICKSTART.md)。

**日更管道**（需要 Tushare 数据）：

```bash
python scripts/daily_update.py --universe alpha --lgbm-model models/lgbm_v6_alpha.pkl --lgbm-top 50 --no-llm --auto-regime --position-sizing hrp
```

**面板**：

```bash
bash start_all.sh
```

这会同时启动 FastAPI（8000）和 Streamlit（8501）。

数据与模型文件不进 git，要用自己的 Tushare 账号下载，并按任务卡里的命令重建。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`METHODOLOGY.md`](METHODOLOGY.md) | 因子体系、模型、回测规范 |
| [`docs/work/tasks/`](docs/work/tasks/) | 任务卡：目标、预登记规则、结果、修订记录 |
| [`docs/work/reports/`](docs/work/reports/) | 研究报告（T-007 LLM 重排消融、T-010 主线 A 真实成交、T-018 主线 B 重训等） |
| [`docs/work/evidence/`](docs/work/evidence/) | 每张卡的证据脚本与结果 JSON |
| [`docs/work/ledger.md`](docs/work/ledger.md) | 任务台账 |
| [`HANDOVER.md`](HANDOVER.md) | 早期交接文档（2026-05，部分内容已过时） |

结果的表述遵循固定措辞：
- 预登记比较只写"检测到增益 / 未检测到增益 / 未检测到差异"；
- 探索性结果明确标注；
- 历史窗口被反复使用过，所以只称"历史滚动样本外"，不称盲测。

---

## 免责声明

本项目仅用于量化研究方法的学习与验证。文中所有收益数字都是历史回测，不代表未来表现，不构成投资建议。
