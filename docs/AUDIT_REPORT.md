# QuantMind 全面审计报告（数据-功能-前端对照）

> 生成时间：2026-06-01 · 只读审计，未修改任何代码 · CodeGraph: 272 文件 / 5,618 节点 / 11,973 边

---

## 摘要（TL;DR）

QuantMind 后端能力**远超**前端展示。核心问题不是"功能不够"，而是**三条数据链路各自为政、前端直接读 parquet 导致字段错位**：

1. **两套推荐数据并行且互不连通**：真实每日推荐（`data/recommendations/` + `reports/investment_pipeline/`）与 30 日模拟（`data/sim30d/`）。**前端首页"今日推荐"展示的其实是模拟盘（2025-10～11 回测），不是当天真实选股。**
2. **最有价值的资产——每只股票的 6-Agent 六维分析——已落盘但前端完全没接**：存在 `reports/investment_pipeline/{date}/strategies.json`，含 `agent_signals`（估值/动量/质量/情绪/风险）+ `investment_thesis`，**没有任何页面读取它**。
3. **PE/ROE 显示错误是真 bug**：`key_factors` 存的是**标准化 z-score**，`market_summary` 把它当原始 PE/ROE 求均值显示 → "平均 PE -0.1x"。
4. **入场价/当前价/行业为空**：推荐 JSON 的 `top10` 根本没有 `industry`/`entry_price`/`name` 字段；数据**在别处存在**（`alpha_universe.parquet` 有行业，`daily_prices_panel` 有价格），只是**没 join 上**。`forward_positions.json` 的 entry_price 字段是 `null`，从未回填。

---

## 第一部分：后端已实现能力盘点

### 1.1 数据产物（data/，共 2,708 个文件）

按"生命体征"分类（今天=2026-06-01，活=7天内更新）：

#### 🟢 活数据（每日/每周更新，前端应优先消费）

| 文件 | 规模 | 内容 | 谁在写 |
|------|------|------|--------|
| `feedback/realized_pnl.parquet` | **80 行 × 16 列** | 已结算收益（季度调仓）as_of: 2024-06~2025-12，exit 至 2026-04 | `track_realized_pnl.py`（周一 cron）|
| `feedback/realized_ic.json` | — | 已结算 IC | 同上 |
| `regime/regime_history.parquet` | 1703 行 × 5 列 | date/regime/bull_prob/neutral_prob/bear_prob | HMM |
| `paper_trading/strategy_config_v2.json` | — | 权重/持仓周期/超配/迭代历史（**刚被迭代优化改过**）| `run_iteration.py` |
| `paper_trading/forward_positions.json` | 20 持仓 | OPEN 持仓，**entry_price/exit_price=null** | `track_realized_pnl.py` |
| `paper_trading/nav_curve_{1w,2w,21d,3m}.parquet` | 8-9 行 | NAV 曲线 | `build_position_nav.py` |
| `loss_signals_v4/{latest,factor_health,action_plan}.json` | — | 损失信号循环 | `dispatch_loss_signals.py`（周一）|
| `recommendations/2026-05-{22,26,27}.json` | top10 | **真实每日推荐**（仅 3 天）| `daily_update.py` |
| `features/alpha_2026-05-{26,27}.parquet` | 1374×74 | 每日因子面板 | `daily_update.py` step4 |
| `iteration/round_002_params.json` | — | 第二轮参数记录 | 本次迭代 |

#### 🟡 半静态/参考数据（PIT 快照，"旧"属正常）

- `panel/alpha_panel_v4.parquet` — **29,406 行 × 77 列**，核心训练面板（16 天前，季度级更新）
- `panel/alpha_panel_v4_regime.parquet` — 97 列（含 regime 特征）
- `raw/daily_prices_panel.parquet` — 728,370 行 × 11 列（前端价格主源）
- `raw/alpha_prices_panel.parquet` — 2,273,529 行 × 13 列
- `alpha_universe/alpha_universe.parquet` — **1374 行，含 ts_code/name/industry/market**（← 行业数据在这里！）
- `alpha_universe/fundamentals/*.parquet` — 1373 个/股财务文件
- `snapshots/{40 个季度}/` — PIT 快照（daily_basic/financials/margin/hk_hold/north_bound），历史回测用

#### 🔴 死数据 / 历史遗留

- `sim30d/` 与 `iteration/round_001_.../` 内容重复（round_001 是 sim30d 的备份副本）
- `loss_signals/`（v1，已被 `loss_signals_v4/` 取代）
- `meta_learner/` 有 v2/v3 两套 meta 文件并存

### 1.2 模型（models/，共 42 个，**仅 5 个在用**）

| 模型 | 状态 | direction | nfeat | 用途 |
|------|------|-----------|-------|------|
| `lgbm_v6_main.pkl` | 🟢 今日重训 | **+1** | 38 | 主板路由 |
| `lgbm_v6_gem.pkl` | 🟢 今日重训 | **+1** | 38 | 创业板路由 |
| `lgbm_v6_star.pkl` | 🟢 今日重训 | **+1** | 38 | 科创板路由 |
| `lgbm_v6_alpha.pkl` | 🟡 15d | +1 | 38 | BoardRouter fallback |
| `factor_cnn_v2_augmented.pkl` | 🟢 7d | — | — | Ensemble CNN 分支 |
| `agents/{valuation,momentum,quality,sentiment,risk}_*.pkl/.pt` | 🟡 16-20d | — | — | 6-Agent ML 后端 |
| 其余 **30 个** `lgbm_v1~v5_*` | 🟡 17-24d | 混乱 | — | **历史实验遗留，无人使用** |

> ⚠️ 37/42 模型是死实验。`models/` 需要一次大扫除（归档到 `models/archive/`）。

### 1.3 核心能力（quantmind/ 包公开接口）

| 子模块 | 关键能力 | 输入 → 输出 |
|--------|---------|------------|
| `data/` | TushareProvider/AkshareProvider, build_snapshot, get_universe | Tushare API → 快照 parquet |
| `features/` | FeaturePipeline + **70+ 因子**（expr_factors 7 个 + fundamental + technical + sentiment + north_flow + em_fundamental + analyst_revision）| 价格/财务 → alpha_panel |
| `models/` | LGBMRankerModel, FactorCNN, LLMListwiseReranker, ensemble_scores | 因子面板 → 排序得分 |
| `models/board_router.py` | **BoardModelRouter**（三门质检 + 板块路由）| ticker → 板块专用模型得分 |
| `selection/` | FunnelSelector（多层漏斗）, LazyDataEngine | universe → 候选池 |
| `regime/` | RegimeHMM（bull/neutral/bear）, **DynamicWeightManager**（三态权重，刚改过 bull）| 指数收益 → regime + 权重 |
| `agents/investment_agents/` | **6-Agent**: Valuation/Momentum/Quality/Sentiment/Risk + Strategy 综合 | snapshot → 六维信号 |
| `agents/debate_orchestrator.py` | **DebateOrchestrator**（fast/full 模式，LLM 可选）| context → DebateResult |
| `portfolio/` | hrp_weights, kelly_weights, blend_weights, **ConstrainedPortfolioOptimizer**（Barra 约束）| 得分 → 仓位 |
| `risk/` | FactorRiskModel, PositionSizer, DrawdownController, **barra.STYLE_MAP** | 持仓 → 风险归因 |
| `iteration/` | **SimulationAnalyzer / ParameterOptimizer / IterationComparator**（本周新建）| sim30d → 诊断+建议 |
| `watchlist/` | WatchlistManager, WatchlistDailyScorer | 用户自选 → 每日打分 |

### 1.4 脚本（scripts/，110 个）

| 脚本 | 输出到 | 触发方式 |
|------|--------|---------|
| `daily_update.py` | `recommendations/{date}.json` + `reports/investment_pipeline/{date}/` + `features/alpha_{date}.parquet` | **cron 16:30 工作日** + 控制台 |
| `track_realized_pnl.py` | `feedback/realized_pnl.parquet` + `forward_positions.json` | cron 周一 |
| `dispatch_loss_signals.py` | `loss_signals_v4/` | cron 周一 |
| `paper_trading_sim.py` + `update_sim_strategy.py` | `paper_trading/` | cron 月末 |
| `run_30day_sim.py` | `data/sim30d/` | 手动 |
| `run_iteration.py` | `strategy_config_v2.json` + `iteration/` | 手动 / 控制台 |
| `train_board_models.py` | `models/lgbm_v6_{main,gem,star}.pkl` | 手动 |
| `build_position_nav.py` | `paper_trading/nav_curve_*.parquet` | 手动 |
| 其余 ~100 个 | 训练/回测/诊断 | 手动一次性 |

---

## 第二部分：前端 12 个页面盘点

| 页面 | 功能意图 | 实际读取 | 数据源类型 | 状态 |
|------|---------|---------|-----------|------|
| **main.py** | 总览导航 | sim_data | 模拟 | ✅ |
| **1_今日推荐** | "每日选股推荐" | `load_sim30d_days()` | **🔴 模拟盘（2025-10~11）非真实当天** | ⚠️ 误导 |
| **2_漏斗选股** | 三系统漏斗 | sim30d daily + stock_returns | 🔴 模拟 | ✅（但模拟数据）|
| **3_单股分析** | 个股详情 | sim30d days + stock_returns | 🔴 模拟 | ✅（但模拟数据）|
| **4_回测表现** | NAV + Realized PnL | sim30d + nav_curves + **realized_pnl** + config | 混合 | ✅ |
| **5_模型管理** | 模型状态 | ic_analysis + config + stock_returns + realized_pnl + 直接 pickle.load(lgbm) | 混合 | ✅ |
| **6_智能问答** | LLM 问答 | **FastAPI `localhost:8000`** + sim_data | API+模拟 | ⚠️ 依赖 API 启动 |
| **7_系统控制台** | 运维中心 | 子进程跑脚本 + agent_stats.json + strategies.json + board_router | 混合 | ✅ 最全 |
| **8_Regime状态** | Regime 监控 | `regime_history.parquet` + RegimeHMM 实时 + config | 🟢 活 | ✅ |
| **9_持仓跟踪** | 进行中持仓 | **forward_positions.json** + daily_prices_panel | 🟢 活 | ⚠️ entry_price=null |
| **10_持仓详情** | 持仓 NAV 明细 | sim30d positions + nav JSON | 🔴 模拟 | ✅（但模拟数据）|
| **11_自选股** | 用户自选 | WatchlistManager/Scorer | 🟢 活 | ✅（刚修完）|
| **12_历史推荐** | 真实推荐追踪 | **recommendations + realized_pnl + forward_positions** | 🟢 真实 | ⚠️ 行业/agent 空 |

**关键观察**：
- 页面 1/2/3/10 标着"今日/选股/持仓"，实际全是 **30 日模拟盘**数据（2025-10-09~11-19 的回测）。
- 只有页面 **12** 接了真实每日推荐，但缺行业 + 缺 6-Agent 分析。
- 只有页面 **6** 用了 FastAPI 后端，其余 11 个页面**直接读 parquet/json**。
- **没有任何页面**展示 `reports/investment_pipeline/{date}/strategies.json` 里的 6-Agent 六维分析。

---

## 第三部分：差距分析（后端有 vs 前端展示）

### 3.1 总对照表

| 后端能力 / 数据 | 前端是否展示 | 差距说明 |
|----------------|------------|---------|
| 真实每日推荐 `recommendations/{date}.json` | ⚠️ 仅页面12 | 首页"今日推荐"反而显示模拟盘 |
| **6-Agent 六维分析** `strategies.json:agent_signals` | ❌ **完全没接** | 最有价值资产被埋没 |
| `investment_thesis`（逐 Agent 文字论证）| ❌ 没接 | 同上 |
| 行业 `alpha_universe.parquet:industry` | ❌ 没 join | 数据存在，未关联 |
| 入场价（推荐日收盘）`daily_prices_panel` | ⚠️ 页面12 算了 | top10 无 entry_price 字段，靠 join |
| `forward_positions.entry_price` | ❌ null | track 脚本从未回填 |
| Realized PnL 80 条 | ✅ 页面4/5/12 | 但 3 个页面**各自用不同 loader** 读同一文件 |
| Regime 实时 + 历史 | ✅ 页面8 | 良好 |
| 损失信号 `loss_signals_v4/` | ❌ 没接 | 周一 cron 在跑，前端看不到 |
| 板块路由状态 BoardRouter | ✅ 页面7 局部 | 仅控制台一角 |
| Ensemble 权重 (LGBM/CNN regime) | ⚠️ 页面5/8 部分 | 无统一视图 |
| Barra 风险归因 `risk/barra.py` | ❌ 没接 | 能力存在，无页面 |
| ConstrainedPortfolioOptimizer | ❌ 没接 | 仓位优化结果不可见 |
| 因子 IC `ic_analysis_30day.json` | ✅ 页面5/7 | 但日期固定 2025-05-17，未随真实推荐更新 |
| 迭代优化诊断/对比 | ✅ 页面7 | 良好（本周新建）|
| 自选股打分 | ✅ 页面11 | 良好 |

### 3.2 用户六问逐条回答

**Q1：realized_pnl 现在有多少条？各页面读的是同一份、最新的吗？**
- **80 条**，as_of 是季度调仓日（2024-06-28 ~ 2025-12-31），exit_date 至 2026-04-14。
- **不是最新的覆盖**：最近的真实日推（2026-05-22/26/27）尚未结算，存在 `forward_positions.json`（20 个 OPEN）。
- **同一份文件，但两条代码路径**：`app/utils/sim_data.py:load_realized_pnl()`（页面 4/5）和 `app/utils/rec_data.py:load_realized_pnl()`（页面 12）。两个 loader 读同一 parquet，逻辑重复。

**Q2：推荐记录里的"入场价/当前价/行业"为什么空？数据没有还是没 join 上？**
- **数据有，但没 join**。`recommendations/{date}.json` 的 `top10` 字段只有：`lgbm_rank/ticker/lgbm_score/lgbm_score_raw/key_factors/cnn_score/ensemble_score/ensemble_rank/weight/llm_rank/rank/reason`。**没有 industry、name、entry_price**。
- 行业在 `alpha_universe/alpha_universe.parquet`（每股一行 industry），价格在 `daily_prices_panel.parquet`。页面 12 已 join 了价格和名称，但**行业字段取自推荐 item 本身 → 永远空**。
- `forward_positions.json` 的 `entry_price`/`exit_price` 字面就是 `null`，`track_realized_pnl.py` 从未回填。

**Q3："平均 PE -0.1x" 是标准化值被当原始值显示了吗？**
- **是，确认是 bug**。`key_factors` 存的是**截面标准化 z-score**：`{'pe_ttm': 0.139, 'pb': 0.44, 'roe_ttm': -0.76, ...}`。
- `market_summary` 字段把这些 z-score 当原始 PE/ROE 求均值 → "平均 PE -0.1x，平均 ROE -73.8%"。原始 PE 应是几十倍、ROE 应是百分数。

**Q4：每只股票的 6-Agent 分析存哪？前端哪个页面能看完整六维？**
- 存在 **`reports/investment_pipeline/{date}/strategies.json`**，每只股票一条：
  ```
  { ticker, rating, composite_signal, confidence,
    target_price_1m/3m, stop_loss_price, position_size, holding_horizon,
    investment_thesis,           # 逐 Agent 文字论证
    agent_signals: {
       ValuationAgent: {signal, confidence, summary},
       MomentumAgent:  {signal, confidence, summary},
       QualityAgent:   {signal, confidence, summary},
       SentimentAgent: {signal, confidence, summary},
       RiskAgent:      {signal, confidence, summary},
    },
    key_risks, key_catalysts, llm_used }
  ```
- （5 个分析 Agent + StrategyAgent 做综合 = "6-Agent"）
- **前端没有任何页面读取它**。页面 7 只读了同目录的 `agent_stats.json`（聚合计数：分析了几只、平均置信度），看不到逐股六维。
- **这是当前最大的"有后端无前端"缺口。**

**Q5：每次选股的完整记录（哪天、选了什么、各子系统打了什么分）存哪？能回溯吗？**
- 散落在**三处且互不连通**：
  1. `recommendations/{date}.json` — LGBM/CNN/Ensemble/LLM 四种排名 + key_factors
  2. `reports/investment_pipeline/{date}/strategies.json` — 6-Agent 分析
  3. `data/sim30d/daily/{date}.json` — **30 日模拟**的 system1/2/3（≠ 真实选股）
- 真实日推只有 3 天（05-22/26/27）+ 历史季度。**无法完整回溯**：前端页面 1/2/3 展示的是模拟盘，页面 12 只接了链路①，链路②（六维）完全没接。

**Q6：各模型/子系统实时状态在哪个页面能一览？**
- **没有统一页面**。分散：
  - 页面 5（模型管理）：LGBM 直接 pickle + IC
  - 页面 7（控制台）：board_router 状态、ollama 探测、pytest、model probe
  - 页面 8（Regime）：HMM regime + DynamicWeightManager 权重
- LGBM/CNN/HMM/Agent/板块路由**没有一处能同时看到健康度**。

---

## 第四部分：架构建议

### 4.1 "前端直接读 parquet"是不是问题根源？

**是，但不是唯一根源。** 真正的根源有三层：

1. **字段契约缺失**：前端假设 parquet/json 里有某字段（industry/entry_price/原始PE），但写入方根本没放或放的是标准化值。没有 schema 层校验 → 错位/空值/z-score 当原始值。
2. **数据链路三套并行**：模拟盘 / 真实日推 / Agent 报告，各写各的目录，前端各读各的，无统一"一次选股 = 一条完整记录"的视图。
3. **重复 loader**：`sim_data.py` 和 `rec_data.py` 各有一份 `load_realized_pnl`，逻辑漂移。

### 4.2 要不要引入数据库（SQLite/DuckDB）？

**建议：引入 DuckDB 作为只读查询层，但不迁移写入。**
- 现有 parquet 是合理的存储格式（列存、PIT 友好），**不需要**把写入改成 SQL。
- DuckDB 可以**直接 `SELECT ... FROM 'data/**/*.parquet'`** 做跨文件 join（推荐 ⋈ 行业 ⋈ 价格 ⋈ pnl），省掉前端手写 pandas join。
- 适合 Q2（行业 join）、Q5（跨链路回溯）这类"多文件关联"痛点。
- **不建议** SQLite（不擅长分析查询、要额外 ETL）。

### 4.3 要不要前后端分离（FastAPI）？现在做值不值得？

**已经有半套 FastAPI**（`app/api/server.py`，仅页面 6 在用）。建议：
- **值得做，但分步**。不要一次性把 11 个页面全改成调 API。
- 第一步只做一件事：把"数据读取"收敛到一个 **DataService**（Python 类，不必先上 HTTP），所有页面和 API 都调它。消除重复 loader + 统一字段契约。
- 等 DataService 稳定后，再决定哪些页面走 HTTP API（移动端/多用户才有必要）。单机自用，HTTP 分离收益有限。

### 4.4 有没有现成 skill 能帮上忙？

- `scientific-figure-skill`（已加载）：NAV 曲线、IC 序列、Barra 归因的出版级图表。
- `backtest-expert` / `position-sizer` / `macro-regime-detector`（CLAUDE.md 中列出）：回测验收、仓位、regime 切换方法论。
- **数据层/API 无专用 skill**，DataService 需手写。

### 4.5 最小改造路径（让前端看到全部数据/模型状态/每次选股/六维分析）

分 4 步，每步独立可上线：

**第 1 步（半天）— 修数据契约 bug（最高优先级，纯修复）**
- 修 `market_summary` 的 PE/ROE：用原始 daily_basic 值，不用 key_factors 的 z-score。
- 推荐写入时补 `industry`（join alpha_universe）、`name`、`entry_price`（join 当日收盘）三个字段。
- `track_realized_pnl.py` 回填 forward_positions 的 entry_price。

**第 2 步（1-2 天）— 建统一 DataService**
- 新建 `app/services/data_service.py`，收敛所有 `load_*`：
  `get_recommendations(date)` / `get_agent_analysis(ticker, date)` / `get_realized_pnl()` / `get_model_status()` / `get_regime()`。
- 内部用 DuckDB 做跨 parquet join（推荐 ⋈ 行业 ⋈ 价格 ⋈ pnl ⋈ agent_signals）。
- 页面 12 和重复 loader 全部改调它。

**第 3 步（1 天）— 接通 6-Agent 六维分析（最高价值）**
- 页面 3（单股分析）或新建页面：读 `strategies.json:agent_signals`，画六维雷达图 + 逐 Agent 论证文字。
- 页面 12 的个股行点开 → 展示该股当日六维。

**第 4 步（半天）— 统一子系统状态页**
- 新建"系统健康"页：LGBM(×4 板块)/CNN/HMM/6-Agent/BoardRouter 的 direction/IC/训练时间/fallback 状态一屏展示。
- 数据来自 DataService.get_model_status()。

> 改造顺序原则：**先修 bug（第1步），再统一读取（第2步），最后补展示（第3/4步）**。不要先动展示，否则在错位数据上做 UI 是返工。

---

## 附录：关键数据结构速查

```
recommendations/{date}.json
  └ top10[]: {lgbm_rank, ticker, lgbm_score, lgbm_score_raw,
              key_factors(z-score!), cnn_score, ensemble_score,
              ensemble_rank, weight, llm_rank, rank, reason}
     ⚠️ 无 name/industry/entry_price

reports/investment_pipeline/{date}/strategies.json
  └ []: {ticker, rating, composite_signal, confidence,
         target_price_1m/3m, stop_loss_price, position_size,
         investment_thesis, key_risks, key_catalysts,
         agent_signals: {Valuation/Momentum/Quality/Sentiment/Risk:
                         {signal, confidence, summary}}}

feedback/realized_pnl.parquet  (80×16)
  └ as_of_date, ticker, entry_date, exit_date, holding_days,
    predicted_rank, predicted_score, entry_price, exit_price,
    actual_return_63d, hit, pnl_vs_median, ...

paper_trading/forward_positions.json
  └ positions[]: {as_of, ticker, predicted_rank, holding_period,
                  estimated_exit_date, entry_price(null!), status}

alpha_universe/alpha_universe.parquet  (1374×8)
  └ ts_code, name, industry, market, list_date, in_csi300/500/800
     ← 行业数据源（Q2 join 目标）
```
