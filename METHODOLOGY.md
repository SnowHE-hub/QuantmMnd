# QuantMind — 工程方法论（Methodology）

> 版本 v1.0 · 2026-05-15  
> 覆盖：数据源、因子体系、模型架构、回测规范、风险管理、Agent 系统

---

## 目录

1. [项目概览](#1-项目概览)
2. [数据工程](#2-数据工程)
3. [因子体系](#3-因子体系)
4. [排名模型](#4-排名模型)
5. [风险管理体系](#5-风险管理体系)
6. [Agent 投资分析系统](#6-agent-投资分析系统)
7. [回测规范](#7-回测规范)
8. [日更流水线](#8-日更流水线)
9. [已知局限与改进方向](#9-已知局限与改进方向)

---

## 1. 项目概览

QuantMind 是一套面向 A 股市场的量化研究平台，涵盖：

| 子系统 | 核心能力 |
|--------|----------|
| 数据底座 | PIT（Point-in-Time）快照，1374 只 Alpha 宇宙，2019–2026 |
| 因子工厂 | 71 个横截面因子（基本面 + 技术 + 扩展），自动 IC 筛选 |
| 排名引擎 | LightGBM LambdaRanker + Regime-Aware 集成模型 |
| Agent 系统 | 6 个专属 Agent（估值/动量/风险/质量/情绪/策略） |
| 回测框架 | 日频真实持仓 NAV + 季度调仓 Walk-Forward |
| 日更管道 | `daily_update.py` 端到端，含 LLM 精排 + 6-Agent 分析 + HRP 仓位优化 |

**投资目标**：Alpha 1374 宇宙中，季度调仓 Top-30 等权/HRP 长仓策略，2020–2024 年实现显著正 Alpha。

---

## 2. 数据工程

### 2.1 数据源

| 来源 | 用途 | 限制 |
|------|------|------|
| **Tushare Pro**（2000 积分账号） | 日线行情、复权因子、财务报表、北向资金、融资融券、指数权重 | 约 200 次/分钟 |
| **AkShare** | 备用：行情、宏观 | 无限频但不稳定 |
| **SSE 交易日历**（Tushare `trade_cal`） | 判断交易日、快照截止日 | 缓存在 `.cache/` |

### 2.2 快照体系（PIT Snapshots）

所有历史数据以**季末交易日**为 `as_of` 日期构建快照，严格避免前视偏差：

```
data/snapshots/
  2020-03-31/
    universe.parquet       # 成份股 + 权重
    daily_prices.parquet   # 过去 280 个交易日行情
    adj_factor.parquet     # 复权因子
    balance_sheet.parquet  # 最新已发布的资产负债表
    income_stmt.parquet    # 最新已发布的利润表
    cashflow_stmt.parquet  # 最新已发布的现金流量表
    daily_basic.parquet    # 技术指标（换手率、市值等）
    hk_hold.parquet        # 北向持股（陆股通）
    margin.parquet         # 融资融券余额
    index_daily.parquet    # 沪深300/中证500 指数行情
  2020-06-30/ ...
```

**财务数据发布延迟处理**：每个财务字段取 `ann_date ≤ as_of` 的最新一条，不使用 `report_date` 排序，确保 PIT 正确性。

**Alpha 宇宙**：`data/alpha_universe/alpha_universe.txt` 中的 1374 只股票，以 `tickers_override_policy="replace"` 覆盖 CSI300 成份股逻辑，每期等权（`weight=1/1374`）。

### 2.3 价格合并面板

`data/raw/alpha_prices_panel.parquet`（长格式）：

| 列 | 类型 | 说明 |
|----|------|------|
| `ts_code` | str | 股票代码 |
| `trade_date` | datetime | 交易日 |
| `open/high/low/close` | float | 不复权价 |
| `adj_factor` | float | 复权因子 |
| `adj_close` | float | 前复权收盘价（= close × adj_factor / adj_factor_today） |
| `vol`, `amount` | float | 成交量、成交额 |
| `pct_chg` | float | 当日涨跌幅（未复权） |

---

## 3. 因子体系

### 3.1 因子分类（共 71 个）

#### 基本面因子（~18 个）

| 因子 | 公式/来源 | 方向 |
|------|-----------|------|
| `earnings_yield` | EBIT / EV | + |
| `book_to_market` | BV / MV | + |
| `pe_ttm` | 市盈率 TTM | − |
| `pb` | 市净率 | − |
| `ps_ttm` | 市销率 TTM | − |
| `roe_ttm` | 净资产收益率 TTM | + |
| `roa_ttm` | 资产收益率 TTM | + |
| `gross_margin` | 毛利率 | + |
| `dividend_yield_ttm` | 股息率 TTM | + |
| `debt_to_equity` | 资产负债率 | − |
| `revenue_yoy` | 营收同比增速 | + |
| `net_profit_yoy` | 净利润同比增速 | + |
| `fcf_yield` | 自由现金流 / 市值 | + |
| `earnings_accel_q` | EPS 单季环比加速度 | + |
| `revenue_accel_q` | 营收单季环比加速度 | + |
| `size_rank` | 流通市值从小到大排名（小盘正向） | + |

#### 技术因子（~22 个）

| 因子 | 计算窗口 | 方向 |
|------|----------|------|
| `momentum_1m` | 过去 21 日收益率 | + |
| `momentum_3m` | 过去 63 日收益率 | + |
| `momentum_6m` | 过去 126 日（跳过近 21 日） | + |
| `momentum_12m` | 过去 252 日（跳过近 21 日） | + |
| `volatility_1m` / `_3m` / `_1y` | 已实现波动率（年化） | − |
| `downside_volatility_3m` | 下行半方差（年化） | − |
| `turnover_3m_avg` | 换手率 3 月均值 | context |
| `turnover_rate_quantile` | 换手率历史百分位（120d） | context |
| `macd_signal` | MACD 柱 / 价格 | + |
| `rsi_14` | RSI(14) | + |
| `bollinger_position` | 布林带位置 | − |
| `atr_ratio` | ATR(14) / 收盘价 | − |
| `amplitude_quantile` | 振幅历史百分位（60d） | − |
| `price_to_52w_low` | 当前价 / 过去252日最低价 | + |
| `turnover_acceleration` | 换手率加速度（近期 vs 历史） | + |

#### 扩展因子（~34 个，含小盘专属）

| 因子 | 说明 | 适用 Regime |
|------|------|-------------|
| `log_market_cap` | 对数总市值 | all |
| `log_circ_market_cap` | 对数流通市值 | all |
| `net_margin` | 净利润率 | all |
| `current_ratio` | 流动比率 | all |
| `equity_multiplier` | 权益乘数 | all |
| `operating_profit_yoy` | 营业利润同比 | all |
| `north_bound_net_inflow_30d` | 北向资金 30 日净流入（市场级） | all |
| `margin_buy_intensity` | 融资净买入强度（融资余额变动） | all |
| `relative_strength_vs_csi500_60d` | 个股相对中证500的超额收益（60d） | small |
| `volume_price_corr_20d` | 量价相关系数（20d） | small |
| `pct_chg_rank_quantile` | 涨跌幅历史百分位 | small |
| `pb_rank_in_sector` | 行业内 PB 排名 | all |
| `roe_rank_in_sector` | 行业内 ROE 排名 | all |

### 3.2 因子选择流程

1. **IC 计算**：对每个 `as_of` 日期截面，计算因子与 `forward_return_63d`（63日远期收益）的 Spearman 相关系数。
2. **ICIR 筛选**：
   - 全局模型：`|mean_IC| ≥ 0.02`，`ICIR ≥ 0.3`（共 18 个入选 `lgbm_v3_top18.pkl`）
   - 大盘子模型：`|mean_IC| ≥ 0.015`（使用 2022-2024 期间）
   - 小盘子模型：`|mean_IC| ≥ 0.010`（使用 2022-2024 期间）
3. **方向校正（auto_flip 已禁用）**：训练前对 IC < 0 的因子取负，保证所有特征单调正向。

---

## 4. 排名模型

### 4.1 全局模型：`lgbm_v3_top18`

| 属性 | 值 |
|------|----|
| 算法 | LightGBM `LambdaRank`（listwise） |
| 特征 | 18 个高 ICIR 因子（主要为技术/资金流向类） |
| 标签 | `forward_return_63d` 的组内排名（按 `as_of` 分组） |
| 训练集 | 2020Q1–2024Q4，Walk-Forward 扩展窗口 |
| 验证集 | 每 fold 留最近 1 期 out-of-sample |
| 参数 | `n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30` |
| 评估 | 全期 IC 均值 = **0.0356**，ICIR = **+0.444** |

### 4.2 Regime-Aware 集成

#### 市场状态检测（`risk_hmm_v3.pkl` · `market_hmm`）

使用保存在 Risk Agent Bundle 中的市场 HMM 状态字典：

```python
market_hmm = {
    "current_regime":    0,                          # 当前状态（0/1/2）
    "regime_labels":     {0:"bull_low_vol", 1:"normal", 2:"bear_crisis"},
    "recent_30d_probs":  {0:1.0, 1:0.0, 2:0.0},    # 最近30日平均概率
    "csi300_ret_63d":    -0.035,                     # CSI300 近63日收益
    "csi300_vol_21d":    0.18                        # CSI300 近21日年化波动
}
```

HMM 使用 CSI300 日收益率序列（2019–2024，共 1721 个交易日）训练，3 个隐状态。

#### 子模型切换规则（`daily_update.py --auto-regime`）

| 市场状态 | 触发条件 | 模型 |
|----------|----------|------|
| `bull_low_vol` | `bull_prob > 0.7` | `lgbm_ensemble_large.pkl`（39 特征） |
| `bear_crisis` | `bear_prob > 0.5` | `lgbm_ensemble_small.pkl`（38 特征） |
| `normal` / 不确定 | 其他 | `lgbm_v3_top18.pkl`（18 特征，默认） |

### 4.3 Regime 面板构建

`scripts/build_regime_panel.py` 将市场级 Regime 指标（`regime_label`, `regime_small_prob`, `csi500_csi300_63d`, `csi300_20d_vol`, `breadth_20d`）与股票级因子做**交叉乘积**，生成交互特征，供大/小盘专属 LightGBM 模型训练。

---

## 5. 风险管理体系

### 5.1 RiskAgent v3（`risk_hmm_v3.pkl`）

| 组件 | 方法 | 输出 |
|------|------|------|
| 个股波动率 | EWMA（λ=0.94，252 日窗口，年化） | `vol_63d_annualized` |
| 市场 Beta | OLS（63日 vs CSI300 + CSI500） | `beta_csi300`, `beta_csi500` |
| 风险状态 | 3-state HMM（个股波动率） | `hmm_regime`: low_vol / normal / crisis |
| 市场状态 | CSI300 收益率 HMM | `market_regime` |
| 尾部风险 | Student-t CVaR（自由度估计） | `cvar_95` |

**模型 Bundle 结构**：
```python
{
    "kind": "risk_hmm_v3",
    "vol_by_ticker": {ticker: [vol_series...]},   # 年化波动率时序
    "market_hmm": {...},                           # 市场 HMM（见 4.2）
    "regime_labels": {0:"low_vol", 1:"normal", 2:"crisis"}
}
```

### 5.2 估值 Agent v3（`valuation_lgbm_v3.pkl`）

| 组件 | 方法 |
|------|------|
| 算法 | LightGBM 回归（`objective=regression`） |
| 特征 | 22 个估值 + 盈利 + 行业相对因子，**截面分位映射**（持久化 `feature_quantiles`） |
| 标签 | `forward_return_63d`（截面分位排名后的相对收益） |
| 训练策略 | 清洗时序交叉验证（Purged TS-CV，`embargo_days=90`） |
| 行业处理 | 计算股票相对行业均值的比率特征（`_sector_relative`） |
| 推理校正 | 推理时用 `feature_quantiles` 将原始值映射为分位排名，对齐训练分布 |

### 5.3 动量 Agent v4（`momentum_patchtst_v4.pkl`）

| 组件 | 方法 |
|------|------|
| 架构 | PatchTST（Patch 时序 Transformer），输入 63 日 OHLCV 序列 |
| 特征 | 5 个：对数收益率、对数成交量变化、振幅 / close、收盘相对最高、RSI(14) |
| 标签 | 未来 5 日涨跌方向（二分类） |
| 正则化 | Dropout=0.2，Label Smooth=0.05，WeightDecay=5e-4，d_model=64，n_layers=2 |
| 中性带 | `|P - 0.5| < 0.07` 时回退规则引擎，减少低置信度误信号 |

---

## 6. Agent 投资分析系统

### 6.1 6-Agent 架构

```
run_six_agents(ticker, context)
    ├── ValuationAgent    → valuation_score (-1~+1)
    ├── MomentumAgent     → momentum_score (-1~+1)  
    ├── RiskAgent         → risk_score    (-1~+1)
    ├── QualityAgent      → quality_score (-1~+1)
    ├── SentimentAgent    → sentiment_score (-1~+1)
    └── StrategyAgent.analyze_with_llm()
            → comprehensive_rating (BUY/HOLD/SELL)
            → confidence (0~1)
            → price_target, key_risks, catalysts
```

### 6.2 日更集成（step7a）

`daily_update.py` 在 LLM 精排（step6）之后调用 step7a：

1. 对 Top-N（默认 10）候选股运行 `run_six_agents`
2. 调用 `StrategyAgent.analyze_with_llm`（支持 Qwen/DeepSeek/Ollama）
3. 输出文件：
   - `reports/investment_pipeline/<date>/strategies.json`
   - `reports/investment_pipeline/<date>/final_recommendations.md`
   - `reports/investment_pipeline/<date>/validations.json`
4. Dashboard `3_单股分析` 页面读取 `strategies.json` 渲染 Agent 雷达图

---

## 7. 回测规范

### 7.1 核心原则

| 原则 | 实现 |
|------|------|
| **无前视偏差** | 所有因子仅用 `as_of` 日前已发布数据；财务因子取 `ann_date ≤ as_of` 的最新报告 |
| **PIT 快照隔离** | 每个季末单独构建快照目录，不共享缓存 |
| **Walk-Forward** | 训练集逐步扩展，测试集始终为下一期（单期 OOS） |
| **调仓成本** | 暂未扣除，换手率约 35–50%/季度（待接入） |
| **等权基准** | 比较基准为 CSI300 买入持有 |

### 7.2 面板回测（Panel Backtest）

`scripts/run_alpha_report.py`：

- 加载 `alpha_panel_v3.parquet`（20 个季度，27,480 行）
- 按 `as_of` 截面打分 → Top-N 持仓
- 用 `forward_return_63d`（面板中的期望 63 日收益）计算组合收益
- 支持 `--weight-method {equal, hrp, kelly, blend}` 权重方法对比

**局限**：`forward_return_63d` 是截面平均的面板估计值，非真实持仓路径收益。

### 7.3 日频真实 NAV 回测

`scripts/run_nav_backtest.py`：

- 从 `alpha_prices_panel.parquet`（长格式日频数据）读取真实价格
- 按季末调仓，持有期间每日用实际 `adj_close` 计算组合收益
- 输出：`reports/alpha_final/nav_daily.csv`（日期、NAV、持仓列表）

**优于面板回测**：完整路径依赖，反映持仓期内的真实波动和最大回撤。

### 7.4 主要绩效指标

| 指标 | 计算公式 |
|------|---------|
| 年化收益 | `(NAV_T / NAV_0)^(252/T) - 1` |
| 年化波动率 | `std(daily_ret) × √252` |
| Sharpe 比率 | `(ann_ret - rf) / ann_vol`，`rf=3%` |
| Calmar 比率 | `ann_ret / |max_drawdown|` |
| 最大回撤 | `max(NAV_t / max(NAV_{0..t}) - 1)` |
| 胜率 | 季度正收益期数 / 总期数 |
| 换手率 | `(新进股票数 / N)` per rebalance |
| IC / ICIR | Spearman IC 均值 / IC 标准差 |

---

## 8. 日更流水线

### 8.1 流程图

```
daily_update.py
  step1  确定当日交易日（SSE 日历）
  step2  检查缓存（是否已有当日快照）
  step3  下载快照（Tushare，~5min）
  step4  构建因子（71 个，~3min）
  step5  LGBM 粗排（Regime-aware 模型选择 → Top-50）
  step5b HRP/Kelly 仓位优化（→ position_weights.json）
  step6  LLM 精排（可选，Listwise Reranker → Top-10）
  step7  保存推荐 JSON（daily_recommendations.json）
  step7a 6-Agent 投资分析（→ strategies.json + final_recommendations.md）
  step8  生成 HTML 报告（daily_report.html）
```

### 8.2 关键 CLI 参数

```bash
python scripts/daily_update.py \
  --universe alpha \             # Alpha 1374 宇宙
  --lgbm-model models/lgbm_v3_top18.pkl \
  --lgbm-top 50 \                # LGBM 粗排取 Top-50
  --no-llm \                     # 跳过 LLM 精排（无 API Key 时）
  --auto-regime \                # 根据 HMM 市场状态自动切换子模型
  --position-sizing hrp \        # HRP 仓位优化
  --agent-top 10 \               # 对 Top-10 做 6-Agent 分析
  --agent-provider none          # 仅 ML+Rules，不调用 LLM
```

### 8.3 Cron 定时运行

```cron
# 每个交易日 16:30（UTC+8）
30 16 * * 1-5 cd /proj && python scripts/daily_update.py \
  --universe alpha --no-llm --auto-regime --position-sizing hrp \
  --agent-top 10 --agent-provider none >> logs/daily_$(date +%F).log 2>&1
```

---

## 9. 已知局限与改进方向

### 9.1 数据层

| 局限 | 说明 | 改进 |
|------|------|------|
| 交易成本未扣除 | 换手率约 35–50%/季 | 接入滑点模型（冲击成本 = 0.1% × 换手） |
| ST/退市处理 | 未过滤停牌日的因子缺失 | 在快照构建时标记 ST 并剔除 |
| 快照颗粒度 | 仅季末，无月末/周末 | 增量月末快照（已有 2025 全年） |
| Token B 到期 | 高频代理 Token 已于 2026-05-19 到期 | 续费或改用 Token A 单线下载 |

### 9.2 模型层

| 局限 | 说明 | 改进 |
|------|------|------|
| 标签泄露风险 | `forward_return_63d` 含部分未来信息 | 改用 63 日前复权价计算严格 PIT 标签 |
| 因子多重共线 | 相关性高的因子（如多个动量）稀释信号 | Barra 风险归因后正交化 |
| PatchTST 高中性带 | 中性带宽泛导致实际 PatchTST 信号覆盖率低 | 增加训练数据多样性；采用集成投票 |
| Regime 切换延迟 | HMM 状态滞后约 5–10 个交易日 | 加入实时波动率/情绪快照更新 |

### 9.3 系统层

| 局限 | 说明 | 改进 |
|------|------|------|
| Dashboard 实时性 | Streamlit 无 WebSocket 推送 | 接入 FastAPI + WebSocket |
| 无实盘对接 | 仅 paper trading | 接入 XTP/CTP/国投等券商 API |
| LLM 成本 | Dashscope Qwen-plus 约 0.04 元/分析 | 本地 Ollama Qwen2.5:7B（已配置） |
| CI 缺失 | 无自动化 IC/Sharpe 回归测试 | 接入 GitHub Actions，`ic_mean ≥ 0.02` 不允许合并 |

---

*文档自动生成工具链：`scripts/run_alpha_report.py`、`scripts/run_nav_backtest.py`、`scripts/daily_update.py`*  
*最后更新：2026-05-15*
