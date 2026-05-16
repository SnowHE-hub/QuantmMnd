# QuantMind — AI-Enhanced Quantitative Investment System

> 基于 LightGBM 因子模型 + 6 Agent 投资研究 + RAG 知识库的 A 股量化投资平台。  
> Alpha 1374 宇宙 · 季度调仓 Top-30 · Regime-Aware 集成模型 · HRP/Kelly 仓位优化 · 端到端日更管道。

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Factor_Model-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📊 核心性能（日频真实持仓 NAV，2020-03 → 2026-05）

| 指标 | 策略（Top-30 等权） | CSI300 基准 | 超额 |
|------|-------------------|------------|------|
| **年化收益** | **+10.36%** | +1.13% | **+9.23%** |
| **Sharpe 比率** | **0.301** | −0.112 | — |
| **最大回撤** | −48.5% | −45.6% | — |
| **月度胜率** | ~57% | — | — |

> 因子模型（`lgbm_v3_top18`）OOS IC 均值 = **+0.036**，ICIR = **+0.444**（2020–2024）  
> 上述为**日频真实价格**驱动的持仓 NAV（非面板期望收益），含完整路径依赖与持仓波动。  
> 未扣除交易成本（季度换手约 35–50%）。

---

## 🏗️ 系统架构

```
Alpha 1374 宇宙（1374 只 A 股）
      │
      ▼ ── 因子工厂 ──────────────────────────────────────────────
      71 个横截面因子（基本面 18 + 技术 22 + 扩展 34）
      IC 筛选 → Top-18 高 ICIR 因子 → LightGBM LambdaRanker
      │
      ▼ ── Regime-Aware 排名引擎 ─────────────────────────────────
      market_hmm 状态检测（bull_low_vol / normal / bear_crisis）
        bull_low_vol → lgbm_ensemble_large（39 特征）
        bear_crisis  → lgbm_ensemble_small（38 特征）
        default      → lgbm_v3_top18（18 特征）
      → Top-50 候选
      │
      ▼ ── HRP/Kelly 仓位优化（step5b）────────────────────────────
      分层风险平价 / 分数 Kelly / 混合 → position_weights.json
      │
      ▼ ── LLM 精排（可选，step6）────────────────────────────────
      Qwen/DeepSeek Listwise Reranker → Top-10
      │
      ▼ ── 6 Agent 投资分析（step7a）─────────────────────────────
      ① ValuationAgent   lgbm_v3（22特征，截面分位映射）
      ② MomentumAgent    PatchTST v4（63日OHLCV序列，二分类）
      ③ QualityAgent     Piotroski F-Score（9 信号）
      ④ SentimentAgent   TF-IDF 语义中心向量
      ⑤ RiskAgent        HMM v3（EWMA波动 + OLS Beta + CVaR）
      ⑥ StrategyAgent    LLM 综合策略（目标价/止损/仓位）
      │
      ▼ ── 输出 ──────────────────────────────────────────────────
      strategies.json / final_recommendations.md / position_weights.json
      daily_report.html / Streamlit Dashboard（6 页）
```

---

## 📁 项目结构

```
quantmind/
├── data/
│   ├── snapshots/          # PIT 季末快照（2019Q1 → 2026Q2）
│   ├── panel/              # 因子面板（alpha_panel_v3.parquet）
│   ├── raw/                # 合并价格面板（alpha_prices_panel.parquet）
│   └── alpha_universe/     # Alpha 1374 股票列表
├── models/
│   ├── lgbm_v3_top18.pkl           # 全局模型（18特征，ICIR=0.444）
│   ├── lgbm_ensemble_large.pkl     # 大盘 Regime 子模型
│   ├── lgbm_ensemble_small.pkl     # 小盘 Regime 子模型
│   └── agents/
│       ├── risk_hmm_v3.pkl         # Risk Agent（HMM+Beta+CVaR）
│       ├── valuation_lgbm_v3.pkl   # Valuation Agent（LGBM+截面分位）
│       └── momentum_patchtst_v4.pkl # Momentum Agent（PatchTST）
├── quantmind/
│   ├── data/               # Tushare/AkShare 数据提供者
│   ├── features/           # 因子计算（fundamental/technical/expansion）
│   ├── models/             # FactorModel / LGBMRankerModel / PatchTST
│   ├── agents/             # 6 个 Investment Agents
│   └── portfolio/          # HRP/Kelly 仓位优化
├── scripts/
│   ├── download_data.py         # 快照批量下载
│   ├── build_full_panel.py      # 因子面板构建
│   ├── train_factor_model.py    # LightGBM 训练
│   ├── train_regime_ensemble.py # Regime 集成模型训练
│   ├── train_risk_agent_v3.py   # Risk HMM v3 训练
│   ├── train_valuation_agent_v3.py # Valuation LGBM v3 训练
│   ├── train_momentum_patchtst.py  # Momentum PatchTST v4 训练
│   ├── run_nav_backtest.py      # 日频真实 NAV 回测 ← 核心报告
│   ├── run_alpha_report.py      # 季度面板回测 HTML 报告
│   ├── run_investment_pipeline.py # 6-Agent 完整投资分析
│   ├── daily_update.py          # 端到端日更管道
│   └── setup_cron.sh            # Cron 定时任务配置
├── app/                     # Streamlit Dashboard（6 页）
├── reports/
│   ├── alpha_final/         # NAV 报告（nav_report.html）
│   └── investment_pipeline/ # 每日 Agent 分析结果
├── METHODOLOGY.md           # 工程方法论文档
└── README.md
```

---

## 🚀 快速开始

### 环境准备

```bash
conda create -n quantmind python=3.11
conda activate quantmind
pip install -r requirements.txt

# 配置 API Keys
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN、DASHSCOPE_API_KEY 等
```

### 一键日更（生产模式）

```bash
python scripts/daily_update.py \
  --universe alpha \
  --auto-regime \               # Regime-Aware 模型自动切换
  --position-sizing hrp \       # HRP 仓位优化
  --no-llm \                    # 不调 LLM（可改为 --provider dashscope）
  --agent-top 10 \              # Top-10 做 6-Agent 分析
  --agent-provider none         # 仅 ML+Rules
```

### 日频 NAV 回测

```bash
python scripts/run_nav_backtest.py \
  --panel  data/panel/alpha_panel_v3.parquet \
  --prices data/raw/alpha_prices_panel.parquet \
  --model  models/lgbm_v3_top18.pkl \
  --top 30 --weight-method equal \
  --out reports/alpha_final/
# 输出：nav_report.html（含 CSI300 基准对比）
```

### 启动 Dashboard

```bash
streamlit run app/主页.py
# 访问 http://localhost:8501
```

### 重建快照（增量更新）

```bash
# 下载最新季度快照
python scripts/download_data.py \
  --rebalance-quarterly-range 2026-04-01 2026-06-30 \
  --tickers-file data/alpha_universe/alpha_universe.txt \
  --tickers-override-policy replace

# 重建因子面板
python scripts/build_full_panel.py \
  --snapshots-dir data/snapshots \
  --out data/panel/alpha_panel_v3.parquet
```

---

## 📈 主要脚本说明

| 脚本 | 功能 | 耗时 |
|------|------|------|
| `download_data.py` | 批量下载 PIT 快照（1374只 × 1季度） | ~14h（Token A 限速） |
| `build_full_panel.py` | 构建 71 因子面板 | ~20min |
| `train_factor_model.py` | 训练 LightGBM LambdaRanker | ~5min |
| `train_regime_ensemble.py` | 训练 Regime 大/小盘集成模型 | ~10min |
| `train_risk_agent_v3.py` | 训练 Risk HMM v3 | ~15min（需 GPU） |
| `train_valuation_agent_v3.py` | 训练 Valuation LGBM v3 | ~10min |
| `train_momentum_patchtst.py` | 训练 Momentum PatchTST v4 | ~30min（GPU 推荐） |
| `run_nav_backtest.py` | 日频真实 NAV 回测 | ~10s |
| `daily_update.py` | 端到端日更 | ~5-30min |

---

## ⚙️ 关键 CLI 参数速查

### `daily_update.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--universe` | `alpha` | Alpha 1374 宇宙 |
| `--lgbm-model` | `lgbm_v3_top18.pkl` | LGBM 粗排模型 |
| `--lgbm-top` | `50` | LGBM 取 Top-N |
| `--auto-regime` | 关 | 自动切换 Regime 子模型 |
| `--position-sizing` | `equal` | `equal/hrp/kelly/blend` |
| `--kelly-fraction` | `0.5` | Kelly 分数（0.25~0.5） |
| `--no-llm` | 关 | 跳过 LLM 精排 |
| `--agent-top` | `10` | 6-Agent 分析只数 |
| `--agent-provider` | `none` | LLM 提供商 |
| `--stop-after` | — | 调试用：在指定步骤后停止 |

---

## 📚 文档

- **[METHODOLOGY.md](METHODOLOGY.md)** — 数据源、因子体系、模型架构、回测规范完整方法论
- **[reports/alpha_final/nav_report.html](reports/alpha_final/nav_report.html)** — 日频真实 NAV 交互式报告
- **[reports/alpha_final/nav_metrics.json](reports/alpha_final/nav_metrics.json)** — 汇总绩效指标 JSON

---

## 🗺️ 路线图

### 已完成 ✅
- Alpha 1374 宇宙 PIT 快照（2019–2026Q1）
- 71 因子面板 + IC 自动筛选
- LightGBM LambdaRanker（ICIR=0.444）
- Regime-Aware 集成模型（大/小盘自动切换）
- Risk HMM v3 / Valuation LGBM v3 / Momentum PatchTST v4
- 6-Agent 投资分析系统 + StrategyValidator
- 端到端 `daily_update.py`（Regime + HRP/Kelly + 6-Agent）
- 日频真实 NAV 回测（vs CSI300 基准）
- METHODOLOGY.md + 完整 Dashboard

### 进行中 🔄
- 2026Q2 快照增量更新（后台下载中）

### 下一步优先级 📌
1. **扣除交易成本**：冲击成本模型（`0.1% × 换手率`）+ 印花税
2. **Barra 风险归因**：因子暴露分解（市场/行业/风格）
3. **实盘对接**：XTP / 海通 API Paper Trading
4. **CI 自动化**：GitHub Actions，IC 不允许跌破阈值

---

*最后更新：2026-05-15 | [METHODOLOGY.md](METHODOLOGY.md)*
