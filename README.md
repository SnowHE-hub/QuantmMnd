# QuantMind — AI 增强量化投资系统

> 基于 LightGBM v6 因子模型 + 三系统选股流水线 + 6-Agent 投资研究的 A 股量化平台。  
> **全A股 5535 只 · 三系统每日筛选 → 15只→10只 · 季度/30日模拟持仓验证 · HRP/Kelly 仓位优化 · 端到端自动化管道**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-v6_38features-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📊 核心绩效

### 日频真实 NAV 回测（Alpha 1374 宇宙，2020-03 → 2026-05）

| 指标 | 策略（Top-30 等权） | CSI300 基准 | 超额 |
|------|-------------------|------------|------|
| **年化收益** | **+10.36%** | +1.13% | **+9.23%** |
| **Sharpe 比率** | **0.301** | −0.112 | — |
| **最大回撤** | −48.5% | −45.6% | — |

### 30日全A股三系统模拟盘（2025-10-09 → 2025-11-19，5535只）

| 持仓期 | 均值 | 期胜率 | IR | 结论 |
|--------|------|--------|-----|------|
| 1 周 | −1.88% | 23.3% | −0.56 | 短线不适用 |
| 2 周 | −2.02% | 30.0% | −0.44 | 短线不适用 |
| 21 日 | −1.95% | 30.0% | −0.28 | 短线不适用 |
| **3 个月** | **+20.22%** | **96.7%** | **+1.87** | ✅ **最优，推荐** |

> 2025Q4–2026Q1 A股 bull market：基本面筛选股短期波动大，3m 完整捕获超额。  
> 行业归因：装修装饰(+252%)、建筑工程(+119%)、机械基件(+58%) 为主要贡献来源。

---

## 🏗️ 三系统流水线架构

```
全A股 5535 只（Tushare 实时拉取）
      │
      ▼ ── 系统一：筛选系统（6层漏斗）─────────────────────────────
      Layer 1  基础质量：排除 ST / 新股(<180天) / 市值<30亿
      Layer 2  流动性：换手率 > 30th 百分位
      Layer 3  趋势：reversal_1w > -15%, momentum_1m > 10th 百分位
      Layer 4  基本面：PE(0-150), PB>0, ROE>0
      Layer 5  LGBM v6 打分（38 个 v4 因子）→ Top 50
      Layer 6  行业分散（每行业最多 3 只）→ Top 15 候选
      │
      ▼ ── 系统二：分析系统（四维评分，v2 IC校准权重）────────────
      价值(24.2%) + 动量(22.3%) + 质量(33.3%) + 技术(20.2%)
      → 综合分 → 投资评级（观望/持有/买入/强烈买入）+ 建议持仓期
      逐日 IC 均值 = +0.137，IC>0 占 70%
      │
      ▼ ── 系统三：回测验证（历史胜率）───────────────────────────
      60日历史窗口 → Sharpe / 最大回撤 / 胜率 → 风险分级
      → 过滤低胜率 → 最终可投资名单（约 10 只）
      │
      ▼ ── 持仓期模拟 ────────────────────────────────────────────
      1w / 2w / 21d / 3m 四期限等权组合 → 收益率 / IC 分析
      │
      ▼ ── 6-Agent 深度分析（重点标的）──────────────────────────
      估值 / 动量 / 质量 / 情绪 / 风险 / 策略 → 个股研究报告
      │
      ▼ ── HRP / Kelly 仓位优化 ──────────────────────────────────
      → position_weights.json / strategies.json / Streamlit Dashboard
```

---

## 📁 关键文件结构

```
quantmind/
├── data/
│   ├── sim30d/
│   │   ├── daily/          # 30 个交易日 JSON（三系统每日输出）
│   │   ├── positions.parquet       # 299 行模拟持仓记录
│   │   ├── stock_returns.parquet   # 450 条逐股票四期限收益
│   │   ├── summary.json    # 汇总绩效指标
│   │   └── raw/            # Tushare 原始缓存（5535只 207日日线）
│   ├── paper_trading/
│   │   ├── strategy_config_v2.json   # 30日模拟驱动的策略参数
│   │   ├── system2_weights_v2.json   # IC 校准后四维权重
│   │   ├── ic_analysis_30day.json    # 全因子 IC 分析表
│   │   ├── positions.parquet         # 季度回测持仓（9期×10只）
│   │   └── forward_positions.json    # 前向跟踪持仓模板
│   ├── feedback/
│   │   └── realized_pnl.parquet  # 379 条实际PnL（80季度+299条30日）
│   ├── panel/
│   │   └── alpha_panel_v4.parquet    # v4 因子面板（1374只，至2024Q2）
│   ├── recommendations/      # 季度推荐 JSON（9期）
│   └── snapshots/            # PIT 快照（2019Q1–2026Q2，29个季度）
├── models/
│   ├── lgbm_v6_alpha.pkl          # 🔑 当前主模型（38特征 v4）
│   ├── lgbm_v5_alpha_63d.pkl      # 备用（37特征）
│   └── lgbm_v3_top18.pkl          # 早期基线（18特征）
├── scripts/
│   ├── run_30day_sim.py           # 🔑 全A股30日三系统模拟主脚本
│   ├── optimize_30day_results.py  # 🔑 逐股归因 + System2 IC 校准
│   ├── paper_trading_sim.py       # 季度 paper trading 回测
│   ├── update_sim_strategy.py     # 基于模拟结果更新策略参数
│   ├── run_nav_backtest.py        # 日频真实 NAV 回测
│   ├── build_full_panel.py        # v4 因子面板构建
│   ├── daily_update.py            # 端到端日更管道
│   └── setup_cron.sh              # Cron 定时任务配置
├── app/                           # Streamlit Dashboard（6 页）
└── quantmind/
    ├── agents/                    # 6 个 Investment Agents
    ├── features/                  # 因子计算（v4 38特征）
    ├── models/                    # FactorModel / meta_learner
    └── portfolio/                 # HRP / Kelly 仓位优化
```

---

## 🚀 快速开始

### 环境准备

```bash
conda create -n quantmind python=3.11
conda activate quantmind
pip install -r requirements.txt
```

### 运行 30 日全A股三系统模拟

```bash
# Step 1: 拉取全A股数据（首次约 30-60min，后续使用本地缓存）
python scripts/run_30day_sim.py --step fetch \
    --token 64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56

# Step 2: 执行 30 日三系统模拟（每日 5535只 → 10只 final picks）
python scripts/run_30day_sim.py --step simulate

# Step 3: 绩效评估 + 系统自动优化
python scripts/run_30day_sim.py --step evaluate
```

### 逐股 IC 归因与系统校准

```bash
python scripts/optimize_30day_results.py
# 输出:
#   data/sim30d/stock_returns.parquet       — 逐股票四期限收益
#   data/paper_trading/system2_weights_v2.json  — 校准后权重
#   data/paper_trading/strategy_config_v2.json  — 完整策略配置
#   data/feedback/realized_pnl.parquet     — 扩充至 379 条
```

### 季度 Paper Trading 回测

```bash
python scripts/paper_trading_sim.py \
  --positions data/paper_trading/positions.parquet \
  --prices    data/raw/alpha_prices_panel.parquet
```

### 日频 NAV 回测

```bash
python scripts/run_nav_backtest.py \
  --panel  data/panel/alpha_panel_v4.parquet \
  --prices data/raw/alpha_prices_panel.parquet \
  --model  models/lgbm_v6_alpha.pkl \
  --top 30 --weight-method equal \
  --out reports/alpha_final/
```

### 启动 Dashboard

```bash
streamlit run app/主页.py
# 访问 http://localhost:8501
```

---

## 🔬 系统优化结论（2025Q4 30日模拟 → 基于数据的 IC 校准）

### System 2 四维权重（v2 校准版）

| 维度 | 原权重 | → | 校准后 | IC vs 3m 收益 |
|------|--------|---|--------|--------------|
| 价值 | 30.0% | → | **24.2%** ↓ | −0.062 |
| 动量 | 25.0% | → | **22.3%** ↓ | +0.066 |
| **质量** | 25.0% | → | **33.3%** ↑ | **+0.141**（p=0.015，显著） |
| 技术 | 20.0% | → | **20.2%** ≈ | +0.069 |

### 关键策略参数

```json
{
  "holding_period": "3m",
  "top_n": 10,
  "equal_weight": true,
  "stop_loss_threshold": -0.15,
  "regime_note": "bull market 下 quality 因子主导，避免短线(<21d)操作"
}
```

---

## 📚 文档体系

| 文档 | 内容 |
|------|------|
| [HANDOVER.md](HANDOVER.md) | Claude Code 接续开发快速上手（含待办清单） |
| [METHODOLOGY.md](METHODOLOGY.md) | 因子体系、模型架构、回测规范完整方法论 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速运行指南 |
| `data/paper_trading/strategy_config_v2.json` | 当前最优策略参数 |
| `data/paper_trading/ic_analysis_30day.json` | 因子 IC 分析结果 |

---

## 🗺️ 路线图

### ✅ 已完成

- Alpha 1374 宇宙 PIT 快照（2019Q1–2026Q2）+ 71 因子面板
- LightGBM v6（38特征）+ Regime-Aware 集成模型
- 6-Agent 投资分析系统（估值/动量/质量/情绪/风险/策略）
- **全A股 5535 只三系统流水线**（筛选→分析→回测）
- **30日全A股模拟盘** — 3m 期胜率 96.7%，均值 +20.22%
- System2 IC 校准（质量因子权重从 25% 提升至 33.3%）
- realized_pnl 扩充至 379 条（原 80 条的 4.7 倍）
- 端到端 `daily_update.py` + Cron 月度自动化
- Streamlit Dashboard（6 页）

### 📌 下一步优先级

1. **[立即]** 379 条 realized_pnl 重训 meta-learner
2. **[立即]** 生产代码应用 System2 v2 校准权重
3. **[本月]** 行业超配：建筑/机械/有色 Layer6 最多 5 只
4. **[本季]** 2026Q2 面板更新 → 重跑序贯验证 Round 10
5. **[下季]** Regime 感知：bull/bear 自动切换 System2 权重

---

*最后更新：2026-05-17 | 基于 30日全A股三系统模拟盘（2025-10-09 ~ 2025-11-19）优化*
