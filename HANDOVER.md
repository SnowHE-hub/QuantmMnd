# QuantMind — Claude Code 接续开发交接文档

> 更新于 2026-05-17，基于 30日全A股三系统模拟盘完成后的系统状态。  
> 本文档供下一位 Claude Code 实例快速接手，无需重读历史对话。

---

## 一、项目简介与当前定位

**QuantMind** 是一套 AI 驱动的 A 股多模态量化投研系统：

| 层次 | 内容 | 状态 |
|------|------|------|
| **三系统选股** | 全A股5535只 → 6层筛选 → 四维分析 → 回测验证 → 10只 final picks | ✅ 已完成+验证 |
| **量化因子模型** | 71因子面板 → LGBM v6（38特征）→ HRP/Kelly 仓位优化 | ✅ 已完成 |
| **6-Agent 研究** | 估值/动量/质量/情绪/风险/策略 Agent × Streamlit 展示 | ✅ 已完成 |
| **模拟盘验证** | 30日全A股模拟（3m期胜率96.7%）+ 9期季度回测 | ✅ 已完成 |
| **系统优化** | IC校准System2权重 + realized_pnl 379条 | ✅ 已完成 |

**仓库**：`git@github.com:SnowHE-hub/QuantmMnd.git`（注意仓库名拼写）  
**主分支**：`main`（最新 commit: 52b1d57）  
**运行环境**：WSL2 Ubuntu / conda env `quantmind` / Python 3.11  
**Tushare Token**：`64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56`

---

## 二、系统当前状态（2026-05-17）

### 2.1 三系统流水线（核心）

脚本：`scripts/run_30day_sim.py`（1124行）

```
全A股 5535 只（Tushare daily 批量拉取，已缓存 data/sim30d/raw/）
  → System 1（筛选）：6层漏斗 → 15只候选
      Layer1 质量/ST/IPO/市值（>30亿）
      Layer2 流动性（换手率>30th pct）
      Layer3 趋势（reversal>-15%, momentum>10th pct）
      Layer4 基本面（PE 0-150, PB>0, ROE>0）
      Layer5 LGBM v6 打分 → Top50
      Layer6 行业分散（每行业最多3只）→ Top15
  → System 2（分析）：四维评分
      价值(24.2%) + 动量(22.3%) + 质量(33.3%) + 技术(20.2%)  ← v2 IC校准权重
      → 综合分 + 投资评级 + 建议持仓期
  → System 3（回测）：60日历史验证 → 最终10只可投资标的
```

**30日模拟盘绩效**（2025-10-09 ~ 2025-11-19）：

| 期限 | 均值 | 期胜率 | IR |
|------|------|--------|-----|
| 1w | −1.88% | 23.3% | −0.56 |
| 2w | −2.02% | 30.0% | −0.44 |
| 21d | −1.95% | 30.0% | −0.28 |
| **3m** | **+20.22%** | **96.7%** | **+1.87** |

### 2.2 数据资产

| 资产 | 状态 | 路径 |
|------|------|------|
| 全A股日线缓存 | 5535只，2025-07-01~2026-05-11 | `data/sim30d/raw/prices_all.parquet`（28MB） |
| 30日模拟结果 | 30个 daily JSON + positions.parquet | `data/sim30d/daily/` |
| 逐股票收益 | 450条，四期限 | `data/sim30d/stock_returns.parquet` |
| Alpha 1374 面板 | v4，至2024Q2 | `data/panel/alpha_panel_v4.parquet` |
| PIT 快照 | 29季度（2019Q1~2026Q2） | `data/snapshots/` |
| 季度推荐 | 9期（2024Q2~2026Q1） | `data/recommendations/` |
| **realized_pnl** | **379条**（80季度+299条30日） | `data/feedback/realized_pnl.parquet` |

### 2.3 模型资产

| 模型 | 说明 | 路径 |
|------|------|------|
| **lgbm_v6_alpha.pkl** | 🔑 主模型，38特征 v4，当前生产用 | `models/lgbm_v6_alpha.pkl` |
| lgbm_v5_alpha_63d.pkl | 备用，37特征 | `models/lgbm_v5_alpha_63d.pkl` |
| lgbm_v3_top18.pkl | 早期基线，18特征，ICIR=0.444 | `models/lgbm_v3_top18.pkl` |
| lgbm_ensemble_large/small | Regime 大/小盘子模型 | `models/lgbm_ensemble_*.pkl` |
| meta_learner | 待用 379 条重训（当前可能为旧版） | `data/meta_learner/` |

### 2.4 策略配置（最新）

```
data/paper_trading/
├── strategy_config_v2.json   ← 当前最优策略参数（持仓期3m，质量因子33.3%）
├── system2_weights_v2.json   ← IC 校准后 System2 四维权重
├── ic_analysis_30day.json    ← 因子 IC 分析报告（vs 四期限实际收益）
└── forward_positions.json    ← 2026Q1 建仓持仓跟踪（到期日 ~2026-06-26）
```

---

## 三、待执行任务（按优先级排序）

### 🔴 高优先级（立即执行）

#### Task 1：重训 meta-learner（379条样本 vs 原80条）

```bash
# realized_pnl 已扩充至 379 条，可运行重训
python scripts/train_meta_learner.py \
  --pnl data/feedback/realized_pnl.parquet \
  --out data/meta_learner/
# 目标: R² 从当前约0.15提升到0.35+（样本量增加4.7倍）
```

检查：`quantmind/models/meta_learner.py` 是否需要更新 feature 列表（新增了 composite_score、value_score 等）。

#### Task 2：将 System2 v2 权重更新到生产三系统代码

`scripts/run_30day_sim.py` 中 `AnalysisSystem.WEIGHTS` 当前为：
```python
WEIGHTS = {"value": 0.30, "momentum": 0.25, "quality": 0.25, "technical": 0.20}
```
需更新为：
```python
WEIGHTS = {"value": 0.242, "momentum": 0.223, "quality": 0.333, "technical": 0.202}
```
同时更新 `data/paper_trading/strategy_config_v2.json` 已包含正确权重，可从该文件动态加载。

### 🟡 中优先级（本月内）

#### Task 3：行业超配额度放宽

`SelectionSystem.run()` 中 Layer 6 行业分散逻辑（`scripts/run_30day_sim.py` 约 Line 370）：
- 当前：每行业最多 3 只 → 结果：小金属 23 只、通信设备 12 只过多
- 建议：保持 3 只上限，但对 30日模拟 Top3 行业（装修/建筑/机械）放宽至 5 只

#### Task 4：前向持仓结算

`data/paper_trading/forward_positions.json` 中有 2026-03-31 建仓持仓，预计到期约 2026-06-26（3m）：
- 届时从 Tushare 拉取出场价格，计算实际收益
- 追加到 `realized_pnl.parquet`
- 运行 `python scripts/update_sim_strategy.py` 更新策略

#### Task 5：System3 Bull Regime 过滤放宽

`BacktestSystem.validate()` 中当前历史 Sharpe IC vs 3m = −0.11*（负相关）：
- Bull market 下历史 Sharpe 低的股票反而表现好
- 建议：在 bull regime 下将 hist_sharpe 阈值从 >1.0 降至 >0.5，或完全不过滤

### 🟢 下季度

#### Task 6：2026Q2 面板更新 + 序贯验证 Round 10

```bash
# 下载 2026Q2 快照
python scripts/download_data.py \
  --rebalance-quarterly-range 2026-04-01 2026-06-30

# 重建 v4 因子面板
python scripts/build_full_panel.py \
  --out data/panel/alpha_panel_v4.parquet

# Round 10 序贯验证
python scripts/validate_strategies.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --rounds 10
```

#### Task 7：Regime 感知动态权重

当市场从 bull 切换到 bear 时，System2 权重应自动调整：
- Bull：quality 权重高（33%），value 低（24%）← 当前 v2
- Bear：value 权重回升（35%），momentum 降权

---

## 四、关键脚本说明

```bash
# 全A股30日三系统模拟（分步执行）
python scripts/run_30day_sim.py --step fetch      # 拉Tushare数据
python scripts/run_30day_sim.py --step simulate   # 每日三系统流水线
python scripts/run_30day_sim.py --step evaluate   # 绩效评估+优化建议

# IC 归因 + System2 校准（可重复执行，幂等）
python scripts/optimize_30day_results.py

# 季度 paper trading 回测
python scripts/paper_trading_sim.py

# 策略参数更新（基于季度模拟结果）
python scripts/update_sim_strategy.py \
  --positions data/paper_trading/positions.parquet \
  --panel data/panel/alpha_panel_v4.parquet

# 日频真实 NAV 回测
python scripts/run_nav_backtest.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --model models/lgbm_v6_alpha.pkl \
  --top 30

# 归因分析（快速查看季度绩效）
python scripts/_sim_attribution.py

# 启动 Streamlit Dashboard
streamlit run app/主页.py
```

---

## 五、LGBM v6 特征说明（38维 v4 因子）

```python
# 技术/动量因子（来自日线价格计算）
momentum_1m, momentum_3m, momentum_6m
reversal_1w
rsi_14
bollinger_position
price_to_52w_low
beta_60d, beta_252d
relative_strength_vs_csi300_60d, relative_strength_vs_csi300_120d
relative_strength_vs_csi500_60d
volume_momentum, volume_trend

# 基本面因子（来自 Tushare daily_basic）
pe_ttm, pb, ps_ttm
book_to_market, earnings_yield, dividend_yield_ttm
log_market_cap
roe_approx  # = pb / pe_ttm（实盘近似ROE）
free_float_ratio

# 零填因子（全A股模拟时无来源，填0）
north_hold_ratio, north_hold_amount
margin_buy, margin_sell
accruals, asset_growth, ...
```

> 模型特征顺序固定，必须用以下方式提取并填充：
> ```python
> model_feats = model._model.feature_name()
> X_df = pd.DataFrame({f: df.get(f, 0.0) for f in model_feats})
> ```

---

## 六、已知问题与技术债

| 问题 | 严重度 | 状态 | 建议 |
|------|--------|------|------|
| System2 逐日 IC 方差大（0.137均值，但单日−0.73~+0.78） | 🟡 中 | 已知 | 样本少（10只/日），扩大至15只可降噪 |
| realized_pnl 新增数据中 actual_rank/pnl_vs_median 为 null | 🟢 低 | 已知 | 不影响 meta-learner 训练，这两列可删 |
| run_30day_sim.py 每次重跑会重算（无增量缓存） | 🟡 中 | 待优化 | 按 date 检查 daily/ 已存在则跳过 |
| alpha_panel_v4 最新 as_of = 2024Q2 | 🔴 高 | 待更新 | Task 6：下载 2026Q2 快照重建 |
| System3 Bull Regime 下 hist_sharpe 负IC | 🟡 中 | 已分析 | Task 5：放宽过滤阈值 |
| 行业分散 Layer6 小金属 23只、通信设备 12只过多 | 🟡 中 | 已发现 | Task 3：Top行业放宽至5只 |

---

## 七、IC 分析核心结论（30日模拟数据）

```
因子               IC_1w    IC_2w   IC_21d    IC_3m
lgbm_score        -0.148** -0.192*** +0.167*** +0.034
composite_score   -0.012   +0.050   +0.012    +0.088
value_score       -0.039   +0.076   +0.012    -0.062
momentum_score    -0.069   -0.087   -0.072    +0.066
quality_score     +0.068   +0.128** +0.124**  +0.141**  ← 最强
technical_score   +0.005   -0.054   -0.062    +0.069
hist_win_rate     -0.014   -0.052   -0.063    +0.049
hist_sharpe       -0.076   -0.112*  -0.061    +0.015
hist_maxdd        -0.090   +0.015   +0.072    -0.029

结论：
  · 3m 最优持仓期，LGBM 21d IC=+0.167*** 有效
  · quality_score 在所有期限均正向，3m 显著（p=0.015）
  · 短线(<21d): LGBM 和 hist_sharpe 表现出反转特性（bull regime）
  · System2 逐日 IC 均值=+0.137，IC>0 占 70% → 方向性存在但噪声大
```

---

## 八、月度自动化（Cron）

```bash
# 已配置（via scripts/setup_cron.sh）
# 每月最后一个交易日 22:00 执行：
0 22 28-31 * * [ "$(date +\%u)" -le 5 ] && cd /home/lenovo/projects/quantmind && \
  conda run -n quantmind python scripts/paper_trading_sim.py >> logs/paper_trading.log 2>&1

# 每月1日 08:00 更新策略配置：
0 8 1 * * cd /home/lenovo/projects/quantmind && \
  conda run -n quantmind python scripts/update_sim_strategy.py >> logs/strategy_update.log 2>&1
```

---

## 九、快速验证命令

```bash
conda activate quantmind
cd /home/lenovo/projects/quantmind

# 验证模型可用
python -c "
import pickle
m = pickle.load(open('models/lgbm_v6_alpha.pkl','rb'))
feats = m._model.feature_name()
print('lgbm_v6: OK, features:', len(feats))
"

# 验证 realized_pnl
python -c "
import pandas as pd
df = pd.read_parquet('data/feedback/realized_pnl.parquet')
print('realized_pnl:', df.shape, '| 3m均值:', df['actual_return_63d'].mean())
"

# 验证 30日模拟结果
python -c "
import pandas as pd
pos = pd.read_parquet('data/sim30d/positions.parquet')
sr = pd.read_parquet('data/sim30d/stock_returns.parquet')
print('positions:', pos.shape, '| stock_returns:', sr.shape)
final = sr[sr['in_final']]
print('3m mean:', final['return_3m'].mean(), '| win_rate:', (final['return_3m']>0).mean())
"

# 快速重跑 IC 分析（约 30s）
python scripts/optimize_30day_results.py 2>&1 | tail -30

# 启动 Dashboard
streamlit run app/主页.py --server.port 8501
```

---

## 十、给下一位 Claude Code 的开场提示语

```
请用中文回复。

继续开发 QuantMind 量化投资系统，项目在 /home/lenovo/projects/quantmind。
conda 环境：quantmind（Python 3.11）。

【项目现状】
- 三系统选股流水线已完成+验证（全A股5535只，30日模拟3m期胜率96.7%）
- LGBM v6（38特征）为当前主模型
- System2 IC 校准完成（质量因子 25%→33.3%）
- realized_pnl 已扩充至 379 条（原80条的4.7倍）

【当前最优策略参数】（见 data/paper_trading/strategy_config_v2.json）
- 持仓期：3m
- System2 权重：价值24.2%/动量22.3%/质量33.3%/技术20.2%
- 止损线：-15%

【待执行任务（按优先级）】
1. [高] 重训 meta-learner（379条样本）：python scripts/train_meta_learner.py
2. [高] 将 System2 v2 权重更新到 run_30day_sim.py AnalysisSystem.WEIGHTS
3. [中] 2026Q2 面板更新：下载快照 → 重建 alpha_panel_v4
4. [中] 前向持仓结算（2026-03-31建仓，~2026-06-26到期）

【安全规则】
- Tushare Token: 64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56
- API Key 从 .env 读取，绝不硬编码
- 提交前 git add 具体文件，不要 git add -A
```

---

*更新时间：2026-05-17 | 对应 commit: 52b1d57*  
*30日全A股模拟盘完成 + 系统优化分析 + realized_pnl 379条*
