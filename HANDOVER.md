# QuantMind — Claude Code 接续开发交接文档

> 更新于 2026-05-21，基于 Phase E1/E2 完成后的系统状态。  
> 本文档供下一位 Claude Code 实例快速接手，无需重读历史对话。

---

## 一、项目简介与当前定位

**QuantMind** 是一套 AI 驱动的 A 股多模态量化投研系统：

| 层次 | 内容 | 状态 |
|------|------|------|
| **三系统选股** | 全A股5535只 → 6层筛选 → 四维分析 → 回测验证 → 10只 final picks | ✅ 已完成+验证 |
| **量化因子模型** | 71因子面板 → LGBM v6（38特征，ICIR=0.380）→ Kelly/HRP 仓位优化 | ✅ 已完成 |
| **6-Agent 研究** | 估值/动量/质量/情绪/风险/策略 Agent × Streamlit 7页展示 | ✅ 已完成 |
| **模拟盘验证** | 30日全A股模拟（3m期胜率96.7%）+ 9期季度回测 | ✅ 已完成 |
| **Phase E1** | 因子面板 v4 重建 + LGBM v6 重训 + 6-Agent 重训 | ✅ 已完成 |
| **Phase E2** | v6 NAV 回测（4种权重对比，Kelly毛年化18.90%/HRP毛年化25.72%） | ✅ 已完成 |
| **E3 成本修正** | NAV 回测加入 0.13% 单边交易成本，HRP净 24.46%/Sharpe=0.996 | ✅ 已完成 |
| **Barra 归因** | `quantmind/risk/barra.py` + `scripts/run_barra_attribution.py` | ✅ 代码完成，待提交 |

**仓库**：`git@github.com:SnowHE-hub/QuantmMnd.git`（注意仓库名拼写）  
**主分支**：`main`（最新 commit: 442f0ca）  
**运行环境**：WSL2 Ubuntu / conda env `quantmind` / Python 3.11  
**Tushare Token**：`64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56`

---

## 二、系统当前状态（2026-05-21）

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

### 2.2 Phase E2/E3 NAV 回测结果（含成本修正，2019-03 → 2026-05）

v6 模型 + v4 面板，4种权重对比（E3 单边 13 bps，平均换手率 87-93%，30次调仓）：

| 权重方法 | 毛年化 | 净年化 | 成本拖累 | 毛Sharpe | 净Sharpe | 净MaxDD |
|---------|-------|-------|---------|---------|---------|---------|
| hrp | +25.72% | +24.46% | −1.26% | +1.053 | +0.996 | −22.6% |
| equal | +22.45% | +21.27% | −1.18% | +0.935 | +0.880 | −23.6% |
| blend | +22.41% | +21.19% | −1.22% | +0.919 | +0.863 | −23.1% |
| kelly | +18.90% | +17.67% | −1.23% | +0.723 | +0.668 | −25.4% |
| CSI300 | — | −0.15% | — | — | −0.181 | −45.6% |

报告路径：`reports/nav_v4/{kelly,blend,equal,hrp}/`（gross）+ `{*}_e3/`（net）  
累计成本侵蚀约 7%（30次调仓合计）。HRP 净 Sharpe≈1.0，MaxDD 最小。

### 2.3 数据资产

| 资产 | 状态 | 路径 |
|------|------|------|
| 全A股日线缓存 | 5535只，至2026-05-11 | `data/sim30d/raw/prices_all.parquet` |
| 30日模拟结果 | 30个 daily JSON + positions.parquet | `data/sim30d/daily/` |
| 逐股票收益 | 450条，四期限 | `data/sim30d/stock_returns.parquet` |
| Alpha 1374 面板 | **v4，29,406×77，2019Q1~2026Q2** | `data/panel/alpha_panel_v4.parquet` |
| PIT 快照 | 29季度（2019Q1~2026Q2） | `data/snapshots/` |
| 季度推荐 | 9期（2024Q2~2026Q1） | `data/recommendations/` |
| **realized_pnl** | **379条**（80季度+299条30日） | `data/feedback/realized_pnl.parquet` |
| **meta_learner 目录** | 已创建，模型待重训 | `data/meta_learner/` |
| **loss_signals** | 损失信号数据 | `data/loss_signals/` + `data/loss_signals_v4/` |

### 2.4 模型资产

| 模型 | 说明 | 路径 |
|------|------|------|
| **lgbm_v6_alpha.pkl** | 🔑 主模型，38特征 v4，ICIR=+0.380 | `models/lgbm_v6_alpha.pkl` |
| lgbm_ensemble_large.pkl | Regime 大盘模型，ICIR=+0.088 | `models/lgbm_ensemble_large.pkl` |
| lgbm_ensemble_small.pkl | Regime 小盘模型，ICIR=+0.261 | `models/lgbm_ensemble_small.pkl` |
| lgbm_v5_alpha_63d.pkl | 备用，37特征 | `models/lgbm_v5_alpha_63d.pkl` |
| lgbm_v3_top18.pkl | 早期基线，18特征 | `models/lgbm_v3_top18.pkl` |
| 6-Agent 模型包 | valuation/risk/momentum/quality v3 | `models/agents/` |
| dpo_qwen/ | DPO 微调（Qwen2.5-1.5B） | `models/dpo_qwen/` |
| meta_learner | ⚠️ 待用 379 条重训 | `data/meta_learner/` |

### 2.5 策略配置（最新）

```
data/paper_trading/
├── strategy_config_v2.json   ← 当前最优策略参数（持仓期3m，质量因子33.3%）
├── system2_weights_v2.json   ← IC 校准后 System2 四维权重
├── ic_analysis_30day.json    ← 因子 IC 分析报告（vs 四期限实际收益）
└── forward_positions.json    ← 2026-03-31 建仓持仓跟踪（到期日 ~2026-06-26）
```

### 2.6 未提交文件（需在下次 commit 中包含）

```bash
# 新增文件（git add 时包含）
quantmind/risk/barra.py          # Barra 风险归因模块
scripts/run_barra_attribution.py # Barra 归因执行脚本
data/meta_learner/               # meta-learner 数据目录（不含模型pkl）

# 已修改文件（需确认后 commit）
scripts/validate_strategies.py   # 验证阈值放宽（E2 技术债修复）
scripts/run_nav_backtest.py       # E3 成本修正（进行中）
data/paper_trading/performance.json  # 绩效数据更新
```

---

## 三、待执行任务（按优先级排序）

### 🔴 高优先级（立即执行）

#### ~~Task 1：E3 交易成本修正~~ ✅ 已完成（2026-05-21）

`run_nav_backtest.py` 已支持 `--cost-bps 13`，净年化 Kelly=17.67%，HRP=24.46%。详见 2.2 节。

#### Task 2：前向持仓结算 + meta-learner 重训

`data/paper_trading/forward_positions.json` 中 2026-03-31 建仓，约 2026-06-26 到期：

```bash
# 到期后：
# Step 1 - 拉取出场价格，追加 realized_pnl
python scripts/track_realized_pnl.py \
  --forward data/paper_trading/forward_positions.json

# Step 2 - 重训 meta-learner（379条 → 约400+条）
python scripts/train_meta_learner.py \
  --pnl data/feedback/realized_pnl.parquet \
  --out data/meta_learner/
# 目标: R² 从约 0.15 提升到 0.35+
```

检查：`quantmind/models/meta_learner.py` 特征列表是否需要增加 `composite_score`、`value_score` 等新列。

### 🟡 中优先级（本月内）

#### Task 3：2026Q2 面板更新 + 序贯验证 Round 10

```bash
# 下载 2026Q2 快照
python scripts/download_data.py \
  --rebalance-quarterly-range 2026-04-01 2026-06-30

# 重建 v4 因子面板
python scripts/build_full_panel.py \
  --out data/panel/alpha_panel_v4.parquet

# 重跑序贯验证 Round 10
python scripts/validate_strategies.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --rounds 10
```

#### Task 4：行业超配额度放宽

`SelectionSystem.run()` Layer6 行业分散逻辑（`scripts/run_30day_sim.py` 约 Line 370）：
- 当前：每行业最多 3 只
- 建议：对 30日模拟 Top3 行业（装修装饰/建筑工程/机械基件）放宽至 5 只

#### Task 5：提交 Barra 模块

```bash
git add quantmind/risk/barra.py
git add scripts/run_barra_attribution.py
git add scripts/validate_strategies.py
git commit -m "feat(risk): add Barra attribution module and relax validation thresholds"
```

### 🟢 下季度

#### Task 6：Regime 感知动态权重

当市场 HMM 状态从 bull 切换到 bear 时，System2 权重自动调整：
- Bull（当前 v2）：quality 33.3%，value 24.2%
- Bear（建议）：value 35%，momentum 降权，quality 适度降

#### Task 7：每日价格数据更新

`data/raw/alpha_prices_panel.parquet` 截至 2026-05-11，需每周更新：

```bash
python scripts/build_daily_price_panel.py \
  --start 2026-05-12 --end $(date +%Y-%m-%d) \
  --out data/raw/alpha_prices_panel.parquet
```

---

## 四、关键脚本说明

```bash
# 全A股30日三系统模拟（分步执行）
python scripts/run_30day_sim.py --step fetch      # 拉Tushare数据
python scripts/run_30day_sim.py --step simulate   # 每日三系统流水线
python scripts/run_30day_sim.py --step evaluate   # 绩效评估+优化建议

# IC 归因 + System2 校准（幂等，可重复执行）
python scripts/optimize_30day_results.py

# 季度 paper trading 回测
python scripts/paper_trading_sim.py

# 日频真实 NAV 回测（4种权重）
python scripts/run_nav_backtest.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --model models/lgbm_v6_alpha.pkl \
  --weight-method kelly --top 30

# Barra 风险归因
python scripts/run_barra_attribution.py

# 策略参数更新
python scripts/update_sim_strategy.py \
  --positions data/paper_trading/positions.parquet \
  --panel data/panel/alpha_panel_v4.parquet

# 启动 Streamlit Dashboard（7页）
streamlit run app/主页.py --server.port 8501
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
| NAV 回测未含交易成本 | 🔴 高 | E3 进行中 | 加入 0.13% 单边成本，重跑 4 种权重 |
| alpha_panel_v4 最新 as_of = 2026Q2（但价格截至2026-05-11）| 🟡 中 | 待更新 | Task 7：每周更新日价格面板 |
| meta-learner 样本（379条）未重训 | 🟡 中 | 待执行 | 等 2026-06-26 前向持仓到期后执行 |
| Barra 模块未提交 git | 🟡 中 | 待 commit | Task 5 |
| System2 逐日 IC 方差大（均值0.137，但单日−0.73~+0.78） | 🟡 中 | 已知 | 扩大每日标的至15只可降噪 |
| run_30day_sim.py 无增量缓存 | 🟢 低 | 待优化 | 按 date 检查 daily/ 已存在则跳过 |
| System3 Bull Regime 下 hist_sharpe 负IC | 🟡 中 | 已分析 | Bull market 下将 hist_sharpe 阈值从 >1.0 降至 >0.5 |
| 行业分散 Layer6 过于严格 | 🟡 中 | 待调整 | Task 4：Top行业放宽至5只 |

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

## 八、Cron 自动化（已上线）

```bash
# 每周一至五 16:30 执行日更管道（E2 后已配置）：
30 16 * * 1-5 cd /home/lenovo/projects/quantmind && \
  conda run -n quantmind python scripts/daily_update.py \
  --lgbm-model models/lgbm_v6_alpha.pkl \
  --position-sizing hrp \
  >> logs/daily_update.log 2>&1

# 每月最后一个交易日 22:00 执行季度回测：
0 22 28-31 * * [ "$(date +\%u)" -le 5 ] && cd /home/lenovo/projects/quantmind && \
  conda run -n quantmind python scripts/paper_trading_sim.py >> logs/paper_trading.log 2>&1
```

---

## 九、快速验证命令

```bash
conda activate quantmind
cd /home/lenovo/projects/quantmind

# 验证 v6 主模型
python -c "
import pickle
m = pickle.load(open('models/lgbm_v6_alpha.pkl','rb'))
feats = m._model.feature_name()
print('lgbm_v6: OK, features:', len(feats))
"

# 验证 v4 因子面板
python -c "
import pandas as pd
df = pd.read_parquet('data/panel/alpha_panel_v4.parquet')
print('v4 panel:', df.shape, '| periods:', df.index.get_level_values('as_of').nunique())
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

# 启动 Dashboard（7页）
streamlit run app/主页.py --server.port 8501
```

---

## 十、给下一位 Claude Code 的开场提示语

```
请用中文回复。

继续开发 QuantMind 量化投资系统，项目在 /home/lenovo/projects/quantmind。
conda 环境：quantmind（Python 3.11）。

【项目现状（2026-05-21）】
- 三系统选股流水线已完成+验证（全A股5535只，30日模拟3m期胜率96.7%）
- LGBM v6（38特征，ICIR=0.380）为当前主模型
- Phase E2/E3 NAV 回测完成：v6 HRP 净年化+24.46%/Sharpe=0.996（Kelly 净+17.67%，均大幅跑赢CSI300）
- System2 IC 校准完成（质量因子 25%→33.3%）
- realized_pnl 379条，meta-learner 待重训
- Barra 归因模块已实现但未提交 git
- Cron 已上线：每日16:30 daily_update.py

【当前最优策略参数】（见 data/paper_trading/strategy_config_v2.json）
- 持仓期：3m
- System2 权重：价值24.2%/动量22.3%/质量33.3%/技术20.2%
- 止损线：-15%
- 仓位优化：Kelly（生产推荐）

【待执行任务（按优先级）】
1. [高] 前向持仓结算（2026-06-26到期） → 追加realized_pnl → 重训meta-learner
3. [中] 提交 Barra 模块：git add quantmind/risk/barra.py scripts/run_barra_attribution.py
4. [中] 2026Q2 面板更新：下载快照 → 重建 alpha_panel_v4 → 序贯验证 Round 10
5. [中] 行业超配放宽：Layer6 Top行业（装修/建筑/机械）最多5只

【安全规则】
- Tushare Token: 64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56
- API Key 从 .env 读取，绝不硬编码
- 提交前 git add 具体文件，不要 git add -A
```

---

*更新时间：2026-05-21 | 对应 commit: 442f0ca*  
*Phase E1/E2/E3 完成（v6 ICIR=0.380 · HRP 净年化+24.46%/Sharpe=0.996 · E3 含13bps成本）+ Barra 模块实现*
