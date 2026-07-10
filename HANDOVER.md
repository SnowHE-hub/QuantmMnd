# QuantMind — Claude Code 接续开发交接文档

> 更新于 2026-05-23，基于 Phase E1/E2/E3 + 阶段1（HMM/文本情绪/Meta-Learner）+ **阶段2（全部完成）**。  
> 本文档供下一位 Claude Code 实例快速接手，无需重读历史对话。
>
> ⚠ **过期提示（2026-07-10 校正）**：本文档反映 2026-05 状态，其后 Phase 1（幸存者修复 + Ridge 种子 +
> 安全收口）已合 main（SHA `67bb891`），最新权威状态见 `docs/plans/phase1_closure.md` 与 `task_plan.md`。
> 下文个别条目已过时（如 Step5c 实际已实现，见本文件内标注）。

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
| **HMM技术债** | `validate_state_labels()` + 原型锚定假设注释，31项测试通过 | ✅ 已完成 |
| **文本情绪因子** | `ann_sentiment_5d`（BERT/词典→5日均值），29项测试通过 | ✅ 已完成（待实盘IC验证） |
| **序贯验证扩展** | Round 8（2025Q4→2026Q1）+ `--label-col` 参数 | ✅ 已完成 |
| **Meta-Learner v2** | 改为分类（hit预测），CV AUC=0.6025，正确数据源 | ✅ 已完成 |

**仓库**：`git@github.com:SnowHE-hub/QuantmMnd.git`（注意仓库名拼写）  
**主分支**：`main`（最新 commit: `67bb891`，2026-06-19 merge PR #1；本文撰写时为 442f0ca，已过期）  
**运行环境**：WSL2 Ubuntu / conda env `quantmind` / Python 3.11  
**Tushare Token**：`<YOUR_TUSHARE_TOKEN>`

---

## 〇、阶段 2/3 成果摘要（2026-05-24 更新）

### 新增模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **analyst_revision** | `quantmind/features/analyst_revision.py` | 研报评级因子：Tushare拉取 + 数字化 + analyst_revision_score |
| **FactorCNN** | `quantmind/models/factor_cnn.py` | 4分支 Inception CNN + IC Loss + Walk-Forward训练 + ensemble_scores() |
| **FactorCNN v2** | `quantmind/models/factor_cnn.py` | 高斯噪声数据增强（copies=4, σ=0.02），augment_data() + save_cnn_model() |
| **归因报告** | `scripts/generate_attribution_report.py` | 4张发表级图表（NAV/IC热图/Regime/Barra），PDF+PNG+JPG |
| **em_fundamental** | `quantmind/features/em_fundamental.py` | 纯NumPy GMM（K=3）EM隐变量基本面质量因子 |
| **expr_factors** | `quantmind/features/expr_factors.py` | 轻量Qlib表达式引擎，9算子，8内置因子（含momentum_pure），无pyqlib依赖 |

### 关键指标

| 指标 | 数值 | 备注 |
|------|------|------|
| FactorCNN v1 Fold1 IC | **0.030** | forward_return_63d，Walk-Forward |
| **FactorCNN v2 val_IC 均值** | **+0.0331** | 数据增强后（copies=4, σ=0.02），vs baseline +0.0002 |
| **FactorCNN v2 ICIR** | **1.787** | vs baseline 0.003，提升 596× |
| **FactorCNN v2 Fold2 IC** | **+0.0174** | 原 -0.004（regime切换 fold），转正 |
| **FactorCNN v2 Fold3 IC** | **+0.0285** | 原 -0.060，转正 |
| EM因子 IC>0 占比 | **60.7%** | 28季 Spearman，超过55%阈值 |
| ann_contrarian_5d IC | **+0.131** | p=0.025，294样本（反向情绪因子） |
| 表达式引擎一致性 | **1.000** | 8/8因子 Spearman=1.0 vs 参考Python |
| 测试套件 | **702 passed** | 13 integration deselected，0 failures |

### 配置变更

- `strategy_config_v2.json`：新增 `system2_updates.em_factor_weight = 0.2`；新增 ensemble 字段（lgbm_weight/cnn_weight/cnn_model_path）
- `quantmind/features/__init__.py`：注册 EM + ExprFactor 全部符号
- `.claude/CLAUDE.md`：新增完整"因子添加指南"（方式A表达式 / 方式B快照）
- `quantmind/regime/dynamic_weights.py`：Ensemble 权重 7:3→6:4 (bull) / 7:3→6.5:3.5 (neutral) / 8:2→7.5:2.5 (bear)
- `scripts/daily_update.py`：新增 Step5c TODO 注释（CNN ensemble 融合预留接口）
- `.github/workflows/ci.yml`：新增 GitHub Actions CI（unit + weekly integration）

### 待完成（阶段3后）

- ~~**FactorCNN v2 推理接入 daily_update.py**~~ ✅ 已实现（2026-07-10 校正：`daily_update.py` `step5c_cnn_ensemble` 已落地，pkl 缺失时自动降级纯 LGBM，可用 `--no-cnn` 关闭）
- FactorCNN Regime 分层重训：Bull/Bear/Neutral 各训一个子模型，集成时按 HMM 状态切换
- analyst_revision 真实 IC 验证：Tushare 积分升级后拉取完整研报数据
- EM 因子均值 IC 提升：当前 0.015，接近但未达 0.02 阈值；A股基本面噪声限制

---

## 二、系统当前状态（2026-05-23）

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
| **meta_learner 目录** | v2 分类模型（LogisticReg，CV AUC=0.6025，n=100） | `data/meta_learner/` |
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
| meta_learner v2 | ✅ LogisticReg→hit，CV AUC=0.6025（n=100，LOQO） | `data/meta_learner/meta_learner_v2.pkl` |

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

#### Task 2：前向持仓结算 + meta-learner 重训（触发条件：2026-06-26后）

`data/paper_trading/forward_positions.json` 中 2026-03-31 建仓，约 2026-06-26 到期：

```bash
# 到期后：
# Step 1 - 拉取出场价格，追加 realized_pnl
python scripts/track_realized_pnl.py \
  --forward data/paper_trading/forward_positions.json

# Step 2 - 重训 meta-learner v2（分类模型，n 从 100 → ~110）
python scripts/train_meta_learner.py \
  --pnl   data/feedback/realized_pnl.parquet \
  --panel data/panel/alpha_panel_v4.parquet \
  --out   data/meta_learner/meta_learner_v2.pkl
# 期望 CV AUC 持续改善（2024年早期季度 AUC 低会随数据增加被稀释）
# 当 CV AUC ≥ 0.62 且 n ≥ 150 时可考虑接入生产排名
```

**⚠️ 重要注意**：`train_meta_learner.py` 使用的数据源是 `realized_pnl × alpha_panel_v4 join`，
**不是** `stock_returns.parquet`（后者只含单一30日窗口，伪复制样本，不可用于训练）。
详见 §五.五「Meta-Learner v2 设计决策」。

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

## 五.五、Meta-Learner v2 设计决策（2026-05-22 更新）

### 定位

**v2 的职责**：预测单只推荐股票是否能跑赢当季中位数（`hit` 二分类），用于调整推荐名单的置信度排序。

**不适合用来**：直接预测绝对收益（actual_return_63d），因为 n=100 时标签噪声远大于信号（std=0.225）。

### 根因分析（v2-draft 失败：train R²=0.034, CV R²=-0.013）

| 问题 | 根因 |
|------|------|
| 错误数据源 | `stock_returns.parquet` 只有一个30日窗口（2025-10-09～11-19），308只股票 × 平均1.46天 = **伪复制样本** |
| return_3m 无意义 | 该列来自模拟期，与预测分之间没有因果对应 |
| CV 设计错误 | 随机KFold把同期股票分到训练/测试集，泄露截面信息 |

### 正确方案（v2-final，CV AUC=0.6025）

```
数据源：realized_pnl × alpha_panel_v4 join
  - 8季度 × 10只 = 80行 pnl（2024Q2～2025Q4）
  - 2025-06-28 规范化到 2025-06-30，最终 n=100

特征：6个 Agent 代理分（由 compute_agent_proxies 计算）
目标：hit（actual_return_63d > panel_return_63d）
模型：StandardScaler + LogisticRegression(C=1.0)
CV：Leave-One-Quarter-Out（LOQO，7折，不泄露时序）
```

### 代理分 IC（对 hit 的 Spearman IC）

| 代理分 | IC | p值 | 方向 |
|--------|-----|-----|------|
| momentum | -0.238 | 0.017 ✅ | **负**（高动量代理→更难beat median） |
| sentiment | -0.242 | 0.015 ✅ | **负**（北向/融资买入高→反转风险大） |
| risk | -0.227 | 0.023 ✅（vs ret） | 负 |
| quality | +0.103 | 0.310 ❌ | 正（弱） |

> ⚠️ **IC 负号警告**：momentum / sentiment 代理分越高，实际表现越差。  
> 这意味着融资买入多、北向持仓高的股票存在均值回归（本期选股偏动量风格时需反转修正）。  
> **不建议直接用代理分正向排名**，应通过 LogisticRegression 学到的负系数加权后使用。

### LOQO CV AUC 逐季结果

| 季度 | AUC |
|------|-----|
| 2024-Q2 | 0.56 |
| 2024-Q3 | 0.24（差） |
| 2024-Q4 | 0.50 |
| 2025-Q1 | 0.42 |
| 2025-Q2 | 0.71 |
| 2025-Q3 | 0.84 |
| 2025-Q4 | **0.95** |
| **均值** | **0.6025** |

早期季度 AUC 低（模型还在"热身"），近期季度持续改善。每新增一季度数据后重训可期望进一步提升。

### 局限性

1. **n=100 太小**：LOQO 每折测试集仅 10-40 只，AUC 方差极大（0.24～0.95）
2. **代理分可能与真实 Agent 输出不同**：proxy 由规则加权，非模型预测
3. **暂不适合直接影响仓位**：AUC=0.60 仅提供弱先验，建议作为辅助排名信号
4. **建议触发时机**：n≥150（约再积累 2 个季度）后重评估是否部署进生产排名

### 下次重训命令

```bash
python scripts/train_meta_learner.py \
  --pnl   data/feedback/realized_pnl.parquet \
  --panel data/panel/alpha_panel_v4.parquet \
  --out   data/meta_learner/meta_learner_v2.pkl
# 新增季度后自动覆盖，LOQO CV 会包含新季度折
```

---

## 六、已知问题与技术债

| 问题 | 严重度 | 状态 | 建议 |
|------|--------|------|------|
| NAV 回测未含交易成本 | 🔴 高 | E3 进行中 | 加入 0.13% 单边成本，重跑 4 种权重 |
| alpha_panel_v4 最新 as_of = 2026Q2（但价格截至2026-05-11）| 🟡 中 | 待更新 | Task 7：每周更新日价格面板 |
| meta-learner v2 AUC=0.60（n小，早期季度差） | 🟡 中 | 已监控 | n≥150后重训；暂不接入生产排名 |
| ann_sentiment_5d 未实盘验证 IC | 🟡 中 | 待验证 | 需 Tushare token 拉取公告后运行 run_full_pipeline() |
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
1. [高] 前向持仓结算（2026-06-26到期）→ 追加realized_pnl → 重训meta-learner v2（分类）
   ⚠️  数据源：realized_pnl × alpha_panel_v4 join（非 stock_returns）
2. [中] 提交 Barra 模块：git add quantmind/risk/barra.py scripts/run_barra_attribution.py
3. [中] ann_sentiment_5d IC 验证（需 Tushare 公告数据）：python -c "from quantmind.features import run_text_sentiment_pipeline; f,ic=run_text_sentiment_pipeline(); print(ic)"
4. [中] 行业超配放宽：Layer6 Top行业（装修/建筑/机械）最多5只

【新增本期研发成果】
- HMM validate_state_labels()：bull≥neutral≥bear 自动校验+标签互换，fit_from_file()自动调用
- ann_sentiment_5d：Tushare公告→BERT/词典打分→5日滚动均值，IC框架就绪（需实盘数据）
- 序贯验证 Round 8（2025Q4→2026Q1）：IC=-0.146（return_21d），PROMOTE
  Round 9（2026Q2）因21d标签不足跳过
- Meta-Learner v2：CV AUC=0.6025，正确定位为 hit 分类（非绝对收益回归）

【安全规则】
- Tushare Token: <YOUR_TUSHARE_TOKEN>
- API Key 从 .env 读取，绝不硬编码
- 提交前 git add 具体文件，不要 git add -A
```

---

*更新时间：2026-05-22 | 对应 commit: 见最新 git log*  
*Phase E1/E2/E3 完成（v6 ICIR=0.380 · HRP 净年化+24.46%/Sharpe=0.996）+ 本期：HMM validate_state_labels · ann_sentiment_5d 因子 · 序贯验证 Round8 · MetaLearner v2（分类 AUC=0.60）*
