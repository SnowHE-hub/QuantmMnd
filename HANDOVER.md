# QuantMind — Claude Code 接续开发交接文档

> 写于 2026-05-16，由 Cursor Agent 生成。  
> 本文档供下一位 Claude Code 实例快速接手，不需要重读历史对话。

---

## 一、项目简介

**QuantMind** 是一套 AI 驱动的 A 股多模态量化投研系统，架构分三层：

| 层次 | 内容 |
|------|------|
| **数据底座** | 29 季度 PIT 快照（2019Q1–2026Q2）× 1374 只 Alpha 股票 |
| **量化选股** | 71 因子面板 → LightGBM LambdaRanker → HRP 仓位优化 |
| **智能分析** | 6-Agent 投资分析（估值/动量/风险/质量/情绪/策略）× Streamlit 展示端 |

**仓库**：`git@github.com:SnowHE-hub/QuantmMnd.git`  
**主分支**：`main`（当前 HEAD = 上次提交后的最新 commit）  
**运行环境**：WSL2 Ubuntu / conda env `quantmind` / Python 3.11

---

## 二、当前系统状态（2026-05-16）

### 2.1 数据系统 ✅ 完整

| 资产 | 状态 | 路径 |
|------|------|------|
| Alpha 宇宙 | 1373 只 A 股（txt 列表） | `data/alpha_universe/alpha_universe.txt` |
| PIT 快照 | **37 个快照目录**，覆盖 2019Q1–2026Q2，每季度 1374 只全覆盖 | `data/snapshots/<date>/` |
| 历史价格面板 | 227 万行，1374 只，2019-01-02–2026-05-11 | `data/raw/alpha_prices_panel.parquet` |
| 宽格式价格 | adj_close 宽表 | `data/alpha_universe/alpha_prices_wide.parquet` |

**已通过审计**：`data/snapshots/2026-06-30/meta.json` 的 `failures: {}` 且所有表 coverage=100%。

> ⚠️ **已知问题**：`alpha_panel_v3.parquet` 最新 `as_of = 2024-12-31`，尚未包含 2025Q1–2026Q2 新数据（待 E1.1 重建）。  
> ⚠️ `alpha_prices_panel.parquet` 中 `adj_close` 截至 2026-05-11，后续拉取用 `scripts/build_daily_price_panel.py`。

### 2.2 System 1：量化选股 ✅ 可用

| 组件 | 文件 | 说明 |
|------|------|------|
| 因子工程（71 因子） | `quantmind/features/fundamental.py`, `technical.py`, `expansion.py` | 含 8 个小盘扩展因子 |
| 因子面板构建 | `scripts/build_full_panel.py` | 产出 `alpha_panel_v3.parquet` |
| 因子 IC 分析 | `scripts/analyze_factor_ic.py` | 产出 `data/features/top_factors_v2.json` |
| **主模型（当前生产）** | `models/lgbm_v5_alpha_63d.pkl` | 37 特征，14 fold，`label = forward_return_63d` |
| Regime 集成模型 | `models/lgbm_ensemble_large.pkl` + `lgbm_ensemble_small.pkl` | 按市场状态切换 |
| HRP 仓位优化 | `quantmind/portfolio/position_sizing.py` | `hrp_weights()` / `kelly_weights()` |
| 全流程演示 | `scripts/run_2025q1_full_demo.py` | 5517 只 → 4 层漏斗 → Top-15 + 6-Agent |

**模型指标（当前）**：`lgbm_v5_alpha_63d` IC mean = -0.019，ICIR = -0.237（**待 E1 重训提升**）

### 2.3 System 2：6-Agent 投资分析 ✅ 可用

| Agent | 模型版本 | 文件 |
|-------|---------|------|
| ValuationAgent | LGBM v3 | `quantmind/agents/investment_agents/valuation_agent.py` |
| MomentumAgent | PatchTST v4 | `quantmind/agents/investment_agents/momentum_agent.py` |
| RiskAgent | HMM v3 | `quantmind/agents/investment_agents/risk_agent.py` |
| QualityAgent | **LGBM v2**（20 财务因子；标签为截面 IC 加权质量合成后的 top/bottom 30%）+ **Piotroski** 兜底 + **rules_v1** | `quantmind/agents/investment_agents/quality_agent.py` |
| SentimentAgent | **finbert_llm_v4**（FinBERT 批量打分 → Top-K → LLM `chat` 合成）；降级 **bert_v3**（TF-IDF 正负语义中心）→ **rules_v1** | `quantmind/agents/investment_agents/sentiment_agent.py` |
| StrategyAgent | LLM（DashScope qwen-plus） | `quantmind/agents/investment_agents/strategy_agent.py` |

**重要修复**（刚完成）：`StrategyAgent` 现在要求 LLM 输出 JSON `{"thesis": "...", "confidence": 0.xx}`，`confidence_score` 属性已加为别名；此前全部输出 0.00，现已正常（0.58–0.78）。

**Quality / Sentiment 升级**（2026-05-16）：原先 README 仍写 Piotroski / TF-IDF，实为 Phase C 未优先迭代这两条链路；现已补齐训练脚本与注册流程。**`models/agents/` 下 `.pkl` 已 gitignore**，换机需本地训练或拷贝权重后再执行注册。

```bash
conda activate quantmind
cd /home/lenovo/projects/quantmind

# Quality LGBM v2（默认 alpha_panel_v3；panel v4 落盘后改路径重训）
python scripts/train_quality_agent_v2.py \
  --panel data/panel/alpha_panel_v3.parquet \
  --out models/agents/quality_lgbm_v2.pkl

# Sentiment bert_v3 bundle（正负种子语料 → TF-IDF 中心向量，供 finbert_llm_v4 降级）
python scripts/train_sentiment_agent_v3.py --out models/agents/sentiment_bert_v3.pkl

# 写入 registry 并设为 active（Quality 须 model_type=ml，已由脚本写好）
python scripts/register_upgraded_agents.py --all
```

**调用入口**：
```bash
# 单股 6-Agent 分析（无 LLM）
python scripts/run_investment_pipeline.py --ticker 600519.SH --as-of 2025-01-02

# 全流程 + DashScope
set -a && . .env && set +a
python scripts/run_2025q1_full_demo.py   # 读 .env 里的 DASHSCOPE_API_KEY
```

### 2.4 System 3：回测验证 ✅ 可用

| 组件 | 文件 | 说明 |
|------|------|------|
| 历史回测验证 | `scripts/validate_strategies.py` | `batch_validate(strategies, price_df, panel_df)` |
| 日频 NAV 回测 | `scripts/run_nav_backtest.py` | 真实持仓 + CSI300 基准 |
| Alpha 报告 | `scripts/run_alpha_report.py` | HTML 报告含换手/行业暴露 |

### 2.5 展示端（Streamlit）⚠️ 部分接线

6 个页面，基本框架完整，部分数据路径已对接：

| 页面 | 文件 | 状态 |
|------|------|------|
| 今日推荐 | `app/pages/1_今日推荐.py` | ✅ 读 `reports/daily/<date>/` |
| 漏斗选股 | `app/pages/2_漏斗选股.py` | ✅ 调 `run_investment_pipeline` |
| 单股分析 | `app/pages/3_单股分析.py` | ✅ Agent 雷达图（读 strategies.json） |
| 回测表现 | `app/pages/4_回测表现.py` | ✅ 读 `reports/alpha_final/` |
| 模型管理 | `app/pages/5_模型管理.py` | ⚠️ 模型注册表展示，训练触发未接 |
| 智能问答 | `app/pages/6_智能问答.py` | ⚠️ RAG 知识库需重建 |

### 2.6 日更流水线 ⚠️ 待完整测试

```bash
python scripts/daily_update.py \
  --as-of 2026-05-16 \
  --auto-regime \
  --position-sizing hrp
```

步骤：`step1`(行情) → `step2`(因子) → `step3`(快照) → `step4`(面板) → `step5`(LGBM) → `step5b`(HRP) → `step6`(LLM rerank) → `step7`(daily 报告) → `step7a`(6-Agent 分析) → `step8`(推送)

Cron 示例：`scripts/setup_cron.sh`

---

## 三、环境与密钥

### 3.1 激活环境
```bash
conda activate quantmind
cd /home/lenovo/projects/quantmind
set -a && source .env && set +a   # 加载 API keys
```

### 3.2 关键环境变量（存于 `.env`，已 gitignore）

| 变量 | 用途 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Token A（官方，2000积分） |
| `DASHSCOPE_API_KEY` | 阿里百炼 qwen-plus（已验证可用） |
| `DEEPSEEK_API_KEY` | DeepSeek 备用 |
| `TUSHARE_HI_URL` | **必须留空！** 原代理已过期 |

> ⚠️ 运行任何 Tushare 相关脚本，必须先 `export TUSHARE_HI_URL=""`，否则走已过期代理报错。

### 3.3 Tushare Token B（已过期，仅记录）
- Token: `5caf9b3022e13d4e915df0af19a076130287cb7837c0b020290691c8`  
- 代理: `http://tsy.xiaodefa.cn`  
- **到期时间: 2026-05-19**（可续费，关键接口不稳定不建议依赖）

---

## 四、目录结构速查

```
quantmind/
├── app/                    # Streamlit 展示端
│   ├── main.py
│   └── pages/              # 6 个页面
├── configs/default.yaml    # 全局配置
├── data/
│   ├── alpha_universe/     # alpha_universe.txt + 价格宽表
│   ├── features/           # top_factors_v2.json
│   ├── panel/              # alpha_panel_v3.parquet（需重建到v4）
│   ├── raw/                # alpha_prices_panel.parquet
│   └── snapshots/          # 37 个 PIT 快照（gitignored）
├── models/                 # 所有训练好的模型（gitignored）
│   └── agents/             # 6 个 Agent 模型
├── quantmind/              # 核心库
│   ├── agents/investment_agents/   # 6-Agent 实现
│   ├── data/               # snapshot.py, tushare_provider.py
│   ├── features/           # 71 因子实现
│   ├── models/             # LGBMRankerModel 包装
│   └── portfolio/          # HRP/Kelly 仓位优化
├── scripts/                # 所有 CLI 脚本
│   ├── build_full_panel.py
│   ├── daily_update.py
│   ├── download_data.py
│   ├── run_2025q1_full_demo.py   # ← 三系统端到端演示
│   ├── register_upgraded_agents.py  # Quality/Sentiment 注册 active
│   ├── train_quality_agent_v2.py
│   ├── train_sentiment_agent_v3.py
│   ├── run_investment_pipeline.py
│   ├── run_nav_backtest.py
│   ├── train_*.py          # 各 Agent 训练脚本
│   └── validate_strategies.py
├── .env                    # API keys（gitignored）
├── METHODOLOGY.md          # 工程方法论文档
└── HANDOVER.md             # 本文档
```

---

## 五、下阶段优先任务（Phase E）

### E1 — 模型增量重训（最高优先，~1天）

2026Q2 快照已落盘，现在有 **29 个季度**训练数据（原 20 个），是模型质量最大的单次提升机会。

#### E1.1 重建因子面板 v4

```bash
export TUSHARE_HI_URL=""
conda activate quantmind
cd /home/lenovo/projects/quantmind

python scripts/build_full_panel.py \
  --snapshots-dir data/snapshots \
  --out data/panel/alpha_panel_v4.parquet
```

预期：`(27480 × 29/20) ≈ 39,831` 行，覆盖 2020Q1–2026Q2 的 20 个完整季度（前几个因数据不完整可能被过滤）。

#### E1.2 重训 LGBM 主模型

```bash
python scripts/train_factor_model.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --label forward_return_63d \
  --out models/lgbm_v6_alpha.pkl \
  --n-folds 18 \
  --report-out reports/wf_alpha_v4/report.html
```

预期 ICIR 从 -0.237 提升到 0.4+（2025 牛市样本会修正 IC 方向）。

#### E1.3 更新 Regime 集成模型

```bash
# 重建 Regime 特征面板
python scripts/build_regime_panel.py --panel data/panel/alpha_panel_v4.parquet

# 重训 Regime 集成（large-cap bull / small-cap）
python scripts/train_regime_ensemble.py \
  --panel data/panel/alpha_panel_v4_regime.parquet
```

#### E1.4 更新 Agent 模型（用新数据重训）

```bash
# Risk HMM v3 — 新增 2025 波动率
python scripts/train_risk_agent_v3.py

# Valuation LGBM v3 — 加入 2025 年报
python scripts/train_valuation_agent_v3.py \
  --panel data/panel/alpha_panel_v4.parquet

# Momentum PatchTST v4 — 2025 牛市正样本比例改善
python scripts/train_momentum_patchtst.py

# Quality LGBM v2 — 建议 panel v4 重建后重训
python scripts/train_quality_agent_v2.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --out models/agents/quality_lgbm_v2.pkl

# Sentiment bert_v3 bundle（FinBERT 不可用时兜底）
python scripts/train_sentiment_agent_v3.py --out models/agents/sentiment_bert_v3.pkl
python scripts/register_upgraded_agents.py --sentiment
```

---

### E2 — 策略归因（E1 后，~2天）

用 2025 数据验证模型有效性：

```bash
# 日频 NAV 回测（含 2025-2026）
python scripts/run_nav_backtest.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --out reports/nav_2020_2026.html

# 对比四种权重方法
for method in equal hrp kelly blend; do
  python scripts/run_nav_backtest.py \
    --weight-method $method \
    --out reports/nav_${method}.html
done

# Alpha 最终报告
python scripts/run_alpha_report.py \
  --panel data/panel/alpha_panel_v4.parquet \
  --weight-method hrp \
  --out reports/alpha_final/report_v4.html
```

---

### E3 — 成本修正（独立，~2天）

在 `run_nav_backtest.py` 中接入交易成本：

```python
# 在 build_daily_nav() 中：
cost_per_trade = 0.001 + 0.0003   # 印花税 + 佣金 = 0.13% 单边（卖方印花税减半后约 0.065%）
turnover_cost = turnover_ratio * cost_per_trade
daily_return -= turnover_cost / holding_days
```

---

### E4 — Barra 风险归因（~1周）

当前无 Barra 模块，需新建 `quantmind/risk/barra.py`：

1. 行业哑变量（申万一级，从 `stock_basic.industry` 映射）
2. 规模/价值/动量/质量 4 个风格因子（已在 `alpha_panel` 中）
3. 计算因子载荷矩阵 → 残差 IC（纯 Alpha 验证）
4. 因子风险预算约束（行业暴露 ≤ 10%）

---

## 六、已知问题 & 技术债

| 问题 | 严重度 | 建议 |
|------|--------|------|
| `lgbm_v5_alpha_63d` ICIR = -0.237（负） | 🔴 高 | E1.2 重训后应转正 |
| `alpha_panel_v3` 最新 as_of = 2024-12-31 | 🔴 高 | E1.1 重建到 v4 |
| `alpha_prices_panel.parquet` 截至 2026-05-11 | 🟡 中 | 每周更新一次 |
| `validate_strategies` AVOID 过多（仅"积极关注"才回测） | 🟡 中 | 放宽阈值或加更多信号 |
| `StrategyAgent` `confidence_score` 曾全为 0 | 🟢 已修复 | 已改为 JSON 结构化输出 |
| Quality LGBM v2 标签为自监督合成（非直接预测收益） | 🟡 中 | E1.1 后用 `alpha_panel_v4` 重训；关注与 `forward_return_63d` 的截面 IC 是否稳定 |
| Sentiment `finbert_llm_v4` 依赖 transformers + 可选 GPU | 🟡 中 | 失败时自动降级 bert_v3 / rules_v1 |
| Registry 中 `model_type` 非 `ml` 时不会加载 pickle | 🟢 已规避 | `register_upgraded_agents.py` 使用 `model_type="ml"` |
| Streamlit 页面 5/6 未接线 | 🟡 中 | E2 后补 |
| 日更流水线未上线 cron | 🟡 中 | `scripts/setup_cron.sh` |
| `alpha_universe.txt` 实际 1373 行（比名字少1） | 🟢 低 | 确认是否有重复/删除 |

---

## 七、快速验证命令

```bash
# 验证环境
conda activate quantmind
cd /home/lenovo/projects/quantmind
export TUSHARE_HI_URL=""
set -a && source .env && set +a

# 1. 检查因子面板
python -c "
import pandas as pd
df = pd.read_parquet('data/panel/alpha_panel_v3.parquet').reset_index()
print('shape:', df.shape, '| as_of range:', df['as_of'].min(), '-', df['as_of'].max())
"

# 2. 端到端三系统演示（~5min，调百炼 LLM）
python scripts/run_2025q1_full_demo.py
# 报告: reports/demo/2025-01-02-fullA/full_A_demo_report.html

# 3. 启动 Dashboard
streamlit run app/main.py
```

---

## 八、本轮提交摘要（Quality / Sentiment 升级）

相对于仓库上一版 `main`，本批变更主要为 **QualityAgent LGBM v2** 与 **SentimentAgent finbert_llm_v4 接线**，不含 `models/` 权重（目录仍 gitignore）。

| 类型 | 路径 |
|------|------|
| Agent 逻辑 | `quantmind/agents/investment_agents/quality_agent.py` — `quality_lgbm_v2` 推理路径 |
| LLM 调用修复 | `quantmind/agents/investment_agents/sentiment_agent.py` — `_get_llm_synthesis` 使用 `llm_client.chat` |
| 训练 / 注册 CLI | `scripts/train_quality_agent_v2.py`，`scripts/train_sentiment_agent_v3.py`（原有），`scripts/register_upgraded_agents.py` |
| 文档 | `README.md`，`HANDOVER.md` |

更早的大规模变更（HRP、daily_update、Streamlit、Risk/Valuation/Momentum 训练脚本等）已在先前 commit（如 `d5481f4`）中。
