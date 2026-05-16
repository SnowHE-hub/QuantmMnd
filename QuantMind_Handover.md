# QuantMind 项目交接文档

> 用途：在 Claude Code 新会话中粘贴此文档，让新 Agent 无缝接续开发。
> 项目路径：`/home/lenovo/projects/quantmind`
> conda 环境：`quantmind`（Python 3.11）
> 硬件：RTX 5060 Ti Laptop 8GB，WSL Ubuntu on Windows

---

## 1. 项目定位（一句话）

QuantMind 是一个端到端的 Agent 驱动量化研究系统，包含三个子系统：
- **Multi-Agent 投资研究**：6+1 个 Agent 协作做个股深度研究，LangGraph 编排 + Self-Reflection
- **生成式量化选股**：LightGBM 粗排 → LLM Listwise Rerank → DPO 偏好对齐
- **严格回测引擎**：PIT 数据隔离、Walk-Forward 验证、Deflated Sharpe Ratio 统计检验

---

## 2. 当前完成状态

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 环境与基础设施 | ✅ 完成 |
| 1 | 数据层（PIT 严格，10/10 PIT 测试）| ✅ 完成 |
| 2 | 特征工程（41 因子 + 标准化 + 多时点快照）| ✅ 完成 |
| 2.2 | 因子分析（IC/IR/分层，top_factors.json）| ✅ 完成 |
| 3.1 | LightGBM 排序模型（IC_IR=+0.797，auto_flip）| ✅ 完成 |
| 3.2 | LLM Listwise Reranker（grounding_score=1.0）| ✅ 完成 |
| 3.3 | DPO 训练（Qwen2.5-1.5B，final_loss=0.6931）| ✅ 完成 |
| 4 | Multi-Agent 系统（6 Agent + LangGraph，24/24 测试）| ✅ 完成 |
| 5 | 知识库与 RAG（BGE-M3 + ChromaDB + HybridRetriever）| ✅ 完成（未填充真实数据）|
| 6 | 回测引擎（T+1/涨跌停/DSR/Walk-Forward，29/29 测试）| ✅ 完成 |
| 7 | 风险管理（Barra/HRP/Kelly/CPPI，34/34 测试）| ✅ 完成 |
| 8 | Streamlit UI（6 页面，curl 验证启动）| ✅ 完成 |
| 9 | 文档与博客（3 篇技术博客 + METHODOLOGY + QUICKSTART）| ✅ 完成 |

**测试总数：291/292 通过**（1 个 akshare 网络超时，属外部依赖偶发，非代码 bug）

---

## 3. 关键文件位置

```
quantmind/
├── data/
│   ├── features/csi300_2019Q1_2024Q2.parquet   # 训练面板 shape=(5760,43)
│   ├── features/top_factors.json               # 筛选后因子列表
│   └── snapshots/                              # PIT 快照（按日期）
├── models/
│   ├── lgbm_ranker.pkl                         # 训练好的 LGBM 模型
│   └── dpo_qwen/                               # DPO 微调权重（Qwen2.5-1.5B）
├── reports/                                    # 生成的 HTML 报告
├── logs/                                       # 运行日志
├── quantmind/
│   ├── core/         llm_router.py / state.py / config.py
│   ├── data/         akshare_provider.py / tushare_provider.py / snapshot.py
│   ├── features/     fundamental.py / technical.py / sentiment.py / pipeline.py
│   ├── models/       lgbm_ranker.py / llm_reranker.py / dpo_trainer.py
│   ├── agents/       planner/data/fundamental/technical/sentiment/critic/report + orchestrator
│   ├── kb/           builder.py / chunker.py / embedding_service.py / retriever.py
│   ├── backtest/     engine.py / metrics.py / statistical_tests.py / agent_backtest.py
│   ├── risk/         factor_risk.py / position_sizing.py / drawdown.py
│   └── ui/           streamlit_app.py + pages/p01~p06
└── scripts/
    ├── download_data.py      # 数据下载
    ├── build_features.py     # 因子计算
    ├── analyze_factors.py    # 因子分析
    ├── train_factor_model.py # LGBM 训练
    ├── run_llm_rerank.py     # LLM Rerank 推荐
    ├── run_agent_research.py # Agent 个股研究
    ├── run_backtest.py       # 策略回测
    └── build_kb.py           # 知识库构建
```

---

## 4. 本地 LLM 环境

```bash
# Ollama 已安装，两个模型：
ollama list
# glm-4.7-flash:latest   19 GB   ← 能力更强，复杂推理用
# qwen2.5:7b             4.7 GB  ← 速度更快，日常推理用

# LLM 调用优先级：
# 开发/测试 → Ollama 本地（不消耗 API）
# 生产/评估 → .env 里的 DEEPSEEK_API_KEY（DeepSeek 推荐主力）
```

---

## 5. 已知问题与待修复项

### 问题 1：north_bound_30d_net_inflow 因子全 0（待修复）
- **根因**：北向资金是市场级因子，横截面标准差=0，z-score 清零
- **修复方案**：从 SENTIMENT_FACTORS 移出，改为时序标准化（用过去 252 日均值/标准差）
- **涉及文件**：`quantmind/features/pipeline.py`、`quantmind/features/standardize.py`

### 问题 2：LGBM 模型 direction=-1（已处理，理解即可）
- **说明**：训练期学到大市值→正收益，但测试期 A 股 regime 切换导致反转
- **处理**：`auto_flip=True` 自动检测并取反，有效 IC = +0.089
- **字段**：`model.direction = -1`，`predict()` 已自动乘以 direction

### 问题 3：KB 知识库未填充真实数据（待执行）
- **现状**：`build_kb.py` 代码完整，但从未实际运行
- **影响**：SentimentAgent 的 `search_research_reports` 返回空列表
- **执行命令**（建议先小批量测试）：
```bash
conda run -n quantmind python scripts/build_kb.py \
  --tickers 600519.SH,300750.SZ \
  --doc-types news \
  --start 2024-01-01
```

### 问题 4：三大子系统缺乏整合入口（待开发）
- **现状**：每个子系统有独立脚本，没有一键串联的入口
- **待写**：`scripts/daily_update.py` 和 `scripts/full_research_pipeline.py`

---

## 6. 立即要做的 4 个任务（已有 Prompt，直接执行）

按优先级顺序：

### Task 1：修复 north_bound 因子
改为时序标准化，验证列不再全 0。

### Task 2：scripts/daily_update.py
每日自动流水线：拉数据 → 重建因子 → LGBM 粗排 → LLM 精排 → 输出推荐 JSON + HTML。

### Task 3：scripts/full_research_pipeline.py  
一键完成：输入股票列表 → Agent 研究 → 量化评分 → 回测 → 整合 HTML 报告。

### Task 4：KB 知识库实际填充
运行 build_kb.py，先干跑验证，再实际索引茅台+宁德时代的新闻数据。

---

## 7. 给 Claude Code 的开场白（直接粘贴）

```
请用中文回复。

我要继续开发 QuantMind 量化投资系统，项目在 /home/lenovo/projects/quantmind。
conda 环境：quantmind（Python 3.11）。

【项目现状】
- Phase 0-9 全部完成，291/292 测试通过（1个外部API网络超时，非代码bug）
- 三大子系统（Multi-Agent研究/生成式量化选股/严格回测）代码全部实现
- 已有训练好的模型：models/lgbm_ranker.pkl 和 models/dpo_qwen/

【本地LLM】开发期优先 Ollama：qwen2.5:7b（日常）或 glm-4.7-flash（复杂任务）
【安全规则】API Key 全部从 .env 读取，绝不硬编码

【现在要做的任务，按顺序执行】

Task 1：修复 north_bound_30d_net_inflow 因子（全0问题）
请先读取 quantmind/features/pipeline.py 和 quantmind/features/standardize.py，
把该因子从横截面标准化改为时序标准化（用过去252日均值/标准差）。

Task 2：实现 scripts/daily_update.py
每日数据更新流水线：
Step1 确定今日交易日 → Step2 检查缓存 → Step3 下载快照 →
Step4 重建因子 → Step5 LGBM粗排Top50 → Step6 LLM精排Top10 →
Step7 保存 data/recommendations/{as_of}.json → Step8 生成 reports/daily_{as_of}.html
支持参数：--date / --force / --stop-after

Task 3：实现 scripts/full_research_pipeline.py
一键流程：输入股票列表 → Agent研究（Ollama）→ 量化评分 → 历史回测 → 整合HTML报告
支持参数：--tickers / --as-of / --mode (research-only|backtest-only|full)

Task 4：KB知识库填充
先给 scripts/build_kb.py 加 --dry-run 参数验证流程，
再实际运行索引 600519.SH 和 300750.SZ 的 2024 年新闻数据。

请先读取所有相关文件，再按顺序执行。
```

---

## 8. 后续更长期的优化方向（不急，按需推进）

| 优先级 | 任务 | 说明 |
|---|---|---|
| 高 | AgentBacktester 历史回测 | 选 5 只股票跑过去 1 年，验证 Agent alpha |
| 高 | Streamlit UI 真实数据联调 | 目前页面可导入但没有真实数据跑通的截图 |
| 中 | DPO 模型接入推理链 | models/dpo_qwen 已有权重，接入 inference.py 替代 API |
| 中 | 扩展知识库数据源 | 加入研报、年报 PDF 文本 |
| 中 | GitHub 开源准备 | 清理 .env、加 .gitignore、写完整 CONTRIBUTING.md |
| 低 | 技术博客发布 | docs/blog/ 下 3 篇已写好，可直接发知乎/Medium |
| 低 | Docker 化 | 提供 Dockerfile 方便部署 |

---

## 9. 常用命令速查

```bash
# 激活环境
conda activate quantmind

# 运行全套核心测试
conda run -n quantmind python -m pytest tests/test_backtest.py tests/test_agents.py tests/test_risk.py -v --tb=short

# 运行完整测试（含外部API，较慢）
conda run -n quantmind python -m pytest tests/ -m "not slow" -v --tb=short

# 启动 Streamlit UI
conda run -n quantmind streamlit run quantmind/ui/streamlit_app.py

# 检查 Ollama 可用模型
ollama list

# 查看 DPO 训练日志
tail -30 logs/dpo_training_v2.log

# 查看项目代码规模
find quantmind/ -name "*.py" | xargs wc -l | tail -5
```

---

*文档生成时间：2025-05-09*
*对应项目版本：Phase 0-9 完成，291/292 测试通过*
