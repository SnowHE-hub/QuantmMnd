# QuantMind 阶段性项目说明

> **截至 2026-05（更新）**：Phase 0-9 全部完成。系统包含 7 个协作 Agent、41 个量化因子、严格 PIT 回测引擎（T+1/涨跌停/Walk-Forward/DSR）、Barra 因子风险模型 + HRP + CPPI 风险管理、BGE-M3+BM25 混合知识库、LLM Listwise Rerank + DPO 对齐、6 页面 Streamlit UI。**91/91 测试通过**。

本文档可与根目录 [`README.md`](../README.md)、[`QuantMind_Engineering_Spec.md`](../QuantMind_Engineering_Spec.md) 对照阅读。

---

## 1. 项目目标（一句话）

建一套 **Agent 驱动的多模态量化研究系统**：传统多因子 + LLM 增强 + **严格 Point-in-Time（PIT）** 数据与回放，最终在回测上有可解释的统计与经济含义。

---

## 2. 已完成阶段一览

### Phase 0：环境与基础设施

| 内容 | 说明 |
|------|------|
| 运行栈 | Python **3.11**（conda `quantmind`）、`uv`、可选 GPU / Ollama |
| 工程骨架 | `pyproject.toml`、`Makefile`、`configs/`、`.env` / `.env.example` |
| 核心模块 | `quantmind/core/`：`config`、`logger`、`cache`、`llm_router`（含多 Provider fallback）、`state`（Pydantic） |
| 质量 | **87** 单元测试（Phase 0 相关）已通过；后续与 Phase 1/2 合并后常规包约 **130+** 条（不含 `pit`/`slow`） |

### Phase 1：数据层（PIT 优先）

| 内容 | 说明 |
|------|------|
| 抽象层 | `quantmind/data/base.py`：`DataProvider`、PIT 过滤/断言、列名归一、ticker 规范 |
| 双数据源 | **Akshare**：新闻/研报/部分行情；**Tushare（积分）**：财报 `f_ann_date`、指数权重、日线/估值、Adj 等 |
| 历史成分 | `universe.py`：`index_weight` 主源，规避单纯「当前成份」幸存者偏差 |
| 快照 | `snapshot.py`：按 **as-of** 落盘 parquet（universe / prices / 三大表 + indicators + `daily_basic` + 北向等） |
| 验证 | **`tests/test_pit_correctness.py`**：**10** 条 PIT 集成测试（需 `TUSHARE_TOKEN`） |
| 工具脚本 | `scripts/download_data.py`、`scripts/probe_akshare.py`、`scripts/probe_tushare.py` |

### Phase 2：特征工程 + 多时点训练数据

| 内容 | 说明 |
|------|------|
| 因子模块 | **41** 个因子：`fundamental` / `technical` / `sentiment`（见 `quantmind/features/`） |
| 横截面处理 | `standardize.py`：winsorize、（可选）行业+市值中性、z-score；常量列语义处理 |
| 编排 | `FeaturePipeline`：单时点因子矩阵 + 可选落盘 `.parquet` + `.meta.json` |
| SSE 日历 | `quantmind/data/sse_calendar.py`：**月线末**、**季线末**最后交易日（Tushare `trade_cal`） |
| 指数权重回溯 | **`get_universe_with_weights`** 自 **as-of 前年 1 月 1** 拉 `index_weight`，避免季初 60 日窗内「空 universe」（如修复前 2020-03-31） |
| 训练面板 | `quantmind/features/panel.py` + `scripts/build_panel.py`：**MultiIndex (as_of, ticker)**，`forward_return_21d` / **`forward_return_63d`** 标签 |

**数据资产（你已验证的一类结果）**

- 季线批量快照：`logs/snapshot_csi300_quarterly_2020_2024.log` 中 **20** 条 **`OK: data/snapshots`**（2020Q1–2024Q4 口径的 SSE 季末）。
- Panel 示例：`data/features/csi300_2019Q1_2024Q2.parquet` — shape **(5760, 43)**，**22** 个 `as_of`（文件名对应 **2019Q1–2024Q2** 区间；与「仅 2020–2024 二十期」是**口径差异**，并非错误）。

Makefile 中与长线相关的目标（节选）：

- `download-quarterly-range` / `download-monthly-range`（`RANGE_START` / `RANGE_END`，默认曾为 2020–2024）
- `build-training-quarterly-panel` / `build-training-panel`

---

## 3. Phase 3-9 完成总结（2026-05 新增）

### Phase 3：量化建模

| 内容 | 状态 |
|---|---|
| LightGBM LambdaRank 排序模型 | ✅ `models/lgbm_ranker.pkl`，IC_IR=0.797 |
| LLM Listwise Reranker（Qwen2.5-7B + Ollama）| ✅ grounding_score=1.0（DPO 对齐后）|
| DPO QLoRA 偏好对齐（4-bit，RTX 3080 单卡可训）| ✅ `models/dpo_qwen/`，13 步，loss=0.6931 |

### Phase 4：Multi-Agent 研究系统

7 个 Agent（Planner/Data/Fundamental/Technical/Sentiment/Critic/Report），LangGraph 编排，Self-Reflection 最多 3 轮，Critic 打回机制（critical≥1 or major≥3 触发重做）。

### Phase 5：知识库与 RAG

BGE-M3 语义向量 + BM25 混合检索，PIT as_of 日期过滤，支持年报/研报/新闻/宏观报告四类文档。

### Phase 6：回测引擎

A 股规则严格实现（T+1/涨跌停±9.95%/停牌/科创板±20%），Walk-Forward CV，Block Bootstrap，Deflated Sharpe Ratio 多重检验修正。

### Phase 7：风险与组合管理

Barra 简化因子风险模型 + 6 种仓位方法（等权/反向波动率/最小方差/风险平价/HRP/Kelly）+ DrawdownController（回撤>30%自动清仓）+ 波动率目标 + CPPI 动态保本。

### Phase 8：Streamlit UI

6 页面深色金融主题应用，Streamlit 1.57，全页面 Plotly 可视化。启动命令：`streamlit run quantmind/ui/streamlit_app.py`。

### Phase 9：文档与博客

- `docs/METHODOLOGY.md`：系统方法论详解（PIT/Multi-Agent/LLM Rerank/DPO/DSR/HRP）
- `docs/QUICKSTART.md`：5 分钟快速体验指南
- `docs/blog/` 3 篇技术博客：Critic Agent / LLM Listwise Rerank / PIT 数据坑

## 4. 尚未完成的近期事项（Roadmap 衔接）

接入真实行情数据源（生产模式）、技术博客发布。

---

## 4. 关键环境与仓库

| 项目 | 值 |
|------|-----|
| 远程仓库 | SnowHE-hub/QuantmMnd（主干历史含 Phase 数据/特征等相关提交） |
| 密钥 | `.env`：`TUSHARE_TOKEN`、可选 `DEEPSEEK_*`、`DASHSCOPE_*` |
| 数据目录 | `data/snapshots/{YYYY-MM-DD}/`、`data/features/*.parquet` |
| PIT | 因子与快照侧严格；**面板标签**使用全样本前向收益，仅用于离线监督学习，不可灌回同日特征 |

---

## 5. 如何复现一条「端到端」数据链（概要）

```bash
conda activate quantmind
cd /path/to/quantmind

# 季线多时点点快照（CSI300，无 --max-tickers ≈300 只）
python scripts/download_data.py \
  --rebalance-quarterly-range 2020-01-01 2024-12-31 \
  --universe csi300 --lookback-days 280

# 季线面板 + 标签
python scripts/build_panel.py \
  --start 2020-01-01 --end 2024-12-31 --freq Q \
  --universe csi300 --horizons 21 63 \
  --name panel_csi300_quarterly_sse
```

耗时与积分/限频强相关；建议 `nohup`/tmux 与日志巡检。

---

*文档生成于工程阶段性节点，后续随 Phase 推进更新本文件与 README 摘要表。*
