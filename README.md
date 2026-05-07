# QuantMind

> **AI Agent-Driven Quantitative Investment Research System**
>
> 多 Agent 协作 × 生成式量化选股 × 严格 PIT 回测

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-pending-lightgrey)]()
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)]()

---

## TL;DR

**QuantMind 不是一个 LLM 投资问答 demo**，而是一个端到端的 Agent 量化研究系统，包含三个相互验证的子系统：

1. **Multi-Agent Research** — 6 个专业 Agent（Planner/Data/Fundamental/Technical/Sentiment/Critic/Report）协作做深度个股研究，基于 LangGraph 状态机 + Self-Reflection 循环
2. **Generative Alpha** — 把"生成式推荐"思想迁移到量化选股：传统多因子粗排 → LLM Listwise Rerank → DPO 偏好对齐
3. **Rigorous Backtest** — 严格 Point-in-Time 数据隔离、Walk-Forward 验证、Deflated Sharpe Ratio 多重检验修正

---

## 当前状态

| Phase | 任务 | 状态 |
|---|---|---|
| 0 | 环境与基础设施 | 🚧 进行中 |
| 1 | 数据层（PIT 严格） | ⏳ 待开始 |
| 2 | 特征工程 | ⏳ |
| 3 | 量化模型（LightGBM + LLM Rerank + DPO） | ⏳ |
| 4 | Agent 系统 | ⏳ |
| 5 | 知识库与 RAG | ⏳ |
| 6 | 回测引擎 | ⏳ |
| 7 | 风险与组合管理 | ⏳ |
| 8 | UI（Streamlit） | ⏳ |
| 9 | 文档与博客 | ⏳ |

---

## Quick Start

### Prerequisites

- **Python 3.11**（推荐 conda 管理；3.13 与部分金融库不兼容）
- **NVIDIA GPU**（可选，DPO 训练 / Embedding 推理用，≥8GB 显存）
- **Ollama**（可选，开发期本地 LLM 节省 API 成本）
- **API Keys**（至少需要其一）：
  - DeepSeek（推荐主力）
  - DashScope / 通义千问
  - OpenAI / Anthropic

### Installation

```bash
# 1. 克隆仓库
git clone <repo-url> quantmind && cd quantmind

# 2. 创建并激活 conda 环境
conda create -n quantmind python=3.11 -y
conda activate quantmind

# 3. 装 uv（10x 快于 pip）
pip install uv

# 4. 装依赖（最小核心）
uv pip install -e .

# 或一键全装（推荐开发机）
uv pip install -e ".[all]"
```

### Configuration

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑 .env，填入你的 API Keys
#    必须：DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY
#    必须：TUSHARE_TOKEN（A 股财报披露日 PIT）
#    可选：OPENAI_API_KEY, ANTHROPIC_API_KEY

# 3. （可选）启动 ollama 跑本地模型
ollama pull qwen2.5:7b
ollama serve
```

### Run

```bash
# Smoke test：验证环境
make smoke

# 下载数据（沪深 300，约 4-8 小时）
make download-data

# 构建因子特征
make build-features

# 跑量化策略回测
make run-backtest

# 跑 Agent 单股票分析
make run-agent

# 启动 UI
make run-ui
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
└──────────────┬──────────────────────────────────────┬───────────┘
               │                                      │
               ▼                                      ▼
   ┌─────────────────────┐              ┌────────────────────────┐
   │   Multi-Agent       │              │   Generative Quant     │
   │   Orchestrator      │              │   Selection            │
   │   (LangGraph)       │              │                        │
   │                     │              │  LightGBM (粗排)        │
   │  Planner → Data →   │              │       ↓                │
   │  Fund/Tech/Sent →   │              │  LLM Listwise Rerank    │
   │  Critic ⟲ Report    │              │       ↓                │
   │                     │              │  DPO-aligned Qwen3-4B   │
   └──────┬──────────────┘              └─────┬──────────────────┘
          │                                   │
          └─────────────┬─────────────────────┘
                        ▼
            ┌──────────────────────────────┐
            │   Rigorous Backtest Engine   │
            │   - PIT-strict snapshot      │
            │   - Walk-Forward CV          │
            │   - Deflated Sharpe / DSR    │
            │   - Agent Decision Backtest  │
            └──────────┬───────────────────┘
                       ▼
            ┌──────────────────────────────┐
            │   Data Layer (PIT)           │
            │   akshare / tushare / SEC    │
            └──────────────────────────────┘
```

---

## Project Structure

```
quantmind/
├── configs/              # YAML 配置（universe、模型超参、LLM provider）
├── data/                 # 数据目录
│   ├── raw/              # 原始下载
│   ├── processed/        # 清洗后
│   ├── features/         # 因子库
│   ├── snapshots/        # PIT 快照（按日期）
│   └── kb/               # RAG 知识库（向量索引）
├── docs/                 # 方法论文档
├── notebooks/            # 探索性分析
├── scripts/              # 命令行工具
├── tests/                # 测试套件（含 PIT 正确性测试）
└── quantmind/            # 主代码包
    ├── core/             # 配置 / 日志 / 缓存 / LLM 路由 / 状态
    ├── data/             # DataProvider / Snapshot / Universe
    ├── features/         # 基本面 / 技术 / 情绪 / LLM 因子
    ├── models/           # LightGBM / LLM Reranker / DPO
    ├── agents/           # 各 Agent + LangGraph Orchestrator
    ├── analysis/         # 财务比率 / DCF / 同业对比
    ├── kb/               # 知识库构建 / 检索器
    ├── backtest/         # 回测引擎 / 指标 / Walk-Forward / 统计检验
    ├── risk/             # 因子风险 / 仓位 / 回撤控制
    └── ui/               # Streamlit 应用
```

---

## Methodology Highlights

### Point-in-Time（项目灵魂）

PIT 是项目区别于 99% 个人量化项目的根本。在 T 时刻，Agent / 模型只能看到 T 时刻市场上真实可获得的数据。

具体防御：
- 财报数据用**披露日**（`f_ann_date`）而非报告期
- Universe 用**历史成分股**而非当前
- 知识库检索强制 `as_of` 过滤
- 专门的 `tests/test_pit_correctness.py` 十几个测试 case

### Critic + Self-Reflection

Critic Agent 不是装饰，会真正打回重做：
- `max_iterations=3` 硬上限
- 每轮迭代必须减少 issue 数量
- Severity 分级（critical/major/minor）

### Agent Decision Backtest（核武器）

让 Agent 对历史每个时点的"当时数据"做投资建议，再用真实未来数据验证 alpha：
- 严格用 snapshot
- 多 baseline 对比（Random / Single ReAct / Pure LightGBM / Analyst Consensus）
- 配对 t-test + Calibration

---

## Roadmap

- [x] 项目骨架（pyproject.toml / configs / Makefile / .env.example）
- [ ] Core 模块（config / logger / cache / llm_router / state）
- [ ] 数据层（akshare / tushare / snapshot / universe）
- [ ] PIT 测试套件
- [ ] 特征工程（传统因子 + LLM 因子）
- [ ] LightGBM 因子模型
- [ ] LLM Listwise Reranker
- [ ] DPO 偏好对齐
- [ ] 6-Agent 系统 + LangGraph 编排
- [ ] 知识库 / RAG
- [ ] 回测引擎 + 统计检验
- [ ] Agent 决策回测
- [ ] Streamlit UI
- [ ] 技术博客 × 3

---

## Contributing

施工中，欢迎 issue 和 PR。

## License

MIT

## Acknowledgements

本项目大量借鉴了以下开源项目的设计：
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [vectorbt](https://github.com/polakowo/vectorbt)
- [empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

---

*"May your Sharpe be high and your drawdown shallow."*
