# QuantMind 快速体验指南（5 分钟）

> 本指南帮助你在 5 分钟内在本地运行 QuantMind，**无需真实 API Key，无需 GPU**。

---

## 前置要求

| 工具 | 版本 | 说明 |
|---|---|---|
| Python | 3.11+ | 推荐 conda 管理 |
| conda/miniforge | 任意 | 环境隔离 |
| Git | 任意 | 克隆仓库 |

> **GPU / API Key 是可选的**。本指南全程使用 Demo 数据和本地模式。

---

## Step 1：克隆与安装（2 分钟）

```bash
# 克隆仓库
git clone <repo-url> quantmind
cd quantmind

# 创建 conda 环境（Python 3.11）
conda create -n quantmind python=3.11 -y
conda activate quantmind

# 安装核心依赖
pip install uv
uv pip install -e ".[all]"
```

> **常见问题**：若 `lightgbm` 安装失败，先运行 `conda install -c conda-forge lightgbm -y`。

---

## Step 2：环境配置（30 秒）

```bash
# 复制环境变量模板
cp .env.example .env

# （可选）如果你有 API Key，编辑 .env 填入：
# DEEPSEEK_API_KEY=sk-...
# TUSHARE_TOKEN=...
```

Demo 模式下 **不需要** 任何 API Key，所有页面都有内置的模拟数据。

---

## Step 3：启动 Streamlit UI（30 秒）

```bash
conda activate quantmind
streamlit run quantmind/ui/streamlit_app.py
```

浏览器自动打开 `http://localhost:8501`，你将看到深色金融主题的 QuantMind 界面。

---

## Step 4：探索各功能页面

### 🏠 Overview（首页）
- 系统架构图（Plotly 纯矢量绘制）
- Phase 0-9 进度一览
- 最新回测结果或 Demo 指标

### 🔍 个股研究
1. 在股票代码框输入 `600519.SH`
2. 点击 **「🎯 运行 Demo」**（无需 API，~1 秒）
3. 查看分析报告：执行摘要、财务分析、技术分析、情绪分析

> 若要运行真实 Multi-Agent 分析，需要 Ollama + `qwen2.5:7b`（见下方可选步骤）。

### 📊 策略回测
1. 保持默认参数（equal_weight，2022-2023，1000万）
2. 点击 **「🚀 运行回测」**
3. 查看：年化收益、夏普比率、最大回撤、净值曲线、月度热力图

### 🤖 Agent 回测
- 自动加载 Demo 决策数据（200 条，多评级）
- 筛选评级/置信度/行业
- 查看 Rating×收益箱线图、Calibration 曲线、命中率表

### 📚 知识库
- **检索测试**：输入自然语言查询，查看 Top-5 结果（Demo 模式）
- **文档浏览**：按股票/文档类型筛选
- **统计**：文档分布图表

### 🧠 模型管理
- LightGBM 特征重要性条形图（Demo 30 个因子）
- DPO 训练曲线（读取 `logs/dpo_training_v2.log` 或 Demo）
- `models/` 目录文件列表

---

## 可选：接入真实能力

### 开启 Multi-Agent 分析（需要 Ollama）

```bash
# 安装 Ollama（https://ollama.ai）
ollama pull qwen2.5:7b
ollama serve  # 保持运行

# 启动 UI 后，个股研究页面点「🚀 开始分析」即可
```

### 接入真实行情数据

```bash
# 编辑 .env，填入 TUSHARE_TOKEN 和 DEEPSEEK_API_KEY
# 下载 CSI 300 历史数据（约 4-8 小时）
python scripts/download_data.py --universe csi300 --start 2020-01-01

# 构建因子特征
python scripts/build_features.py

# 运行真实回测
python scripts/run_backtest.py --strategy equal_weight --start 2022-01-01 --end 2023-12-31
```

---

## 运行测试套件

```bash
# 全部测试（91 个，约 30 秒）
conda activate quantmind
pytest tests/ -v --tb=short

# 仅风险管理测试
pytest tests/test_risk.py -v

# 仅回测测试
pytest tests/test_backtest.py -v

# 仅 Agent 测试
pytest tests/test_agents.py -v
```

---

## 项目结构速查

```
quantmind/
├── quantmind/
│   ├── core/          # 配置、日志、LLM 路由
│   ├── data/          # 数据层（PIT 严格）
│   ├── features/      # 41 个量化因子
│   ├── models/        # LightGBM + LLM Reranker + DPO
│   ├── agents/        # 7 个 Agent + LangGraph 编排
│   ├── knowledge/     # BGE-M3 + BM25 混合检索
│   ├── backtest/      # 回测引擎 + 统计检验
│   ├── risk/          # 风险管理（Barra/HRP/CPPI）
│   └── ui/            # Streamlit UI（6 页面）
├── tests/             # 91 个测试
├── docs/              # METHODOLOGY.md / QUICKSTART.md
├── scripts/           # 数据下载 / 回测脚本
└── models/            # 模型权重（lgbm + dpo_qwen）
```

---

## 遇到问题？

| 问题 | 解决方案 |
|---|---|
| `streamlit: command not found` | `pip install streamlit` |
| `No module named 'lightgbm'` | `conda install -c conda-forge lightgbm -y` |
| `No module named 'torch'` | Demo 模式不需要 torch；训练用 `pip install torch` |
| Ollama 连接失败 | 确认 `ollama serve` 已运行，端口 11434 |
| 回测失败 | 检查数据是否已下载（`data/raw/` 目录） |

更多问题请提交 [GitHub Issue]()。

---

*文档版本：v1.0 — 2026-05 — QuantMind Phase 9*
