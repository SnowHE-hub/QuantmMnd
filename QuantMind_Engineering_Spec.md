# AI Agent驱动的多模态量化投资系统 —— Claude Code工程执行手册

> **项目代号**：QuantMind
> **版本**：v1.0 Engineering Spec
> **目标**：本文档是给Claude Code的逐任务执行指令集。每个Phase内的Task都是独立可执行单元，按顺序粘贴Prompt到Claude Code即可推进。
> **对标岗位**：Qwen Agent / 阿里云Serverless Agent / 蚂蚁Plan A / 乾象MetaSummer量化研究 / 招行AI数据科学家 / 快手生成式推荐 / 腾讯混元

---

## 0. 项目全景与核心创新

### 0.1 项目定位

这不是一个"LLM问答投资助手Demo"，而是一个**端到端的Agent驱动量化研究系统**，包含三个相互验证的子系统：

1. **Agent投资研究子系统**（Research Agents）：多Agent协作做基本面/技术面/情绪面深度研究，输出结构化投资建议
2. **生成式量化选股子系统**（Generative Alpha）：把"生成式推荐"思想迁移到量化——用LLM做Listwise股票排序+理由生成，传统因子做粗排
3. **严格回测与归因子系统**（Rigorous Backtest）：解决look-ahead bias、survivorship bias、统计显著性，让Agent建议和Alpha策略接受真实市场检验

### 0.2 与市面上Agent投资Demo的本质区别

| 维度 | 网上烂大街Demo | QuantMind |
|---|---|---|
| Agent架构 | 单Agent ReAct | 多Agent DAG + Self-Reflection循环 |
| 数据真实性 | LLM瞎编财务数据 | 真实akshare/Tushare/yfinance + 时间快照 |
| 计算可靠性 | LLM算财务比率 | Python严格计算+LLM只解读 |
| 回测严格度 | 无回测 | Point-in-time严格回测+多个偏差控制 |
| 选股策略 | 无 | 传统因子+LLM Listwise Rerank+DPO对齐 |
| 评测可信度 | 主观 | Sharpe/IC/IR/MaxDD/胜率全套+统计检验 |

### 0.3 简历与面试杀手锏

完成本项目后，简历可写：

> **QuantMind: Agent-Driven Quantitative Research System**
> - 设计6-Agent协作的投资研究系统（Planner/Data/Fundamental/Technical/Sentiment/Critic），基于LangGraph实现Self-Reflection循环与严格的Point-in-Time数据隔离
> - 提出"生成式量化选股"框架：传统多因子模型粗排+LLM Listwise Rerank with Reasoning精排+DPO偏好对齐，在沪深300成分股回测IC达0.0X，Sharpe X.XX
> - 实现工业级回测引擎：解决look-ahead/survivorship/data-snooping bias，集成Walk-Forward验证与Deflated Sharpe Ratio统计检验
> - 开源代码X stars，技术博客全网阅读XXk+，受邀某Workshop口头报告

---

## 1. 工程总目录结构

```
quantmind/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Makefile
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── backtest_methodology.md
│   └── agent_prompts.md
├── configs/
│   ├── default.yaml
│   ├── universe_csi300.yaml
│   ├── universe_sp500.yaml
│   └── llm_providers.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── snapshots/        # Point-in-time快照
│   └── kb/               # 知识库向量索引
├── quantmind/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py            # 全局状态定义
│   │   ├── config.py           # 配置加载
│   │   ├── llm_router.py       # 多LLM路由
│   │   ├── cache.py            # 缓存层
│   │   └── logger.py           # 日志
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py             # DataProvider抽象
│   │   ├── akshare_provider.py
│   │   ├── tushare_provider.py
│   │   ├── yfinance_provider.py
│   │   ├── sec_edgar_provider.py
│   │   ├── snapshot.py         # PIT快照管理器
│   │   └── universe.py         # 股票池管理
│   ├── features/
│   │   ├── __init__.py
│   │   ├── fundamental.py      # 基本面因子
│   │   ├── technical.py        # 技术因子
│   │   ├── sentiment.py        # 情绪因子
│   │   ├── llm_features.py     # LLM特征
│   │   └── pipeline.py         # 特征工程pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factor_model.py     # 多因子模型
│   │   ├── lgbm_ranker.py      # LightGBM排序
│   │   ├── llm_reranker.py     # LLM Listwise Rerank
│   │   └── dpo_trainer.py      # DPO对齐训练
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py             # Agent基类
│   │   ├── planner.py
│   │   ├── data_agent.py
│   │   ├── fundamental_agent.py
│   │   ├── technical_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── critic_agent.py
│   │   ├── report_agent.py
│   │   └── orchestrator.py     # LangGraph编排
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── ratios.py           # 财务比率
│   │   ├── dcf.py              # DCF估值
│   │   ├── comparable.py       # 同业对比
│   │   └── technical_indicators.py
│   ├── kb/
│   │   ├── __init__.py
│   │   ├── builder.py          # 知识库构建
│   │   ├── retriever.py        # 检索器
│   │   └── chunker.py          # 文档切分
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           # 回测引擎
│   │   ├── portfolio.py        # 组合管理
│   │   ├── execution.py        # 撮合模拟
│   │   ├── metrics.py          # 评测指标
│   │   ├── walk_forward.py     # Walk-Forward验证
│   │   └── statistical_tests.py # 统计检验
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── factor_risk.py      # 因子风险
│   │   ├── drawdown.py         # 回撤控制
│   │   └── position_sizing.py  # 仓位管理
│   └── ui/
│       ├── __init__.py
│       ├── streamlit_app.py    # Streamlit UI
│       └── components/
├── scripts/
│   ├── download_data.py
│   ├── build_features.py
│   ├── train_factor_model.py
│   ├── train_dpo.py
│   ├── run_backtest.py
│   ├── run_agent_research.py
│   └── benchmark_agents.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   ├── 03_agent_case_studies.ipynb
│   └── 04_backtest_results.ipynb
└── tests/
    ├── test_data_providers.py
    ├── test_features.py
    ├── test_agents.py
    ├── test_backtest.py
    └── test_pit_correctness.py  # 关键：PIT正确性测试
```

---

## 2. 环境与依赖（Phase 0：基础设施）

### Task 0.1 项目初始化

**Claude Code Prompt（直接复制使用）：**

```
请为我创建一个名为quantmind的Python量化Agent项目，要求：

1. 使用pyproject.toml（PEP 621），Python版本要求>=3.11
2. 使用uv或pip作为包管理器，但配置文件用pyproject.toml
3. 项目代码主目录为quantmind/
4. 创建以下顶级目录结构（空目录用.gitkeep占位）：
   configs/, data/{raw,processed,features,snapshots,kb}, docs/, scripts/,
   notebooks/, tests/, quantmind/{core,data,features,models,agents,analysis,kb,backtest,risk,ui}

5. pyproject.toml需要包含以下依赖分组：
   - core: pandas>=2.2, numpy>=1.26, pydantic>=2.5, pyyaml, python-dotenv,
           loguru, tenacity, joblib, polars>=0.20
   - data: akshare>=1.12, tushare, yfinance>=0.2.40, requests, beautifulsoup4,
           lxml, openpyxl, pyarrow
   - ml: scikit-learn>=1.4, lightgbm>=4.1, xgboost, statsmodels, scipy
   - dl: torch>=2.2, transformers>=4.40, accelerate, peft, trl, datasets,
         sentence-transformers
   - llm: langchain>=0.2, langgraph>=0.1, langchain-openai, langchain-community,
          openai, anthropic, dashscope
   - rag: chromadb>=0.5, rank-bm25, FlagEmbedding
   - backtest: vectorbt, empyrical, quantstats, pyfolio-reloaded
   - viz: matplotlib, seaborn, plotly, altair
   - ui: streamlit>=1.30, gradio>=4.0
   - dev: pytest, pytest-cov, pytest-asyncio, ruff, mypy, ipykernel, jupyter

6. 创建.env.example，包含以下变量：
   DEEPSEEK_API_KEY=
   DASHSCOPE_API_KEY=    # 阿里云Qwen
   OPENAI_API_KEY=
   ANTHROPIC_API_KEY=
   TUSHARE_TOKEN=
   DATA_ROOT=./data
   LOG_LEVEL=INFO
   CACHE_DIR=./.cache
   DEFAULT_LLM_PROVIDER=deepseek
   DEFAULT_LLM_MODEL=deepseek-chat
   EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
   EMBEDDING_DEVICE=cuda

7. 创建.gitignore，忽略：
   .env, __pycache__, *.pyc, .ipynb_checkpoints, data/raw/*, data/processed/*,
   data/features/*, data/snapshots/*, data/kb/*, .cache/, *.parquet,
   *.feather, .venv, venv, *.log, mlruns/, wandb/

8. 创建Makefile，包含targets：
   install, install-dev, test, lint, format, clean, run-ui, build-features,
   run-backtest, run-agent

9. 创建README.md主框架（详细内容后续Task补充），先有：
   项目标题、Badges占位、TL;DR一段、Quick Start三步、目录结构、贡献指南占位

10. 创建configs/default.yaml，包含默认配置框架：
    data:
      universe: csi300
      start_date: '2018-01-01'
      end_date: '2024-12-31'
      pit_strict: true
    features:
      lookback_days: 252
      rebalance_freq: M
    models:
      factor_model: lgbm
      use_llm_rerank: true
    backtest:
      initial_capital: 1000000
      commission_bps: 3
      slippage_bps: 5
      benchmark: 000300.SH
    llm:
      provider: ${DEFAULT_LLM_PROVIDER}
      model: ${DEFAULT_LLM_MODEL}
      temperature: 0.1
      max_tokens: 4096

请生成所有上述文件的完整内容。
```

### Task 0.2 核心基础模块

**Claude Code Prompt：**

```
请实现quantmind/core/下的所有基础模块：

1. quantmind/core/config.py
   - 使用pydantic-settings加载.env和configs/*.yaml
   - 实现Settings类，支持环境变量覆盖
   - 实现load_config(config_name: str)函数
   - 配置项分组：DataConfig/FeatureConfig/ModelConfig/BacktestConfig/LLMConfig

2. quantmind/core/logger.py
   - 基于loguru，输出到console+文件
   - 不同级别使用不同颜色
   - 日志文件按日切分，保留7天
   - 暴露get_logger(name)函数
   - 关键操作（数据下载、模型训练、回测）单独输出到operations.log

3. quantmind/core/cache.py
   - 基于joblib + diskcache
   - 装饰器@cached(ttl_hours=24, key_func=None)
   - 支持parquet/feather/pickle三种序列化
   - 自动按函数名+参数hash生成key
   - 支持手动invalidate

4. quantmind/core/llm_router.py
   核心：统一不同LLM API的调用接口，避免代码绑定到特定provider
   - 抽象基类BaseLLMProvider，方法chat(messages, **kwargs) -> str
   - 实现以下providers：
     * DeepSeekProvider（用openai SDK，base_url切换）
     * QwenProvider（用dashscope）
     * AnthropicProvider（用anthropic SDK）
     * OpenAIProvider
   - 实现LLMRouter类：
     * 根据config自动选provider
     * 支持fallback链：deepseek失败→qwen→openai
     * 集成tenacity做exponential backoff重试
     * 记录token消耗到日志
     * 实现token_usage_tracker，按provider/model聚合

5. quantmind/core/state.py
   定义全局Pydantic Schema：
   - InvestmentQuery: 用户输入
   - TaskNode: DAG节点
   - DataSnapshot: PIT数据快照
   - FundamentalAnalysis: 基本面分析结果
   - TechnicalAnalysis: 技术分析结果
   - SentimentAnalysis: 情绪分析结果
   - CriticFeedback: Critic反馈
   - InvestmentReport: 最终报告
   - AgentState: LangGraph State（包含上述所有字段+iteration_count+history）

每个文件都要：
- 完整type hints
- 详尽docstring（Google风格）
- 关键函数有简单使用示例（在docstring或if __name__ == "__main__"中）
- 输出可读日志

请按顺序生成上述5个文件的完整代码。
```

---

## 3. 数据层（Phase 1：可信数据基石）

> **本Phase的核心理念**：数据层的正确性决定整个项目的可信度。任何回测和Agent输出的可信度都建立在"在T时刻Agent能看到的数据严格等于T时刻市场上真实可获得的数据"这一基石上。这是与99%个人项目最大的区别。

### Task 1.1 DataProvider抽象与实现

**Claude Code Prompt：**

```
请实现quantmind/data/下的数据层，这是整个项目的基石，必须严格遵守Point-in-Time原则。

1. quantmind/data/base.py
   抽象基类DataProvider：
   - get_price(ticker, start, end, as_of=None, freq='D') -> DataFrame
     * as_of：关键参数，PIT约束，只返回as_of日期前可用的数据
     * 列：[date, open, high, low, close, volume, adj_close]
   - get_financials(ticker, statement_type, as_of=None) -> DataFrame
     * statement_type: balance_sheet | income | cashflow
     * 关键：财报有"报告期"和"披露日期"两个时间，PIT必须用披露日期约束
   - get_consensus_estimate(ticker, as_of=None) -> DataFrame
     * 一致预期数据
   - get_corporate_actions(ticker, as_of=None) -> DataFrame
     * 分红、配股、拆股、停复牌
   - get_index_constituents(index_code, as_of=None) -> List[str]
     * 关键：处理survivorship bias，必须返回as_of时点的真实成分股
   - get_news(ticker, start, end, as_of=None) -> DataFrame
   - is_tradable(ticker, date) -> bool
     * 是否可交易（停牌/退市检查）

   每个方法都必须：
   - 强制要求as_of参数（默认None表示用当前最新数据，但要warning）
   - 返回前断言：所有数据的"可用日期" <= as_of
   - 返回DataFrame带元数据attrs: {data_source, fetched_at, as_of}

2. quantmind/data/akshare_provider.py
   实现AkshareProvider继承DataProvider：
   - 用akshare获取A股数据
   - 处理akshare的常见坑：
     * 列名不一致（中英文混用）→ 标准化为统一英文列名
     * 数据缺失值用NaN而非空字符串
     * 复权处理（前复权qfq用于回测，不复权用于事件研究）
     * 财报披露日期：用stock_yjbb_em获取业绩快报披露日，用作PIT约束
   - 实现内部_to_pit(df, as_of)方法：把"报告期"数据用披露日做PIT过滤
   - 用@cached装饰器缓存所有外部调用

3. quantmind/data/tushare_provider.py
   实现TushareProvider继承DataProvider：
   - Tushare在财报披露日字段更准确（f_ann_date字段）
   - 实现get_financials时，必须用f_ann_date而非end_date做PIT过滤
   - Token从.env读取
   - Pro级API需要积分，普通用户用免费部分（限频）

4. quantmind/data/yfinance_provider.py
   实现YfinanceProvider继承DataProvider：
   - 美股数据，用于扩展性
   - SEC披露日期：从EDGAR的filings数据获取
   - 处理美股特殊情况：split调整、dividend调整

5. quantmind/data/snapshot.py
   实现SnapshotManager类：
   核心：把任意as_of日期的"全市场快照"持久化到parquet，保证回测可复现。
   - build_snapshot(date, universe, force_rebuild=False)
     * 拉取该date之前可用的所有数据：行情、财报、新闻
     * 按ticker分文件存到data/snapshots/{date}/
     * 用元数据manifest.json记录data_source、build_time、coverage
   - load_snapshot(date) -> Dict[str, DataFrame]
   - validate_snapshot(date) -> SnapshotValidationReport
     * 检查PIT正确性：所有数据日期<=as_of
     * 检查覆盖率：每个universe股票是否都有数据
     * 检查异常值：极端return、负价格等

6. quantmind/data/universe.py
   实现UniverseManager类：
   核心：解决survivorship bias。回测中T日的universe必须是T日真实存在的股票。
   - get_universe(name, as_of) -> List[str]
     * name: csi300 | csi500 | sp500 | nasdaq100 | custom
     * 必须返回as_of时点的真实成分股，包括后来退市的
   - get_listing_date(ticker) -> date
   - get_delisting_date(ticker) -> Optional[date]
   - is_in_universe(ticker, universe_name, date) -> bool
   - 内置历史成分股数据（akshare的index_component_em可拉沪深300历史）

每个文件必须：
- 关键路径有assert防御PIT bug（这是最容易出错的地方）
- 用loguru记录所有数据获取操作和异常
- 单元测试可执行（创建对应的tests/test_xxx.py）
- 提供__main__示例，可独立运行验证

请生成全部6个文件的完整代码，包括数据列名标准化字典和测试样例。
```

### Task 1.2 PIT正确性测试套件

**Claude Code Prompt：**

```
请实现tests/test_pit_correctness.py，这是项目最重要的测试，要严防look-ahead bias：

需要的测试用例：

1. test_price_pit_strict
   - 拉取as_of='2023-06-15'的300750.SZ数据，验证返回数据date都<=2023-06-15

2. test_financials_use_disclosure_date
   - 验证财报数据用披露日期而非报告期：
     * 2022年Q4财报报告期2022-12-31，但披露日2023-04-25
     * 在as_of='2023-04-20'时不应该看到该财报
     * 在as_of='2023-04-26'时应该看到

3. test_universe_no_survivorship_bias
   - 拉取2018-01-01的沪深300成分股
   - 验证列表中包含后来退市的股票（如某些ST股）
   - 验证不包含2018年还未上市的股票

4. test_corporate_action_adjustment
   - 测试拆股/送股的复权处理
   - 验证前复权数据连续

5. test_snapshot_reproducibility
   - 同一as_of日期build两次snapshot，结果必须bit-level一致
   - 用md5校验

6. test_news_pit
   - 验证新闻只返回发布日<=as_of的

7. test_multi_provider_consistency
   - 同一股票同一日期，akshare和tushare的close价应一致（容忍0.5%差异）

8. test_holiday_handling
   - as_of=节假日，应该返回最后一个交易日的数据

9. test_suspended_stock_handling
   - 停牌期间的股票，is_tradable应返回False
   - 停牌期间不应有volume>0的数据

10. test_no_future_leak_in_kb
    - 知识库构建时，加入的研报/新闻必须严格按发布日期切分
    - as_of=2023-01-01不应检索到2023-02的研报

每个测试必须：
- 使用真实API（pytest mark slow，可skip）+ mock版本
- 失败时打印详细诊断信息（具体哪一行数据违反PIT）
- 一旦失败CI必须fail

请生成完整测试代码，并在tests/conftest.py添加pytest fixtures。
```

### Task 1.3 数据下载脚本

**Claude Code Prompt：**

```
请实现scripts/download_data.py，作为命令行工具批量下载初始数据：

用法：
    python scripts/download_data.py --universe csi300 --start 2018-01-01 --end 2024-12-31
    python scripts/download_data.py --build-snapshots --dates 2020-01-01,2021-01-01,2022-01-01
    python scripts/download_data.py --kb-corpus --tickers 300750.SZ,002594.SZ

功能：
1. 用argparse解析参数（--universe, --start, --end, --provider, --build-snapshots, --dates, --kb-corpus, --tickers）
2. 进度条用rich.progress
3. 自动跳过已下载文件（除非--force）
4. 三类下载模式：
   a) bulk：批量下载某universe某时间段所有数据
   b) snapshot：构建特定日期的全市场快照
   c) kb-corpus：下载特定股票的研报、新闻、年报PDF文本（用于RAG）
5. 错误处理：
   - 单只股票失败不影响其他
   - 失败列表写到failed_downloads.log
   - 提供--retry-failed选项
6. 限频处理：
   - akshare/tushare都有限频，用ratelimit装饰器
   - akshare每分钟≤200次
   - tushare按token积分等级动态调整
7. 输出汇总：
   - 总耗时、成功数、失败数、数据大小
   - 按ticker的覆盖度报告

请生成完整可执行脚本。
```

---

## 4. 特征工程（Phase 2：因子库构建）

> **本Phase核心理念**：特征是模型的天花板。我们要构建包含**传统量化因子+LLM抽取的非结构化特征**的混合因子库，这是与传统量化基金的最大差异化。

### Task 2.1 传统因子库

**Claude Code Prompt：**

```
请实现quantmind/features/下的因子库：

1. quantmind/features/fundamental.py
   实现FundamentalFactors类，至少包含以下基本面因子（每个都是PIT安全的）：

   估值因子：
   - PE_TTM, PB, PS_TTM, EV_EBITDA, FCF_Yield
   - PEG（PE/3年净利润CAGR）

   质量因子：
   - ROE_TTM, ROA_TTM, GrossMargin, OperatingMargin, NetMargin
   - 杜邦三因子分解：净利率 × 总资产周转 × 权益乘数
   - DebtToEquity, CurrentRatio, QuickRatio, InterestCoverage
   - Accruals（应计项目，盈余质量代理）
   - 现金流/净利润比

   成长因子：
   - Revenue_CAGR_3Y, Revenue_CAGR_5Y
   - NetProfit_CAGR_3Y, NetProfit_CAGR_5Y
   - YoY_Revenue_Growth, YoY_NetProfit_Growth
   - 季度同比加速度（本季同比 - 上季同比）

   规模因子：
   - LogMarketCap, LogEnterpriseValue

   每个因子函数签名：
   def factor_name(financials: Dict[str, DataFrame], price: DataFrame, as_of: date) -> float | NaN

2. quantmind/features/technical.py
   实现TechnicalFactors类：

   动量因子：
   - Momentum_1M, 3M, 6M, 12M（剔除最近1月）
   - Industry-Adjusted Momentum（行业中性化动量）
   - 52周高点距离

   反转因子：
   - Reversal_1M, Reversal_5D

   波动率因子：
   - Volatility_3M, 1Y
   - IdiosyncraticVolatility（CAPM残差波动率）
   - DownsideVolatility

   流动性因子：
   - Amihud_Illiquidity
   - Turnover_3M_Avg
   - Volume_Spike（最近5日成交量/前30日均量）

   技术形态：
   - RSI_14, RSI_28
   - MACD_Signal
   - Bollinger_Position
   - 量价背离指标

   每个因子用vectorbt或pandas-ta实现，必须矢量化计算。

3. quantmind/features/sentiment.py
   实现SentimentFactors类：

   分析师情绪：
   - Analyst_Rating_Score（一致预期评级）
   - Estimate_Revision_3M（预期上调比例）
   - Earnings_Surprise（最近季度EPS超预期幅度）

   新闻情绪：
   - News_Sentiment_30D（用FinBERT/snownlp打分）
   - News_Volume_Spike（新闻数量异常）
   - Negative_News_Ratio

   市场情绪：
   - 北向资金流入（A股专属，akshare的stock_em_hsgt_north_net_flow_in）
   - 融资融券余额变化
   - 大单净流入

4. quantmind/features/llm_features.py
   核心创新：让LLM从财报/研报/新闻抽取结构化特征
   实现LLMFeatureExtractor类：

   - extract_management_tone(annual_report_text) -> Dict
     * 用LLM读年报"管理层讨论与分析"，输出：
       {confidence_score: 0-1, growth_outlook: bullish/neutral/bearish,
        risk_emphasis: high/med/low, strategic_focus: List[str]}

   - extract_competitive_moat(company_description, industry_data) -> Dict
     * 输出：{moat_type: cost/network/brand/switching/regulatory,
              moat_strength: 1-5, durability_years: int}

   - extract_news_event_impact(news_text, ticker) -> Dict
     * 输出：{event_type: enum, impact_direction: pos/neg/neutral,
              impact_magnitude: 1-5, time_horizon: short/med/long}

   - extract_research_consensus(reports_list) -> Dict
     * 综合多份研报，输出统一观点：
       {consensus_target_price, divergence_score, key_disagreements}

   关键工程要点：
   - 用structured output（OpenAI function calling或Qwen的tool_call）
   - Pydantic模型严格校验
   - 缓存：相同文本相同prompt不重复调LLM
   - 批量处理：用asyncio + asyncio.Semaphore控制并发
   - 成本追踪：每次调用记录token消耗

5. quantmind/features/pipeline.py
   实现FeaturePipeline类：
   - run(universe, start_date, end_date, freq='M') -> DataFrame
     * MultiIndex: (date, ticker)
     * Columns: 所有因子
   - 每个rebalance日（月初）：
     1. 加载该日snapshot
     2. 并行计算所有因子（用joblib.Parallel）
     3. 标准化（cross-sectional rank or zscore）
     4. 中性化（行业中性、市值中性，用回归残差法）
     5. 异常值处理（3 sigma winsorize）
     6. 缺失值填充（中位数，按行业分组）
   - 输出存到data/features/factors_{universe}_{start}_{end}.parquet
   - 自动版本化：每次pipeline变更生成新版本号

请生成全部5个文件的完整代码，每个因子都要有详细docstring说明经济含义和参考文献。
```

### Task 2.2 因子分析与有效性检验

**Claude Code Prompt：**

```
请实现notebooks/02_factor_analysis.ipynb的对应Python脚本：scripts/analyze_factors.py

这是量化研究的标准流程，必须严格执行：

1. 单因子IC分析
   - Information Coefficient (Spearman相关系数)
   - 每个rebalance日计算因子值与未来1M/3M/6M收益的IC
   - 输出IC时间序列、IC均值、IC IR (mean/std)、IC衰减曲线

2. 分组回测（Quintile Backtest）
   - 按因子值分5组（Q1最低到Q5最高）
   - 每组等权持有，月度调仓
   - 输出每组累计收益曲线、年化收益、Sharpe
   - 多空组合（Q5-Q1）的Sharpe是核心指标

3. 因子稳健性检验
   - 不同时期分组：牛市/熊市/震荡市表现
   - 不同universe：沪深300/中证500/全市场
   - 不同行业：行业内排序+行业间对比

4. 因子相关性矩阵
   - 计算所有因子两两相关性
   - 识别冗余因子（|corr|>0.7需要降维）
   - 用PCA分解共线性

5. 因子衰减分析
   - 计算因子从生成到失效的时间
   - 用于决定因子的rebalance频率

6. 多空组合最大回撤
   - 不仅看Sharpe，要看尾部风险
   - 计算最大回撤、calmar ratio

输出：
- factor_analysis_report.html（用plotly生成，可交互）
- factor_ranking.csv（所有因子按IC IR排名）
- top_factors.json（IC IR > 0.3的因子列表，作为下一步建模输入）

实现要点：
- 用alphalens-reloaded库做底层IC计算
- 多进程并行加速
- 必须避免lookahead bias（IC计算时T日因子值预测T+1的收益）

请生成完整脚本+必要的可视化辅助函数。
```

---

## 5. 量化模型层（Phase 3：从传统多因子到LLM Listwise Rerank）

> **本Phase核心创新**：实现"传统多因子模型粗排 + LLM Listwise Rerank with Reasoning + DPO对齐"三阶段策略。

### Task 3.1 LightGBM多因子排序模型

**Claude Code Prompt：**

```
请实现quantmind/models/factor_model.py和quantmind/models/lgbm_ranker.py：

1. quantmind/models/factor_model.py
   FactorModel基类抽象：
   - fit(X, y, **kwargs)
   - predict(X) -> np.ndarray
   - rank(X) -> np.ndarray  # 返回每只股票的排名
   - feature_importance() -> pd.Series
   - save(path), load(path)

2. quantmind/models/lgbm_ranker.py
   LGBMRanker类（继承FactorModel）：

   核心要点：
   - 使用LightGBM的lambdarank objective（learning to rank）
   - target: 未来N日收益的cross-sectional rank
   - group: 每个rebalance日的所有股票为一组
   - early_stopping_rounds=50

   实现方法：
   - prepare_data(features_df, prices_df, holding_period=20):
     * 构造(X, y, group)：
       X: 因子值（标准化后）
       y: 未来holding_period日收益的rank
       group: 每个调仓日的股票数（lambdarank要求）

   - fit(X, y, group, eval_set=None):
     * 默认参数：
       num_leaves=63, learning_rate=0.05, n_estimators=500,
       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
       lambda_l1=0.1, lambda_l2=0.1
     * 支持类别特征（行业one-hot或Lightgbm原生类别支持）

   - predict_at_date(date, features_df) -> pd.DataFrame:
     * 输出：[ticker, score, rank, percentile]

   - explain(date, features_df, top_k=10) -> Dict:
     * 用SHAP计算top_k股票的因子贡献
     * 返回结构化结果，给LLM Reranker使用

   - cross_validate(X, y, group, n_splits=5):
     * 时序CV（不能用普通KFold，会泄露未来）
     * 用sklearn.model_selection.TimeSeriesSplit变体

3. 训练脚本scripts/train_factor_model.py：
   - 加载features parquet
   - Walk-Forward训练：
     * 训练窗口：rolling 3年
     * 验证窗口：6个月
     * 测试窗口：3个月
     * 每3个月重训
   - 保存每个时期的模型到 models/lgbm/{train_end_date}.pkl
   - 输出训练日志：每期的IC、Sharpe、Top10股票示例

请生成完整代码，包含详细注释解释每个超参数选择的理由。
```

### Task 3.2 LLM Listwise Reranker（核心创新）

**Claude Code Prompt：**

```
请实现quantmind/models/llm_reranker.py，这是项目最核心的创新：把生成式推荐的Listwise Rerank思想用到量化选股。

核心idea：
- LightGBM粗排出Top-N候选（如N=50）
- LLM对Top-N做Listwise Rerank：不仅给排序，还给每只股票的推理过程
- 关键：让LLM做reasoning而非简单打分，这是"生成式"的核心

1. quantmind/models/llm_reranker.py
   LLMReranker类：

   核心方法：
   - rerank(date, candidates: List[StockCandidate], top_k=10) -> RerankResult

   StockCandidate数据结构：
   - ticker, name, sector
   - factor_values: Dict[str, float]
   - lgbm_score, lgbm_rank
   - shap_explanation: Dict[str, float]  # 因子贡献
   - recent_news_summary: str
   - financial_highlights: Dict[str, str]
   - peer_comparison: Dict[str, Dict]

   RerankResult数据结构：
   - ranked_tickers: List[str]
   - reasoning: Dict[ticker, str]  # 每只股票的推荐理由
   - portfolio_thesis: str  # 整体组合逻辑
   - risk_warnings: List[str]

   关键prompt设计：

   SYSTEM_PROMPT = '''
   你是一位资深量化基金经理，擅长结合量化模型信号和基本面定性判断。

   现在给你以下信息：
   1. 量化模型（LightGBM）粗排出的Top-N候选股票（按模型分数降序）
   2. 每只股票的SHAP因子贡献（量化模型为什么看好它）
   3. 每只股票的近期重大事件、财务亮点、同业对比

   你的任务：
   1. 综合定量+定性，对这N只股票重新排序，输出Top-K
   2. 对每只股票给出120字以内的推荐理由（必须引用具体数字）
   3. 对Top-K整体组合给出投资逻辑（为什么这K只能形成有效组合）
   4. 列出3条主要风险

   严格要求：
   - 不能仅复述量化模型的判断，必须加入额外洞察
   - 不能编造数据，所有数字必须来自输入
   - 推理必须可验证（哪些因子+哪个事件+什么逻辑→什么结论）
   - 优先考虑市场未充分定价的信息

   输出严格JSON格式：
   {
     "ranked_tickers": ["...", "..."],
     "reasoning": {"ticker1": "...", "ticker2": "..."},
     "portfolio_thesis": "...",
     "risk_warnings": ["...", "...", "..."]
   }
   '''

   USER_PROMPT_TEMPLATE = '''
   日期：{date}
   候选股票（{N}只）：

   {candidates_table}

   每只股票详细信息：
   {detailed_cards}

   请输出Top-{K}的Listwise Rerank结果。
   '''

   工程实现要点：
   - 用structured output（function calling）保证JSON有效
   - 候选股票太多时分批：N>30则分两批，最后再合并
   - 实现批量推理：用asyncio并行多次调用减少延迟
   - 缓存：相同date+candidates的结果缓存
   - 成本控制：估算每次rerank的token消耗，超过阈值警告

2. 单元测试tests/test_llm_reranker.py：
   - 用mock LLM测试（返回固定JSON）
   - 验证输出格式严格符合RerankResult schema
   - 测试空候选、单候选、N>30的分批逻辑
   - 测试数字一致性：reasoning里的数字必须出现在输入中

3. 实现evaluation：
   quantmind/models/llm_reranker_eval.py：
   - reasoning_coverage_score：reasoning中提到的具体数字占比
   - reasoning_grounding_score：reasoning中数字与输入的一致性（用LLM-as-Judge）
   - rerank_lift：LLM rerank相比LGBM粗排的IC提升

请生成完整实现+测试+评估代码。
```

### Task 3.3 DPO偏好对齐（让LLM学会"好的"投资推理）

**Claude Code Prompt：**

```
请实现quantmind/models/dpo_trainer.py，用DPO（Direct Preference Optimization）让LLM学会优秀的投资推理风格：

核心idea：
- 收集"chosen"（好的reasoning）和"rejected"（差的reasoning）pair
- 用DPO让一个小LLM（如Qwen3-4B）学会偏好chosen风格
- 这个小LLM作为线下大规模rerank的成本优化方案
- 这是Qwen Agent JD里"GRPO/GSPO/DAPO"等RL训练经验的强对应

1. quantmind/models/dpo_data_builder.py
   构造DPO训练数据：

   方法一：基于backtest结果的自动标注
   - 对历史每个rebalance日，用GPT-4或DeepSeek-V3生成多份reasoning（不同temperature）
   - 60天后真实收益>10%且原reasoning推荐了→chosen
   - 60天后真实收益<-5%但原reasoning大力推荐→rejected
   - 这样自动构造10000+ pairs

   方法二：基于人类反馈（少量精标）
   - 准备100-200对最关键的case，自己手动标注
   - 用作golden test set

   方法三：基于规则的合成
   - chosen: 引用具体财务数字+多维度论证+风险讨论
   - rejected: 空泛、矛盾、忽略风险
   - 用GPT-4生成各500条

2. quantmind/models/dpo_trainer.py
   基于trl库实现DPOTrainer：

   - 基模型：Qwen3-4B-Instruct（可选Qwen2.5-7B根据显存）
   - LoRA配置：
     r=16, lora_alpha=32, target_modules=q_proj, k_proj, v_proj, o_proj
     (gate_proj, up_proj, down_proj)
     dropout=0.05
   - DPO配置：
     beta=0.1, max_length=4096, max_prompt_length=3072
     batch_size=2 (RTX 5060 Ti 16GB约束)
     gradient_accumulation_steps=8
     learning_rate=5e-7
     warmup_steps=100
     num_epochs=3

   - 训练监控：
     用wandb或tensorboard记录loss、rewards/chosen, rewards/rejected, accuracy

   - 显存优化：
     QLoRA 4bit量化、gradient_checkpointing、flash_attention_2

3. quantmind/models/inference.py
   推理服务封装：
   - load_dpo_model() -> 加载训练好的LoRA权重
   - rerank_with_dpo_model(...) 替代API调用，本地推理
   - 性能基准：单次rerank latency vs API调用
   - 用vllm做生产部署（可选，加分项）

4. 评估scripts/evaluate_dpo.py：
   - 在golden test set上对比：
     * Base Model (Qwen3-4B) vs DPO Model
     * DPO Model vs DeepSeek-V3 API
   - 评估维度：
     * 数字一致性
     * 推理深度（用GPT-4打分）
     * 与真实60天收益的IC
   - 输出对比表+可视化

请生成完整代码，包括数据构造、训练、推理、评估全流程。

注意：训练前先生成1000条mock数据跑通pipeline，确认训练能正常进行不OOM，再扩展到10000条正式训练。
```

---

## 6. Agent层（Phase 4：多Agent协作研究系统）

> **本Phase核心**：构建可独立运行的多Agent投资研究系统，与量化模型层互补——量化做横截面选股，Agent做深度个股研究。

### Task 4.1 Agent基类与工具集

**Claude Code Prompt：**

```
请实现quantmind/agents/base.py和工具集：

1. quantmind/agents/base.py
   BaseAgent抽象基类：
   - 属性：name, description, llm, tools, max_iterations, system_prompt
   - 方法：
     * run(state: AgentState) -> AgentState（核心方法）
     * pre_check(state) -> bool（前置条件检查）
     * post_validate(output) -> bool（输出校验）
     * format_input(state) -> str
     * parse_output(llm_response) -> Dict

   关键设计：
   - 每个Agent输入输出都是AgentState，不可变更新
   - 错误处理：tool调用失败、LLM输出格式错误、超时
   - 日志：每次run记录输入摘要、调用工具、token消耗、输出摘要
   - Trace：保留完整执行trace用于debug

2. quantmind/agents/tools/
   实现Agent可调用的工具集（每个工具是一个Pydantic函数）：

   2.1 data_tools.py
   - fetch_stock_basics(ticker) -> Dict
   - fetch_financials_pit(ticker, as_of) -> Dict
   - fetch_price_history(ticker, start, end, as_of) -> DataFrame
   - fetch_industry_peers(ticker, as_of, top_n=10) -> List[Dict]
   - fetch_recent_news(ticker, days=30, as_of) -> List[Dict]

   2.2 analysis_tools.py
   - compute_financial_ratios(financials) -> Dict
   - compute_dcf_valuation(financials, wacc, growth, years) -> Dict
   - compute_comparable_multiples(target, peers) -> Dict
   - compute_technical_indicators(prices) -> Dict
   - run_factor_screening(criteria) -> List[str]

   2.3 kb_tools.py
   - search_research_reports(query, date_range, top_k=5) -> List[Dict]
   - search_news(query, date_range, top_k=10) -> List[Dict]
   - search_company_filings(ticker, filing_type, as_of) -> List[Dict]

   2.4 quant_tools.py
   - get_factor_signal(ticker, as_of) -> Dict  # LightGBM分数
   - get_llm_rerank_thesis(ticker, as_of) -> Dict
   - get_backtest_performance(strategy_id) -> Dict

   每个工具：
   - 用Pydantic定义input/output schema
   - 输出符合OpenAI function calling规范
   - 内置错误处理和retry
   - 添加@tool装饰器（langchain）

3. quantmind/agents/tools/registry.py
   工具注册中心：
   - register_tool(tool) / get_tool(name)
   - 提供get_tools_for_agent(agent_name)按需返回工具子集
   - 工具描述自动生成符合LLM理解的格式

请生成全部代码。
```

### Task 4.2 各专业Agent实现

**Claude Code Prompt：**

```
请实现quantmind/agents/下的所有具体Agent：

1. quantmind/agents/planner.py
   PlannerAgent：
   - 输入：用户query
   - 输出：DAG任务列表
   - System Prompt：

   '''
   你是顶级投资研究主管。给定用户查询，把复杂任务拆解为DAG子任务。

   每个子任务必须明确：
   - task_id: T1, T2, T3...
   - agent_type: data | fundamental | technical | sentiment | quant
   - action: 具体动作
   - params: 参数字典
   - depends_on: 依赖的task_id列表
   - priority: 1-5

   严格要求：
   - 任务粒度要适中：不能太粗（一个任务做太多）也不能太细（pipeline爆炸）
   - 必须有数据收集任务在最前面
   - 必须有Critic任务在最后
   - 任务总数控制在5-15个

   常见任务模板：
   - "分析单只股票" → [取数, 财务分析, 技术分析, 情绪分析, 综合, Critic, 报告]
   - "对比多只股票" → 上述每只各做一份 + 对比任务
   - "行业研究" → [行业数据, 龙头分析, 趋势, 竞争格局]
   - "风险评估" → [风险因子识别, 历史相似事件, 压力测试]
   '''

2. quantmind/agents/data_agent.py
   DataAgent：
   - 接受Planner派发的data任务
   - 调用data_tools取数
   - 输出DataSnapshot到state
   - 失败重试逻辑

3. quantmind/agents/fundamental_agent.py
   FundamentalAgent：

   核心职责：基于真实财务数据做深度基本面分析
   注意：所有定量计算（财务比率、DCF）必须用Python函数算，LLM只负责解读

   工作流：
   1. 从state读取财务数据
   2. 调用analysis_tools计算所有比率、DCF、可比
   3. LLM基于计算结果生成解读：
      - 盈利能力分析
      - 偿债能力分析
      - 成长性分析
      - 估值合理性分析（DCF结果vs当前市值）
      - 关键风险点

   System Prompt要点：
   - 必须引用具体数字
   - 不能算数（算数交给Python）
   - 必须对比同业
   - 必须区分短期波动vs长期趋势
   - 输出结构化FundamentalAnalysis对象

4. quantmind/agents/technical_agent.py
   TechnicalAgent：

   核心职责：技术面分析+量化信号整合

   工作流：
   1. 从state读取价格数据
   2. 计算技术指标（MA、MACD、RSI、Bollinger）
   3. 识别技术形态（突破、反转、整理）
   4. 调用quant_tools获取量化模型信号
   5. LLM综合输出技术观点

5. quantmind/agents/sentiment_agent.py
   SentimentAgent：

   核心职责：情绪面分析（新闻、研报、社交媒体）

   工作流：
   1. 调用kb_tools检索研报
   2. 调用data_tools取最近新闻
   3. 用LLM从新闻提取重要事件、情绪倾向
   4. 综合输出SentimentAnalysis

6. quantmind/agents/critic_agent.py
   CriticAgent（Self-Reflection的核心）：

   核心职责：审查前面所有Agent的输出，检测问题

   System Prompt：

   '''
   你是严苛的投资研究主管，审查下属递交的研究报告。

   审查维度：
   1. 数据完整性：关键数据是否齐全？最新季报有没有？
   2. 推理一致性：各部分结论是否互相矛盾？
   3. 数字准确性：财务计算是否合理？（不要重新算，看是否在合理范围）
   4. 风险覆盖：是否充分讨论了下行风险？
   5. 论证严谨性：观点是否有具体证据支撑？

   输出严格JSON：
   {
     "passed": true/false,
     "issues": [
       {
         "severity": "critical|major|minor",
         "type": "data_missing|inconsistency|calculation_error|risk_missing|weak_argument",
         "description": "...",
         "fix_action": {
           "agent": "data|fundamental|technical|sentiment",
           "instruction": "..."
         }
       }
     ],
     "overall_quality_score": 1-10,
     "approval_message": "..."
   }

   严格标准：
   - critical issue任何一个 → passed=false
   - major issues >= 3 → passed=false
   - 否则 passed=true
   '''

   迭代控制：
   - max_iterations=3
   - 每次迭代必须减少issue数量，否则强制终止

7. quantmind/agents/report_agent.py
   ReportAgent：

   核心职责：把所有Agent的输出整合成最终报告

   输出格式：
   - Markdown文本（可直接渲染）
   - 章节：
     1. 执行摘要（3-5要点）
     2. 公司概况
     3. 财务分析（含图表）
     4. 估值分析（DCF+Comparable）
     5. 技术面与量化信号
     6. 情绪面与催化剂
     7. 风险评估
     8. 投资建议（评级+目标价+持有期）
     9. 附录（数据来源、方法论）

   - 自动生成图表：
     * 股价走势+均线
     * 财务比率vs同业
     * DCF敏感性分析
     * 因子贡献雷达图

请生成全部7个Agent的完整代码，每个Agent都要有完整的System Prompt和unit test。
```

### Task 4.3 LangGraph编排

**Claude Code Prompt：**

```
请实现quantmind/agents/orchestrator.py，用LangGraph编排所有Agent：

1. 状态机设计：

   START
     │
     ▼
   ┌─────────┐
   │ Planner │  生成DAG
   └────┬────┘
        │
        ▼
   ┌─────────────────────────────┐
   │ Task Dispatcher (动态路由)   │
   └────┬────────┬────────┬──────┘
        │        │        │
        ▼        ▼        ▼
    Data Agt  Fund Agt  Tech Agt  ... (并行)
        │        │        │
        └────────┴────────┘
                 │
                 ▼
            ┌─────────┐
            │ Critic  │
            └────┬────┘
                 │
        ┌────────┴────────┐
        │                 │
   passed=true       passed=false
        │                 │
        ▼                 ▼
   ┌─────────┐     回到对应Agent
   │ Report  │     iteration_count++
   └────┬────┘     若>3则强制结束
        │
        ▼
       END

2. 实现要点：

   - 使用StateGraph(AgentState)
   - add_node添加所有Agent
   - add_edge / add_conditional_edges连接
   - Conditional逻辑：
     * task_dispatcher根据state["plan"]中的pending tasks决定下一步
     * critic_router根据state["critic_passed"]决定是否进入report
   - 并行执行：
     * data任务可以并行（不同股票分别取）
     * fundamental/technical/sentiment可以并行（同一股票不同维度）
     * 用LangGraph的Send API实现fan-out

3. Checkpointing：
   - 用LangGraph的SqliteSaver保存中间状态
   - 支持中断恢复（用户中途退出可重启）
   - 支持回放（debug时重放某次执行）

4. Streaming：
   - 实现stream模式：每个Agent完成立即输出
   - 用于Streamlit UI实时展示

5. 实现QuantMindOrchestrator类：
   - __init__(config)：构建graph
   - run(query, as_of=None) -> InvestmentReport（同步）
   - astream(query, as_of=None) -> AsyncIterator（流式）
   - run_with_checkpoint(query, thread_id) -> InvestmentReport（断点续传）

6. 测试用例：
   - 单股票分析：宁德时代（300750.SZ）
   - 对比分析：宁德时代 vs 比亚迪
   - 行业研究：新能源车行业
   - 风险事件：某只股票暴跌后的归因

请生成完整orchestrator代码+测试用例+使用文档。
```

---

## 7. 知识库与RAG（Phase 5：金融知识增强）

### Task 5.1 知识库构建

**Claude Code Prompt：**

```
请实现quantmind/kb/下的知识库系统：

1. quantmind/kb/builder.py
   KBBuilder类：

   功能：从多源构建按时间索引的金融知识库

   数据源：
   - 上市公司年报/季报（PDF文本，akshare的stock_yjbb_em可获取）
   - 券商研报（雪球公开研报+东方财富研报）
   - 新闻（财联社、东方财富、雪球热门讨论）
   - 监管公告（巨潮资讯、SEC EDGAR）
   - 行业报告（艾瑞、易观等公开报告）

   关键设计：
   - 每篇文档必须有published_date字段（PIT约束）
   - 文档级metadata：{ticker, doc_type, source, published_date, fiscal_period}
   - chunk级metadata继承文档+chunk_index

2. quantmind/kb/chunker.py
   智能分块：
   - 普通文本：RecursiveCharacterTextSplitter，chunk_size=1000，overlap=200
   - PDF（年报）：按章节分（用unstructured库识别章节标题）
   - 研报：保留段落完整性
   - 表格：保留完整不切分（pandas DataFrame转Markdown）
   - 中文优化：用jieba做边界检测

3. quantmind/kb/retriever.py
   HybridRetriever类（混合检索）：

   - Dense检索：用BGE-M3 embedding + ChromaDB
   - Sparse检索：BM25（rank_bm25库）
   - Reranker：BGE-Reranker-v2 cross-encoder做最终排序

   关键：PIT过滤
   - search(query, as_of, top_k=10, filters=None)
   - 在检索后强制过滤published_date <= as_of
   - 这是防止look-ahead bias的最后一道防线

   支持复合查询：
   - 多条件AND/OR：ticker AND date_range AND keyword
   - Metadata过滤：{ticker: "300750.SZ", doc_type: "annual_report"}

4. quantmind/kb/embedding_service.py
   EmbeddingService类：
   - 加载BGE-M3（多语言、多粒度、混合检索三合一）
   - 支持GPU推理
   - 批量embed
   - 缓存（同样文本不重复embed）

5. scripts/build_kb.py
   命令行：
   python scripts/build_kb.py --tickers 300750.SZ,002594.SZ --doc-types annual_report,research_report --start 2018-01-01

   功能：
   - 拉取数据→清洗→chunk→embed→入ChromaDB
   - 进度条
   - 支持增量更新（只embed新文档）
   - 输出统计：文档数、chunk数、平均长度、embedding维度

请生成全部文件代码。
```

### Task 5.2 RAG质量评估

**Claude Code Prompt：**

```
请实现quantmind/kb/evaluation.py，评估RAG质量：

参考RAGAS框架，实现以下指标：

1. Retrieval Quality
   - Hit Rate@K
   - MRR (Mean Reciprocal Rank)
   - NDCG@K
   - 需要构建golden test set：50-100个query+正确文档对

2. Answer Quality
   - Faithfulness：答案是否忠实于检索文档
   - Answer Relevance：答案是否回答了问题
   - Context Precision：检索到的内容是否都相关
   - Context Recall：相关内容是否都检索到

3. PIT Correctness（核心，独有）
   - 验证检索结果中没有as_of之后的文档
   - 验证测试：用历史日期查询，结果不能含未来信息

4. 评测脚本scripts/evaluate_kb.py：
   - 加载golden set
   - 跑全套评估
   - 生成可视化报告

请生成完整代码+示例golden set（10条）。
```

---

## 8. 回测引擎（Phase 6：严格量化评测）

> **本Phase核心**：这是项目区别于普通Agent demo的最大差异化。量化研究的可信度80%来自回测的严格性。

### Task 6.1 回测引擎核心

**Claude Code Prompt：**

```
请实现quantmind/backtest/下的工业级回测引擎：

1. quantmind/backtest/engine.py
   BacktestEngine类：

   核心：事件驱动回测，严格的时间线模拟

   - run(strategy, start_date, end_date, universe, **kwargs) -> BacktestResult

   策略接口Strategy基类：
   - on_market_open(date, snapshot) -> List[Order]
   - on_market_close(date, snapshot) -> None
   - on_corporate_action(ticker, action) -> None
   - on_signal(date, signals) -> List[Order]

   关键功能：
   - 严格PIT：每个decision time只能用<=该时刻的数据
   - 撮合模拟：开盘价/收盘价/VWAP多种成交方式
   - 涨跌停限制（A股专属）
   - 停牌处理
   - T+1限制（A股）
   - 双向交易（支持做空，融券限制）

2. quantmind/backtest/portfolio.py
   Portfolio类：

   - positions: Dict[ticker, Position]
   - cash, total_value, mv_history
   - update(date, prices, orders) -> None
   - rebalance_to_weights(target_weights, prices) -> List[Order]
   - get_returns() -> pd.Series
   - get_turnover() -> float

   Position：
   - ticker, shares, avg_cost, market_value, unrealized_pnl, weight

3. quantmind/backtest/execution.py
   ExecutionSimulator类：

   - simulate_order(order, market_state) -> Fill
   - 成交价模型：
     * SimpleOpenPrice（用开盘价成交）
     * VWAPSlippage（VWAP+滑点）
     * TWAPSlippage（按时间分段）
     * MarketImpact（基于Almgren-Chriss模型）
   - 交易成本：
     * 佣金（默认万3）
     * 印花税（卖出千1，A股）
     * 过户费（沪市万0.2）
     * 滑点（默认10bp）
   - 成交量约束：
     * 单笔不超过当日成交额的5%
     * 涨跌停不能成交

4. quantmind/backtest/metrics.py
   PerformanceMetrics类：

   收益指标：
   - total_return, annualized_return, cagr
   - 月度/年度收益分布

   风险指标：
   - volatility (annualized)
   - max_drawdown, drawdown_duration
   - VaR_95, CVaR_95
   - downside_deviation, sortino_ratio

   风险调整收益：
   - sharpe_ratio (rf=2.5%)
   - sortino_ratio
   - calmar_ratio
   - information_ratio (vs benchmark)

   交易指标：
   - turnover_rate (annual)
   - win_rate, profit_loss_ratio
   - avg_holding_period
   - hit_rate by holding_period

   高阶指标：
   - alpha, beta (vs CSI300)
   - Jensen's alpha, Treynor ratio
   - Fama-French 3因子/5因子归因
   - tail_ratio, skew, kurtosis

5. quantmind/backtest/walk_forward.py
   WalkForwardValidator：

   核心：避免数据窥探

   - 实现Rolling Window CV：
     * 训练窗口：3年
     * 验证窗口：6个月（参数选择）
     * 测试窗口：3个月（OOS评估）
     * 滚动步长：3个月
   - 每个window独立训练独立测试
   - 输出每个window的OOS指标
   - 计算指标的稳定性（mean、std、IR）

请生成全部5个文件的完整代码，每个指标都附经济含义注释。
```

### Task 6.2 统计显著性检验

**Claude Code Prompt：**

```
请实现quantmind/backtest/statistical_tests.py：

这是区别于业余项目的关键。简单的Sharpe数字没用，必须有统计检验。

实现以下检验：

1. Deflated Sharpe Ratio (DSR)
   参考：Bailey & López de Prado 2014
   修正多次试验的data snooping bias
   - compute_dsr(strategies_sharpe_list, candidate_sharpe) -> p_value

2. White's Reality Check / SPA Test
   多策略比较中的最优策略是否真的优于benchmark
   - white_reality_check(returns_matrix, benchmark) -> p_value

3. Bootstrap Confidence Interval
   - 对Sharpe、CAGR、MaxDD做bootstrap CI
   - 用block bootstrap（保留时序相关性）

4. Probabilistic Sharpe Ratio (PSR)
   计算Sharpe>某阈值的概率，考虑skew和kurtosis

5. Hsu's Test
   时间序列特异性检验

6. T-test for IC
   - 因子IC是否显著>0
   - Newey-West修正异方差和自相关

实现要求：
- 每个函数都有详细数学推导注释（引用论文）
- 输出p-value、置信区间、解释文本
- 提供visualize函数生成检验过程图

测试：
- 对随机噪声策略（应该不显著）
- 对真实有效策略（应该显著）
- 对人工构造的过拟合策略（DSR应该detect出来）

请生成完整代码。
```

### Task 6.3 Agent决策回测（项目核武器）

**Claude Code Prompt：**

```
请实现quantmind/backtest/agent_backtest.py，这是面试时的核武器：

核心思想：
让Agent对历史每个时点的"当时数据"做投资建议，然后用真实未来数据验证Agent的alpha。
99%的Agent项目没有这个，这是质的飞跃。

1. AgentBacktester类：

   核心方法：
   run(
     orchestrator: QuantMindOrchestrator,  # Agent系统
     query_template: str,  # 如"分析{ticker}未来60天投资价值"
     tickers: List[str],
     start_date, end_date,
     freq: str = "M",  # 多久让Agent判断一次
     holding_days: int = 60,
   ) -> AgentBacktestResult

   流程：
   for each rebalance_date in dates:
     for each ticker in tickers:
       # 关键：as_of=rebalance_date，Agent严格只能看历史数据
       advice = orchestrator.run(
         query=query_template.format(ticker=ticker),
         as_of=rebalance_date  # 严格PIT
       )
       # 等待holding_days，记录真实表现
       actual_return = data_provider.get_return(
         ticker, rebalance_date, rebalance_date + holding_days
       )
       results.append({
         "date": rebalance_date,
         "ticker": ticker,
         "agent_rating": advice.rating,  # buy/hold/sell
         "agent_target_price": advice.target_price,
         "agent_confidence": advice.confidence,
         "agent_thesis": advice.thesis,
         "actual_return": actual_return,
         "actual_max_dd": ...,
         "agent_predicted_correctly": ...
       })

2. 评估指标：

   分类准确率：
   - 按rating（buy/hold/sell）分组
   - 计算每组的平均后续收益
   - buy组应显著>0，sell组应显著<0

   Top-K Hit Rate：
   - Agent confidence Top-K股票，60天上涨比例

   Calibration：
   - Agent宣称70%概率上涨的股票，实际上涨比例是多少？
   - Reliability Diagram

   归因分析：
   - 哪些类型的股票Agent判断准？（大盘股/小盘股、价值/成长）
   - 哪些市场环境下Agent判断准？（牛/熊/震荡）

   错误模式：
   - 错误判断的case集中分析
   - 是否存在系统性偏差（如过度乐观、忽略宏观）

3. 与Baseline对比：

   - Baseline 1: 随机推荐
   - Baseline 2: 单LLM ReAct（无multi-agent）
   - Baseline 3: 纯LightGBM量化模型
   - Baseline 4: 卖方分析师一致预期

   QuantMind Agent vs 这些Baseline，做配对t-test。

4. 可视化报告：

   - rating × actual_return 箱线图
   - Calibration曲线
   - 累计胜率随时间变化
   - 错误案例展示（最大10个）
   - Top-K回测净值曲线vs benchmark

5. 输出：
   - HTML report
   - 关键数字写入README

实现要点：
- 严格用snapshot数据，禁止意外的look-ahead
- 用checkpoint支持断点续传（一次回测可能跑几小时）
- 多线程并行（不同股票不同日期独立）
- 成本控制：每次Agent运行预估token，总成本可控

请生成完整代码+一个端到端示例（10个股票×12个月回测）。
```

---

## 9. 风险与组合管理（Phase 7：实盘可用度提升）

### Task 7.1 风险管理

**Claude Code Prompt：**

```
请实现quantmind/risk/下的模块：

1. quantmind/risk/factor_risk.py
   FactorRiskModel类（基于Barra多因子风险模型简化版）：

   - estimate_factor_returns(returns, exposures) -> DataFrame
   - estimate_specific_risk(returns, factor_returns, exposures) -> DataFrame
   - portfolio_risk(weights, factor_cov, specific_var) -> Dict
   - factor_attribution(returns, exposures) -> Dict

   实现风险因子：
   - 行业（28个申万一级）
   - 风格（10个：Beta、Momentum、Size、EarningsYield、Volatility、
           Growth、Value、Leverage、Liquidity、NonLinearSize）

2. quantmind/risk/position_sizing.py

   PositionSizer类（多种仓位管理方法）：
   - equal_weight(tickers) -> Dict
   - market_cap_weight(tickers, mcaps) -> Dict
   - inverse_volatility(returns) -> Dict
   - minimum_variance(cov) -> Dict
   - max_diversification(cov) -> Dict
   - risk_parity(cov) -> Dict
   - kelly_criterion(expected_returns, cov) -> Dict
   - hierarchical_risk_parity(returns) -> Dict（HRP，参考López de Prado）

   每个方法都有详细论文引用。

3. quantmind/risk/drawdown.py
   DrawdownController类（动态回撤控制）：
   - 当回撤>X%时降低杠杆
   - 当回撤>Y%时清仓
   - Volatility Targeting：保持组合年化波动率=10%
   - CPPI（Constant Proportion Portfolio Insurance）

4. 集成到回测：
   - 修改Strategy基类，支持插入risk filter
   - 回测中实时风险监控

请生成完整代码。
```

---

## 10. UI层（Phase 8：可演示前端）

### Task 8.1 Streamlit应用

**Claude Code Prompt：**

```
请实现quantmind/ui/streamlit_app.py，构建一个真正能演示的UI：

页面结构（多页面应用）：

1. 首页 / Overview
   - 项目介绍、架构图
   - 最新回测结果概览
   - 项目里程碑

2. 个股深度研究 / Single Stock Research
   - 股票代码输入框（autocomplete）
   - as_of日期选择器（默认今天，可选历史日期做PIT演示）
   - 分析维度多选：财务/技术/情绪/估值
   - "开始分析"按钮
   - 流式展示Agent执行过程（每个Agent的输入/输出/状态）
   - 最终报告渲染（Markdown+图表）
   - 下载报告（PDF/Markdown）

3. 多股票对比 / Compare Stocks
   - 输入多个ticker（最多5个）
   - 并排展示关键指标
   - 雷达图对比
   - LLM生成对比分析

4. 量化策略回测 / Strategy Backtest
   - 选择策略：基础多因子/LightGBM/LLM Rerank/DPO Model
   - 选择universe + 时间段
   - "运行回测"
   - 实时进度条
   - 结果可视化：净值曲线、回撤曲线、月度热力图、因子归因
   - 与benchmark对比
   - 下载详细报告

5. Agent决策回测 / Agent Backtest
   - 加载预先跑好的Agent回测结果
   - 互动式探索：filter by rating/confidence/sector
   - 错误案例deep dive
   - Calibration可视化

6. 知识库 / Knowledge Base
   - 浏览已索引文档（按ticker、日期）
   - 测试检索（输入query+as_of，看返回结果）
   - 添加新文档（拖拽上传PDF）

7. 模型管理 / Models
   - 查看所有训练好的模型
   - 模型详情：训练时间、参数、性能
   - SHAP可解释性可视化

实现要点：
- 用st.session_state管理状态
- 长任务用st.spinner+进度条
- 流式输出用st.write_stream
- 图表用plotly（可交互）+ matplotlib（静态）
- 缓存用@st.cache_data和@st.cache_resource
- 多页面用pages/目录
- 主题：使用深色专业主题（金融风）
- 错误处理：友好的error message

部署：
- 本地：streamlit run quantmind/ui/streamlit_app.py
- 云端：可部署到streamlit cloud（免费）或自建服务器
- Docker：提供Dockerfile

请生成全部代码（主入口+各pages）。
```

---

## 11. 文档与博客（Phase 9：影响力建设）

### Task 9.1 完整README

**Claude Code Prompt：**

```
请生成完整的README.md，包含以下sections：

1. Header
   - Banner image（占位）
   - Badges：License, Python version, Tests, Coverage, Stars
   - 一句话核心定位

2. TL;DR
   3-5句话讲清楚项目做什么、和别人不一样在哪

3. Demo
   - 屏幕录制GIF/视频链接
   - 在线Demo URL（如有）
   - 关键截图3-5张

4. Key Results
   - 表格展示核心指标
   - 与baseline对比

5. Architecture
   - Mermaid或Excalidraw架构图
   - 三大子系统介绍

6. Quick Start
   - Prerequisites（Python版本、CUDA、API Keys）
   - Installation（用uv或pip）
   - Configuration（.env配置）
   - Running examples：
     * 跑Agent单股票分析
     * 跑量化策略回测
     * 启动UI
   - Run tests

7. Project Structure
   - 树形结构说明

8. Methodology
   - 核心方法论简述
   - 链接到docs/详细文档

9. Reproducibility
   - 数据来源
   - 随机种子
   - 环境锁定（poetry.lock或requirements.txt）

10. Roadmap
    - Done items（已完成）
    - In Progress
    - Future Work

11. Citations
    - BibTeX占位（投paper后填）
    - 主要参考文献列表

12. Contributing
    - 简单的贡献指南

13. License
    - MIT或Apache-2.0

14. Acknowledgements
    - 致谢使用的开源项目

15. Contact
    - 你的联系方式

要求：
- 中英文双语（英文主，中文章节标注）
- 所有代码块要可直接复制运行
- 用emoji但不滥用
- 关键数字加粗
- 长度控制在让人愿意读完（3000-5000词）

请生成完整README。
```

### Task 9.2 技术博客系列

**Claude Code Prompt：**

```
请帮我撰写3篇技术博客，发到知乎/小红书/Medium：

博客1：《我用LangGraph做了一个AI投资分析Agent，但它告诉我Agent都在讲故事》

主题：Critic Agent + Self-Reflection的工程实践
重点：
- 讲清楚为什么单Agent ReAct在复杂金融分析里会胡说
- Critic Agent如何识别幻觉（具体case）
- Self-Reflection的工程坑（无限循环、迭代退化）
- 真实case复盘：宁德时代分析迭代3次的演化

博客2：《把生成式推荐用到量化选股：LLM Listwise Rerank的真实回测结果》

主题：LLM Rerank在量化的应用
重点：
- 介绍传统多因子模型的瓶颈
- Listwise Rerank的核心思想（vs pointwise）
- 工程实现：候选生成→特征卡片构造→LLM Rerank
- DPO对齐让小模型超越API
- 真实回测数据：IC、Sharpe、超额收益
- 失败案例：什么时候LLM Rerank会变差

博客3：《Point-in-Time数据是量化研究的玻璃心：踩过的8个look-ahead bias坑》

主题：PIT数据正确性
重点：
- 财报披露日 vs 报告期：最容易翻车的地方
- Survivorship Bias：用沪深300成分股就能"高预测准确率"
- Future Information泄漏到Embedding：RAG的隐藏坑
- 财务调整重述：业绩预告、业绩快报、正式报告的时间线
- 复权数据的细节
- 节假日、停牌、退市的边界条件
- 我的测试套件如何捕获这些bug
- 一个完整的PIT validator工具开源

要求每篇：
- 3000-5000字
- 至少5张图（架构图、代码截图、回测结果）
- 真实代码段（可copy）
- 清晰的故事线（提出问题→分析→方案→验证）
- 文末导流到GitHub仓库

请按这个结构生成3篇博客的完整内容（每篇可分多次输出）。
```

---

## 12. 测试与CI（Phase 10：工程质量）

### Task 10.1 完整测试套件

**Claude Code Prompt：**

```
请补全tests/下所有测试，目标是测试覆盖率>70%：

1. tests/test_data_providers.py
   - 各provider的接口一致性
   - PIT约束测试
   - Mock测试（不依赖真实API）

2. tests/test_features.py
   - 每个因子的正确性（用手工算的小样本验证）
   - 因子的数学性质（如momentum的反转）
   - 缺失值处理

3. tests/test_models.py
   - LightGBM训练能跑通
   - 预测shape正确
   - LLM Reranker输出schema validation

4. tests/test_agents.py
   - 每个Agent的input/output合约
   - Agent串联的integration test
   - Critic的issue识别能力

5. tests/test_backtest.py
   - 简单策略（buy and hold）的回测正确性
   - 多策略并行的隔离性
   - 各项指标的边界case

6. tests/test_pit_correctness.py（已实现）

7. tests/test_kb.py
   - Chunker的边界
   - 检索的PIT过滤
   - 中英文混合

8. tests/conftest.py
   - 共享fixtures（mock data、temp dir、test config）

CI配置.github/workflows/test.yml：
- Python 3.11 / 3.12 矩阵
- Ubuntu / MacOS
- 运行ruff + mypy + pytest
- 上传coverage到codecov

请生成所有测试文件+CI配置。
```

---

## 13. 执行顺序与里程碑

### 推荐执行顺序

按以下顺序逐个让Claude Code执行Task：

```
Week 1（基础建设）：
  Day 1-2: Task 0.1-0.2（环境与core）
  Day 3-4: Task 1.1-1.3（数据层）
  Day 5: Task 1.2 PIT测试（这是关键）
  Day 6-7: Task 2.1（特征工程）

Week 2（量化模型）：
  Day 8-9: Task 2.2 因子分析
  Day 10-11: Task 3.1 LightGBM
  Day 12-13: Task 3.2 LLM Reranker
  Day 14: 测试+调优

Week 3（Agent系统）：
  Day 15-16: Task 4.1-4.2 Agent实现
  Day 17-18: Task 4.3 LangGraph编排
  Day 19-20: Task 5.1-5.2 知识库

Week 4（回测与评估）：
  Day 21-22: Task 6.1 回测引擎
  Day 23: Task 6.2 统计检验
  Day 24-25: Task 6.3 Agent回测（核武器）
  Day 26-27: Task 7.1 风险管理

Week 5（前端与文档）：
  Day 28-29: Task 8.1 Streamlit UI
  Day 30: Task 9.1 README
  Day 31-33: Task 9.2 三篇博客
  Day 34: Task 10.1 测试与CI

Week 6（高阶可选）：
  - Task 3.3 DPO训练（最难，最值钱）
  - 部署到云
  - 投Workshop
```

### 关键里程碑Checklist

**Milestone 1：数据基石**（Week 1结束）
- [ ] 沪深300所有成分股2018-2024的全数据可加载
- [ ] PIT测试全部通过
- [ ] 任意历史日期的snapshot可重现

**Milestone 2：因子库**（Week 2结束）
- [ ] 50+因子可计算
- [ ] 因子分析报告生成
- [ ] LightGBM模型在OOS有正Sharpe

**Milestone 3：Agent系统**（Week 3结束）
- [ ] 6-Agent协作可端到端跑通
- [ ] Critic能识别>80%人工注入的错误
- [ ] 单股票分析<5分钟完成

**Milestone 4：回测**（Week 4结束）
- [ ] 至少3个baseline策略完成回测
- [ ] LLM Rerank vs LightGBM有显著IC提升
- [ ] Agent回测100+样本完成

**Milestone 5：演示**（Week 5结束）
- [ ] Streamlit UI部署可访问
- [ ] README完整
- [ ] 至少1篇技术博客发布
- [ ] GitHub Stars > 30

---

## 14. 给Claude Code执行时的工程纪律

### 14.1 每次Task开始前的自检Prompt

```
开始本Task前，请：
1. 查看当前项目结构（运行tree -L 3）
2. 检查依赖是否安装（如pyproject.toml里有相关包）
3. 确认前置Task是否完成
4. 列出本Task将创建/修改的文件
5. 说明本Task完成后如何验证

如果发现前置不满足，立刻指出并暂停。
```

### 14.2 每次Task完成后的自检

```
本Task完成，请运行以下验证：
1. ruff check：代码规范
2. mypy：类型检查
3. pytest tests/test_当前模块.py：单元测试
4. 实际运行一个smoke test
5. git diff统计：新增/修改的行数

输出验证结果摘要。
```

### 14.3 工程纪律红线

让Claude Code严格遵守：

1. **不准mock数据冒充真实**：所有声称"已实现"的功能必须真能跑出真实数据，mock只在标明了TODO或测试中允许
2. **不准跳过PIT检查**：任何涉及历史数据的代码，必须有as_of参数和断言
3. **不准忽略错误**：try/except必须有具体处理，不能bare except: pass
4. **不准硬编码路径**：用configs和环境变量
5. **不准复制粘贴大段代码**：相同逻辑必须抽函数
6. **每个公开函数必须有docstring + type hint**
7. **关键路径必须有日志**：数据下载、模型训练、回测、Agent调用
8. **代码风格统一**：用ruff + black强制
9. **PR级别的commit**：每个Task一个commit，message清晰

### 14.4 性能优化提示

针对RTX 5060 Ti 16GB约束：

- LLM训练：QLoRA 4bit + gradient checkpointing
- LLM推理：vLLM + AWQ量化
- Embedding：批量处理，BGE-M3约2GB显存
- 因子计算：用polars替代pandas（快3-5倍）
- 回测：用numba JIT关键循环
- 数据：parquet替代csv（小10倍，快20倍）
- 缓存：积极使用@cached，避免重复LLM调用

---

## 15. 简历与面试呈现

### 15.1 简历版本（一段话精华版）

```
QuantMind: Agent-Driven Quantitative Investment Research System          [Personal Project]

Designed and built an end-to-end multi-agent investment research system integrating
LLM-driven qualitative analysis with rigorous quantitative backtesting on Chinese A-share
and US markets.

• Architected 6-agent collaborative pipeline (Planner/Data/Fundamental/Technical/
  Sentiment/Critic) with LangGraph state machine and Self-Reflection loop, achieving
  85% issue detection rate vs single-agent baseline
• Developed "Generative Quant Selection" framework: LightGBM factor model for coarse
  ranking + LLM Listwise Reranker with reasoning + DPO-aligned Qwen3-4B for inference;
  on CSI300 OOS backtest 2022-2024 achieved IC 0.0X (vs LGBM-only 0.0Y), Sharpe X.XX
• Engineered industrial-grade backtest infrastructure with strict Point-in-Time data
  isolation, Walk-Forward validation, and Deflated Sharpe Ratio for multiple-testing
  correction; identified and fixed 8 categories of look-ahead bias in pipeline
• Built hybrid RAG knowledge base over 5000+ research reports, annual reports and news
  with BGE-M3 embedding + BM25 + cross-encoder reranking, with PIT-strict retrieval
• Tech: Python, LangGraph, PyTorch, LightGBM, Transformers/PEFT/TRL, Polars, vLLM,
  ChromaDB, Streamlit
• Open-source: github.com/<your-handle>/quantmind  | XX stars | 3 technical blogs (15K+ reads)
```

### 15.2 高频面试问题与防御

**Q1: 你的Agent怎么避免幻觉？**
答：三层防御。第一层，所有数字必须来自工具调用，禁止LLM自己算财务比率，Python严格计算。第二层，Critic Agent专门审查推理一致性和数字grounding。第三层，最终在backtest上验证Agent建议的真实alpha——如果Agent在讲故事，回测会暴露。

**Q2: Critic Agent怎么避免无限循环？**
答：三个约束。一是max_iterations=3硬上限。二是每轮迭代必须减少issue数量，否则强制exit。三是每个issue有severity分级，只有critical+major合计达到阈值才触发回流。还做了compaction：迭代3次后未通过的，标记部分通过+列出已知遗留问题，不阻塞最终交付。

**Q3: 你的回测有没有look-ahead bias？**
答：这是我重点防御的。具体做了：(1) 财报数据用披露日不用报告期；(2) Universe用历史成分股而非当前；(3) 知识库检索强制as_of过滤；(4) 写了专门的test_pit_correctness.py十几个测试case；(5) 还做了一个对照实验——故意引入look-ahead后Sharpe从X.XX涨到Y.YY，验证测试套件能检测出来。

**Q4: LLM Listwise Rerank的IC提升有统计意义吗？**
答：做了三个检验。(1) Newey-West修正的IC t-test；(2) Deflated Sharpe Ratio，控制了我尝试过的多个模型变体；(3) Block Bootstrap CI，95% CI不含0。p值<0.05。同时做了stratified分析，confirmed提升主要来自小盘股（信息效率低，LLM补充alpha更明显）。

**Q5: 你的DPO训练数据怎么来？**
答：三类。(1) 自动标注：基于真实60天后股价表现给历史reasoning打chosen/rejected；(2) 规则合成：chosen有具体数字+多维度+风险讨论，rejected空泛；(3) 人工精标：100对最关键case自己标。总计约8000对。我做了消融：只用自动标注效果最差，加上规则合成最好，少量人工精标主要提升尾部case。

**Q6: 你的Agent比单纯调GPT-4 API好在哪？**
答：(1) 数据grounding：通过工具调用接真实数据，GPT-4直接调没有；(2) 推理可验证：Critic可以追溯每个结论到具体数据源；(3) 成本可控：DPO模型本地推理比API便宜10倍；(4) 可定制：我对中文金融场景做了专门优化，包括A股特有的T+1、涨跌停、披露规则；(5) 可回测：完整pipeline支持PIT回测，验证真实alpha。

**Q7: 项目的局限性？**
答：诚实回答。(1) 数据深度：研报覆盖只有公开部分，没买卖方研究终端，深度差；(2) 高频信号缺失：日级别factor模型，无法捕捉intraday alpha；(3) 中型样本：CSI300约300只，OOS两年样本，统计功效有限；(4) Agent成本：单股完整分析约$0.5-1，规模化用需要进一步优化；(5) 没考虑实盘交易摩擦的全部细节，如借券难度、大单冲击。下一步会重点解决前两个。

---

## 附录A：常用Claude Code Prompt模板

### A.1 重构请求模板

```
请重构 {file_path}，要求：
1. 保持外部接口不变（其他模块不需要修改）
2. 拆分超过200行的函数
3. 添加缺失的type hints和docstring
4. 改进错误处理（具体异常而非bare except）
5. 添加日志
6. 用ruff+mypy验证
7. 运行相关测试确保不破坏

输出：完整重构后代码 + 改动摘要 + 测试结果
```

### A.2 性能优化模板

```
请分析并优化 {file_path} 的性能：
1. 用cProfile定位瓶颈
2. 优先用vectorization > parallel > caching
3. 给出优化前后的benchmark对比
4. 不能改变API语义
5. 不能损失精度

输出：优化代码 + benchmark + 改动说明
```

### A.3 Bug修复模板

```
当前问题：{描述bug现象}

错误信息：
{完整traceback}

复现步骤：
{minimum reproducible example}

期望行为：
{描述应该是什么样}

请：
1. 分析根本原因（不是表面症状）
2. 给出最小修复
3. 添加回归测试防止再发
4. 检查是否其他地方有同类问题

输出：修复代码 + 测试 + 根因分析
```

### A.4 代码审查模板

```
请审查 {file_path}，从以下角度：
1. 正确性：逻辑bug、边界case
2. 鲁棒性：异常处理、资源清理
3. 性能：可优化点
4. 可读性：命名、结构、注释
5. 可测试性：依赖注入、mock友好
6. 安全性：注入、敏感信息
7. PIT合规：金融项目特殊关注

输出：分级问题清单（critical/major/minor）+ 具体修改建议
```

---

## 附录B：依赖详细清单

### pyproject.toml完整版

```toml
[project]
name = "quantmind"
version = "0.1.0"
description = "AI Agent-Driven Quantitative Investment Research System"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "you@example.com"}]

dependencies = [
    # Core
    "pandas>=2.2",
    "numpy>=1.26",
    "polars>=0.20",
    "pydantic>=2.5",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "loguru>=0.7",
    "tenacity>=8.2",
    "joblib>=1.3",
    "diskcache>=5.6",
    "rich>=13.0",
    "typer>=0.12",
    # Data
    "akshare>=1.12",
    "tushare>=1.4",
    "yfinance>=0.2.40",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "openpyxl>=3.1",
    "pyarrow>=14.0",
    "fastparquet>=2024.2",
    # ML
    "scikit-learn>=1.4",
    "lightgbm>=4.1",
    "xgboost>=2.0",
    "statsmodels>=0.14",
    "scipy>=1.12",
    "numba>=0.59",
    # DL
    "torch>=2.2",
    "transformers>=4.40",
    "accelerate>=0.30",
    "peft>=0.10",
    "trl>=0.9",
    "datasets>=2.18",
    "sentence-transformers>=2.7",
    "bitsandbytes>=0.43",
    # LLM
    "langchain>=0.2",
    "langgraph>=0.1",
    "langchain-openai>=0.1",
    "langchain-community>=0.2",
    "openai>=1.30",
    "anthropic>=0.25",
    "dashscope>=1.17",
    # RAG
    "chromadb>=0.5",
    "rank-bm25>=0.2",
    "FlagEmbedding>=1.2",
    "unstructured>=0.13",
    # Backtest
    "vectorbt>=0.26",
    "empyrical-reloaded>=0.5",
    "quantstats>=0.0.62",
    "alphalens-reloaded>=0.4",
    # Viz
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.20",
    "altair>=5.3",
    # UI
    "streamlit>=1.30",
    "gradio>=4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.23",
    "pytest-xdist>=3.5",
    "ruff>=0.4",
    "mypy>=1.10",
    "ipykernel>=6.29",
    "jupyter>=1.0",
    "pre-commit>=3.6",
]
serve = [
    "vllm>=0.4",
    "fastapi>=0.110",
    "uvicorn>=0.29",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=quantmind --cov-report=term-missing"
markers = [
    "slow: requires real API calls",
    "gpu: requires GPU",
]
```

---

## 附录C：项目最终交付清单

完成项目后，应有以下产出：

**代码层面**：
- [ ] GitHub仓库公开，README完整
- [ ] 测试覆盖率>70%
- [ ] CI绿色
- [ ] 至少5000行高质量代码（不含测试）
- [ ] 至少30+ stars

**文档层面**：
- [ ] 完整README
- [ ] docs/下5+技术文档
- [ ] 3篇技术博客（知乎+小红书+Medium）
- [ ] arXiv技术报告（可选但加分）

**演示层面**：
- [ ] 部署的Streamlit Demo（公网可访问）
- [ ] 5分钟演示视频
- [ ] 关键case study notebook

**研究层面**：
- [ ] 完整回测报告（HTML+PDF）
- [ ] 因子分析报告
- [ ] Agent能力评估报告
- [ ] 对外展示的关键数字

**简历层面**：
- [ ] 一段话精华版
- [ ] 详细版（适合不同公司）
- [ ] 项目portfolio页面（可选个人网站）

---

# 结语

这份Engineering Spec总长约15000字，覆盖了从环境搭建到论文级回测的全流程。每个Task都是Claude Code可以独立执行的粒度。

执行的关键纪律：
1. **严格按Phase顺序**：数据→特征→模型→Agent→回测，每一步是下一步的基石
2. **PIT是项目灵魂**：宁可慢也不能违反PIT
3. **Critic不是装饰**：真实使用，真实迭代
4. **回测是核武器**：所有claim必须有回测支撑
5. **开源+博客是放大器**：好项目要让世界看见

遇到任何执行卡点，参考附录A的Prompt模板与Claude Code对话。

Good luck. May your Sharpe be high and your drawdown shallow.
