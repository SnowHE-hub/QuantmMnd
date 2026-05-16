# Point-in-Time 数据是量化研究的玻璃心：踩过的 8 个 look-ahead bias 坑

> 作者：QuantMind 项目笔记 | 系列：AI Agent 量化研究实践

---

## 为什么 PIT 是量化的命门

先做一个简单的实验：

用"2023 年三季报 ROE"选股，买入 ROE 排名前 20% 的股票，卖出后 20%。

如果你用**报告期（2023-09-30）**作为数据截止日期，回测 Sharpe 大约是 1.2。

如果你用**实际披露日（三季报通常在 10-10 月底披露）**作为截止日期，Sharpe 降到约 0.85。

差距来自于：一些公司的三季报要到 10 月 25-31 日才真正披露，但你的策略在 10 月 1 日就已经"看到"这些数据并交易了。这 25-30 天的"未来泄漏"就是这 0.35 的 Sharpe 差距的来源。

**40% 的虚高 Sharpe**。这是最简单的一个坑，但现实中的坑远不止这一个。

QuantMind 的 `tests/test_pit_correctness.py` 收集了我在搭建这个系统时遇到的所有 look-ahead bias 坑，这篇文章讲 8 个最典型的。

---

## 8 个具体的坑

### 坑 1：财报报告期 vs 披露日（最常见）

**场景**：你想在 2024 年 1 月用 2023 年三季报数据选股。

**错误做法**：
```python
# 以报告期为截止日期——这是 99% 的量化教程里的写法
financials = fin_df[fin_df['end_date'] <= '2024-01-01']
```

**正确做法**：
```python
# 以实际公告日（f_ann_date）为截止日期
financials = fin_df[fin_df['f_ann_date'] <= '2024-01-01']
```

一般情况下，2023 年三季报（报告期 2023-09-30）的实际披露日（`f_ann_date`）在 2023 年 10 月。如果你在 2024 年 1 月回测，这没问题——两种方法结果一样。

但如果你在 2023 年 10 月 3 日（国庆节后第一个交易日）选股，就会发现：大约有 40% 的公司三季报在这天还没披露，它们的财务数据不应该被使用。

测试代码：
```python
def test_tushare_financials_f_ann_date_pit(statement: str) -> None:
    """断言所有财报行的 f_ann_date 严格 ≤ as_of."""
    p = TushareProvider()
    as_of = date(2024, 4, 1)  # 2023年报多在4月中下旬披露
    df = p.get_financials("300750.SZ", statement, as_of=as_of)
    assert df["f_ann_date"].max() <= pd.Timestamp(as_of), (
        f"{statement} has f_ann_date > {as_of}: {df['f_ann_date'].max()}"
    )
```

### 坑 2：Survivorship Bias（用当前成分股代替历史）

**场景**：你要回测 2019-2023 年对沪深 300 成分股的策略。

**错误做法**：直接用**今天**的沪深 300 成分股跑历史回测。

**问题**：今天的沪深 300 包含了 2019-2023 年表现好的公司（正因为表现好才留在指数里），排除了那些被剔除的差公司。这就是幸存者偏差——你只看到了活下来的样本。

**测试代码**：
```python
def test_universe_changes_over_time() -> None:
    """csi300 在 2020 vs 2024 应有显著差异（>=20只换仓）."""
    u_2020 = set(get_universe("csi300", as_of=date(2020, 6, 30)))
    u_2024 = set(get_universe("csi300", as_of=date(2024, 6, 30)))
    diff = u_2020.symmetric_difference(u_2024)
    assert len(diff) >= 40, (
        f"csi300 universe changed too little ({len(diff)}); survivorship bias suspect"
    )
    only_2020 = u_2020 - u_2024
    assert len(only_2020) >= 20, "几乎没有 2020 在 2024 退出的票"
```

正确做法是用 tushare 的 `index_weight` 接口，这个接口支持历史日期查询，返回**该日期当天**的实际成分股。

### 坑 3：业绩预告 → 业绩快报 → 正式报告的三个时间点

**场景**：某公司 2024 年 1 月 15 日发布业绩预告，说 2023 年净利润增长 30%。2 月 5 日发布业绩快报，给出具体数字。3 月 20 日发布正式年报。

这三个事件有三个不同的披露日期，对应三种不同的信息状态。

业绩预告的时间点很重要：很多量化研究者知道"用 f_ann_date 而不是 end_date"，但忘记了业绩预告是一个**独立事件**，有自己的披露日（`pre_date`），而且往往是最早能看到"财年利润信号"的时间点。

这层信息如果处理不好，在回测里要么"太早用了正式年报数字"，要么"错过了业绩预告这个催化剂"。

在 QuantMind 里的处理：`DataProvider.get_financials()` 返回三张报表时，字段 `f_ann_date` 统一映射到每条数据**真正可获得的最早时间点**——年报用正式披露日，有预告的优先用预告日。

### 坑 4：财务重述（公司修改历史数据）

**场景**：某公司 2023 年年报披露后，2024 年因会计准则调整，追溯重述了 2022 年的财务数据。如果你在 2024 年底访问"2022 年报数据"，可能拿到的是**修订后的版本**，而 2023 年初做投资决策时用的是原始版本。

这个问题特别隐蔽，因为大多数数据提供商的 API 只保留最新版本的财务数据。

tushare 的 `fina_indicator` 接口有一个相对可靠的处理：它会保留不同版本的记录（通过 `update_flag` 字段区分），你可以筛选出特定时间点的有效版本。

实际操作中，除非数据提供商明确支持版本追溯，否则这个坑很难完全避开，只能尽量使用靠近首次披露的数据（更新频率低的财务数据，重述风险相对小）。

### 坑 5：复权数据（前复权价格随时间变化）

**场景**：2020 年 1 月，某股票价格是 50 元。该股票在 2022 年 10 送 10（股本扩张一倍）。

如果你今天拉这只股票的"前复权"历史价格，2020 年 1 月的价格会显示为 **25 元**（因为复权系数变了）。

但在 2020 年 1 月做决策时，你看到的价格是 50 元。如果你的信号是"当前价格 vs 历史均价"，用今天的前复权数据计算 2020 年的"历史均价"，会得到一个和当时不同的数字，从而产生 look-ahead bias。

解决方案：存储的是**后复权价格**（上市以来一直累计复权），计算技术指标时用后复权；或者存储原始价格 + 复权系数，按需计算。QuantMind 使用后复权，并在 snapshot 里记录当时的复权系数，确保回测时计算的收益率和实际一致。

### 坑 6：RAG 知识库的隐藏泄漏（Embedding 索引含未来信息）

这是 QuantMind 独有的坑，传统量化系统不会遇到，但 AI Agent 系统必须处理。

**场景**：你构建了一个向量知识库，里面有 2019-2024 年的所有研报、新闻。当你在做 2022 年 3 月的历史回测时，Agent 调用 RAG 检索，但知识库没有做时间过滤，返回了 2023 年的分析师报告（里面可能包含"2022 年全年业绩已经公布，增长 X%"这样的信息）。

Agent 就看到了"未来"。

**正确实现**：

```python
def search(query: str, as_of: str, top_k: int = 5):
    candidates = vector_store.similarity_search(query, k=top_k * 3)
    # 严格过滤：只返回 as_of 时点之前发布的文档
    filtered = [
        doc for doc in candidates
        if doc.metadata["date"] <= as_of
    ]
    return filtered[:top_k]
```

每篇文档在入库时必须记录 `date` 字段（发布日期），检索时强制 `doc.date <= as_of`。

这在技术上不难，但容易在项目初期被忽视，因为"知识库泄漏"在单次查询时不明显，只在系统性回测时才会体现为虚高的 alpha。

### 坑 7：节假日 / 停牌的边界条件

**场景**：A股有涨跌停和停牌机制。如果某股票今天停牌（`volume=0`），但你的策略生成了买入信号，这个信号在回测里如何处理？

**错误做法**：允许在停牌日成交，实际上相当于以"复牌第一天的价格"成交，但用的是停牌前的信号，这是一种隐蔽的 look-ahead bias（你知道它会复牌，但真实情况中，复牌价可能有巨大跳空）。

**正确做法**：停牌日所有买卖操作全部跳过，顺延到复牌后。

```python
# quantmind/backtest/engine.py 中的核心检查
def _is_tradable(self, bar: dict) -> bool:
    """停牌（volume=0）返回 False."""
    return float(bar.get("volume", 0)) > 0
```

类似地，涨跌停也有边界问题：涨停板上，买入可能无法成交（没有卖家），这在回测里要明确处理，不能默认"信号发出就能成交"。

### 坑 8：指数成分股调整（每季度调整）

沪深 300 每年 6 月和 12 月进行成分股调整。调整公告在调整前大约 5 个交易日发出（调整预告），实际执行在月底。

**坑**：在回测中用"6月1日"的成分股跑 1-5 月的回测，但这批成分股是"6月调整后"的版本，已经用了调整预告的信息（调整预告是 5 月 25 日发出的），相当于在 5 月就知道了 6 月会调入哪些股票。

正确做法是在每个回测日期，精确使用**该日期有效的成分股版本**，而不是最近一次调整后的版本。tushare 的 `index_weight` 表里每次调整都有独立记录，可以精确还原历史。

---

## 我的测试套件如何捕获这些 Bug

`tests/test_pit_correctness.py` 包含 10 个专项测试，每个测试聚焦一个特定的 PIT 边界。

关键测试案例：

**财报披露日测试**：
```python
@pytest.mark.parametrize("statement", ["income", "balance_sheet", "cashflow"])
def test_tushare_financials_f_ann_date_pit(statement: str) -> None:
    """断言所有财报行的 f_ann_date 严格 ≤ as_of."""
    p = TushareProvider()
    as_of = date(2024, 4, 1)
    df = p.get_financials("300750.SZ", statement, as_of=as_of)
    assert "f_ann_date" in df.columns
    assert df["f_ann_date"].max() <= pd.Timestamp(as_of), (
        f"{statement} has f_ann_date > {as_of}: max={df['f_ann_date'].max()}"
    )
```

这个测试的核心逻辑：选 2024-04-01 作为 as_of，故意处于"2023 年报（报告期 2023-12-31）已经部分披露但未全部披露"的时间点。测试验证返回的数据里没有未来日期的记录。

**成分股变化测试（反幸存者偏差）**：
```python
def test_universe_changes_over_time() -> None:
    u_2020 = set(get_universe("csi300", as_of=date(2020, 6, 30)))
    u_2024 = set(get_universe("csi300", as_of=date(2024, 6, 30)))
    diff = u_2020.symmetric_difference(u_2024)
    # 4 年间至少有 40 只票发生了变化
    assert len(diff) >= 40
    # 2020 年在 2024 年退出的票至少有 20 只
    assert len(u_2020 - u_2024) >= 20
```

**PIT 边界精确性测试**（已知数据点验证）：
```python
def test_pit_cutoff_boundary() -> None:
    """2023 年报（f_ann_date ≈ 2024-04-25）在边界两侧表现正确."""
    p = TushareProvider()
    # 在报告披露前：看不到年报
    df_before = p.get_financials("600519.SH", "income", as_of=date(2024, 4, 20))
    # 在报告披露后：能看到年报
    df_after = p.get_financials("600519.SH", "income", as_of=date(2024, 5, 1))
    # 披露前看不到 2023 年报
    assert not any(
        df_before["end_date"] == pd.Timestamp("2023-12-31")
    )
    # 披露后能看到 2023 年报
    assert any(
        df_after["end_date"] == pd.Timestamp("2023-12-31")
    )
```

这个测试用已知的真实披露日期构造了一个"时间机器测试"：在披露前看不到，披露后能看到，精确验证了数据层的时间边界。

---

## 一个可复用的 PIT Validator 工具

构建了一个通用工具，可以对任意 DataFrame 做 PIT 合规性检查：

```python
def validate_no_lookahead(
    df: pd.DataFrame,
    as_of: str | date,
    date_col: str = "f_ann_date",
    raise_on_violation: bool = True,
) -> dict:
    """
    检查 DataFrame 里是否存在超出 as_of 时点的数据。
    
    返回：
        {
            "passed": bool,
            "violation_count": int,
            "max_date": Timestamp,
            "violations": DataFrame  # 违规行
        }
    """
    as_of_ts = pd.Timestamp(as_of)
    if date_col not in df.columns:
        return {"passed": True, "violation_count": 0, "max_date": None, "violations": df.iloc[:0]}
    
    violations = df[df[date_col] > as_of_ts]
    result = {
        "passed": len(violations) == 0,
        "violation_count": len(violations),
        "max_date": df[date_col].max(),
        "violations": violations,
    }
    
    if not result["passed"] and raise_on_violation:
        raise ValueError(
            f"PIT violation detected: {len(violations)} rows with {date_col} > {as_of}.\n"
            f"Max date found: {result['max_date']}.\n"
            f"Sample violations:\n{violations[[date_col]].head()}"
        )
    
    return result
```

使用示例：

```python
# 在任何数据拉取后都可以加这个检查
df = provider.get_financials("600519.SH", "income", as_of="2024-01-01")
result = validate_no_lookahead(df, as_of="2024-01-01", date_col="f_ann_date")
# 如果有 look-ahead bias，立即抛出 ValueError，而不是静默地产生错误结果
```

在生产系统里，这个函数是数据层和特征层之间的"PIT 防火墙"，任何数据在被用于特征计算之前都要过这个检查。

---

## 结语：PIT 是区别专业和业余的分水岭

传统量化教程里，数据处理通常是这样的：

```python
df = yfinance.download("AAPL", start="2020-01-01")
# 直接用，没有任何时间边界处理
```

这种写法在研究阶段可以，但把它搬到实盘会遇到灾难性的结果。

A 股的情况比美股更复杂：A 股有更严格的停牌制度，涨跌停限制，成分股调整机制，以及三套独立的财务披露时间轴（预告/快报/正式）。每一层都可能产生 look-ahead bias。

QuantMind 的 PIT 测试套件（10 个测试，覆盖 tushare + akshare 双源）是项目里最先写完的东西之一，早于任何模型和 Agent 的开发。

原则很简单：**如果你不确定数据是否有 look-ahead bias，假设它有**。先写测试，然后用测试驱动数据层的实现。

这比在回测结果出来之后再回溯 debug 要便宜得多。

完整测试代码：`tests/test_pit_correctness.py`，数据层实现：`quantmind/data/`。

---

*这是 QuantMind 系列的第三篇，前两篇：*
- *[《我用 LangGraph 做了一个 AI 投资分析 Agent，但它告诉我 Agent 都在讲故事》](01_critic_agent_self_reflection.md)*
- *[《把生成式推荐用到量化选股：LLM Listwise Rerank 的真实回测结果》](02_llm_listwise_rerank.md)*
