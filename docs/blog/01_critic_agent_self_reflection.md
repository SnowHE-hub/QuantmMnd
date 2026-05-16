# 我用 LangGraph 做了一个 AI 投资分析 Agent，但它告诉我 Agent 都在讲故事

> 作者：QuantMind 项目笔记 | 系列：AI Agent 量化研究实践

---

## 引子：单 Agent 为什么会在金融分析里胡说

2023 年底，我第一次让 ReAct Agent 分析宁德时代（300750.SZ）。结果出来得很快，格式也很漂亮：

> "宁德时代 2023 年 ROE 为 28.5%，PE 为 35 倍，考虑到新能源赛道景气度，建议**买入**，目标价 250 元。"

我一查实际数据：当年 ROE 大约是 **16.8%**，PE（TTM）大约是 **20 倍**。目标价更是无从考证。

数字**几乎全是编的**。

这不是个例。在金融分析任务里，单 Agent 的幻觉问题有三个根源：

**第一，LLM 的记忆混淆**。训练数据里有大量关于宁德时代的分析报告，模型会"记住"某年的某个数字，却不区分那是 2020 年的数据还是 2023 年的。当你没有强制让它从工具调用结果中读取数据时，它会从"记忆"里凭感觉填数字。

**第二，没有验证机制**。ReAct 的循环是：思考 → 行动 → 观察 → 思考……但整个链条里没有任何节点问"这个数字合理吗？"如果工具返回了数据，Agent 直接采信，连合理性检查都不做。如果工具调用失败（比如 API 超时），Agent 会从"常识"里编一个。

**第三，任务本身的复杂性**。一份完整的投资分析报告需要整合：财务数据（ROE/PE/PB/DCF）+ 技术面（MA/MACD/RSI）+ 情绪（北向资金/新闻）+ 估值（历史分位、同业对比）。这些维度相互依赖，单 Agent 同时处理很容易产生内在矛盾，比如"财务面说增长放缓"但结论却是"强烈买入"。

于是我开始设计 CriticAgent，这是 QuantMind 系统里最有意思的一个组件。

---

## Critic Agent 的设计

一句话：**CriticAgent 是一个专职挑错的 Agent，它看着其他 Agent 的输出，找出所有有问题的地方**。

### 审查的五个维度

从工程实践看，金融分析里的错误集中在这五类：

```
1. 数据完整性：关键财务数据是否齐全？是否有 None/缺失？
2. 推理一致性：基本面、技术面、情绪面结论是否相互矛盾？
3. 数字合理性：财务数字是否在合理范围？
   （如 ROE>100% 可疑，PE<0 需要解释）
4. 风险覆盖：是否充分讨论下行风险？（至少3个具体风险点）
5. 论证严谨性：观点是否有具体数字支撑？
   是否有"这只股票很好"这类空话？
```

这五个维度对应五种最常见的分析缺陷。设计时有意把"数字合理性"单独列出来——这是对抗幻觉最直接的武器：只要 LLM 给出了一个"ROE=28.5%"，我们就可以检查这个数字是否在工具调用返回的实际数据范围内。

### 如何识别幻觉

CriticAgent 拿到的输入包含**原始工具调用结果**（实际财务数字）和**分析文本**（Agent 的分析内容）。通过对比，它可以发现：

- 分析文本里的数字在原始数据里找不到 → **数据捏造**
- ROE 显示 115% 但没有解释原因 → **数字异常未处理**
- 财务面结论"盈利能力恶化"但投资建议"买入" → **推理矛盾**

在 System Prompt 里我专门加了一条安全规则：

> "严禁基于训练数据中的股票历史知识做主观判断，只能依据表格中的因子值。reason 中引用的具体数字必须来自输入数据，不得捏造。"

### 输出结构：passed / issues / severity

CriticAgent 的输出是结构化 JSON：

```json
{
  "passed": false,
  "issues": [
    {
      "severity": "critical",
      "type": "hallucination",
      "location": "FundamentalAgent.profitability_analysis",
      "description": "分析中提到ROE=28.5%，但工具返回的实际ROE_TTM=16.8%",
      "suggestion": "重新调用 get_financial_ratios 工具并以返回值为准"
    },
    {
      "severity": "major",
      "type": "risk_missing",
      "location": "FundamentalAgent.key_risks",
      "description": "未讨论宁德时代的锂价波动风险和海外竞争风险",
      "suggestion": "增加至少3个具体的下行风险点，引用具体数据"
    }
  ],
  "overall_quality_score": 4.5,
  "approval_message": "数据幻觉问题严重，需重新拉取财务数据后重做"
}
```

**触发规则**（关键设计决策）：

```python
# 强制执行触发规则——不信任 LLM 的 passed 字段
critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
major_count = sum(1 for i in issues if i.severity == IssueSeverity.MAJOR)
rule_based_passed = (critical_count == 0) and (major_count < 3)
passed = rule_based_passed and llm_passed
```

注意这里**双重验证**：规则检查 AND LLM 判断。即使 LLM 说"passed=true"，只要有一个 critical issue，代码层面强制置为 False。这是为了防止 LLM "放水"——它有时候会因为不确定而保守地判通过。

---

## Self-Reflection 的工程坑

这才是最难的部分。

### 坑 1：无限循环

最直接的坑：Critic 打回，分析 Agent 重做，Critic 再审……如果没有终止条件，这个循环可以跑到你破产。

解决方案是硬上限：`max_iterations=3`。在路由逻辑里：

```python
def _route_after_critic(self, d: dict) -> str:
    state = self._to_state(d)
    feedback = state.critic_feedback

    # 强制退出条件
    if state.iteration_count >= self.max_iterations:
        logger.warning(
            f"[Orchestrator] max_iterations={self.max_iterations} reached, "
            "forcing exit"
        )
        state.should_terminate = True
        state.terminate_reason = f"达到最大迭代次数 {self.max_iterations}"
        return END  # LangGraph 内置终止节点

    if feedback is None or feedback.passed:
        return "report"  # 通过 → 生成报告

    # ... 否则路由回对应 Agent 重做
```

3 次迭代大约对应 LLM API 调用 15-20 次（每次分析有多个专项 Agent），在成本和质量间取了个平衡。

### 坑 2：迭代退化

更隐蔽的坑：分析 Agent 重做之后，问题没有减少，反而变多了。比如：

- 第一轮：Critic 发现 2 个 major issue
- 第二轮：分析 Agent "修复"了，但引入了新的矛盾，Critic 发现 3 个 major issue
- 死循环

解决方案是**单调递减检查**：

```python
# Compaction：本轮 issue 数必须少于上轮，否则强制退出
prev_count = getattr(state, "_prev_issue_count", None)
current_count = len(feedback.issues)
if prev_count is not None and current_count >= prev_count:
    logger.warning(
        f"[Orchestrator] issue count not decreasing "
        f"({prev_count} → {current_count}), forcing exit"
    )
    state.should_terminate = True
    return END
state._prev_issue_count = current_count
```

每轮迭代的 issue 数必须严格小于上一轮，否则认为 Self-Reflection 没有效果，强制退出并生成当前最优报告。

### 坑 3：Critic 自己也会错

最根本的问题：CriticAgent 本身也是一个 LLM，**它自己也会产生幻觉**。比如：

- 指出"ROE 数据缺失"，但实际上数据已经在输入里了，只是格式不符合 LLM 的预期
- 把"PE=15 倍"标记为"异常低"，但实际上银行股 PE 就应该是 10-15 倍

目前的处理方式有两个层面：

1. **格式化输入**：`format_input()` 方法把所有数据以标准表格形式输入给 Critic，减少"找不到数据"的假阳性
2. **规则兜底**：issue 的解析完全由代码控制，不相信 LLM 的 "passed" 字段（上面的双重验证）

一个理想的解法是**Critic 置信度校准**：在 CriticFeedback 里加入 `confidence_score`，低置信度的 issue 只记录不触发回流。这是后续版本要做的事。

---

## 真实 Case 复盘

用一个真实的测试 case 来感受一下完整流程。股票：600519.SH（贵州茅台），as_of：2024-06-30。

**第 1 轮**

Planner → Data Agent 获取数据 → Fundamental/Technical/Sentiment 三个 Agent 分析

Critic 审查结果：

```
passed: false
issues:
  - severity: critical, type: data_missing
    location: FundamentalAgent.dcf_value
    description: DCF公允值为 None，未完成估值计算
    suggestion: 调用 dcf_valuation 工具重新计算，使用WACC=8%假设
  - severity: major, type: risk_missing
    description: 未讨论白酒行业政策风险和消费降级风险
overall_quality_score: 5.2
```

路由决策：critical issue 在 FundamentalAgent → 回流到 `fundamental` 节点重做。

**第 2 轮**

FundamentalAgent 重新调用 DCF 工具，补充了 2 个风险点。

Critic 再次审查：

```
passed: false
issues:
  - severity: major, type: inconsistency
    description: 技术面显示短期MACD金叉，但情绪面提到北向资金净流出，
                 报告结论未解释这个矛盾
  - severity: minor, type: weak_argument
    description: 催化剂部分缺少具体时间节点
overall_quality_score: 7.1
```

issue 数从 2 降到 2（但 severity 降了：critical→0，major→1），按规则继续。

**第 3 轮**

FundamentalAgent 和 SentimentAgent 协同补充了对技术/情绪分歧的解释。

Critic：

```
passed: true
overall_quality_score: 8.4
approval_message: "三维数据一致，风险覆盖充分，论证有数字支撑，通过审查"
```

最终 Report Agent 生成报告。相比第 1 轮，质量分从 5.2 → 8.4，关键改进是：DCF 估值补全、风险点覆盖、技术/情绪分歧有了明确解释。

---

## 代码片段

CriticAgent 的完整 System Prompt：

```python
_SYSTEM_PROMPT = """你是一位严格的投资研究质量审查官。
你的任务是审查投资分析报告的质量，从5个维度逐一评估：

1. **数据完整性**：关键财务数据是否齐全？是否有 None/缺失？
2. **推理一致性**：基本面、技术面、情绪面结论是否相互矛盾？
3. **数字合理性**：财务数字是否在合理范围？（如 ROE>100% 可疑，PE<0 需解释）
4. **风险覆盖**：是否充分讨论下行风险？（至少3个具体风险点）
5. **论证严谨性**：观点是否有具体数字支撑？是否有"这只股票很好"这类空话？

触发规则（严格执行）：
- critical ≥ 1 条 → passed=false
- major ≥ 3 条 → passed=false
- 否则 → passed=true

输出严格 JSON：
{
  "passed": true/false,
  "issues": [
    {
      "severity": "critical|major|minor",
      "type": "data_missing|inconsistency|calculation_error|risk_missing|
               weak_argument|hallucination|pit_violation",
      "location": "指出问题所在（如 FundamentalAgent.valuation_analysis）",
      "description": "问题描述（具体，引用实际内容）",
      "suggestion": "修复建议（具体可操作）"
    }
  ],
  "overall_quality_score": 0-10,
  "approval_message": "通过/不通过的说明（50字内）"
}"""
```

解析输出并强制规则：

```python
def _execute(self, state: AgentState) -> AgentState:
    # ... 省略调用 LLM 的部分 ...

    # 解析 issue 列表，映射到 IssueSeverity enum
    issues: list[CriticIssue] = []
    for raw_issue in parsed.get("issues", []):
        severity = IssueSeverity(raw_issue.get("severity", "minor").lower())
        issue_type = IssueType(raw_issue.get("type", "weak_argument").lower())
        issues.append(CriticIssue(
            severity=severity,
            type=issue_type,
            description=raw_issue.get("description", ""),
            fix_action=...,  # 根据 location 推断回流 Agent
        ))

    # 不信任 LLM 的 passed 字段，代码层面强制执行触发规则
    critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
    major_count = sum(1 for i in issues if i.severity == IssueSeverity.MAJOR)
    rule_based_passed = (critical_count == 0) and (major_count < 3)
    passed = rule_based_passed and bool(parsed.get("passed", True))

    state.critic_feedback = CriticFeedback(
        passed=passed,
        issues=issues,
        overall_quality_score=float(parsed.get("overall_quality_score", 5.0)),
        approval_message=parsed.get("approval_message", ""),
    )
    return state
```

---

## 结语

Self-Reflection 不是一个"加上去会更好"的特性，而是金融 AI Agent 在生产环境里的**生存条件**。

数字错一个，分析的公信力就崩了。投资者不会因为"AI 时代偶尔幻觉正常"而原谅一个写错 ROE 的系统。

三个关键工程原则，用一行话总结：
- **硬上限**（max_iterations=3）：防无限循环，不要相信 LLM 会自己收敛
- **单调性检查**：每轮必须有改进，否则强制退出
- **规则优于 LLM**：触发逻辑不能全靠 LLM 判断，代码层面的规则是最后防线

完整代码在 GitHub（`quantmind/agents/critic_agent.py` 和 `quantmind/agents/orchestrator.py`）。

---

*下一篇：《把生成式推荐用到量化选股：LLM Listwise Rerank 的真实回测结果》*
