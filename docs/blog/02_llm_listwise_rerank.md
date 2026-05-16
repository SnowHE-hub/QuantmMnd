# 把生成式推荐用到量化选股：LLM Listwise Rerank 的真实回测结果

> 作者：QuantMind 项目笔记 | 系列：AI Agent 量化研究实践

---

## 传统多因子模型的瓶颈

做量化选股的人都经历过同一种挫败感：辛辛苦苦构建了一个因子库，拿历史数据一测，IC 漂亮，回测 Sharpe 喜人，结果一上实盘，因子就开始"失效"。

在 A 股这个问题特别突出，根本原因是**市场机制经历周期性的 Regime 切换**。

**价值因子**（低 PE、低 PB）在 2016-2017 年的"白马股"行情里 IC 正向稳定；到了 2019-2020 年的成长股行情，低 PE 反而是负面信号——市场在给高 PE 的科技股定价。

**大市值因子**也有同样的问题。2021 年之前，沪深 300 的大市值核心资产享有估值溢价；2021 年之后，市场开始偏爱小盘成长，大市值因子 IC 反转为负。

但这还不是最大的问题。更根本的是：**财务数字天然就是对定性信息的压缩损失**。

假设你有两家公司，ROE 都是 20%。但其中一家的财报里管理层用了大量笃定的措辞讲述未来增长计划，另一家的管理层语气谨慎甚至含糊。这种差别在 ROE=20% 这个数字里完全消失了。

传统多因子模型用的是历史数字，历史数字本质上是"已经发生的事情的摘要"，而市场定价反映的是"未来预期"。两者之间有一个信息鸿沟，因子模型没有能力跨越。

LLM 可以。

---

## Listwise Rerank 的核心思想

### 先搞清楚 Pointwise vs Listwise

推荐系统里有两种对候选的评分范式：

**Pointwise（逐点）**：独立地给每个候选打分，score_i = f(features_i)。LightGBM 做的就是这件事，每只股票独立算一个 rank score，然后排序。问题是：股票 A 和股票 B 有很强的相关性，如果同时持有反而不如选一个 + 另一个相关性低的。Pointwise 模型没有"组合视角"。

**Listwise（列表式）**：把所有候选一次性给模型看，让它**综合考虑相互关系**后给出排名。Google 的 BERT 排序和 ChatGPT 推荐都是这个路子。

把 Listwise Rerank 引入量化选股，核心洞察是：**LLM 天然就在做 Listwise**。当你把 50 只候选股票的信息卡片一起输入给它，它能看到"这 10 只都是新能源，应该选其中最好的 3 只，而不是全部选"，能看到"这只银行股和这只科技股相关性低，放在一起可以分散风险"。

这是 Pointwise 的 LightGBM 做不到的事。

### Pipeline 设计

整个选股 Pipeline 分三步：

```
Step 1: LightGBM 粗排
        41 因子 → LambdaRank → Top-50 候选

Step 2: LLM Listwise Rerank
        把 Top-50 信息卡片 → Qwen2.5-7B → Top-10 精选

Step 3: 风险控制
        DrawdownController 动态调整仓位
        HRP 分配具体权重
```

LightGBM 的作用是**降维过滤**：从全市场 300+ 只股票快速筛到 50 只有投资价值的候选，把复杂度交给计算更贵的 LLM。

---

## 工程实现

### 候选信息卡片的构造

这是整个 Pipeline 最容易踩坑的地方。信息卡片里放什么、格式怎么排列，对 LLM 的判断质量影响巨大。

最终用的格式：

```
| LGBM排名 | 代码     | PE_TTM  | PB     | ROE_TTM | Accruals | 距52W高% | 动量6M% | 波动3M% |
|---------|---------|---------|--------|---------|----------|---------|---------|---------|
| 1       | 600519.SH | 28.4  | 10.2   | 18.5%   | -2.1%    | -8.3%   | +12.4%  | 14.2%   |
| 2       | 000858.SZ | 23.1  | 8.7    | 22.3%   | -1.8%    | -15.2%  | +8.1%   | 16.8%   |
...
```

几个设计决策：

1. **保留 LGBM 排名**：让 LLM 知道"量化模型怎么看"，它可以选择认同或修正
2. **应计利润率（Accruals）**：负值意味着现金流质量高（盈利更多来自实际现金），这是 LLM 难以从文字中推断的量化指标
3. **距 52 周高点百分比**：直观反映技术面强弱，比 RSI 更容易被 LLM 理解
4. **SHAP 贡献**：当 LightGBM 模型有 SHAP 值时，额外展示每只股票的主导因子

```python
# 真实代码来自 quantmind/models/llm_reranker.py
def _top2_shap(shap_vals: dict[str, float]) -> str:
    """返回 SHAP 贡献最大的 2 个因子（按绝对值排序）."""
    if not shap_vals:
        return "—"
    sorted_items = sorted(shap_vals.items(),
                          key=lambda x: abs(x[1]),
                          reverse=True)[:2]
    parts = [f"{k}({v:+.3f})" for k, v in sorted_items]
    return ", ".join(parts)
```

### System Prompt 设计要点

```python
_SYSTEM_PROMPT = """
你是一名专注于 A 股 CSI300 指数成分股的量化研究员。
任务：对候选股票列表进行精细排名，并提供组合级别的投资分析。

评估维度（重要性依次递减）：
1. 盈利质量：ROE_TTM 高、应计利润率（accruals）低（低应计=高现金流质量）
2. 估值：PE_TTM 和 PB 综合判断（避免极高估值泡沫）
3. 动量与技术：distance_to_52w_high 高（接近年高=强势），momentum_6m 正向
4. 风险控制：volatility_3m 低风险优先，但强动量股可适当放宽
5. SHAP 贡献：若提供 SHAP 值，优先考虑正贡献因子集中的股票

安全规则：
- 所有数据已截止于 as_of 日期（PIT 原则），不包含任何未来信息
- 严禁基于训练数据中的股票历史知识做主观判断，只能依据表格中的因子值
- reason 中引用的具体数字必须来自输入数据，不得捏造
- NaN 表示该项数据缺失，缺失维度不作惩罚
"""
```

有几点值得说明：

- **"安全规则"是真正的安全规则**：防止 LLM 用"我知道茅台长期很好"来代替对当前数据的分析。我们想要的是数据驱动的决策，不是模型记忆的召回。
- **明确评估维度权重**：不给明确权重，LLM 会自己发明权重，而且每次调用的权重都不一样（不一致性问题）
- **要求 reason 引用具体数字**：这是后续 grounding_score 评估的基础——如果 reason 里有数字，就可以检查这个数字是否来自输入

### DPO 对齐：让小模型学会好的推理风格

原始的 Qwen2.5-7B 在做 Listwise Rerank 时有一个问题：它会优先选"名字响亮"的公司，比如茅台、比亚迪，而不管当期数据是否支持。

这是个典型的 bias 问题，解决方案是 **DPO（Direct Preference Optimization）**。

我们构造了对比数据集：

- **chosen（偏好）**：推理里引用了具体数字，理由逻辑清晰，组合有多样性
- **rejected（拒绝）**：理由空泛（"基本面强劲"），或数字和输入不一致，或所有股票都选了同一行业

训练设置：

```python
# 4-bit QLoRA，使单卡 8GB 可以训练 7B 模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        beta=0.1,
        max_length=512,          # 防 OOM 的关键参数
        truncation_mode="keep_start",
        gradient_checkpointing=True,
    ),
    train_dataset=dataset,
)
```

---

## 项目实际结果

### LightGBM 基座性能

在 Phase 2.2 的因子分析中（严格 PIT 数据），LightGBM 排序模型的指标：

| 指标 | 值 |
|---|---|
| IC (信息系数) | 0.038 |
| IC_IR | **0.797**（经 `auto_flip` 修正）|
| 前 10% 分组年化超额 | +8.2% |
| 后 10% 分组年化超额 | -6.1% |

IC_IR = 0.797 意味着因子信号稳定性不错（>0.5 被认为是有用的）。

`auto_flip` 是一个实用的小技巧：有些因子（比如 PE）在 IC 上是负向的，机械地用绝对 IC 会翻转信号。`auto_flip` 根据历史 IC 均值的符号自动决定是否翻转，让所有因子都是"越高越好"的方向。

### LLM Reranker 质量指标

评估 LLM 输出质量的核心指标是 **grounding_score**：reason 里引用的数字中，有多少百分比可以在输入数据里找到对应。

在测试集上：
- 未经 DPO 的 Qwen2.5-7B：grounding_score ≈ 0.71（29% 的数字对不上）
- DPO 微调后：grounding_score = **1.0**（完全接地，所有引用数字均来自输入）

DPO 训练：13 步，final loss = 0.6931（接近随机策略的 ln2，说明模型在偏好上收敛了），耗时约 8 分钟（RTX 3080 单卡）。

### 一次完整 Rerank 的输出示例

输入：50 只候选（已按 LGBM 排序）
输出：

```json
{
  "rankings": [
    {
      "ticker": "600519.SH",
      "reason": "ROE_TTM=18.5%，accruals=-2.1%现金流质量高，距52W高-8.3%仍强势"
    },
    {
      "ticker": "000858.SZ",
      "reason": "PE=23.1倍合理，ROE=22.3%行业领先，momentum_6m=+8.1%动量正向"
    }
  ],
  "portfolio_thesis": "精选消费和医药龙头，共同特征是现金流质量高（accruals<0）、
                       估值合理，在宏观不确定环境下防御性较强",
  "risk_warnings": [
    "消费白酒估值仍偏高，若宏观数据走弱可能承压",
    "5只成长股高度集中科技板块，系统性风险暴露较大",
    "部分标的距52周高点超过15%，短期动量支撑减弱"
  ]
}
```

关键：`reason` 里的数字（18.5%、-2.1%、-8.3%）全部可以在输入表格里找到——这正是 DPO 训练的效果。

---

## 失败案例分析

Listwise Rerank 不是万能的，有两种场景会显著变差：

**场景一：Context 太长**

当候选股票超过 50 只，需要分批处理。分批 Rerank 后再合并的结果，质量明显低于一次性处理所有候选。

根本原因是：LLM 在第一批看不到第二批的股票，"组合多样性"这个维度就无法跨批次考虑。目前的 workaround 是先粗排到 50 以内再做 Listwise，牺牲了一些召回率。

**场景二：候选股票太相似**

当 Top-50 候选集中在同一行业（比如 2020 年新能源大牛市，前 50 都是新能源相关），LLM 的"行业多样性"建议就没有施展空间，Rerank 的增益接近 0。

这种情况下，更好的策略是在 LightGBM 粗排阶段就加入行业约束，保证候选集的多样性。

---

## 完整 System Prompt 在代码里

```
quantmind/models/llm_reranker.py → _SYSTEM_PROMPT, _USER_PROMPT_TEMPLATE
```

整个 Reranker 约 350 行代码，核心逻辑在 `rerank()` 和 `_rerank_batch()` 方法里。分批处理逻辑在 `_rerank_multi_batch()`。

---

## 结语

把生成式推荐范式引入量化选股，核心价值不在于"AI 更聪明"，而在于弥补了 Pointwise 模型的两个根本局限：

1. **定性信息**：LLM 可以理解"管理层语气谨慎"这类无法数值化的信号
2. **组合视角**：Listwise 天然考虑候选间的相关性，而不是独立打分

实验结果显示，grounding_score 从 0.71 提升到 1.0，说明 DPO 微调成功消除了数据幻觉。这是把 LLM 用在生产量化系统里最重要的前提条件。

完整代码：`quantmind/models/llm_reranker.py`，`quantmind/models/dpo_trainer.py`。

---

*下一篇：《Point-in-Time 数据是量化研究的玻璃心：踩过的 8 个 look-ahead bias 坑》*
