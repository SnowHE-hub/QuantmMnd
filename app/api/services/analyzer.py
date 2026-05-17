"""数据分析服务 — 用 LLM 对数据/图表生成中文解读."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # quantmind/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.services import llm_client as llm


# ── 系统提示模板 ──────────────────────────────────────────────────────────────
_SYSTEM_ANALYST = """你是 QuantMind 量化投资系统的专业分析师。
基于提供的数据，用简洁专业的中文给出：
1. 核心结论（2-3条要点，用 • 开头）
2. 值得关注的风险或异常
3. 具体可行的建议

风格：数据驱动、精炼、直接，不要废话。回答控制在150字以内。"""

_SYSTEM_COMMANDER = """你是 QuantMind 量化投资系统的 AI 助手。

可用系统命令：
- simulate: 运行30日全A股三系统选股模拟
- simulate_evaluate: 评估当前模拟绩效
- optimize: 因子IC分析 + System2权重校准 + realized_pnl扩充
- backtest: 运行NAV季度回测
- train_meta: 重训 meta-learner（需要379条realized_pnl）
- build_features: 重建v4特征面板

系统数据概况：
- 30日模拟覆盖：5535只全A股，已完成
- 3月期均收益：+20.22%，胜率96.7%，IR=+1.87
- realized_pnl：379条记录（80条季度+299条30日模拟）
- 主模型：LGBM v6 alpha，38特征

判断用户意图，返回严格JSON格式（不要Markdown代码块）：
{
  "action": "none|simulate|simulate_evaluate|optimize|backtest|train_meta|build_features",
  "needs_data": false,
  "data_types": [],
  "reply_hint": "给用户的简短提示（如需执行命令，说明正在执行什么）"
}

data_types 可选: ["sim30d", "nav", "ic", "pnl", "config"]"""


def build_data_context() -> str:
    """读取关键指标，构建给 LLM 的数据上下文."""
    try:
        from app.utils.sim_data import (
            load_sim30d_days, load_sim30d_stock_returns,
            load_ic_analysis, load_realized_pnl,
            horizon_portfolio_ts,
        )
        days = load_sim30d_days()
        sr   = load_sim30d_stock_returns()
        ic   = load_ic_analysis()
        rpnl = load_realized_pnl()

        ctx: dict = {
            "simulation_days": len(days),
            "date_range": f"{days[0]['date']} ~ {days[-1]['date']}" if days else "N/A",
            "total_stocks_analyzed": len(sr),
            "realized_pnl_count": len(rpnl),
        }

        # 各持仓期绩效
        for hz in ["1w", "21d", "3m"]:
            ts = horizon_portfolio_ts(days, hz)
            if not ts.empty:
                mean_ret = ts["mean"].mean()
                win_rate = (ts["mean"] > 0).mean()
                ir = mean_ret / ts["mean"].std() if ts["mean"].std() > 0 else 0
                ctx[f"perf_{hz}"] = {
                    "mean_return": f"{mean_ret*100:+.2f}%",
                    "win_rate": f"{win_rate*100:.1f}%",
                    "ir": f"{ir:+.3f}",
                }

        # IC 分析
        ic_final = ic.get("ic_final_picks", {})
        if ic_final:
            top_factor = max(ic_final.items(),
                             key=lambda x: abs(x[1].get("ic_3m", 0) or 0),
                             default=(None, {}))
            ctx["top_factor_3m"] = {
                "name": top_factor[0],
                "ic": top_factor[1].get("ic_3m"),
            }

        return json.dumps(ctx, ensure_ascii=False, indent=2)
    except Exception as e:
        return f'{{"error": "{e}"}}'


def analyze_data(data_summary: dict, data_type: str = "general",
                 provider: str = "dashscope") -> str:
    """用 LLM 分析数据，返回中文解读."""
    prompt = f"""以下是 QuantMind 系统的 {data_type} 数据摘要：

```json
{json.dumps(data_summary, ensure_ascii=False, indent=2)}
```

请给出专业分析。"""

    messages = [
        {"role": "system", "content": _SYSTEM_ANALYST},
        {"role": "user",   "content": prompt},
    ]
    return llm.chat(messages, provider=provider)


def analyze_execution_result(output: str, cmd_label: str,
                              provider: str = "dashscope") -> str:
    """分析命令执行输出，提取关键结果并解读."""
    # 截取最后 3000 字符（避免超 token）
    if len(output) > 3000:
        output = "...(前段省略)...\n" + output[-3000:]

    prompt = f"""刚刚执行了 QuantMind 系统命令「{cmd_label}」，输出如下：

```
{output}
```

请：
1. 提取关键数字结果（收益、IC、胜率等）
2. 评估执行是否成功，结果是否符合预期
3. 给出下一步建议"""

    messages = [
        {"role": "system", "content": _SYSTEM_ANALYST},
        {"role": "user",   "content": prompt},
    ]
    return llm.chat(messages, provider=provider)


def parse_user_intent(user_message: str, history: list[dict],
                      provider: str = "dashscope") -> dict:
    """解析用户意图：是否需要执行命令、需要哪些数据."""
    messages = [
        {"role": "system", "content": _SYSTEM_COMMANDER},
        *history[-6:],  # 保留最近6条历史
        {"role": "user",   "content": user_message},
    ]
    raw = llm.chat(messages, provider=provider, temperature=0.1)

    # 清理可能的 markdown 代码块
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    try:
        return json.loads(raw)
    except Exception:
        return {"action": "none", "needs_data": False, "data_types": [], "reply_hint": ""}


def generate_full_response(user_message: str, history: list[dict],
                            data_context: str = "",
                            provider: str = "dashscope") -> str:
    """生成完整对话回复（含数据上下文）."""
    system_prompt = f"""你是 QuantMind 量化投资系统的专业 AI 助手。

当前系统数据：
{data_context or build_data_context()}

职责：
- 回答关于系统绩效、因子、选股结果的问题
- 提供专业的量化分析解读
- 如果执行了某个命令，解释结果含义
- 语言：简洁中文，数据驱动，不超过300字
- 适当使用 Markdown（加粗重点、列表）"""

    messages = [
        {"role": "system", "content": system_prompt},
        *history[-8:],
        {"role": "user",   "content": user_message},
    ]
    return llm.chat(messages, provider=provider)


def generate_chart_commentary(chart_title: str, data_points: dict,
                               provider: str = "dashscope") -> str:
    """为图表生成简短 AI 解读."""
    prompt = f"""图表「{chart_title}」的关键数据：
{json.dumps(data_points, ensure_ascii=False)}

用1-2句话给出专业解读，重点说明最值得关注的信息。"""

    messages = [
        {"role": "system", "content": "你是量化分析专家，用简洁中文解读图表数据。"},
        {"role": "user",   "content": prompt},
    ]
    return llm.chat(messages, provider=provider)
