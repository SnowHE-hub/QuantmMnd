"""quantmind.models.dpo_data_builder — DPO 偏好数据构造.

两种数据来源
============
方法一（开发期主力）：基于规则的合成数据
  - chosen：具体财务数字 + 多维度论证 + 风险讨论的高质量 reasoning
  - rejected：空泛、矛盾、忽略风险的低质量 reasoning
  - build_synthetic_pairs(n=100)

方法二（后期用）：基于回测结果自动标注
  - 60天涨幅>10% 且 reasoning 推荐 → chosen
  - 60天跌幅>5%  且 reasoning 强烈推荐 → rejected
  - label_by_returns(reasoning, realized_return_60d)

输出格式（HuggingFace datasets 兼容 JSONL）
==========================================
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DPOPair",
    "build_synthetic_pairs",
    "label_by_returns",
    "save_pairs",
    "load_pairs",
]

# 固定随机种子保证可复现
_RNG = random.Random(42)
_NP_RNG = np.random.default_rng(42)


# ============================================================================
# 数据容器
# ============================================================================


@dataclass
class DPOPair:
    """单条 DPO 偏好对."""
    prompt: str     # 系统提示 + 候选表格（输入部分）
    chosen: str     # 高质量 reasoning（人类偏好）
    rejected: str   # 低质量 reasoning（人类厌恶）
    meta: dict      # 附加元信息（as_of、ticker、source 等）


# ============================================================================
# 合成股票数据工具
# ============================================================================


def _random_ticker() -> str:
    prefix = _RNG.choice(["600", "000", "300", "002", "601"])
    suffix = _RNG.choice([".SH", ".SZ"])
    num = _RNG.randint(1, 999)
    return f"{prefix}{num:03d}{suffix}"


def _random_factor_row() -> dict[str, float]:
    """生成一行随机因子数据（尽量贴近 A 股真实分布）."""
    return {
        "pe_ttm":               round(_NP_RNG.uniform(8, 60), 1),
        "pb":                   round(_NP_RNG.uniform(0.8, 8.0), 2),
        "roe_ttm":              round(_NP_RNG.uniform(0.03, 0.40), 4),
        "accruals":             round(float(_NP_RNG.normal(-0.02, 0.06)), 4),
        "distance_to_52w_high": round(float(_NP_RNG.uniform(-0.30, 0.02)), 4),
        "momentum_6m":          round(float(_NP_RNG.normal(0.05, 0.20)), 4),
        "volatility_3m":        round(_NP_RNG.uniform(0.12, 0.45), 4),
    }


def _fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def _factor_table(rows: list[dict[str, Any]]) -> str:
    """生成候选表格 Markdown."""
    header = (
        "| LGBM排名 | 代码     | PE_TTM | PB   | ROE_TTM | Accruals | "
        "距52W高% | 动量6M% | 波动3M% |\n"
        "|---------|---------|--------|------|---------|----------|"
        "---------|---------|--------|\n"
    )
    body_lines = []
    for r in rows:
        f = r["factors"]
        body_lines.append(
            f"| {r['lgbm_rank']:>7} | {r['ticker']:<8} "
            f"| {f['pe_ttm']:>6.1f} | {f['pb']:>4.2f} "
            f"| {_fmt_pct(f['roe_ttm']):>7} | {f['accruals']:>8.4f} "
            f"| {_fmt_pct(f['distance_to_52w_high']):>7} "
            f"| {_fmt_pct(f['momentum_6m']):>7} "
            f"| {_fmt_pct(f['volatility_3m']):>7} |"
        )
    return header + "\n".join(body_lines)


def _build_prompt_from_rows(rows: list[dict], as_of: str, top_n: int) -> str:
    """构造标准 prompt（与 run_llm_rerank.py 风格一致）."""
    table = _factor_table(rows)
    return (
        f"截止日期（as_of）：{as_of}\n"
        f"候选股票数量：{len(rows)}\n"
        f"请从以下 {len(rows)} 只股票中选出并排序前 {top_n} 名（第1名=最优）。\n\n"
        f"{table}\n\n"
        "请输出 JSON：\n"
        '{"rankings": [{"ticker":"代码","reason":"理由（引用具体数字，30字内）"}, ...],'
        '"portfolio_thesis": "组合逻辑（100字内）",'
        '"risk_warnings": ["风险1", "风险2", "风险3"]}'
    )


# ============================================================================
# Chosen（高质量）模板
# ============================================================================

_CHOSEN_TEMPLATES = [
    # 模板 1：ROE 主导 + 低应计
    (
        "选择 {ticker1}（ROE_TTM={roe1:.1%}，应计利润率={accruals1:.3f}）作为首选，"
        "高 ROE 叠加负应计表明盈利由现金流驱动而非会计操纵。{ticker2}"
        "（PE={pe2:.1f}，PB={pb2:.1f}）估值处于合理区间，"
        "距52周高点{dist2:.1%}，上行空间充足。"
        "组合风险：行业集中度约40%，需关注政策调整；整体波动3M均值{avg_vol:.1%}，"
        "市场下行时回撤可能加剧。"
    ),
    # 模板 2：动量 + 低估值
    (
        "{ticker1} 动量6M={mom1:.1%}，远超组合均值，配合距52周高点{dist1:.1%}，"
        "技术形态强势；PB={pb1:.2f} 处于历史低位，安全边际充足。"
        "{ticker2} ROE_TTM={roe2:.1%}，应计利润率{accruals2:.3f}，"
        "盈利质量排名组合前列。主要风险：动量反转概率约15%-20%，"
        "建议设置止损线；流动性风险需监控换手率变化。"
    ),
    # 模板 3：多因子综合
    (
        "综合盈利质量与估值分析：{ticker1}（ROE={roe1:.1%}，PE={pe1:.1f}，Accruals={accruals1:.3f}）"
        "三指标均优于组合中位数，选为首位。{ticker2} 动量因子{mom2:.1%} 为组合最强，"
        "但波动率{vol2:.1%} 偏高，持仓比例控制在 8% 以内。"
        "组合 PE 中位数约 {med_pe:.1f} 倍，整体估值偏高是最大风险；"
        "建议分批建仓，首仓 60%，跌破 5% 止损触发再评估。"
    ),
]

_REJECTED_TEMPLATES = [
    # 模板 1：空泛
    "这些股票看起来都不错，应该能涨，建议买入。公司基本面整体良好。",
    # 模板 2：矛盾（PE 极高却说估值便宜）
    (
        "{ticker1} 的 PE 虽然高达 {pe1:.0f} 倍，但估值非常便宜，建议重仓。"
        "未来预期增长很快，不需要担心估值问题。其他股票也都可以买。"
    ),
    # 模板 3：忽略风险
    (
        "{ticker1} ROE 高，直接全仓买入。{ticker2} 动量好，也买。"
        "没有什么风险，市场一定会涨的，不用设止损。"
    ),
    # 模板 4：捏造数字（与输入数据不符）
    (
        "{ticker1} 的 ROE 高达 999%（超强），PE 只有 0.1（极度低估），"
        "未来三年预计涨 500%，强烈推荐买入。"
    ),
]


def _fill_chosen(template: str, rows: list[dict]) -> str:
    if len(rows) < 2:
        return template
    r1, r2 = rows[0]["factors"], rows[1]["factors"]
    t1, t2 = rows[0]["ticker"], rows[1]["ticker"]
    avg_vol = float(np.mean([r["factors"]["volatility_3m"] for r in rows]))
    med_pe  = float(np.median([r["factors"]["pe_ttm"] for r in rows]))
    try:
        return template.format(
            ticker1=t1, ticker2=t2,
            roe1=r1["roe_ttm"], pe1=r1["pe_ttm"], pb1=r1["pb"],
            accruals1=r1["accruals"], dist1=r1["distance_to_52w_high"],
            mom1=r1["momentum_6m"], vol1=r1["volatility_3m"],
            roe2=r2["roe_ttm"], pe2=r2["pe_ttm"], pb2=r2["pb"],
            accruals2=r2["accruals"], dist2=r2["distance_to_52w_high"],
            mom2=r2["momentum_6m"], vol2=r2["volatility_3m"],
            avg_vol=avg_vol, med_pe=med_pe,
        )
    except KeyError:
        return template  # 模板字段不匹配时直接返回原模板


def _fill_rejected(template: str, rows: list[dict]) -> str:
    if len(rows) < 2:
        return template
    r1 = rows[0]["factors"]
    t1, t2 = rows[0]["ticker"], rows[1]["ticker"]
    try:
        return template.format(
            ticker1=t1, ticker2=t2,
            pe1=r1["pe_ttm"], roe1=r1["roe_ttm"],
        )
    except KeyError:
        return template


# ============================================================================
# 主要公开 API
# ============================================================================


def build_synthetic_pairs(
    n: int = 100,
    n_candidates_per_pair: int = 10,
    top_n: int = 5,
    as_of: str = "2024-06-30",
) -> list[DPOPair]:
    """构造 n 条合成 DPO 偏好对.

    Args:
        n:                    生成对数（开发期用 100，正式用 1000+）
        n_candidates_per_pair: 每个 prompt 的候选股票数
        top_n:                每次重排的目标数量
        as_of:                数据截止日期（仅写入 prompt）

    Returns:
        list[DPOPair]
    """
    pairs: list[DPOPair] = []
    chosen_tmpl_idx = 0
    rejected_tmpl_idx = 0

    for i in range(n):
        # 生成候选行
        rows = []
        for rank in range(1, n_candidates_per_pair + 1):
            rows.append({
                "lgbm_rank": rank,
                "ticker":    _random_ticker(),
                "factors":   _random_factor_row(),
            })

        prompt = _build_prompt_from_rows(rows, as_of=as_of, top_n=top_n)

        # 交替使用不同模板，增加多样性
        chosen_tmpl  = _CHOSEN_TEMPLATES[chosen_tmpl_idx % len(_CHOSEN_TEMPLATES)]
        rejected_tmpl = _REJECTED_TEMPLATES[rejected_tmpl_idx % len(_REJECTED_TEMPLATES)]
        chosen_tmpl_idx += 1
        rejected_tmpl_idx += 1

        chosen   = _fill_chosen(chosen_tmpl, rows)
        rejected = _fill_rejected(rejected_tmpl, rows)

        pairs.append(DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            meta={"index": i, "as_of": as_of, "source": "synthetic_rule"},
        ))

    return pairs


def label_by_returns(
    reasoning: str,
    realized_return_60d: float,
    positive_threshold: float = 0.10,
    negative_threshold: float = -0.05,
) -> str | None:
    """基于 60 天实际收益对单条 reasoning 进行标注.

    Args:
        reasoning:            LLM 生成的 reasoning 文本
        realized_return_60d:  60 天实际收益率
        positive_threshold:   涨幅 > 此值 → "chosen"
        negative_threshold:   跌幅 < 此值 → "rejected"

    Returns:
        "chosen" | "rejected" | None（无法确定时）
    """
    # 检测 reasoning 是否包含推荐信号（简单关键词检测）
    recommend_keywords = ["买入", "推荐", "首选", "配置", "持有"]
    strong_recommend   = ["强烈推荐", "重仓", "全仓", "大幅配置"]

    text_lower = reasoning.lower()
    has_recommend = any(k in reasoning for k in recommend_keywords)
    has_strong    = any(k in reasoning for k in strong_recommend)

    if realized_return_60d > positive_threshold and has_recommend:
        return "chosen"
    if realized_return_60d < negative_threshold and has_strong:
        return "rejected"
    return None


def save_pairs(
    pairs: list[DPOPair],
    output_path: str | Path,
    format: str = "jsonl",
) -> Path:
    """保存 DPO 偏好对到文件.

    Args:
        pairs:       DPOPair 列表
        output_path: 输出文件路径（.jsonl 或 .json）
        format:      "jsonl"（每行一条）或 "json"（整个列表）

    Returns:
        实际保存的 Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected, **p.meta}
        for p in pairs
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        if format == "jsonl":
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return output_path


def load_pairs(input_path: str | Path) -> list[dict]:
    """加载 DPO 偏好对文件（JSONL 或 JSON）.

    Returns:
        list of dicts with keys: prompt, chosen, rejected
    """
    input_path = Path(input_path)
    records: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        records = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records
