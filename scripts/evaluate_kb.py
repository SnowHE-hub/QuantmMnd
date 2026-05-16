"""scripts/evaluate_kb.py — 知识库 RAG 质量评估脚本.

内置 10 条黄金测试用例（Golden Cases），涵盖：
  - 基本面关键词检索
  - 技术指标新闻检索
  - PIT 正确性验证（截止日期过滤）
  - 跨股票检索
  - 公告/年报检索

用法：
    python scripts/evaluate_kb.py
    python scripts/evaluate_kb.py --as-of 2024-06-30 --top-k 5 --output reports/kb_eval.html
    python scripts/evaluate_kb.py --collection my_kb --chroma-dir .cache/chromadb
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger


# ── 10 条黄金测试用例 ─────────────────────────────────────────────────────────

def make_golden_cases(as_of: date) -> list:
    """构造 10 条黄金测试用例."""
    from quantmind.kb.evaluation import GoldenCase
    from datetime import timedelta

    return [
        # Case 1: 茅台基本面 — 财务关键词
        GoldenCase(
            query="贵州茅台 ROE 净利率 财务分析",
            expected_tickers=["600519"],
            expected_keywords=["茅台", "ROE", "净利率"],
            as_of=as_of,
            description="茅台基本面财务指标检索",
        ),
        # Case 2: 茅台新闻 — 近期事件
        GoldenCase(
            query="贵州茅台 2024 提价 涨价 新闻",
            expected_tickers=["600519"],
            expected_keywords=["茅台"],
            as_of=as_of,
            source_type="news",
            description="茅台提价新闻检索",
        ),
        # Case 3: 五粮液 行业对比
        GoldenCase(
            query="五粮液 白酒行业竞争 市场份额",
            expected_tickers=["000858"],
            expected_keywords=["五粮液", "白酒"],
            as_of=as_of,
            description="五粮液行业竞争检索",
        ),
        # Case 4: 银行股 — 不良贷款率
        GoldenCase(
            query="招商银行 工商银行 不良贷款率 资产质量",
            expected_keywords=["银行", "不良"],
            as_of=as_of,
            description="银行股资产质量检索",
        ),
        # Case 5: 新能源 — 政策与补贴
        GoldenCase(
            query="新能源汽车 补贴政策 2024 电动车",
            expected_keywords=["新能源", "补贴"],
            as_of=as_of,
            source_type="news",
            description="新能源政策新闻检索",
        ),
        # Case 6: PIT 测试 — 截止日期过滤
        GoldenCase(
            query="A股 市场行情 分析报告",
            expected_keywords=["A股"],
            as_of=as_of,
            pit_forbidden_date=as_of,  # 截止日期后的文档不应出现
            description="PIT 正确性验证（截止日期过滤）",
        ),
        # Case 7: 茅台年报 — 公告检索
        GoldenCase(
            query="贵州茅台 年度报告 2023 营业收入",
            expected_tickers=["600519"],
            expected_keywords=["茅台", "营业收入"],
            as_of=as_of,
            source_type="filing",
            description="茅台年报检索",
        ),
        # Case 8: 科技股 — 半导体
        GoldenCase(
            query="半导体 芯片 国产替代 中芯国际",
            expected_keywords=["半导体", "芯片"],
            as_of=as_of,
            description="半导体行业检索",
        ),
        # Case 9: 宏观经济 — GDP
        GoldenCase(
            query="中国经济 GDP 增速 宏观政策 2024",
            expected_keywords=["GDP", "经济"],
            as_of=as_of,
            description="宏观经济数据检索",
        ),
        # Case 10: 茅台 + 情绪 — 北向资金
        GoldenCase(
            query="北向资金 外资流入 贵州茅台 持仓",
            expected_tickers=["600519"],
            expected_keywords=["北向", "外资"],
            as_of=as_of,
            description="北向资金持仓检索",
        ),
    ]


# ── HTML 报告生成 ─────────────────────────────────────────────────────────────

def build_html_report(eval_dict: dict, as_of: date, top_k: int) -> str:
    """生成 HTML 评估报告."""
    summary = eval_dict["summary"]
    cases = eval_dict["cases"]

    def _bar(value: float, color: str = "#00b894") -> str:
        pct = int(value * 100)
        return (
            f'<div style="background:#eee;border-radius:4px;height:16px;width:200px;display:inline-block">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div></div>'
            f' <span style="margin-left:8px">{value:.3f} ({pct}%)</span>'
        )

    # 指标颜色
    def _color(v: float) -> str:
        if v >= 0.7:
            return "#00b894"
        if v >= 0.4:
            return "#fdcb6e"
        return "#d63031"

    # 用例行
    case_rows = ""
    for i, c in enumerate(cases):
        status = "✅" if c["hit"] else "❌"
        pit = "⚠️" if c["pit_violation"] else "—"
        err = f'<span style="color:red">{c["error"][:40]}</span>' if c["error"] else "—"
        snippets = "<br>".join(
            f'<small style="color:#636e72">{s[:80]}...</small>'
            for s in c["retrieved_snippets"]
        )
        case_rows += f"""
<tr>
  <td style="text-align:center">{i+1}</td>
  <td style="text-align:center;font-size:1.2em">{status}</td>
  <td>{c['description'] or c['query'][:30]}</td>
  <td style="text-align:center">{c['reciprocal_rank']:.2f}</td>
  <td style="text-align:center">{c['faithfulness']:.2f}</td>
  <td style="text-align:center">{pit}</td>
  <td style="text-align:center">{c['retrieved_count']}</td>
  <td>{snippets or err}</td>
</tr>"""

    hr = summary["hit_rate"]
    mrr_v = summary["mrr"]
    faith_v = summary["faithfulness"]
    pit_v = summary["pit_correctness"]

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantMind KB 评估报告</title>
<style>
  body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f6fa;margin:0;padding:20px}}
  h1,h2{{color:#1a1a2e}} table{{width:100%;border-collapse:collapse}}
  th{{background:#1a1a2e;color:#fff;padding:10px;text-align:left}}
  td{{padding:8px;border-bottom:1px solid #eee;vertical-align:top}}
  .card{{background:#fff;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
  .metric{{display:inline-block;padding:16px 24px;border-radius:8px;margin:8px;text-align:center;min-width:140px;color:#fff}}
</style>
</head>
<body>
<h1>🔍 QuantMind 知识库 RAG 评估报告</h1>
<p style="color:#636e72">评估日期：{date.today()} | 截止日期：{as_of} | Top-K：{top_k} | 用例数：{summary["total_cases"]}</p>

<div>
  <span class="metric" style="background:{_color(hr)}">
    <div style="font-size:0.8em">Hit Rate@{top_k}</div>
    <div style="font-size:1.6em;font-weight:bold">{hr:.1%}</div>
  </span>
  <span class="metric" style="background:{_color(mrr_v)}">
    <div style="font-size:0.8em">MRR</div>
    <div style="font-size:1.6em;font-weight:bold">{mrr_v:.3f}</div>
  </span>
  <span class="metric" style="background:{_color(faith_v)}">
    <div style="font-size:0.8em">Faithfulness</div>
    <div style="font-size:1.6em;font-weight:bold">{faith_v:.1%}</div>
  </span>
  <span class="metric" style="background:{_color(pit_v)}">
    <div style="font-size:0.8em">PIT Correctness</div>
    <div style="font-size:1.6em;font-weight:bold">{pit_v:.1%}</div>
  </span>
</div>

<div class="card">
  <h2>指标详情</h2>
  <table>
    <tr><th>指标</th><th>值</th><th>说明</th></tr>
    <tr><td>Hit Rate@{top_k}</td>
        <td>{_bar(hr, _color(hr))}</td>
        <td>top-{top_k} 中至少命中1个相关文档的比例</td></tr>
    <tr><td>MRR</td>
        <td>{_bar(mrr_v, _color(mrr_v))}</td>
        <td>Mean Reciprocal Rank，衡量首个命中位置</td></tr>
    <tr><td>Faithfulness</td>
        <td>{_bar(faith_v, _color(faith_v))}</td>
        <td>检索结果包含期望关键词的平均覆盖率</td></tr>
    <tr><td>PIT Correctness</td>
        <td>{_bar(pit_v, _color(pit_v))}</td>
        <td>截止日期过滤正确率（无未来信息泄露）</td></tr>
  </table>
</div>

<div class="card">
  <h2>用例详情</h2>
  <table>
    <tr>
      <th>#</th><th>状态</th><th>描述</th><th>RR</th>
      <th>Faith</th><th>PIT</th><th>命中数</th><th>检索片段</th>
    </tr>
    {case_rows}
  </table>
</div>

<footer style="text-align:center;padding:20px;color:#b2bec3;font-size:0.85em">
  QuantMind Phase 5 | KB RAG Evaluation | ChromaDB + BGE-M3 + BM25 + Reranker
</footer>
</body>
</html>"""
    return html


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind Phase 5 — 知识库 RAG 质量评估"
    )
    parser.add_argument(
        "--as-of", default=None,
        help="截止日期 YYYY-MM-DD（默认今天）"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="检索数量（默认 5）"
    )
    parser.add_argument(
        "--collection", default="default",
        help="ChromaDB 集合名称（默认 default）"
    )
    parser.add_argument(
        "--chroma-dir", default=".cache/chromadb",
        help="ChromaDB 目录（默认 .cache/chromadb）"
    )
    parser.add_argument(
        "--output", default=None,
        help="HTML 报告输出路径（默认 reports/kb_eval_{date}.html）"
    )
    parser.add_argument(
        "--json-output", default=None,
        help="JSON 结果输出路径（可选）"
    )
    parser.add_argument(
        "--use-reranker", action="store_true", default=False,
        help="启用 BGE-Reranker 精排（默认关闭，加速评估）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    as_of: date
    if args.as_of:
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    else:
        as_of = date.today()

    logger.info(f"[EvalKB] 开始评估: as_of={as_of}, top_k={args.top_k}")

    # 初始化 Retriever
    from quantmind.kb.retriever import HybridRetriever
    from quantmind.kb.evaluation import KBEvaluator

    retriever = HybridRetriever(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        use_reranker=args.use_reranker,
    )

    count = retriever.count()
    logger.info(f"[EvalKB] 知识库文档数：{count}")

    if count == 0:
        logger.warning(
            "[EvalKB] 知识库为空！请先运行 scripts/build_kb.py 构建知识库。\n"
            "评估将在空库上运行（所有指标为 0）。"
        )

    # 构造测试用例
    cases = make_golden_cases(as_of)
    logger.info(f"[EvalKB] 黄金测试用例：{len(cases)} 条")

    # 运行评估
    evaluator = KBEvaluator(retriever, top_k=args.top_k)
    results = evaluator.run(cases)

    # 控制台报告
    evaluator.print_report(results)

    # 序列化结果
    eval_dict = evaluator.to_dict(results)

    # JSON 输出
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(
            json.dumps(eval_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[EvalKB] JSON 报告 → {args.json_output}")

    # HTML 输出
    output_path = args.output
    if not output_path:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"kb_eval_{as_of}.html")

    html = build_html_report(eval_dict, as_of, args.top_k)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"[EvalKB] HTML 报告 → {output_path}")

    # 打印最终摘要
    s = eval_dict["summary"]
    print(f"\n评估完成：Hit@{args.top_k}={s['hit_rate']:.1%}  "
          f"MRR={s['mrr']:.3f}  "
          f"Faith={s['faithfulness']:.1%}  "
          f"PIT={s['pit_correctness']:.1%}")
    print(f"报告已保存 → {output_path}")


if __name__ == "__main__":
    main()
