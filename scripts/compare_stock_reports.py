"""从 context JSON 提取 snapshot 指标并生成横向对比表与排名 Markdown。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def extract_field(text: str, key: str) -> str | None:
    """从结构化文本中提取 key: value（与 RAGReportAgent._extract_field 一致）."""
    if not text:
        return None
    patterns = [
        rf"{re.escape(key)}[：:=]\s*([^\n,，]+)",
        rf"{re.escape(key)}=([^\s,，\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pick_latest_snapshot(items: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not items:
        return None

    def sort_key(d: dict[str, Any]) -> tuple:
        ds = (d.get("as_of") or d.get("published_date") or "")[:10]
        try:
            return (date.fromisoformat(ds),)
        except ValueError:
            return (date.min,)

    return max(items, key=sort_key)


def pct_display_val(stored: float | None) -> float | None:
    """context 中 roe 等存为百分数点（如 26.83 表示 26.83%），原样作为展示值."""
    return stored


def extract_row_from_context(ctx: dict[str, Any]) -> dict[str, Any]:
    ticker = ctx.get("ticker") or ""
    news_count = ctx.get("news_count")
    report_count = ctx.get("report_count")
    try:
        news_c = int(news_count) if news_count is not None else None
    except (TypeError, ValueError):
        news_c = None
    try:
        report_c = int(report_count) if report_count is not None else None
    except (TypeError, ValueError):
        report_c = None

    cp = pick_latest_snapshot(ctx.get("snapshot_company_profile"))
    mm = pick_latest_snapshot(ctx.get("snapshot_latest_market_metrics"))
    fi = pick_latest_snapshot(ctx.get("snapshot_financial_indicator_summary"))
    nb = pick_latest_snapshot(ctx.get("snapshot_northbound_summary"))
    mg = pick_latest_snapshot(ctx.get("snapshot_margin_summary"))

    cp_text = (cp or {}).get("text") or ""
    mm_text = (mm or {}).get("text") or ""
    fi_text = (fi or {}).get("text") or ""
    nb_text = (nb or {}).get("text") or ""
    mg_text = (mg or {}).get("text") or ""

    as_of = None
    if mm:
        as_of = (mm.get("as_of") or mm.get("published_date") or "")[:10] or None
    elif fi:
        as_of = (fi.get("as_of") or fi.get("published_date") or "")[:10] or None

    company_name = extract_field(cp_text, "名称")
    industry = extract_field(cp_text, "行业")

    pe = parse_optional_float(extract_field(mm_text, "pe"))
    pb = parse_optional_float(extract_field(mm_text, "pb"))
    total_mv_wan = parse_optional_float(extract_field(mm_text, "total_mv"))
    total_mv_yi = (total_mv_wan / 10000.0) if total_mv_wan is not None else None

    roe = parse_optional_float(extract_field(fi_text, "roe"))
    gross = parse_optional_float(extract_field(fi_text, "grossprofit_margin"))
    npm = parse_optional_float(extract_field(fi_text, "netprofit_margin"))
    dta = parse_optional_float(extract_field(fi_text, "debt_to_assets"))
    or_yoy = parse_optional_float(extract_field(fi_text, "or_yoy"))

    hold_ratio = parse_optional_float(extract_field(nb_text, "hold_ratio"))
    rzye_raw = parse_optional_float(extract_field(mg_text, "rzye"))
    rzye_yi = (rzye_raw / 1e8) if rzye_raw is not None else None

    return {
        "ticker": ticker,
        "company_name": company_name,
        "industry": industry,
        "as_of": as_of,
        "pe": pe,
        "pb": pb,
        "total_mv_yi": total_mv_yi,
        "roe": pct_display_val(roe),
        "grossprofit_margin": pct_display_val(gross),
        "netprofit_margin": pct_display_val(npm),
        "debt_to_assets": pct_display_val(dta),
        "or_yoy": pct_display_val(or_yoy),
        "hold_ratio": hold_ratio,
        "rzye_yi": rzye_yi,
        "news_count": news_c,
        "report_count": report_c,
    }


def load_all_contexts(context_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(context_dir.glob("*_context.json"))
    for p in paths:
        try:
            ctx = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(extract_row_from_context(ctx))
    return rows


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def write_comparison_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        ("ticker", "ticker"),
        ("company_name", "公司名"),
        ("industry", "行业"),
        ("as_of", "as_of"),
        ("total_mv_yi", "总市值(亿)"),
        ("pe", "PE"),
        ("pb", "PB"),
        ("roe", "ROE(%)"),
        ("grossprofit_margin", "毛利率(%)"),
        ("netprofit_margin", "净利率(%)"),
        ("debt_to_assets", "资产负债率(%)"),
        ("or_yoy", "营收同比(%)"),
        ("hold_ratio", "北向持股比(%)"),
        ("rzye_yi", "融资余额(亿)"),
        ("news_count", "news_count"),
        ("report_count", "report_count"),
    ]
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("total_mv_yi") is None, -(r.get("total_mv_yi") or 0)),
    )
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [
        "# 批量股票 snapshot 指标对比（最新 as_of 期）",
        "",
        head,
        sep,
    ]
    for r in sorted_rows:
        cells = [_fmt_cell(r.get(k)) for k, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 数据说明",
        "",
        "- **数据来源**：各股 `context` 中 snapshot 列表的**最新 as_of** 条目，`extract_field` 从文本解析。",
        "- **总市值**：文中 `total_mv` 为**万元**，表格「总市值(亿)」= 万元 / 10000。",
        "- **PE/PB**：来自 `latest_market_metrics` 文本。",
        "- **ROE、毛利率、净利率、资产负债率、营收同比**：来自 `financial_indicator_summary`，"
        "与 KB 文本一致，**数值为百分数点**（如 26.83 表示 26.83%）。",
        "- **北向持股比**：`hold_ratio`，百分数点。",
        "- **融资余额**：`rzye` 为**元**，「融资余额(亿)」= 元 / 1e8。",
        "- **缺失**显示为 —，**非 0**。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def valuation_comment(pe: float | None, roe: float | None) -> str:
    if pe is None and roe is None:
        return "—"
    if pe is not None and roe is not None and pe < 20 and roe > 15:
        return "低估高质"
    if pe is not None and pe > 40:
        return "高估值"
    if roe is not None and roe < 8:
        return "盈利偏弱"
    return "估值合理"


def northbound_comment(hold: float | None) -> str:
    if hold is None:
        return "—"
    if hold > 10:
        return "外资重仓"
    if hold > 5:
        return "外资关注"
    return "外资配置偏低"


def write_valuation_ranking_md(path: Path, rows: list[dict[str, Any]], as_of_label: str) -> None:
    def pe_key(r: dict) -> float:
        v = r.get("pe")
        return float("inf") if v is None else float(v)

    ranked = sorted(rows, key=pe_key)
    lines = [
        f"# 估值横向对比（PE 排名，as_of={as_of_label}）",
        "",
        "| 排名 | ticker | 公司名 | 行业 | PE | PB | ROE(%) | 毛利率(%) | 评注 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in enumerate(ranked, start=1):
        pe = r.get("pe")
        roe = r.get("roe")
        gm = r.get("grossprofit_margin")
        comment = valuation_comment(pe, roe)
        lines.append(
            f"| {i} | {r.get('ticker','')} | {_fmt_cell(r.get('company_name'))} | "
            f"{_fmt_cell(r.get('industry'))} | {_fmt_cell(pe)} | {_fmt_cell(r.get('pb'))} | "
            f"{_fmt_cell(roe)} | {_fmt_cell(gm)} | {comment} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_northbound_ranking_md(path: Path, rows: list[dict[str, Any]], as_of_label: str) -> None:
    def h_key(r: dict) -> float:
        v = r.get("hold_ratio")
        return float("-inf") if v is None else float(v)

    ranked = sorted(rows, key=h_key, reverse=True)
    lines = [
        f"# 北向资金关注度排名（as_of={as_of_label}）",
        "",
        "| 排名 | ticker | 公司名 | 行业 | 北向持股比(%) | 融资余额(亿) | 评注 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for i, r in enumerate(ranked, start=1):
        h = r.get("hold_ratio")
        comment = northbound_comment(h)
        lines.append(
            f"| {i} | {r.get('ticker','')} | {_fmt_cell(r.get('company_name'))} | "
            f"{_fmt_cell(r.get('industry'))} | {_fmt_cell(h)} | {_fmt_cell(r.get('rzye_yi'))} | {comment} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_compare(
    *,
    context_dir: Path,
    output_dir: Path,
    as_of_label: str = "2024-12-31",
    write_summary: bool = True,
) -> list[dict[str, Any]]:
    rows = load_all_contexts(context_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_table.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_comparison_table_md(output_dir / "comparison_table.md", rows)
    write_valuation_ranking_md(output_dir / "valuation_ranking.md", rows, as_of_label)
    write_northbound_ranking_md(output_dir / "northbound_ranking.md", rows, as_of_label)
    if write_summary:
        build_comparison_summary(
            rows,
            comparison_md_path=output_dir / "comparison_table.md",
            out_path=output_dir / "comparison_summary.md",
            as_of_label=as_of_label,
        )
    return rows


def build_comparison_summary(
    rows: list[dict[str, Any]],
    *,
    comparison_md_path: Path,
    out_path: Path,
    as_of_label: str,
) -> None:
    """写入 comparison_summary.md（含复制对比表 + Top5 排名与共性观察）."""
    table_body = ""
    if comparison_md_path.is_file():
        text = comparison_md_path.read_text(encoding="utf-8")
        idx = text.find("## 数据说明")
        table_body = text[:idx].strip() if idx >= 0 else text.strip()

    def top5_pe_low(rws: list[dict]) -> list[dict]:
        have = [r for r in rws if r.get("pe") is not None]
        return sorted(have, key=lambda r: float(r["pe"]))[:5]

    def top5_pe_high(rws: list[dict]) -> list[dict]:
        have = [r for r in rws if r.get("pe") is not None]
        return sorted(have, key=lambda r: float(r["pe"]), reverse=True)[:5]

    def top5_roe(rws: list[dict]) -> list[dict]:
        have = [r for r in rws if r.get("roe") is not None]
        return sorted(have, key=lambda r: float(r["roe"]), reverse=True)[:5]

    def top5_nb(rws: list[dict]) -> list[dict]:
        have = [r for r in rws if r.get("hold_ratio") is not None]
        return sorted(have, key=lambda r: float(r["hold_ratio"]), reverse=True)[:5]

    def top5_rz(rws: list[dict]) -> list[dict]:
        have = [r for r in rws if r.get("rzye_yi") is not None]
        return sorted(have, key=lambda r: float(r["rzye_yi"]), reverse=True)[:5]

    def md_block(title: str, items: list[dict], col: str, field: str) -> list[str]:
        out = [f"### {title}", ""]
        if not items:
            out.append("（无足够数据）")
            out.append("")
            return out
        out.append(f"| ticker | 公司名 | {col} |")
        out.append("| --- | --- | ---: |")
        for r in items:
            v = r.get(field)
            out.append(
                f"| {r.get('ticker','')} | {_fmt_cell(r.get('company_name'))} | {_fmt_cell(v)} |"
            )
        out.append("")
        return out

    by_ind: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ind = r.get("industry")
        pe = r.get("pe")
        if ind and pe is not None:
            by_ind[str(ind)].append(float(pe))
    ind_avg = {k: mean(v) for k, v in by_ind.items() if v}
    high_ind = max(ind_avg, key=lambda k: ind_avg[k]) if ind_avg else None
    low_ind = min(ind_avg, key=lambda k: ind_avg[k]) if ind_avg else None

    hold_vals = [float(r["hold_ratio"]) for r in rows if r.get("hold_ratio") is not None]
    max_hold = max(hold_vals) if hold_vals else None

    low5 = top5_pe_low(rows)
    high5 = top5_pe_high(rows)

    parts = [
        "# 横向对比汇总报告",
        "",
        f"- **context as_of 标签（展示用）**: {as_of_label}",
        f"- **股票数**: {len(rows)}",
        "",
        "## 1. 完整对比表",
        "",
        table_body,
        "",
        "## 2. PE：低估 Top5 / 高估 Top5",
        "",
    ]
    parts += md_block("PE 从低到高（样本 Top5）", low5, "PE", "pe")
    parts += md_block("PE 从高到低（样本 Top5）", high5, "PE", "pe")
    parts.extend(["## 3. ROE 质量 Top5", ""])
    parts += md_block("ROE 从高到低", top5_roe(rows), "ROE(%)", "roe")
    parts.extend(["## 4. 北向持股比 Top5", ""])
    parts += md_block("北向持股比(%)", top5_nb(rows), "北向持股比(%)", "hold_ratio")
    parts.extend(["## 5. 融资余额 Top5（亿元）", ""])
    parts += md_block("融资余额(亿)", top5_rz(rows), "融资余额(亿)", "rzye_yi")
    parts.extend(["## 6. 共性观察（仅基于本表提取值）", ""])
    if ind_avg:
        parts.append(
            f"- 按**行业平均 PE**：当前样本中 **{high_ind}** 平均 PE 最高（{ind_avg[high_ind]:.2f}），"
            f"**{low_ind}** 平均 PE 最低（{ind_avg[low_ind]:.2f}）。"
        )
    else:
        parts.append("- 行业平均 PE：数据不足。")
    if hold_vals and max_hold is not None:
        parts.append(
            f"- **北向持股比**：样本内最高为 {max_hold:.2f}% ；共 {len(hold_vals)} 只股票有可解析北向数据。"
        )
    else:
        parts.append("- 北向持股比：无可解析数据。")
    parts.extend([
        "",
        "## 7. 下一步建议",
        "",
        "- 对比表随 context 更新可重复运行 `compare_stock_reports.py`。",
        "- 若需与行情实时对齐，可更新 snapshot 再生成 context。",
        "- 评注规则均为阈值机械分类，非投资建议。",
        "",
    ])
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-dir", type=Path, default=Path("reports/batch/contexts"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/comparison"))
    p.add_argument("--as-of-label", default="2024-12-31", help="Markdown 标题中的 as_of 文案")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_compare(
        context_dir=args.context_dir,
        output_dir=args.output_dir,
        as_of_label=args.as_of_label,
        write_summary=True,
    )
    print(f"[compare_stock_reports] wrote {len(rows)} rows → {args.output_dir}")


if __name__ == "__main__":
    main()
