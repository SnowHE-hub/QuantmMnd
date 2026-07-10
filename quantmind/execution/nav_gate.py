"""quantmind.execution.nav_gate — Executable NAV formal gate 判定（设计 §5）.

判定标准（`executable_nav_design.md` §5，阈值为设计文档建议值，全部可配置并写入报告）：
  1. 年化净超额（真实成交后） ≥ +5%
  2. 最大回撤 ≤ 15%（12d）/ ≤ 12%（63d）
  3. 超额信息比率 ≥ 1.0
  4. 换手可承受 —— 设计未定阈值：默认仅报告不阻断（设计缺口，报告中注明）
  5. 分年净超额 > 0（"或可解释"属人工评审，本引擎不允许人工强制通过 → 任一年 ≤0 即 fail）

registry 纪律：
  - **fail → registry 一个字节都不改**（保持 research_candidate_pending_nav）；
  - pass → 仅更新该 model_id 的 metrics（executable_nav 结果 + gate_pass=True）；
    gate_status **不自动升 production**（"签收"是人工决策，见设计 §5/registry 注释）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    """formal gate 阈值（默认=设计文档 §5 建议值）。"""
    min_ann_net_excess: float = 0.05
    max_drawdown: float = 0.15           # 12d 默认；63d 调 0.12
    min_information_ratio: float = 1.0
    max_ann_twoway_turnover: float | None = None   # 设计未定阈 → None=仅报告
    require_yearly_positive: bool = True


def evaluate_gate(summary: dict, thresholds: GateThresholds) -> dict[str, Any]:
    """对 NAV summary 应用 gate。返回机器可读判定（含逐项依据）。"""
    navm = summary["nav"]
    to = summary["turnover"]
    checks: list[dict] = []

    def check(name: str, actual: float, op: str, threshold, passed: bool, note: str = ""):
        checks.append({"criterion": name, "actual": actual, "op": op,
                       "threshold": threshold, "passed": bool(passed), "note": note})

    check("ann_net_excess", navm["ann_net_excess"], ">=", thresholds.min_ann_net_excess,
          navm["ann_net_excess"] >= thresholds.min_ann_net_excess)
    check("max_drawdown_net", navm["max_drawdown_net"], ">=", -thresholds.max_drawdown,
          navm["max_drawdown_net"] >= -thresholds.max_drawdown)
    check("information_ratio", navm["information_ratio"], ">=",
          thresholds.min_information_ratio,
          navm["information_ratio"] >= thresholds.min_information_ratio)
    if thresholds.max_ann_twoway_turnover is not None:
        check("ann_twoway_turnover", to["annualized_twoway"], "<=",
              thresholds.max_ann_twoway_turnover,
              to["annualized_twoway"] <= thresholds.max_ann_twoway_turnover)
    else:
        check("ann_twoway_turnover", to["annualized_twoway"], "<=", None, True,
              "设计未定阈值 → 仅报告不阻断（设计缺口）")
    if thresholds.require_yearly_positive:
        yearly = navm["yearly_net_excess"]
        bad = {y: v for y, v in yearly.items() if v <= 0}
        check("yearly_net_excess_all_positive", yearly, "all>0", 0.0, len(bad) == 0,
              f"负年份: {bad}" if bad else "")

    passed = all(c["passed"] for c in checks)
    return {"gate_pass": passed, "checks": checks,
            "thresholds": asdict(thresholds),
            "evaluated_at": datetime.now().isoformat(timespec="seconds")}


def apply_gate_to_registry(model_id: str, gate_result: dict, nav_summary: dict,
                           *, dry_run: bool = False) -> dict[str, Any]:
    """仅 gate_pass=True 时更新 registry（metrics 附加 executable_nav + gate_pass）。

    fail 或 dry_run → 不触碰 registry 文件。gate_status 一律不自动改（升 production 属人工签收）。
    """
    from quantmind.contracts import model_registry as MR  # noqa: N812

    if not gate_result["gate_pass"] or dry_run:
        return {"registry_updated": False,
                "reason": "dry_run" if dry_run else "gate_failed_keep_pending_nav"}

    rec = MR.get(model_id)
    rec.metrics = dict(rec.metrics)
    rec.metrics["gate_pass"] = True
    rec.metrics["executable_nav"] = {
        "ann_net_excess": nav_summary["nav"]["ann_net_excess"],
        "max_drawdown_net": nav_summary["nav"]["max_drawdown_net"],
        "information_ratio": nav_summary["nav"]["information_ratio"],
        "ann_return_net": nav_summary["nav"]["ann_return_net"],
        "annualized_twoway_turnover": nav_summary["turnover"]["annualized_twoway"],
        "total_cost_nav_units": nav_summary["costs"]["total_cost_nav_units"],
        "period": nav_summary["period"],
        "evaluated_at": gate_result["evaluated_at"],
    }
    MR.register(rec)
    return {"registry_updated": True, "model_id": model_id}


def write_gate_reports(out_dir: Path, model_id: str, gate_result: dict,
                       summary: dict) -> None:
    """机器可读 JSON + 人读 Markdown。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": model_id, "gate": gate_result, "nav_summary": summary}
    (out_dir / "gate_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [f"# Executable NAV Gate 报告 — `{model_id}`", "",
             f"- 判定时间：{gate_result['evaluated_at']}",
             f"- **gate_pass = {gate_result['gate_pass']}**", "",
             "## 逐项判定", "",
             "| 指标 | 实际值 | 条件 | 阈值 | 通过 | 备注 |",
             "|---|---|---|---|---|---|"]
    for c in gate_result["checks"]:
        actual = c["actual"]
        if isinstance(actual, dict):
            actual = "; ".join(f"{k}:{v:+.2%}" for k, v in actual.items())
        elif isinstance(actual, float):
            actual = f"{actual:+.4f}"
        lines.append(f"| {c['criterion']} | {actual} | {c['op']} | {c['threshold']} "
                     f"| {'✅' if c['passed'] else '❌'} | {c.get('note','')} |")
    n = summary["nav"]
    lines += ["", "## NAV 摘要", "",
              f"- 区间：{summary['period']['start']} → {summary['period']['end']}"
              f"（{summary['period']['n_trading_days']} 交易日，{summary['period']['n_rebalances']} 次再平衡）",
              f"- 净年化 {n['ann_return_net']:+.2%} ｜ 毛年化 {n['ann_return_gross']:+.2%}"
              f" ｜ 基准年化 {n['ann_return_bench']:+.2%}",
              f"- **年化净超额 {n['ann_net_excess']:+.2%}** ｜ IR {n['information_ratio']:.2f}"
              f" ｜ 净 MaxDD {n['max_drawdown_net']:.2%}",
              f"- 年化单边换手 {summary['turnover']['annualized_oneway']:.2f}"
              f" ｜ 累计成本 {summary['costs']['total_cost_nav_units']:.4f}（NAV 单位）",
              f"- 拒单事件 {summary['execution']['n_rejection_events']}："
              f"{summary['execution']['rejection_reason_counts']}",
              "", "## 设计缺口（保守实现，见 nav_engine 模块 docstring）", "",
              "- Top-N/基准取研究层同口径（top-quintile of PIT top-1500 / 同池等权）；",
              "- 换手阈值设计未定 → 仅报告不阻断；",
              "- ST 无 PIT 标记 → 板块阈值 + 一字板兜底；",
              "- gate 通过也**不自动升 production**（人工签收）。"]
    (out_dir / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = ["GateThresholds", "evaluate_gate", "apply_gate_to_registry", "write_gate_reports"]
