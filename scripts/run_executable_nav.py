"""Executable NAV 回测 CLI（阶段 1：研究→可执行 NAV 闭环）.

对两条 Ridge 种子跑真实成交 NAV 并按 formal gate 判定（executable_nav_design.md §5）。
不接入 Streamlit / FastAPI / daily_update / Agent / PostgreSQL。

用法：
  python scripts/run_executable_nav.py --model 12d          # 只跑 Ridge 12d
  python scripts/run_executable_nav.py --model 63d          # 只跑 Ridge 63d
  python scripts/run_executable_nav.py --model all          # 两者全跑
  python scripts/run_executable_nav.py --model all --dry-run       # 校验输入+打印计划，不写任何文件
  python scripts/run_executable_nav.py --model 12d --report-only   # 用已有 summary 重生成报告，不动 registry
  python scripts/run_executable_nav.py --model all --no-registry   # 跑全量但绝不更新 registry

输出：reports/executable_nav/<model_id>/{nav_daily,holdings_daily,trades,rejected_trades,
targets,filtered,turnover}.parquet + nav_daily.csv + summary.json + gate_report.{json,md}
（独立目录，不覆盖 nav_v4 / sim30d。）

registry 纪律：只有 gate 真实通过且未加 --no-registry/--dry-run/--report-only 时才更新
metrics（gate_pass + executable_nav）；gate_status 不自动升 production。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig  # noqa: E402
from quantmind.execution.nav_gate import (  # noqa: E402
    GateThresholds,
    apply_gate_to_registry,
    evaluate_gate,
    write_gate_reports,
)

V6_PRICES = REPO / "data/raw/alpha_prices_panel_v6.parquet"
V6_PANEL = REPO / "data/panel/alpha_panel_weekly_v6.parquet"

MODELS = {
    "12d": {
        "model_id": "ridge_full_12d_v6_seed",
        "preds": REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet",
        "horizon_td": 12, "rebalance_step": 3,
        "thresholds": GateThresholds(max_drawdown=0.15),
    },
    "63d": {
        "model_id": "ridge_full_fnd_63d_v6_seed",
        "preds": REPO / "data/bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet",
        "horizon_td": 63, "rebalance_step": 13,
        "thresholds": GateThresholds(max_drawdown=0.12),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["12d", "63d", "all"], required=True)
    p.add_argument("--top-n", type=int, default=300,
                   help="组合股票数（默认 300 = PIT top-1500 的 top-quintile，与研究口径对齐）")
    p.add_argument("--universe-size", type=int, default=1500)
    p.add_argument("--out-root", default=str(REPO / "reports/executable_nav"))
    p.add_argument("--dry-run", action="store_true",
                   help="只校验输入并打印执行计划，不写任何文件、不更新 registry")
    p.add_argument("--report-only", action="store_true",
                   help="用已有 summary.json 重生成 gate 报告，不重跑回测、不更新 registry")
    p.add_argument("--no-registry", action="store_true",
                   help="跑全量但不更新 registry（即使 gate 通过）")
    return p.parse_args()


def load_inputs(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = pd.read_parquet(spec["preds"])
    assert set(preds.columns) >= {"as_of", "ticker", "score"}, f"preds 缺列: {preds.columns}"
    prices = pd.read_parquet(
        V6_PRICES,
        columns=["ts_code", "trade_date", "open", "high", "low", "close",
                 "pre_close", "adj_factor", "amount"])
    amihud = pd.read_parquet(V6_PANEL, columns=["amihud_illiquidity"])
    return preds, prices, amihud


def run_model(key: str, args: argparse.Namespace) -> dict:
    spec = MODELS[key]
    model_id = spec["model_id"]
    out_dir = Path(args.out_root) / model_id

    if args.report_only:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        gate = evaluate_gate(summary, spec["thresholds"])
        write_gate_reports(out_dir, model_id, gate, summary)
        print(f"[{key}] report-only：gate_pass={gate['gate_pass']}（registry 未动）")
        return {"model_id": model_id, "gate_pass": gate["gate_pass"], "registry_updated": False}

    preds, prices, amihud = load_inputs(spec)
    n_asof = preds["as_of"].nunique()
    plan = (f"[{key}] model_id={model_id} horizon={spec['horizon_td']}td "
            f"step={spec['rebalance_step']} n_as_of={n_asof} "
            f"n_rebalances≈{(n_asof + spec['rebalance_step'] - 1) // spec['rebalance_step']} "
            f"top_n={args.top_n} universe={args.universe_size} "
            f"preds={spec['preds'].name} prices={V6_PRICES.name}")
    print(plan)
    if args.dry_run:
        print(f"[{key}] dry-run：输入校验通过，未写任何文件。")
        return {"model_id": model_id, "dry_run": True}

    cfg = NavConfig(horizon_td=spec["horizon_td"], rebalance_step=spec["rebalance_step"],
                    top_n=args.top_n, universe_size=args.universe_size)
    engine = ExecutableNavEngine(preds, prices, cfg, amihud=amihud)
    result = engine.run()
    summary = result["summary"]
    summary["inputs"] = {"model_id": model_id, "preds": str(spec["preds"].relative_to(REPO)),
                         "prices": str(V6_PRICES.relative_to(REPO)),
                         "amihud_panel": str(V6_PANEL.relative_to(REPO)),
                         "n_pred_as_of": int(n_asof)}

    out_dir.mkdir(parents=True, exist_ok=True)
    result["nav_daily"].to_parquet(out_dir / "nav_daily.parquet", index=False)
    result["nav_daily"].to_csv(out_dir / "nav_daily.csv", index=False)
    for name in ("holdings_daily", "trades", "rejected_trades", "targets",
                 "filtered", "turnover"):
        df = result[name]
        (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_parquet(
            out_dir / f"{name}.parquet", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    gate = evaluate_gate(summary, spec["thresholds"])
    write_gate_reports(out_dir, model_id, gate, summary)
    reg = apply_gate_to_registry(model_id, gate, summary,
                                 dry_run=args.no_registry)
    n = summary["nav"]
    print(f"[{key}] 完成：净年化 {n['ann_return_net']:+.2%} 基准 {n['ann_return_bench']:+.2%} "
          f"净超额 {n['ann_net_excess']:+.2%} IR {n['information_ratio']:.2f} "
          f"MaxDD {n['max_drawdown_net']:.2%} → gate_pass={gate['gate_pass']} "
          f"registry_updated={reg['registry_updated']}")
    print(f"[{key}] 输出目录：{out_dir}")
    return {"model_id": model_id, "gate_pass": gate["gate_pass"],
            "registry_updated": reg["registry_updated"], "summary": summary}


def main() -> None:
    args = parse_args()
    keys = ["12d", "63d"] if args.model == "all" else [args.model]
    results = [run_model(k, args) for k in keys]
    print("\n=== 汇总 ===")
    for r in results:
        if r.get("dry_run"):
            print(f"  {r['model_id']}: dry-run OK")
        else:
            print(f"  {r['model_id']}: gate_pass={r['gate_pass']} "
                  f"registry_updated={r['registry_updated']}")


if __name__ == "__main__":
    main()
