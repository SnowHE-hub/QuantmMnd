"""scripts/line_a/export_reference_contracts.py — 从冻结代码导出主线 A v2 的"原样参考策略"契约（T-035 P0）.

参考臂必须与研究期的策略逐项相同（股票池、信号、调仓、持有期、持仓数、权重、成本、成交规则）。
本脚本不手写这些参数，而是直接从冻结的模块里读出来，并记录源文件的 sha256 和 git 提交，
使契约与代码可以逐项核对。任何一个源文件的哈希变了，契约就要重新导出并重新评审。

用法（仓库根目录）：
    PYTHONPATH=. python scripts/line_a/export_reference_contracts.py --out configs/line_a_v2/reference_strategies.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs/work/evidence/T-030"))

FROZEN_SOURCES = [
    "docs/work/evidence/T-030/eval_primary.py",
    "quantmind/execution/nav_engine.py",
    "quantmind/backtest/wf_costs.py",
    "quantmind/research/gpu/ensemble.py",
    "quantmind/research/gpu/folds.py",
    "docs/work/evidence/T-030/build_daily_base.py",
    "docs/work/evidence/T-030/build_model_ready.py",
    "docs/work/evidence/T-034/forward_bundle_fold18.py",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import eval_primary as ep  # noqa: E402  (frozen T-030 evaluation module)

    from quantmind.backtest import wf_costs
    from quantmind.execution.nav_engine import NavConfig

    nav_defaults = {f.name: (f.default if f.default is not dataclasses.MISSING else None)
                    for f in dataclasses.fields(NavConfig) if f.name != "tiers"}
    tiers = dataclasses.asdict(wf_costs.SlippageTiers())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()

    strategies = {}
    for h in ("20d", "63d"):
        hold_td, step, nav_block = ep.NAV[h]
        common = {
            "horizon": h,
            "universe": {
                "entry_rule": "PIT 入池：上市满 120 个交易日且未退市（build_daily_base.py）",
                "pool": f"按时点的 adv20 前 {nav_defaults['universe_size']}（NavConfig.universe_size；只用决策日及之前的数据）",
                "decision_filters": {"min_recent_bar_td": nav_defaults["min_recent_bar_td"],
                                     "adv20_min_periods": nav_defaults["adv20_min_periods"]},
            },
            "decision_grid": "v6_1 周频网格（folds.calendar_and_grid / build_asof_grid）；测试 as_of 来自 make_folds(方案 C)",
            "rebalance_step_weekly_periods": step,
            "holding_td": hold_td,
            "portfolio": {"top_n": nav_defaults["top_n"], "weighting": "equal（nav_engine：target_weight_each = 1/n）"},
            "costs": {"commission_bp_each_side": nav_defaults["commission_bp"],
                      "transfer_bp_each_side": nav_defaults["transfer_bp"],
                      "stamp_duty_sell": {"before": wf_costs.STAMP_DUTY_BEFORE, "after": wf_costs.STAMP_DUTY_AFTER,
                                          "change_date": str(wf_costs.STAMP_DUTY_HALVING_DATE.date())},
                      "slippage_tiers_bp": tiers},
            "execution": {"engine": "quantmind.execution.nav_engine.ExecutableNavEngine",
                          "delist_writeoff_td": nav_defaults["delist_writeoff_td"],
                          "rules": "T+1 成交；涨停不可买、跌停不可卖、停牌无 bar 不成交；未成交买单到下一再平衡日撤单"},
            "nav_paired_block_days": nav_block,
        }
        strategies[f"ridge_native_{h}"] = dict(common, signal={
            "model": "Ridge C（α = 10，方案 C 连续样本外）",
            "historical_predictions": f"data/{ep.BASE[h]}"})
        strategies[f"ens_native_{h}"] = dict(common, signal={
            "model": "ENS：四族等权截面排名平均（ensemble.rank_average_frames）",
            "members": list(ep.ENS_MEMBERS),
            "historical_predictions": f"data/gpu_v1/preds/<member>_{h}/fold_*.parquet",
            "forward_bundle": "data/gpu_v1/forward_models/20d_fold18/" if h == "20d" else None,
            "forward_bundle_note": None if h == "20d" else "63d 还没有冻结的前瞻模型包，前瞻打分不可用，需要另行登记"})

    out = {
        "what": "主线 A v2 原样参考策略契约（从冻结代码导出，不手写参数）",
        "exported_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "source_sha256": {p: _sha(REPO / p) for p in FROZEN_SOURCES},
        "strategies": strategies,
        "rules": [
            "参考臂只读本契约；改动任何一项就是挑战臂，必须单独登记",
            "20 日与 63 日分开比较；不把 ENS 20 日对 Ridge 63 日的差异称为算法增益",
            "源文件哈希变化时契约失效，需要重新导出并评审",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {a.out} strategies={list(strategies)}")


if __name__ == "__main__":
    main()
