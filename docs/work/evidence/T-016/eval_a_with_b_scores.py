"""T-016 研究评估：主线 A 的选股改用主线 B 的打分，在 T-010 同一真实成交口径下的绩效（不改任何生产代码）.

打分来源
  A_own  T-010 主结果的样本外分数（v4_1 标签、42 特征的 LightGBM，严格 walk-forward）
  B_seed 主线 B `ridge_full_fnd_63d_v6_seed` 的样本外预测（`data/bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet`，
         周频、全市场；方向只由验证集决定，见 `scripts/bakeoff/p3b_tabular.py:102`）。
         A 的每个季度调仓日取 B **最近一个 ≤ 调仓日**的 as_of；相隔超过 14 个自然日的调仓日不纳入。

变体（每个都在 `ExecutableNavEngine` 下评估：T+1 开盘、涨跌停/停牌拒单、分层滑点、时变印花税、PIT adv20 前 1500、
Top 30 等权、季度调仓；基准 = 同一选股池等权；价格用 T-015 的 `alpha_prices_panel_v6_20261009.parquet`）
  1 A_own                 A 自己的分数（同一调仓日子集）——对照
  2 B_on_A_candidates     B 的分数，只在 A 当期有分数的同一批股票里选（只换打分模型，候选集合不变）
  3 B_on_static_1374      B 的分数，限定在 A 的静态 1374 票池
  4 B_full_market_top30   B 的分数，全市场（引擎选股池仍是 PIT adv20 前 1500）

用法（t/016-eval 工作区根目录，需要 t/010 的 scripts/run_line_a_oos.py）：
  PYTHONPATH=. python docs/work/evidence/T-016/eval_a_with_b_scores.py --data-root ~/projects/quantmind \
      --a-scores docs/work/evidence/T-010/run_v4_1/oos_scores.csv --out docs/work/evidence/T-016/result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import scripts.run_line_a_oos as lao

B_PREDS = "data/bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet"
UNIVERSE = "data/alpha_universe/alpha_universe.txt"
V6_PRICES = "data/raw/alpha_prices_panel_v6_20261009.parquet"
MAX_STALE_DAYS = 14


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def align_b(b: pd.DataFrame, dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[dict]]:
    """每个 A 调仓日取 B 最近一个 ≤ 该日的 as_of；过旧的调仓日剔除。返回 (as_of=A 调仓日 的分数, 对齐记录)."""
    b_dates = pd.DatetimeIndex(sorted(b["as_of"].unique()))
    parts, log = [], []
    for d in dates:
        prior = b_dates[b_dates <= d]
        if not len(prior) or (d - prior[-1]).days > MAX_STALE_DAYS:
            log.append({"a_date": str(d.date()), "b_as_of": str(prior[-1].date()) if len(prior) else None,
                        "included": False})
            continue
        cs = b[b["as_of"] == prior[-1]][["ticker", "score"]].copy()
        cs["as_of"] = d
        parts.append(cs)
        log.append({"a_date": str(d.date()), "b_as_of": str(prior[-1].date()), "included": True,
                    "stale_days": int((d - prior[-1]).days), "n_scored": int(len(cs))})
    return pd.concat(parts, ignore_index=True), log


def summarize(res: dict) -> dict:
    n = res["summary"]["nav"]
    return {"ann_net": n["ann_return_net"], "ann_bench": n["ann_return_bench"], "excess": n["ann_net_excess"],
            "ir": n["information_ratio"], "mdd": n["max_drawdown_net"], "yearly_excess": n["yearly_net_excess"],
            "turnover_oneway": res["summary"]["turnover"]["annualized_oneway"],
            "cost_drag": res["summary"]["costs"]["cost_drag_ann"], "period": res["summary"]["period"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--a-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.data_root
    lao.V6_PRICES_REL = V6_PRICES

    a_sc = pd.read_csv(a.a_scores, parse_dates=["as_of"])
    b = pd.read_parquet(root / B_PREDS)
    b["as_of"] = pd.to_datetime(b["as_of"])
    b_sc, align_log = align_b(b, sorted(a_sc["as_of"].unique()))
    dates = sorted(b_sc["as_of"].unique())
    a_sub = a_sc[a_sc["as_of"].isin(dates)]
    static = set(Path(root / UNIVERSE).read_text().split())

    on_a = b_sc.merge(a_sub[["as_of", "ticker"]], on=["as_of", "ticker"], how="inner")
    variants = {
        "A_own": a_sub[["as_of", "ticker", "score"]],
        "B_on_A_candidates": on_a,
        "B_on_static_1374": b_sc[b_sc["ticker"].isin(static)],
        "B_full_market_top30": b_sc,
    }
    out = {"inputs": {"a_scores": str(a.a_scores), "a_scores_sha256": sha(a.a_scores),
                      "b_preds": B_PREDS, "b_preds_sha256": sha(root / B_PREDS),
                      "v6_prices": V6_PRICES, "v6_prices_sha256": sha(root / V6_PRICES),
                      "static_universe": UNIVERSE, "static_universe_size": len(static)},
           "alignment": align_log, "rebalance_dates": [str(d.date()) for d in dates],
           "candidates_per_date": {k: v.groupby("as_of").size().describe()[["min", "50%", "max"]].to_dict()
                                   for k, v in variants.items()},
           "results": {}}
    for name, sc in variants.items():
        res = lao.executable_nav(sc[["as_of", "ticker", "score"]], 30, 1500, root)
        out["results"][name] = summarize(res)
        r = out["results"][name]
        print(f"{name:22s} net {r['ann_net']:+.4f} bench {r['ann_bench']:+.4f} excess {r['excess']:+.4f} "
              f"IR {r['ir']:+.3f} MDD {r['mdd']:+.3f}", flush=True)
    # 同一批调仓日上 A 与 B 的截面秩相关（只看 A 有分数的票）
    m = a_sub.merge(b_sc, on=["as_of", "ticker"], suffixes=("_a", "_b"))
    out["rank_corr_a_vs_b"] = m.groupby("as_of").apply(
        lambda g: g["score_a"].rank().corr(g["score_b"].rank()), include_groups=False).round(4).astype(float).to_dict()
    out["rank_corr_a_vs_b"] = {str(k.date()): v for k, v in out["rank_corr_a_vs_b"].items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
