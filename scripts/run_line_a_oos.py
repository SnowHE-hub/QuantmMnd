"""scripts/run_line_a_oos.py — 主线 A 的严格样本外分数与两种口径的 NAV（T-010）.

README 首页的"v6 HRP 净 Sharpe 0.996"用单个模型（lgbm_v6_alpha，序贯验证第 4 轮，
train_end=2024-12-31）给 2019–2026 全部截面打分，其中 24 次调仓落在训练期内。
本脚本对每个测试截面 t 只用 t 之前、且标签窗口在 t 之前结束的截面训练，再评估：

  legacy      run_nav_backtest 的口径：Top-N、扁平单边 13bp、无成交约束（与 0.996 同口径）
  executable  ExecutableNavEngine：T+1 开盘成交、涨跌停/停牌拒单、分层滑点、时变印花税、PIT adv20 选股池

切分（每个测试截面 t = dates[i]）：
  val   = dates[i-gap-n_val : i-gap]
  train = dates[: i-gap-n_val]
gap=1 个季度：63 交易日标签约一个季度，gap 保证训练/验证样本的标签在 t 之前结束。

用法（仓库根目录）：
  python scripts/run_line_a_oos.py --out reports/line_a_oos/run1
输出目录已存在则拒绝运行（不覆盖任何产物）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PANEL_REL = "data/panel/alpha_panel_v4_1.parquet"
META_REL = "models/lgbm_v6_alpha.meta.json"
V4_PRICES_REL = "data/raw/alpha_prices_panel.parquet"
V6_PRICES_REL = "data/raw/alpha_prices_panel_v6.parquet"
V6_PANEL_REL = "data/panel/alpha_panel_weekly_v6.parquet"
LABEL_TD = 63


@dataclass(frozen=True)
class OOSFold:
    test: pd.Timestamp
    train: tuple[pd.Timestamp, ...]
    val: tuple[pd.Timestamp, ...]


def oos_folds(
    dates: list[pd.Timestamp], *, gap: int = 1, n_val: int = 2, min_train: int = 8,
) -> list[OOSFold]:
    """为每个可测截面生成 (train, val, test)。train 与 val 都在 test 之前至少 gap 个截面."""
    if gap < 0 or n_val < 1 or min_train < 1:
        raise ValueError("gap>=0, n_val>=1, min_train>=1")
    dates = sorted(set(pd.Timestamp(d) for d in dates))
    folds = []
    for i, t in enumerate(dates):
        val_end = i - gap
        val_start = val_end - n_val
        if val_start < min_train:
            continue
        folds.append(OOSFold(t, tuple(dates[:val_start]), tuple(dates[val_start:val_end])))
    return folds


def label_ends_before(
    as_of: pd.Timestamp, test: pd.Timestamp, calendar: pd.DatetimeIndex, horizon_td: int = LABEL_TD,
) -> bool:
    """as_of 的 h 日标签目标日是否严格早于 test 的基准日（PIT：test 当日收盘决策）."""
    base = calendar[calendar <= as_of]
    if len(base) == 0:
        return True
    pos = calendar.get_loc(base[-1])
    if pos + horizon_td >= len(calendar):
        return False
    test_base = calendar[calendar <= test][-1]
    return calendar[pos + horizon_td] <= test_base


def fit_and_score(
    panel: pd.DataFrame, fold: OOSFold, features: list[str], label: str, params: dict,
) -> pd.Series:
    from quantmind.models.factor_model import build_lgbm_arrays
    from quantmind.models.lgbm_ranker import LGBMRankerModel

    X, y, g = build_lgbm_arrays(panel, list(fold.train), features, label, 5)
    Xv, yv, gv = build_lgbm_arrays(panel, list(fold.val), features, label, 5)
    model = LGBMRankerModel(auto_flip=False, **params)
    model.fit(X, y, groups=g, X_val=Xv, y_val=yv, groups_val=gv, feature_names=features)
    xs = panel.xs(fold.test, level="as_of")
    x = xs[features].fillna(0.0).to_numpy(dtype=np.float32)
    return pd.Series(model.predict(x), index=xs.index.astype(str), name="score")


def rank_ic(scores: pd.Series, labels: pd.Series) -> float:
    df = pd.DataFrame({"s": scores, "y": labels}).dropna()
    if len(df) < 20:
        return float("nan")
    return float(df["s"].rank().corr(df["y"].rank()))


def run_oos(panel: pd.DataFrame, features: list[str], label: str, params: dict,
            gap: int, n_val: int, min_train: int,
            calendar: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, list[dict]]:
    dates = sorted(panel.index.get_level_values("as_of").unique())
    rows, log = [], []
    for fold in oos_folds(dates, gap=gap, n_val=n_val, min_train=min_train):
        if calendar is not None:
            late = [d for d in fold.train + fold.val if not label_ends_before(d, fold.test, calendar)]
            if late:
                raise AssertionError(f"test={fold.test.date()}: labels of {late} end after the test date")
        s = fit_and_score(panel, fold, features, label, params)
        y = panel.xs(fold.test, level="as_of")[label]
        y.index = y.index.astype(str)
        ic = rank_ic(s, y)
        rows.append(pd.DataFrame({"as_of": fold.test, "ticker": s.index, "score": s.values}))
        log.append({"test": str(fold.test.date()), "n_train": len(fold.train),
                    "train_end": str(fold.train[-1].date()), "val": [str(d.date()) for d in fold.val],
                    "n_scored": int(len(s)), "rank_ic": ic})
        print(f"  test={fold.test.date()} train_end={fold.train[-1].date()} "
              f"val={[str(d.date()) for d in fold.val]} n={len(s)} RankIC={ic:+.4f}", flush=True)
    return pd.concat(rows, ignore_index=True), log


def legacy_nav(panel: pd.DataFrame, scores: pd.DataFrame, top: int, weight: str,
               cost_bps: float, data_root: Path) -> dict:
    from scripts.run_nav_backtest import build_daily_nav, calc_metrics, load_price_wide

    asofs = sorted(scores["as_of"].unique())
    sub = panel[panel.index.get_level_values("as_of").isin(asofs)]
    sc = scores.set_index(["as_of", "ticker"])["score"].reindex(sub.index)
    price_wide = load_price_wide(data_root / V4_PRICES_REL)
    nav, _, turnover_log = build_daily_nav(sub, sc, price_wide, top, weight_method=weight,
                                           kelly_fraction=0.5, cost_bps=cost_bps)
    m = calc_metrics(nav, rf=0.03, turnover_log=turnover_log, cost_bps=cost_bps)
    return {"metrics": m, "nav": nav}


def executable_nav(scores: pd.DataFrame, top: int, universe_size: int, data_root: Path) -> dict:
    from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig

    prices = pd.read_parquet(data_root / V6_PRICES_REL, columns=["ts_code", "trade_date", "open", "high", "low",
                                                 "close", "pre_close", "adj_factor", "amount"])
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(prices["trade_date"]).unique()))
    p = scores.copy()
    # 引擎只在交易日决策：季末日历日 → 最后一个 ≤ as_of 的交易日（与标签基准日一致）
    p["as_of"] = [cal[cal <= pd.Timestamp(a)][-1] for a in p["as_of"]]
    amihud = pd.read_parquet(data_root / V6_PANEL_REL, columns=["amihud_illiquidity"])
    week = pd.DatetimeIndex(sorted(amihud.index.get_level_values("as_of").unique()))
    parts = []
    for d in sorted(p["as_of"].unique()):
        prior = week[week <= d]
        if len(prior) == 0:
            continue
        cs = amihud.xs(prior[-1], level="as_of").copy()
        cs.index = pd.MultiIndex.from_product([[d], cs.index], names=["as_of", "ticker"])
        parts.append(cs)
    amihud_q = pd.concat(parts) if parts else None
    cfg = NavConfig(horizon_td=LABEL_TD, rebalance_step=1, top_n=top, universe_size=universe_size)
    result = ExecutableNavEngine(p, prices, cfg, amihud=amihud_q).run()
    return result


def main(argv: list[str] | None = None) -> int:
    global V4_PRICES_REL, V6_PRICES_REL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO,
                    help="data/ 与 models/ 所在目录（worktree 缺被 gitignore 的数据时指向主 checkout）")
    ap.add_argument("--panel", type=Path, default=None, help=f"默认 <data-root>/{PANEL_REL}")
    ap.add_argument("--model-meta", type=Path, default=None,
                    help="取 feature_cols、label_col 与超参的来源（默认 lgbm_v6_alpha）")
    ap.add_argument("--features-from-pkl", type=Path, default=None,
                    help="用该模型 pkl 的特征列表代替 meta 的 feature_cols（线上 lgbm_v6_alpha.pkl 为 38 个，meta 为 42 个）")
    ap.add_argument("--label", default=None, help="默认取 meta 的 label_col")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--min-train", type=int, default=8)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--universe-size", type=int, default=1500)
    ap.add_argument("--cost-bps", type=float, default=13.0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-executable", action="store_true")
    ap.add_argument("--v4-prices", default=V4_PRICES_REL,
                    help="旧口径 NAV 与交易日历用的价格表（相对 data-root 或绝对路径）")
    ap.add_argument("--v6-prices", default=V6_PRICES_REL, help="真实成交 NAV 用的价格表")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    V4_PRICES_REL, V6_PRICES_REL = a.v4_prices, a.v6_prices
    root = a.data_root
    a.panel = a.panel or root / PANEL_REL
    a.model_meta = a.model_meta or root / META_REL
    out = a.out if a.out.is_absolute() else REPO / a.out
    if out.exists():
        print(f"refusing to overwrite existing output dir: {out}")
        return 1
    meta = json.loads(a.model_meta.read_text())
    features, label = list(meta["feature_cols"]), a.label or meta["label_col"]
    if a.features_from_pkl is not None:
        from quantmind.models.factor_model import FactorModel
        features = list(FactorModel.load(a.features_from_pkl)._feature_names)
    mp = meta.get("model_params", {})
    params = {"n_estimators": mp.get("n_estimators", 400), "num_leaves": mp.get("num_leaves", 31),
              "learning_rate": mp.get("learning_rate", 0.05),
              "early_stopping_rounds": mp.get("early_stopping_rounds", 50),
              "n_jobs": a.n_jobs, "random_state": 42}
    panel = pd.read_parquet(a.panel)
    missing = [c for c in features + [label] if c not in panel.columns]
    if missing:
        print(f"panel lacks columns: {missing[:5]}")
        return 1

    print(f"[line_a_oos] panel={a.panel.name} features={len(features)} label={label} "
          f"gap={a.gap} n_val={a.n_val} min_train={a.min_train} params={params}")
    px_dates = pd.read_parquet(root / V4_PRICES_REL, columns=["trade_date"])["trade_date"]
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(px_dates).unique()))
    scores, fold_log = run_oos(panel, features, label, params, a.gap, a.n_val, a.min_train, calendar)
    out.mkdir(parents=True)
    scores.to_parquet(out / "oos_scores.parquet", index=False)

    ics = pd.Series([f["rank_ic"] for f in fold_log], dtype=float).dropna()
    summary: dict = {
        "inputs": {"panel": str(a.panel), "model_meta": str(a.model_meta), "label": label,
                   "n_features": len(features), "gap": a.gap, "n_val": a.n_val,
                   "min_train": a.min_train, "params": params,
                   "v4_prices": V4_PRICES_REL, "v6_prices": V6_PRICES_REL},
        "folds": fold_log,
        "rank_ic": {"n": int(len(ics)), "mean": float(ics.mean()), "std": float(ics.std()),
                    "icir": float(ics.mean() / ics.std()) if ics.std() > 0 else None,
                    "win_rate": float((ics > 0).mean())},
    }
    for w in ("equal", "hrp"):
        r = legacy_nav(panel, scores, a.top, w, a.cost_bps, root)
        r["nav"].to_csv(out / f"legacy_nav_{w}.csv")
        summary[f"legacy_{w}"] = r["metrics"]
        print(f"  legacy {w}: ann={r['metrics']['ann_return']:+.2%} sharpe={r['metrics']['sharpe_ratio']:.3f} "
              f"mdd={r['metrics']['max_drawdown']:.2%}")
    if not a.skip_executable:
        res = executable_nav(scores, a.top, a.universe_size, root)
        res["nav_daily"].to_csv(out / "executable_nav_daily.csv", index=False)
        for name in ("trades", "rejected_trades", "filtered", "turnover"):
            df = res[name]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_parquet(out / f"exec_{name}.parquet", index=False)
        summary["executable"] = res["summary"]
        n = res["summary"]["nav"]
        print(f"  executable: net ann={n['ann_return_net']:+.2%} bench={n['ann_return_bench']:+.2%} "
              f"excess={n['ann_net_excess']:+.2%} IR={n['information_ratio']:.2f} mdd={n['max_drawdown_net']:.2%}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(f"[line_a_oos] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
