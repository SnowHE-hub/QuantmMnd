"""P4d：v6 上 Ridge(full) 预测的两口径评估（quantmind env）.

复用 scripts/bakeoff/evaluate_bakeoff（IC / 含成本净超额 / 换手 / DD），但：
  (1) panel 上下文 = v6；(2) 中性化加 UNKNOWN 行业桶（缺行业不剔除，归显式 dummy）；
  (3) 两口径：① 全市场 ② PIT 流动性 top-1500（每 as_of 按 adv20 降序，trailing-20td 成交额，≤as_of）；
  (4) regime-conditional neut IC（market_momentum_60d 三档，build_regime，OOS 2022-01-01）；
  (5) 逐 as_of neut IC 分布分位。
输出 data/bakeoff/p4_verdict_v6.json + 打印对照表（含 v5 基线行）。判定对 gate 线，不与 v5 攀比。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import scripts.bakeoff.evaluate_bakeoff as EB
from scripts.bakeoff.evaluate_bakeoff import _spear, score_predictions
from scripts._diag_wf_v2 import build_regime

V6_PANEL = REPO / "data/panel/alpha_panel_weekly_v6.parquet"
V6_PRICES = REPO / "data/raw/alpha_prices_panel_v6.parquet"
PREDS = REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet"
DIRS = REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.dirs.json"


# ---- UNKNOWN 行业桶中性化（替换 evaluate_bakeoff._neutralize_date）----
def _neutralize_unknown(g):
    d = g.copy()
    d["industry"] = d["industry"].astype("object").where(d["industry"].notna(), "UNKNOWN")
    d = d.dropna(subset=["score", "realized", "log_mktcap"])      # 不对 industry dropna
    if len(d) < 30 or d["industry"].nunique() < 2:
        return None
    D = pd.get_dummies(d["industry"], drop_first=True).astype(float)  # UNKNOWN 作为显式 dummy
    z = (d["log_mktcap"] - d["log_mktcap"].mean()) / (d["log_mktcap"].std() + 1e-12)
    X = np.column_stack([np.ones(len(d)), z.values, D.values])
    y = d["score"].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    raw = _spear(pd.Series(y, index=d.index), d["realized"])
    neu = _spear(pd.Series(resid, index=d.index), d["realized"])
    return pd.Series({"raw_ic": raw, "neu_ic": neu, "n": len(d)})


EB._neutralize_date = _neutralize_unknown   # 打补丁（score_predictions 内部引用模块全局）


def _v6_context():
    panel = pd.read_parquet(V6_PANEL)
    db = pd.read_parquet(REPO / "data/lake/daily_basic.parquet",
                         columns=["ts_code", "trade_date", "circ_mv"]).rename(columns={"ts_code": "ticker"})
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    db = db[db["circ_mv"] > 0]
    db["log_mktcap"] = np.log(db["circ_mv"].astype(float))
    lmc = {pd.Timestamp(d): s.set_index("ticker")["log_mktcap"] for d, s in db.groupby("trade_date")}
    return panel, lmc


def _merge_ctx(preds, panel, lmc):
    df = preds.copy(); df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.merge(panel["forward_return_12d"].rename("realized"),
                  left_on=["as_of", "ticker"], right_index=True, how="left")
    df = df.merge(panel["exposure_industry"].rename("industry"),
                  left_on=["as_of", "ticker"], right_index=True, how="left")
    df = df.merge(panel["amihud_illiquidity"].rename("amihud"),
                  left_on=["as_of", "ticker"], right_index=True, how="left")
    rows = []
    for a, sub in df.groupby("as_of"):
        s = lmc.get(pd.Timestamp(a).normalize())
        rows.append(sub.assign(log_mktcap=sub["ticker"].map(s) if s is not None else np.nan))
    return pd.concat(rows, ignore_index=True)


def _adv20_top1500_mask(as_of_list):
    """每 as_of 按 trailing-20td 成交额(amount)降序取 top-1500（PIT：只用 ≤as_of）。返回 set 字典."""
    px = pd.read_parquet(V6_PRICES, columns=["ts_code", "trade_date", "amount"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    amt = (px.drop_duplicates(["trade_date", "ts_code"])
           .pivot(index="trade_date", columns="ts_code", values="amount").sort_index())
    adv20 = amt.rolling(20, min_periods=10).mean()
    out = {}
    for a in as_of_list:
        a = pd.Timestamp(a)
        prior = adv20.index[adv20.index <= a]
        if len(prior) == 0:
            out[a] = set(); continue
        row = adv20.loc[prior[-1]].dropna()
        out[a] = set(row.sort_values(ascending=False).head(1500).index)
    return out


def _per_asof_neut(df):
    per = df.groupby("as_of").apply(_neutralize_unknown, include_groups=False).dropna(how="all").reset_index()
    per["neu_ic"] = per["neu_ic"].astype(float); per["raw_ic"] = per["raw_ic"].astype(float)
    return per


def _regime_breakdown(per, regime):
    per = per.copy(); per["regime"] = per["as_of"].map(lambda a: regime.get(pd.Timestamp(a), "unknown"))
    return {r: round(float(per.loc[per.regime == r, "neu_ic"].mean()), 4)
            for r in ("bear", "neutral", "bull")}, per["regime"].value_counts().to_dict()


def _eval_lens(df, panel, lmc, regime, name):
    m = score_predictions(df[["as_of", "ticker", "score"]].copy(), period="12d", panel=panel, lmc=lmc)
    per = _per_asof_neut(df)
    reg, reg_n = _regime_breakdown(per, regime)
    q = per["neu_ic"].quantile([.25, .5, .75]).round(4).tolist()
    return {"lens": name, "n_tickers_total": int(df["ticker"].nunique()),
            "raw_ic": m["raw_ic"], "neut_ic": m["neut_ic"], "neut_icir": m["neut_icir"],
            "neut_ic_posfrac": m["neut_ic_posfrac"], "net_excess_cost": m["net_excess_cost"],
            "avg_oneway_turnover": m["avg_oneway_turnover"], "max_drawdown": m["max_drawdown"],
            "regime_neut_ic": reg, "regime_counts": reg_n,
            "per_asof_neut_ic_p25_50_75": q, "n_oos_dates": m["n_oos_dates"]}


def main():
    panel, lmc = _v6_context()
    preds = pd.read_parquet(PREDS)
    df = _merge_ctx(preds, panel, lmc)
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    regime, reg_meta = build_regime(panel, as_of_list)
    train_s = json.loads(DIRS.read_text()).get("train_seconds")

    res = {"model": "ridge_full_12d_quarterly_v6", "train_seconds": train_s,
           "regime_cut": {"q33": round(reg_meta["q33"], 4), "q67": round(reg_meta["q67"], 4)},
           "lenses": []}
    # 口径①全市场
    res["lenses"].append(_eval_lens(df, panel, lmc, regime, "full_market"))
    # 口径②PIT top-1500
    top = _adv20_top1500_mask(as_of_list)
    df2 = df[df.apply(lambda r: r["ticker"] in top.get(pd.Timestamp(r["as_of"]), set()), axis=1)]
    res["lenses"].append(_eval_lens(df2, panel, lmc, regime, "pit_top1500"))

    (REPO / "data/bakeoff/p4_verdict_v6.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
