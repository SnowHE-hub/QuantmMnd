"""P63-3 硬化检验（数字出来后必做）：逐fold稳健 + 流动性分桶 + Diag B + in/OOS.

对 B(full+fnd)/C(no-fnd)/D(fnd-only) 三变体（PIT top-1500）：
 1) 逐 fold neut IC 分布 + 丢最好 2 后均值（多数 fold 正？）
 2) 流动性分桶 IC：top500/mid500-1500/small1500+（63d 关注是否小盘撑起）
 3) Diag B：限制到 v5 的 1373 票看 IC（排除 v6 universe-effect；C/D >0.07 必做）
 4) D 特别：in-sample 因子 vs OOS（引 P63-2 screen：16 survivors 仅 5 个 OOS 可信）
输出 data/bakeoff/p63_3_hardening.json。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.survivorship.p4d_eval_v6 import _v6_context, _per_asof_neut, _adv20_top1500_mask
from scripts.survivorship.p63_3_eval_v6 import _merge_ctx
from scripts.survivorship.p63_3_train_v6 import make_semiannual_cutoffs
from quantmind.data.lake import load_trading_calendar
from quantmind.backtest.wf_split import PurgedWalkForwardSplit

PREDS = REPO / "data/bakeoff/preds"
VARIANTS = {"full_fnd": "B", "full_nofnd": "C", "fnd_only": "D"}


def main():
    panel, lmc = _v6_context()
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    top = _adv20_top1500_mask(as_of_list)
    v5_tickers = set(pd.read_parquet(REPO / "data/panel/alpha_panel_weekly_v5.parquet",
                                     columns=["forward_return_63d"]).index.get_level_values("ticker").astype(str))
    # adv20 排名（分桶）
    px = pd.read_parquet(REPO / "data/raw/alpha_prices_panel_v6.parquet", columns=["ts_code", "trade_date", "amount"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    amt = px.drop_duplicates(["trade_date", "ts_code"]).pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    adv = amt.rolling(20, min_periods=10).mean()
    rk = {}
    for a in as_of_list:
        prior = adv.index[adv.index <= pd.Timestamp(a)]
        rk[pd.Timestamp(a)] = adv.loc[prior[-1]].dropna().rank(ascending=False, method="first") if len(prior) else pd.Series(dtype=float)
    # fold 映射
    cal = load_trading_calendar()
    sp = PurgedWalkForwardSplit(cal, horizon=63, embargo=63, mode="rolling", rolling_lookback_td=756, n_val=2)
    cut = make_semiannual_cutoffs(cal, "2022-04-01", "2025-10-31")
    folds = sp.make_folds(as_of_list, cut, oos_start=pd.Timestamp("2022-01-01"), oos_end=as_of_list[-1])
    asof2fold = {pd.Timestamp(dt): i for i, f in enumerate(folds) for dt in f.test_dates}

    res = {}
    for variant, row in VARIANTS.items():
        df = _merge_ctx(pd.read_parquet(PREDS / f"ridge_{variant}_63d_quarterly_v6.parquet"), panel, lmc)
        df["ticker"] = df["ticker"].astype(str)
        df["_rank"] = df.apply(lambda r: rk[pd.Timestamp(r["as_of"])].get(r["ticker"], np.nan), axis=1)
        dtop = df[df.apply(lambda r: r["ticker"] in top.get(pd.Timestamp(r["as_of"]), set()), axis=1)]

        # 1) 逐 fold
        per = _per_asof_neut(dtop); per["fold"] = per["as_of"].map(lambda a: asof2fold.get(pd.Timestamp(a)))
        fic = per.dropna(subset=["fold"]).groupby("fold")["neu_ic"].mean(); v = fic.values
        fold_d = {"n_folds": int(len(v)), "per_fold": [round(x, 4) for x in np.sort(v)[::-1]],
                  "n_pos": int((v > 0).sum()), "drop_best2": round(float(np.sort(v)[:-2].mean()), 4) if len(v) > 2 else None}

        # 2) 流动性分桶
        bucket = {}
        for name, m in [("top500", df["_rank"] <= 500), ("mid500_1500", (df["_rank"] > 500) & (df["_rank"] <= 1500)),
                        ("small1500+", df["_rank"] > 1500)]:
            p = _per_asof_neut(df[m]); bucket[name] = round(float(p["neu_ic"].mean()), 4)

        # 3) Diag B：限制 v5 1373 票
        dv5 = dtop[dtop["ticker"].isin(v5_tickers)]
        ic_v5 = round(float(_per_asof_neut(dv5)["neu_ic"].mean()), 4)

        res[row] = {"variant": variant, "fold": fold_d, "liquidity_bucket": bucket,
                    "diagB_v5universe_neut_ic": ic_v5, "diagB_n_tickers": int(dv5["ticker"].nunique())}
        print(f"[{row}] {variant}: fold(n={fold_d['n_folds']} pos={fold_d['n_pos']} drop2={fold_d['drop_best2']}) "
              f"buckets={bucket} diagB_v5={ic_v5}", flush=True)

    (REPO / "data/bakeoff/p63_3_hardening.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n[save] data/bakeoff/p63_3_hardening.json")


if __name__ == "__main__":
    main()
