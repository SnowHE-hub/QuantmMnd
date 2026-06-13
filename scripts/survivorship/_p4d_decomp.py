"""P4d 决策诊断（非调参，纯分解）：v6 neut IC 高于 v5 的来源拆解.

问题：v6 neut IC(0.052-0.057) > v5(0.034)，与"上界拿掉应更低"相反。
假设：退市票真实暴跌标签 = 可预测的崩盘簇，抬高 IC。
做法：PIT top-1500 与全市场，分别报 (a)全部行 (b)剔除"horizon内将退市"行 的 neut IC。
若剔除后仍≥0.02 → 信号在真实可交易票上站得住；差额 = 退市规避 alpha。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.survivorship.p4d_eval_v6 import (
    _v6_context, _merge_ctx, _adv20_top1500_mask, _neutralize_unknown, _per_asof_neut)

V6_PRICES = REPO / "data/raw/alpha_prices_panel_v6.parquet"
SB_FULL = REPO / "data/lake/stock_basic_full.parquet"
PREDS = REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet"


def main():
    panel, lmc = _v6_context()
    df = _merge_ctx(pd.read_parquet(PREDS), panel, lmc)
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())

    # 每票末交易日（v6 价格末日）
    px = pd.read_parquet(V6_PRICES, columns=["ts_code", "trade_date"])
    last_trade = px.groupby("ts_code")["trade_date"].max()
    last_trade.index = last_trade.index.astype(str)
    cal = np.array(sorted(px["trade_date"].unique()))

    def td_gap(a, t):  # a→t 之间交易日数（t 为末交易日）
        lt = last_trade.get(t)
        if pd.isna(lt):
            return 10**9
        i = np.searchsorted(cal, np.datetime64(pd.Timestamp(a)))
        j = np.searchsorted(cal, np.datetime64(pd.Timestamp(lt)))
        return j - i

    sb = pd.read_parquet(SB_FULL)
    delisted = set(sb.loc[sb.list_status == "D", "ts_code"].astype(str))
    df["ticker"] = df["ticker"].astype(str)
    # "horizon内将退市"：退市票 且 末交易日距 as_of ≤ 12 交易日（12d 标签捕捉到崩盘）
    df["delist_in_h"] = df.apply(
        lambda r: (r["ticker"] in delisted) and (0 <= td_gap(r["as_of"], r["ticker"]) <= 12), axis=1)
    n_dih = int(df["delist_in_h"].sum())

    def neut_ic(d):
        per = _per_asof_neut(d)
        return round(float(per["neu_ic"].mean()), 4), len(per)

    top = _adv20_top1500_mask(as_of_list)
    df_top = df[df.apply(lambda r: r["ticker"] in top.get(pd.Timestamp(r["as_of"]), set()), axis=1)]

    print(f"horizon内将退市行: 全市场={n_dih} ({n_dih/len(df):.3%})  top1500={int(df_top['delist_in_h'].sum())}")
    for name, d in [("full_market", df), ("pit_top1500", df_top)]:
        ic_all, n_all = neut_ic(d)
        ic_clean, n_cl = neut_ic(d[~d["delist_in_h"]])
        print(f"\n[{name}] neut IC 全部={ic_all} (n_asof={n_all})  剔除将退市行后={ic_clean} (n_asof={n_cl})  "
              f"Δ={round(ic_all-ic_clean,4)}")


if __name__ == "__main__":
    main()
