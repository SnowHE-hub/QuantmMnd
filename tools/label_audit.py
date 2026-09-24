# -*- coding: utf-8 -*-
"""label_audit.py — 用与复权基准无关的真值审计面板的 forward_return 标签。

真值：快照 prices.parquet 的 pct_change（相对除权参考价的日收益）按交易日连乘。
horizon 定义与 build_full_panel.compute_forward_returns 相同：
以 as_of 当天或之前最后一个交易日为基准，取合并交易日历上其后第 h 个交易日为目标日。

用法（在 WSL 的仓库根目录）：
    python tools/label_audit.py --panel data/panel/alpha_panel_v4.parquet
    python tools/label_audit.py --panel data/panel/alpha_panel_v4_1.parquet \
        --min-spearman 0.999 --max-bad-frac 0.001      # T-003 验收门槛
退出码：0 = 全部截面满足门槛；1 = 不满足；2 = 输入错误。
只读，不写任何文件。
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd


def load_prices(snap_root: str, tickers: set[str]) -> pd.DataFrame:
    parts = []
    for p in sorted(glob.glob(f"{snap_root}/*/prices.parquet")):
        d = pd.read_parquet(p, columns=["trade_date", "ticker", "close", "pct_change"])
        d = d[d["ticker"].isin(tickers)]
        d["snap"] = p.replace("\\", "/").split("/")[-2]
        parts.append(d)
    if not parts:
        raise SystemExit(2)
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    return px


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--horizons", default="21,63")
    ap.add_argument("--bad-threshold", type=float, default=0.005, help="|存储−真值| 超过多少算错（默认 0.5%）")
    ap.add_argument("--min-spearman", type=float, default=None, help="每个截面 Spearman(存储, 真值) 的下限")
    ap.add_argument("--max-bad-frac", type=float, default=None, help="全样本错误占比上限")
    a = ap.parse_args()
    hs = [int(x) for x in a.horizons.split(",")]

    cols = [f"forward_return_{h}d" for h in hs]
    panel = pd.read_parquet(a.panel, columns=cols)
    tickers = set(panel.index.get_level_values("ticker"))
    px = load_prices(a.snapshots, tickers)

    incons = px.groupby(["trade_date", "ticker"])["pct_change"].nunique()
    print(f"[真值自检] pct_change 跨快照不一致: {(incons > 1).sum()} / {len(incons):,}")

    px = px.sort_values("snap", kind="stable").sort_values(["trade_date", "ticker"], kind="stable")
    pv = px.drop_duplicates(["trade_date", "ticker"], keep="last")
    close = pv.pivot(index="trade_date", columns="ticker", values="close").sort_index()
    growth = (1 + pv.pivot(index="trade_date", columns="ticker", values="pct_change") / 100) \
        .reindex(close.index).fillna(1.0)
    cum = growth.cumprod()

    rows = []
    for as_of in sorted(panel.index.get_level_values("as_of").unique()):
        ob = close.index[close.index <= pd.Timestamp(as_of)]
        if not len(ob):
            continue
        base = ob[-1]
        fut = close.index[close.index > base]
        for h in hs:
            if len(fut) < h:
                continue
            tgt = fut[h - 1]
            truth = cum.loc[tgt] / cum.loc[base] - 1
            stored = panel.xs(as_of, level="as_of")[f"forward_return_{h}d"]
            df = pd.DataFrame({"stored": stored, "true": truth.reindex(stored.index)}).dropna()
            df = df[close.loc[tgt].reindex(df.index).notna()]
            if len(df) < 5:
                continue
            err = (df["stored"] - df["true"]).abs()
            rows.append({"h": h, "as_of": pd.Timestamp(as_of).date(), "n": len(df),
                         "spearman": df["stored"].rank().corr(df["true"].rank()),
                         "bad_frac": float((err > a.bad_threshold).mean()),
                         "bad10_frac": float((err > 0.10).mean()),
                         "max_err": float(err.max())})
    if not rows:
        print("没有可比对的截面"); return 2
    r = pd.DataFrame(rows)
    for h in hs:
        x = r[r.h == h]
        tot_bad = (x.bad_frac * x.n).sum() / x.n.sum()
        print(f"\n[h={h}d] 截面 {len(x)} · 样本 {x.n.sum():,} · 错误占比(>{a.bad_threshold:.1%}) {tot_bad:.3%} · "
              f"Spearman 均值 {x.spearman.mean():.4f} 最低 {x.spearman.min():.4f}")
    print("\n最差 8 个截面：")
    print(r.sort_values("spearman").head(8).round(4).to_string(index=False))

    ok = True
    if a.min_spearman is not None and (r.spearman < a.min_spearman).any():
        print(f"\nFAIL: {(r.spearman < a.min_spearman).sum()} 个截面 Spearman < {a.min_spearman}"); ok = False
    if a.max_bad_frac is not None:
        tot = (r.bad_frac * r.n).sum() / r.n.sum()
        if tot > a.max_bad_frac:
            print(f"\nFAIL: 全样本错误占比 {tot:.4%} > {a.max_bad_frac:.4%}"); ok = False
    if a.min_spearman is not None or a.max_bad_frac is not None:
        print("\nPASS" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
