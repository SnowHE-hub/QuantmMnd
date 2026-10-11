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
from pathlib import Path

import numpy as np
import pandas as pd

Coverage = list[tuple[pd.DatetimeIndex, frozenset]]


def load_prices(snap_root: str, tickers: set[str], extra_daily: str | None = None) -> tuple[pd.DataFrame, Coverage]:
    """快照（以及可选的日线长表）的 (trade_date, ticker, close, pct_change, snap)，按 tickers 过滤；
    另返回每个来源**过滤前**的覆盖：(该来源有数据的交易日, 该来源出现过的股票)。
    日线长表的 snap 记为 "_daily"，排序时排在快照之前，同日同票以快照为准。"""
    parts, coverage = [], []

    def take(d: pd.DataFrame, snap: str) -> None:
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        coverage.append((pd.DatetimeIndex(d["trade_date"].unique()).sort_values(), frozenset(d["ticker"].unique())))
        d = d[d["ticker"].isin(tickers)].copy()
        d["snap"] = snap
        parts.append(d)

    if extra_daily:
        d = pd.read_parquet(extra_daily, columns=["trade_date", "ts_code", "close", "pct_chg"])
        take(d.rename(columns={"ts_code": "ticker", "pct_chg": "pct_change"}), "_daily")
    for p in sorted(glob.glob(f"{snap_root}/*/prices.parquet")):
        take(pd.read_parquet(p, columns=["trade_date", "ticker", "close", "pct_change"]),
             p.replace("\\", "/").split("/")[-2])
    if not parts:
        raise SystemExit(2)
    return pd.concat(parts, ignore_index=True), coverage


def trading_axis(coverage: Coverage, calendar: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    """交易日轴 = 各来源有数据的日期（不按股票过滤）∪ 来源区间内的独立交易日历；与面板票池无关。"""
    src = coverage[0][0]
    for dates, _ in coverage[1:]:
        src = src.union(dates)
    if calendar is None:
        return src
    inside = src[(src >= calendar.min()) & (src <= calendar.max())]
    bad = inside.difference(calendar)
    if len(bad):
        raise SystemExit(f"来源中有 {len(bad)} 个不在交易日历里的日期，例如 {list(bad[:5])}")
    return src.union(calendar[(calendar >= src.min()) & (calendar <= src.max())])


def observed_mask(coverage: Coverage, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """某票某日是否被至少一个来源覆盖：该来源当天有数据、且该票出现在该来源里。
    覆盖范围内无行 = 停牌（收益 0）；不被任何来源覆盖 = 缺数据，相关样本不参与比较。"""
    arr = np.zeros((len(index), len(columns)), dtype=bool)
    colpos = {c: i for i, c in enumerate(columns)}
    for dates, names in coverage:
        pos = index.get_indexer(dates)
        rows = pos[pos >= 0]
        cols = [colpos[t] for t in names if t in colpos]
        if len(rows) and cols:
            arr[np.ix_(rows, cols)] = True
    return pd.DataFrame(arr, index=index, columns=columns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--horizons", default="21,63")
    ap.add_argument("--bad-threshold", type=float, default=0.005, help="|存储−真值| 超过多少算错（默认 0.5%%）")
    ap.add_argument("--min-spearman", type=float, default=None, help="每个截面 Spearman(存储, 真值) 的下限")
    ap.add_argument("--max-bad-frac", type=float, default=None, help="全样本错误占比上限")
    ap.add_argument("--extra-daily", default=None,
                    help="可选：日线长表（trade_date, ts_code, close, pct_chg），补快照未覆盖的 (日, 票)")
    ap.add_argument("--trade-calendar", default="data/lake/trade_cal_sse.parquet",
                    help="独立交易日历（列 cal_date）；不存在时交易日轴只由各来源的日期（不按股票过滤）构成")
    a = ap.parse_args()
    hs = [int(x) for x in a.horizons.split(",")]

    cols = [f"forward_return_{h}d" for h in hs]
    panel = pd.read_parquet(a.panel, columns=cols)
    tickers = set(panel.index.get_level_values("ticker"))
    px, coverage = load_prices(a.snapshots, tickers, a.extra_daily)
    calendar = None
    if a.trade_calendar and Path(a.trade_calendar).is_file():
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(a.trade_calendar)["cal_date"])).sort_values()
    else:
        print(f"[日历] 未找到 {a.trade_calendar}：交易日轴只用来源日期")

    incons = px.groupby(["trade_date", "ticker"])["pct_change"].nunique()
    print(f"[真值自检] pct_change 跨快照不一致: {(incons > 1).sum()} / {len(incons):,}")

    px = px.sort_values("snap", kind="stable").sort_values(["trade_date", "ticker"], kind="stable")
    pv = px.drop_duplicates(["trade_date", "ticker"], keep="last")
    axis = trading_axis(coverage, calendar)
    close = pv.pivot(index="trade_date", columns="ticker", values="close").reindex(axis)
    growth = (1 + pv.pivot(index="trade_date", columns="ticker", values="pct_change") / 100) \
        .reindex(axis).fillna(1.0)
    cum = growth.cumprod()
    observed = observed_mask(coverage, close.index, close.columns)
    print(f"[日历] 交易日轴 {len(axis)} 天；没有任何来源有数据的天数 {int((~observed.any(axis=1)).sum())}")

    rows = []
    n_uncovered = 0
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
            covered = observed.loc[(observed.index > base) & (observed.index <= tgt)].all(axis=0)
            stored = panel.xs(as_of, level="as_of")[f"forward_return_{h}d"]
            df = pd.DataFrame({"stored": stored, "true": truth.reindex(stored.index)}).dropna()
            df = df[close.loc[tgt].reindex(df.index).notna()]
            ok = covered.reindex(df.index).fillna(False).astype(bool)
            n_uncovered += int((~ok).sum())
            df = df[ok]
            if len(df) < 5:
                continue
            err = (df["stored"] - df["true"]).abs()
            rows.append({"h": h, "as_of": pd.Timestamp(as_of).date(), "n": len(df),
                         "spearman": df["stored"].rank().corr(df["true"].rank()),
                         "bad_frac": float((err > a.bad_threshold).mean()),
                         "bad10_frac": float((err > 0.10).mean()),
                         "max_err": float(err.max())})
    if not rows:
        print("没有可比对的截面")
        return 2
    r = pd.DataFrame(rows)
    for h in hs:
        x = r[r.h == h]
        tot_bad = (x.bad_frac * x.n).sum() / x.n.sum()
        print(f"\n[h={h}d] 截面 {len(x)} · 样本 {x.n.sum():,} · 错误占比(>{a.bad_threshold:.1%}) {tot_bad:.3%} · "
              f"Spearman 均值 {x.spearman.mean():.4f} 最低 {x.spearman.min():.4f}")
    print(f"\n[覆盖] 窗口内有未被任何来源覆盖的交易日、因而不参与比较的样本：{n_uncovered:,}"
          + ("（可用 --extra-daily 补充日线长表）" if n_uncovered and not a.extra_daily else ""))
    print("\n最差 8 个截面：")
    print(r.sort_values("spearman").head(8).round(4).to_string(index=False))

    ok = True
    if a.min_spearman is not None and (r.spearman < a.min_spearman).any():
        print(f"\nFAIL: {(r.spearman < a.min_spearman).sum()} 个截面 Spearman < {a.min_spearman}")
        ok = False
    if a.max_bad_frac is not None:
        tot = (r.bad_frac * r.n).sum() / r.n.sum()
        if tot > a.max_bad_frac:
            print(f"\nFAIL: 全样本错误占比 {tot:.4%} > {a.max_bad_frac:.4%}")
            ok = False
    if a.min_spearman is not None or a.max_bad_frac is not None:
        print("\nPASS" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
